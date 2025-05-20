from utils import log_util
from vis import show_single
from torch.utils.tensorboard import SummaryWriter
import datetime
import os
from torchvision import transforms
import torch
from PIL import Image
import numpy as np
import random

from OCC.Display.OCCViewer import rgb_color
from OCC.Core.V3d import  V3d_DirectionalLight
from OCC.Core.Quantity import Quantity_Color, Quantity_TOC_RGB, Quantity_NOC_BLACK
from OCC.Core.Graphic3d import Graphic3d_NameOfMaterial

from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeEdge, BRepBuilderAPI_MakeWire, BRepBuilderAPI_MakeFace
from OCC.Core.gp import gp_Pnt, gp_Vec, gp_Dir
from OCC.Core.GProp import GProp_GProps
from OCC.Core.BRepGProp import brepgprop
from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeBox
from OCC.Core.BRepExtrema import BRepExtrema_DistShapeShape
from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Fuse, BRepAlgoAPI_Common
from OCC.Core.Bnd import Bnd_Box
from OCC.Core.BRepBndLib import brepbndlib
from OCC.Display.SimpleGui import init_display
from OCC.Core.TopAbs import TopAbs_FACE
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
from OCC.Core.Quantity import Quantity_NOC_RED
from OCC.Core.Image import Image_AlienPixMap
from OpenGL.GL import glReadPixels, GL_DEPTH_COMPONENT, GL_FLOAT, GL_COLOR_INDEX, GL_RGB
from OCC.Display.OCCViewer import Viewer3d
from OCC.Core.AIS import AIS_ColoredShape
from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeBox
from OCC.Display.OCCViewer import rgb_color
from OCC.Display.SimpleGui import init_display
from OCC.Extend.TopologyUtils import TopologyExplorer

from vis.show_single import show_BRep
from cadlib.Brep_utils import get_BRep_from_file, get_wireframe

from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
import time
from  math import *
import numpy
import sys


def shape_to_normal_image(shape, resolution=(512, 512)) -> Image.Image:
    # 三角剖分形状
    from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
    from OCC.Core.BRep import BRep_Tool
    mesh = BRepMesh_IncrementalMesh(shape, 0.001)
    mesh.Perform()

    explorer = TopExp_Explorer(shape, TopAbs_FACE)
    normals = []

    while explorer.More():
        face = explorer.Current()
        triangulation = BRep_Tool.Triangulation(face, None)
        if triangulation is None:
            explorer.Next()
            continue

        nodes = triangulation.Nodes()
        triangles = triangulation.Triangles()
        for i in range(1, triangulation.NbTriangles()+1):
            tri = triangles.Value(i)
            n1, n2, n3 = tri.Get()
            p1 = nodes.Value(n1)
            p2 = nodes.Value(n2)
            p3 = nodes.Value(n3)

            v1 = np.array([p1.X(), p1.Y(), p1.Z()])
            v2 = np.array([p2.X(), p2.Y(), p2.Z()])
            v3 = np.array([p3.X(), p3.Y(), p3.Z()])

            # 三角面法向量
            normal = np.cross(v2 - v1, v3 - v1)
            normal = normalize(normal)
            normals.append(normal)

        explorer.Next()

    # 将法向量映射为颜色图像（简化示意，仅取平均法向量）
    avg_normal = normalize(np.mean(normals, axis=0))
    rgb = ((avg_normal + 1) / 2 * 255).astype(np.uint8)
    img = Image.new("RGB", resolution, color=tuple(rgb))
    return img


def triangulate_shape(shape, deflection=0.1):
    from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
    from OCC.Extend.TopologyUtils import TopologyExplorer
    BRepMesh_IncrementalMesh(shape, deflection).Perform()
    verts = []
    tris = []

    for face in TopologyExplorer(shape).faces():
        location = face.Location()
        triangulation = face.Triangulation(location)

        if triangulation is None:
            continue

        nodes = triangulation.Nodes()
        triangles = triangulation.Triangles()

        for i in range(1, nodes.Size() + 1):
            pnt = nodes.Value(i)
            verts.append((pnt.X(), pnt.Y(), pnt.Z()))

        for i in range(1, triangles.Size() + 1):
            tri = triangles.Value(i)
            # Note: OpenCascade uses 1-based indices
            tris.append((
                tri.Value(1) - 1,
                tri.Value(2) - 1,
                tri.Value(3) - 1
            ))

    return np.array(verts, dtype=np.float32), np.array(tris, dtype=np.uint32)


def display_surface_normals(shape, display, scale=5.0, samples_u=5, samples_v=5):
    '''计算并可视化法向量'''
    exp = TopExp_Explorer(shape, TopAbs_FACE)
    while exp.More():
        face = topods_Face(exp.Current())
        surface_adaptor = BRepAdaptor_Surface(face)

        u_min, u_max = surface_adaptor.FirstUParameter(), surface_adaptor.LastUParameter()
        v_min, v_max = surface_adaptor.FirstVParameter(), surface_adaptor.LastVParameter()

        for i in range(samples_u):
            for j in range(samples_v):
                u = u_min + (u_max - u_min) * i / (samples_u - 1)
                v = v_min + (v_max - v_min) * j / (samples_v - 1)

                try:
                    pnt = surface_adaptor.Value(u, v)
                    normal = surface_adaptor.Normal(u, v)
                    vec = gp_Vec(normal.XYZ())
                    vec.Scale(scale)

                    ais_vec = AIS_Vector(pnt, gp_Dir(vec))
                    ais_vec.SetColor(Quantity_NOC_RED)
                    display.Context.Display(ais_vec, False)

                except Exception as e:
                    # 有些点可能在几何边界上无法正确计算法向量，跳过
                    print(f"Warning: Failed at u={u}, v={v} -> {e}")

        exp.Next()


def create_2_box():
    p1_1 = gp_Pnt(0.0, 0.0, 0.0)
    p1_2 = gp_Pnt(1.0, 1.0, 1.0)
    box1 = BRepPrimAPI_MakeBox(p1_1, p1_2)
    shape1 = box1.Shape()

    p2_1 = gp_Pnt(0.0, 0.0, 0.0)
    p2_2 = gp_Pnt(2.0, 2.0, 2.0)    
    box2 = BRepPrimAPI_MakeBox(p2_1, p2_2)
    shape2 = box2.Shape()

    return shape1, shape2


def create_AABB_box(shape):
    bbox = Bnd_Box()
    brepbndlib.Add(shape, bbox, True)
    xmin, ymin, zmin, xmax, ymax, zmax = bbox.Get()
    lower = gp_Pnt(xmin, ymin, zmin)
    upper = gp_Pnt(xmax, ymax, zmax)  
    print(xmax-xmin, ymax-ymin, zmax-zmin)
    return BRepPrimAPI_MakeBox(lower, upper).Shape()


def get_faces_and_normals(shape):
    from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
    from OCC.Core.TopLoc import TopLoc_Location
    from OCC.Core.BRep import BRep_Tool

    # 先进行三角剖分
    mesh = BRepMesh_IncrementalMesh(shape, 0.0001)
    mesh.Perform()

    faces_info = []

    exp = TopExp_Explorer(shape, TopAbs_FACE)
    while exp.More():
        face = exp.Current()
        loc = TopLoc_Location()
        triangulation = BRep_Tool.Triangulation(face, loc)
        if triangulation is None:
            exp.Next()
            continue

        nb_nodes = triangulation.NbNodes()
        points = [triangulation.Node(i + 1) for i in range(nb_nodes)]
        triangles = triangulation.Triangles()

        for i in range(triangulation.NbTriangles()):
            tri = triangles.Value(i + 1)
            n1, n2, n3 = tri.Get()
            p1 = points[n1 - 1]
            p2 = points[n2 - 1]
            p3 = points[n3 - 1]

            # 构建顶点坐标
            v1 = np.array([p1.X(), p1.Y(), p1.Z()])
            v2 = np.array([p2.X(), p2.Y(), p2.Z()])
            v3 = np.array([p3.X(), p3.Y(), p3.Z()])

            # 计算法线
            normal = np.cross(v3 - v1, v2 - v1)
            normal = normal / np.linalg.norm(normal)
            color = normal / 2.0 +0.5

            faces_info.append({
                'verts': [v1, v2, v3],
                'normal': normal,
                'color': color
            })
        exp.Next()
    return faces_info


import numpy as np

def look_at(eye, target, up=[0, 0, 1]):
    eye = np.array(eye, dtype=np.float32)
    target = np.array(target, dtype=np.float32)
    up = np.array(up, dtype=np.float32)

    forward = target - eye
    forward /= np.linalg.norm(forward)

    right = np.cross(forward, up)
    right /= np.linalg.norm(right)

    true_up = np.cross(right, forward)

    view = np.eye(4, dtype=np.float32)
    view[0, :3] = right
    view[1, :3] = true_up
    view[2, :3] = -forward
    view[:3, 3] = eye

    return view



def render_faces_to_image(faces_info, image_path="rendered.png", resolution=(512, 512)):
    import numpy as np
    import trimesh
    import pyrender
    from PIL import Image
    vertices = []
    faces = []
    face_colors = []

    vertex_count = 0
    for face in faces_info:
        v = face['verts']
        c = face['color']
        vertices.extend(v)
        faces.append([vertex_count, vertex_count+1, vertex_count+2])
        vertex_count += 3

        rgb = (np.clip(c * 255, 0, 255)).astype(np.uint8)
        face_colors.append(rgb.tolist() + [255])  # RGBA

    vertices = np.array(vertices)
    faces = np.array(faces)
    face_colors = np.array(face_colors, dtype=np.uint8)

    # 创建 mesh
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    mesh.visual.face_colors = face_colors

    # 转为 pyrender Mesh
    pyrender_mesh = pyrender.Mesh.from_trimesh(mesh, smooth=False)

    # 创建场景
    scene = pyrender.Scene()
    scene.add(pyrender_mesh)

    # 摄像机看向原点
    camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
    cam_pose = look_at(
        eye=[1, 1, 1],
        target=[0, 0, 0],
        up=[0, 0, 1]
    )
    scene.add(camera, pose=cam_pose)

    # 添加光源
    light = pyrender.DirectionalLight(color=np.ones(3), intensity=2.0)
    scene.add(light, pose=cam_pose)

    # 渲染
    r = pyrender.OffscreenRenderer(*resolution)
    color, _ = r.render(scene)
    r.delete()

    # 保存
    Image.fromarray(color).save(image_path)
    print(f"已保存图像: {image_path}")


def build_plane_frame(normal):
    # 以 normal 为法向量，在该平面上建立局部 u,v 坐标系
    normal = normal / np.linalg.norm(normal)
    if np.allclose(normal, [0, 0, 1]):
        u = np.array([1, 0, 0])
    else:
        u = np.cross(normal, [0, 0, 1])
        u /= np.linalg.norm(u)
    v = np.cross(normal, u)
    return u, v


def project_point_to_plane(P, plane_normal, point_on_plane):
    """
    P: 要投影的3D点
    plane_normal: 平面法向量 (A, B, C)，必须是单位向量
    point_on_plane: 平面上的任意一点
    """
    P = np.array(P)
    n = np.array(plane_normal)
    n = n / np.linalg.norm(n)  # 单位化
    v = P - point_on_plane
    distance = np.dot(v, n)
    projection = P - distance * n
    return projection


def project_to_custom_plane(verts3d, plane_normal=(1, 1, 1), plane_offset=3):
    plane_normal = np.array(plane_normal)
    point_on_plane = plane_normal / np.dot(plane_normal, plane_normal) * plane_offset
    u, v = build_plane_frame(plane_normal)

    points_2d = []
    for P in verts3d:
        P_proj = project_point_to_plane(P, plane_normal, point_on_plane)
        local_u = np.dot(P_proj - point_on_plane, u)
        local_v = np.dot(P_proj - point_on_plane, v)
        points_2d.append((local_u, local_v))
    return points_2d


# def project_to_2d(verts3d):
#     return [(v[0], v[2]) for v in verts3d]

def project_to_2d(verts3d):
    return project_to_custom_plane(verts3d, plane_normal=(1, 1, 1), plane_offset=3)


def normal_to_rgb(normal):
    # normal 是一个长度为3的 numpy 向量，值域为 [-1, 1]
    rgb = ((normal + 1) / 2 * 255).astype(np.uint8)  # 映射到 [0, 255]
    return tuple(int(c) for c in rgb[::-1]) 

def render_normal_map(faces, img_size=(512, 512)):
    import cv2
    img = np.zeros((img_size[1], img_size[0], 3), dtype=np.uint8)  # 创建图片

    # 坐标变换（将模型投影到图像中心）
    all_pts = np.concatenate([np.array(project_to_2d(f['verts'])) for f in faces])  # 所有点坐标
    min_xy = all_pts.min(axis=0)
    max_xy = all_pts.max(axis=0)
    dxy = max_xy - min_xy
    scale = min(img_size[0] / (max_xy[0] - min_xy[0] + 1e-5),
                img_size[1] / (max_xy[1] - min_xy[1] + 1e-5)) * 0.75  # 最长包围边缩放到图像大小的0.75

    offset = (np.array(img_size) - dxy * scale)  / 2.0 


    for f in faces:
        pts2d = np.array(project_to_2d(f['verts']))
        pts2d = (pts2d - min_xy) * scale
        pts2d = offset + pts2d  

        pts_int = np.array(pts2d, dtype=np.int32)
        color = normal_to_rgb(f['normal'])

        cv2.fillConvexPoly(img, pts_int, color) # 填充颜色

    return img


def visualize_with_open3d(faces_info):
    import open3d as o3d
    # 创建Open3D三角形网格对象
    mesh = o3d.geometry.TriangleMesh()
    
    # 收集所有顶点
    vertices = np.vstack([face['verts'] for face in faces_info])
    # 创建三角形索引 (每3个顶点构成一个三角形)
    triangles = np.array([[i, i+1, i+2] for i in range(0, len(vertices), 3)])
    
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(triangles)
    
    # 计算顶点法线 (可选)
    mesh.compute_vertex_normals()
    
    # 可视化
    o3d.visualization.draw_geometries([mesh])


def visualize_with_plt(faces_info):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # 为每个三角形创建多边形
    for face in faces_info:
        verts = face['verts']
        color = face['color']
        # 创建三角形多边形
        tri = Poly3DCollection([verts], alpha=1.0)
        # 设置颜色 (需要确保颜色值在0-1之间)
        tri.set_color(color) # 黑色边线
        ax.add_collection3d(tri)
    
    # 自动缩放视图
    ax.auto_scale_xyz([v[0] for face in faces_info for v in face['verts']],
                     [v[1] for face in faces_info for v in face['verts']],
                     [v[2] for face in faces_info for v in face['verts']])
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.axis('off')
    plt.show()


def visualize_with_matplotlib(faces_info):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # 收集所有三角形
    triangles = []
    for face in faces_info:
        triangles.append(face['verts'])
    
    # 创建多边形集合
    poly = Poly3DCollection(triangles, alpha=0.8)
    ax.add_collection3d(poly)
    
    # 设置坐标轴范围
    all_verts = np.vstack([face['verts'] for face in faces_info])
    ax.auto_scale_xyz(all_verts[:,0], all_verts[:,1], all_verts[:,2])
    
    # 可选：显示法线
    # for face in faces_info[:100]:  # 限制显示的法线数量以免太密集
    #     centroid = np.mean(face['verts'], axis=0)
    #     ax.quiver(*centroid, *face['normal'], length=0.1, color='r')
    
    plt.tight_layout()
    plt.show()



def test_AABB():
    '''测试并可视化包围盒'''
    name = '00902881'
    file_dir = "/home/lkh/siga/dataset/deepcad/data/cad_json/"
    file_dir = os.path.join(file_dir, name[:4])
    file_path = os.path.join(file_dir, name+'.json')
    shape = get_BRep_from_file(file_path)
    aabb_box = create_AABB_box(shape)

    aabb_wires = get_wireframe(aabb_box)
    
    display, start_display, _, _ = init_display()

    for edge in aabb_wires:
        display.DisplayShape(edge, update=True, color="black")


    display.DisplayShape(shape, update=True)
    start_display()


def test_volume():
    '''测试相交体积'''
    shape1, shape2 = create_2_box()
    body = BRepAlgoAPI_Common(shape1, shape2).Shape()
    props = GProp_GProps()
    brepgprop.VolumeProperties(body, props)
    print(props.Mass())


def test_shape_distance():
    '''测试两个shape之间的最短距离'''
    '''创建两个box作为shape'''
    shape1, shape2 = create_2_box()

    # 计算两个box的最小距离
    dist = BRepExtrema_DistShapeShape(shape1, shape2)
    if dist.IsDone() and dist.Value() < 1e-6:  # 考虑浮点误差
        print(dist.Value())
    else: print(dist.Value())


def test_normal():
    name = '00902881'
    file_dir = "/home/lkh/siga/dataset/deepcad/data/cad_json/"
    file_dir = os.path.join(file_dir, name[:4])
    file_path = os.path.join(file_dir, name+'.json')
    shape = get_BRep_from_file(file_path)

    display, start_display, _, _ = init_display()
    display.DisplayShape(shape, update=True)

    # 添加法向量显示

    start_display()


def test_depth():
    '''测试计算深度图'''
    import os

    name = '00902881'
    file_dir = "/home/lkh/siga/dataset/deepcad/data/cad_json/"
    file_dir = os.path.join(file_dir, name[:4])
    file_path = os.path.join(file_dir, name+'.json')
    shape = get_BRep_from_file(file_path)

    display, start_display, _, _ = init_display()
    display.DisplayShape(shape, update=False)
    # 提取 depth map
    depth_map = render_depth_map(display, width=640, height=480)

    if depth_map is not None:
        import matplotlib.pyplot as plt
        plt.imshow(depth_map, cmap='gray')
        plt.title('Depth Map')
        plt.colorbar()
        plt.show()

    start_display()


def test_show():
    '''测试并可视化包围盒'''
    from OCC.Core.Image import Image_PixMap
    from OCC.Core.V3d import V3d_View
    from OCC.Core.Image import Image_PixMap, Image_Format

    shape, _ = create_2_box()

    offscreen_renderer = Viewer3d()  # 离线渲染
    offscreen_renderer.Create()  # 初始化
    # 设置渲染模式为阴影模式，显示面信息
    offscreen_renderer.SetModeShaded()
    offscreen_renderer.SetSize(512, 512)



    from OCC.Core.Quantity import Quantity_Color, Quantity_TOC_RGB, Quantity_NOC_BLACK
    bg_color = Quantity_Color(1, 1, 1, Quantity_TOC_RGB)
    offscreen_renderer.View.SetBgGradientColors(bg_color,bg_color)
    offscreen_renderer.DisplayShape(shape, update=True)


    offscreen_renderer.View.SetEye(10, 10, 10)  
    offscreen_renderer.View.SetAt(1, 1, 1)  
    offscreen_renderer.View.SetScale(50)


    offscreen_renderer.Repaint()
    pix_map = Image_PixMap()
    success = offscreen_renderer.View.DumpZBuffer(pix_map)

    width, height = 512, 512

    pixel_data = np.zeros((height, width), dtype=np.float32)
    glReadPixels(
        0, 0,                  # 起始坐标
        width, height,         # 读取区域大小
        GL_DEPTH_COMPONENT,                # 格式：RGB
        GL_FLOAT,      # 数据类型：0-255
        pixel_data             # 目标缓冲区
    )
    pixel_data = np.flipud(pixel_data)
    image = Image.fromarray(pixel_data, 'RGB')
    image.save('/home/lkh/siga/CADIMG/test/output.png')

    offscreen_renderer.View.Dump('/home/lkh/siga/CADIMG/test/output0.png')


def test_mesh():
    from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
    from OCC.Core.BRep import BRep_Tool

    name = '00004108'
    file_dir = "/home/lkh/siga/dataset/deepcad/data/cad_json/"
    file_dir = os.path.join(file_dir, name[:4])
    file_path = os.path.join(file_dir, name+'.json')
    shape, _ = create_2_box()

    offscreen_renderer = Viewer3d()  # 离线渲染
    offscreen_renderer.Create(create_default_lights=False, phong_shading=True)  # 初始化
    offscreen_renderer.SetModeShaded()
    offscreen_renderer.default_drawer.SetFaceBoundaryDraw(False) 

    offscreen_renderer.View.SetEye(0, 0, 10)  # 设置摄像机位置
    offscreen_renderer.View.SetAt(0.0, 0.0, 0.0) 
    offscreen_renderer.View.SetScale(1)


    offscreen_renderer.SetSize(512, 512)
    #offscreen_renderer.DisplayShape(shape, update=True, material=Graphic3d_NameOfMaterial.Graphic3d_NOM_METALIZED,color=rgb_color(128/255, 0/255, 0/255))
    ais_shp = AIS_ColoredShape(shape)

    for fc in TopologyExplorer(shape).faces():
        # set a custom color per-face
        ais_shp.SetCustomColor(fc, rgb_color(0.5, 0.5, 1))
    offscreen_renderer.Context.Display(ais_shp, True)

    offscreen_renderer.Repaint()

    offscreen_renderer.View.Dump('/home/lkh/siga/CADIMG/test/output1.png')


def test_stl():
    from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeBox
    from OCC.Core.StlAPI import StlAPI_Writer
    from OCC.Core.TopoDS import TopoDS_Shape
    from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
    from OCC.Core.BRep import BRep_Tool
    import cv2
    name = '00902881'
    file_dir = "/home/lkh/siga/dataset/deepcad/data/cad_json/"
    file_dir = os.path.join(file_dir, name[:4])
    file_path = os.path.join(file_dir, name+'.json')
    shape = get_BRep_from_file(file_path)

    # mesh = BRepMesh_IncrementalMesh(shape, 0.1)
    # mesh.Perform()

    # display, start_display, _, _ = init_display()
    # display.DisplayShape(shape, update=True)
    # start_display()

    faces_info = get_faces_and_normals(shape)
    render_faces_to_image(faces_info, '/home/lkh/siga/CADIMG/test/1.png')

    # visualize_with_matplotlib(faces_info)



def test_face():
    name = '00902881'
    file_dir = "/home/lkh/siga/dataset/deepcad/data/cad_json/"
    file_dir = os.path.join(file_dir, name[:4])
    file_path = os.path.join(file_dir, name+'.json')
    shape, _ = create_2_box()

    from random import random

    from OCC.Core.AIS import AIS_ColoredShape
    from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeBox
    from OCC.Display.OCCViewer import rgb_color
    from OCC.Display.SimpleGui import init_display
    from OCC.Extend.TopologyUtils import TopologyExplorer

    display, start_display, add_menu, add_function_to_menu = init_display()
    display.Create(create_default_lights=False)


    ais_shp = AIS_ColoredShape(shape)
    
    for k , fc in enumerate(TopologyExplorer(shape).faces()):
        # print(k)
        # set a custom color per-face
        # if k==0:
        #     ais_shp.SetCustomColor(fc, Quantity_Color(0.1, 0.1, 0.1, Quantity_TOC_RGB))

        ais_shp.SetCustomColor(fc, Quantity_Color(0.5, 0.5, 1.0, Quantity_TOC_RGB))

    display.Context.Display(ais_shp, True)
    display.FitAll()

    start_display()


def test_output_stl():
    from OCC.Core.StlAPI import StlAPI_Writer
    from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
    name = '00902881'
    file_dir = "/home/lkh/siga/dataset/deepcad/data/cad_json/"
    file_dir = os.path.join(file_dir, name[:4])
    file_path = os.path.join(file_dir, name+'.json')
    shape = get_BRep_from_file(file_path)

    mesh = BRepMesh_IncrementalMesh(shape, 0.001)
    mesh.Perform()

    stl_writer = StlAPI_Writer()
    stl_writer.SetASCIIMode(True)
    stl_writer.Write(shape, "/home/lkh/siga/CADIMG/test/output1.stl")

def test_makeface():
    from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeEdge, BRepBuilderAPI_MakeWire, BRepBuilderAPI_MakeFace
    from OCC.Core.gp import gp_Pnt

    v1 = gp_Pnt(0,0,0)
    v2 = gp_Pnt(1,2,3)
    v3 = gp_Pnt(1,1,3)
    edge1 = BRepBuilderAPI_MakeEdge(v1, v2).Edge()
    edge2 = BRepBuilderAPI_MakeEdge(v2, v3).Edge()
    edge3 = BRepBuilderAPI_MakeEdge(v3, v1).Edge()

    wire = BRepBuilderAPI_MakeWire(edge1, edge2, edge3).Wire()
    face = BRepBuilderAPI_MakeFace(wire).Face()

    display, start_display, add_menu, add_function_to_menu = init_display()
    display.default_drawer.SetFaceBoundaryDraw(False)
    display.DisplayShape(face, update=True)
    display.FitAll()
    start_display()

def test_wire():
    from OCC.Display.SimpleGui import init_display
    from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeBox

    display, start_display, add_menu, add_function_to_menu = init_display()
    my_box = BRepPrimAPI_MakeBox(10., 20., 30.).Shape()

    display.default_drawer.SetFaceBoundaryDraw(False)
    display.DisplayShape(my_box, update=True)
    
    start_display()


def test_co():
    shape, _ = create_2_box()
    display, start_display, add_menu, add_function_to_menu = init_display()
    
    # display.SetModeShaded()


    # random_dir = gp_Dir(1, 0, 0)
    # light = V3d_DirectionalLight(random_dir, Quantity_Color(1, 0, 0, Quantity_TOC_RGB), True)  # 创建光源
    # light.SetIntensity(1.0)  # 设置光强
    # display.Viewer.AddLight(light)  # 将光源添加到渲染器
    # display.Viewer.SetLightOff()  # 打开光源


    display.DisplayShape(shape, update=True, color=rgb_color(0.5, 0.5, 1))

    display.View.SetEye(10, 10, 10)  # 设置摄像机位置
    display.View.SetAt(1, 1, 0) 
    display.View.SetScale(400)

    display.Repaint()
    start_display()




def test_exp():
    from random import random

    from OCC.Core.AIS import AIS_ColoredShape
    from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeBox
    from OCC.Display.OCCViewer import rgb_color
    from OCC.Display.SimpleGui import init_display
    from OCC.Extend.TopologyUtils import TopologyExplorer
    from OCC.Core.Quantity import Quantity_Color, Quantity_TOC_RGB, Quantity_NOC_BLACK
    from OCC.Core.BRep import BRep_Tool
    from OCC.Core.TopLoc import TopLoc_Location
    from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeEdge, BRepBuilderAPI_MakeWire, BRepBuilderAPI_MakeFace
    from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh

    display, start_display, add_menu, add_function_to_menu = init_display()

    name = '00902881'
    file_dir = "/home/lkh/siga/dataset/deepcad/data/cad_json/"
    file_dir = os.path.join(file_dir, name[:4])
    file_path = os.path.join(file_dir, name+'.json')
    shape = get_BRep_from_file(file_path)

    # mesh = BRepMesh_IncrementalMesh(shape, 0.0001)
    # mesh.Perform()
    for k,fc in enumerate(TopologyExplorer(shape).faces()):
        mesh = BRepMesh_IncrementalMesh(fc, 0.0001).Perform()
        loc = TopLoc_Location()
        triangulation = BRep_Tool.Triangulation(fc, loc)
        tr = loc.Transformation()
        # nodes = [triangulation.Node(i + 1) for i in range(triangulation.NbNodes())]
        nodes = []
        for i in range(1, triangulation.NbNodes() + 1):
            node = triangulation.Node(i)
            # 将局部坐标转换为全局坐标
            transformed_node = node.Transformed(tr)
            nodes.append(transformed_node)
        triangles = triangulation.Triangles()
        

        for i in range(triangulation.NbTriangles()):
            tri = triangles.Value(i + 1)
            n1, n2, n3 = tri.Get()
            p1, p2, p3 = nodes[n1 - 1], nodes[n2 - 1], nodes[n3 - 1]
            # 创建顶点
            gp1, gp2, gp3 = gp_Pnt(p1.X(), p1.Y(), p1.Z()), gp_Pnt(p2.X(), p2.Y(), p2.Z()), gp_Pnt(p3.X(), p3.Y(), p3.Z())

            # 创建边和面
            edge1 = BRepBuilderAPI_MakeEdge(p1, p2).Edge()
            edge2 = BRepBuilderAPI_MakeEdge(p2, p3).Edge()
            edge3 = BRepBuilderAPI_MakeEdge(p3, p1).Edge()
            wire = BRepBuilderAPI_MakeWire(edge1, edge2, edge3).Wire()
            mk_face = BRepBuilderAPI_MakeFace(wire).Face()
            display.DisplayShape(mk_face, update=False)

    # builder = BRep_Builder()
    # compound = TopoDS_Compound()
    # builder.MakeCompound(compound)

    # ais_shp = AIS_ColoredShape(my_box)

    # i = 0
    # for fc in TopologyExplorer(my_box).faces():
    #     triangulation = BRep_Tool.Triangulation(fc, TopLoc_Location())
    #     nodes = [triangulation.Node(i + 1) for i in range(triangulation.NbNodes())]
    #     triangles = triangulation.Triangles()

    #     for i in range(triangulation.NbTriangles()):
    #         tri = triangles.Value(i + 1)
    #         n1, n2, n3 = tri.Get()
    #         p1, p2, p3 = nodes[n1 - 1], nodes[n2 - 1], nodes[n3 - 1]

    #         # 创建顶点
    #         gp1, gp2, gp3 = gp_Pnt(p1.X(), p1.Y(), p1.Z()), gp_Pnt(p2.X(), p2.Y(), p2.Z()), gp_Pnt(p3.X(), p3.Y(), p3.Z())

    #         # 创建边和面
    #         edge1 = BRepBuilderAPI_MakeEdge(gp1, gp2).Edge()
    #         edge2 = BRepBuilderAPI_MakeEdge(gp2, gp3).Edge()
    #         edge3 = BRepBuilderAPI_MakeEdge(gp3, gp1).Edge()
    #         wire = BRepBuilderAPI_MakeWire(edge1, edge2, edge3).Wire()
    #         mk_face = BRepBuilderAPI_MakeFace(wire)

    #     if mk_face.IsDone():
    #         builder.Add(compound, mk_face.Face())
    #     # set a custom color per-face
    #     ais_shp.SetCustomColor(fc, Quantity_Color(random(), random(), random(), Quantity_TOC_RGB))
    #     i+=1
    # print(i)
    # display.Context.Display(ais_shp, True)
    display.FitAll()
    display.Repaint()
    start_display()
    R_A = np.array([[1, 0, 0],
                    [0, 1, 0],
                    [0, 0, 1]])  # 单位矩阵，世界坐标

    R_B = np.array([[0, 1, 0],
                    [-1, 0, 0],
                    [0, 0, 1]])  # 绕 Z 轴逆时针旋转 90 度

def test_ma():
    def coordinate_transform(A, B, v_a):
    
        B_inv = np.linalg.inv(B)
        # 计算转换矩阵：B⁻¹ * A
        transformation_matrix = B_inv @ A
        # 应用转换矩阵到原坐标
        v_b = transformation_matrix @ v_a
        return v_b
    R_A = np.array([[1, 0, 0],
                    [0, 1, 0],
                    [0, 0, 1]])  # 单位矩阵，世界坐标

    R_B = np.array([[0, -1, 0],
                    [1, 0, 0],
                    [0, 0, 1]])  # 绕 Z 轴逆时针旋转 90 度
    p_A = np.array([2, 0, 0])
    p_world = R_A @ p_A
    p_B =  R_B.T @  p_world
    print("点在 B 坐标系下的坐标:", p_B)
    print(coordinate_transform(R_A, R_B, p_A))

    a = (sqrt(6)-sqrt(2))/4
    b=1
    print()



def main():
    test_mesh()


if __name__ == "__main__":
    main()


# img_path = "/home/lkh/siga/CADIMG/infer"
# imgs = []
# for i in range(0,4):
#     img_p = os.path.join(img_path, f"{i:06d}.png")
#     imgs.append(np.array(Image.open(img_p)))
# images_np = [np.array(img) for img in imgs]
# result_np = np.concatenate(images_np, axis=1)  # axis=1 表示水平

# # 转回 PIL.Image 并保存
# result = Image.fromarray(result_np)
# result.save(os.path.join(img_path, "02.png"))

# img = Image.open(img_path).convert("L")
# img0 = np.array(img)
# print(img0.shape)
# transform = transforms.Compose([
#     transforms.ToTensor()
# ])
# img = transform(img)
# print(img.shape)
# img = transforms.ToPILImage()(img)
# img1 = np.array(img)
# print(img1.shape)
# transform = transforms.Compose([
#                 transforms.Resize((128,128)),
#                 transforms.ToTensor()
#             ])

# mask = transform(mask)
# print(mask[0,:,64])
