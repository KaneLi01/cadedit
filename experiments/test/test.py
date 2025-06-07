

from torch.utils.tensorboard import SummaryWriter
import datetime
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from CADIMG.utils import log_utils
from utils.vis import show_single
import utils.cadlib.Brep_utils as Brep_utils
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



def test_sketch_render(shape_name, view_num=0):
    # 查看sketch是否和normal匹配
    '''
    先查看合适的形状，然后检查normal是否和sketch重合
    '''
    import pickle
    import copy

    def get_pkl(path, name):
        with open(path, 'rb') as f:
            data_dict = pickle.load(f)
        return data_dict[name]

    VIEW_CORNERS_6 = {
        'front': [ # front
            [2.0,-2*sqrt(2),0.0], [2.0,-2.0,2.0], [2.0,0.0,2*sqrt(2)], [2.0,2.0,2.0], [2.0,2*sqrt(2),0.0], [2.0,0.0,-2*sqrt(2)]  
        ],

        'back': [ # back
            [-2.0,-2*sqrt(2),0.0], [-2.0,-2.0,2.0], [-2.0,0.0,2*sqrt(2)], [-2.0,2.0,2.0], [-2.0,2*sqrt(2),0.0], [-2.0,0.0,-2*sqrt(2)]
        ],

        'right': [ # right
            [0.0,2.0,2*sqrt(2)], [2.0,2.0,2.0], [2*sqrt(2),2.0,0.0], [2.0,2.0,-2.0], [0.0,2.0,-2*sqrt(2)], [-2*sqrt(2),2.0,0.0]
        ],
        
        'left': [ # left
            [0.0,-2.0,2*sqrt(2)], [2.0,-2.0,2.0], [2*sqrt(2),-2.0,0.0], [2.0,-2.0,-2.0], [0.0,-2.0,-2*sqrt(2)], [-2*sqrt(2),-2.0,0.0]
        ],

        'up': [ # up
            [0.0,-2*sqrt(2),2.0], [2.0,-2.0,2.0], [2*sqrt(2),0,2.0], [2.0,2.0,2.0], [0.0,2*sqrt(2),2.0], [-2*sqrt(2),0.0,2.0]
        ],
        
        'down': [ # down
            [0.0,-2*sqrt(2),-2.0], [2.0,-2.0,-2.0], [2*sqrt(2),0,-2.0], [2.0,2.0,-2.0], [0.0,2*sqrt(2),-2.0], [-2*sqrt(2),0.0,-2.0]
        ]
    }

    
    json_dir = '/home/lkh/siga/dataset/deepcad/data/cad_json'
    shape_path = os.path.join(json_dir, shape_name[:4], shape_name+'.json')
    
    pkl_dir = '/home/lkh/siga/CADIMG/dataset/create_dataset/render_normal'
    center_path = os.path.join(pkl_dir, 'centers_correct.pkl')
    view_path = os.path.join(pkl_dir, 'views_correct.pkl')

    seeat = get_pkl(center_path, shape_name)
    view = get_pkl(view_path, shape_name)
    campos = VIEW_CORNERS_6[view][view_num]

    seq = Brep_utils.get_seq_from_json(shape_path)
    seq1 = copy.deepcopy(seq)
    shape = Brep_utils.get_BRep_from_seq(seq1.seq[-1:])
    wires = Brep_utils.get_wireframe(shape)
    output_path = os.path.join('/home/lkh/siga/output/test/sketch/', shape_name+'_sketch.png')
    show_single.save_BRep_wire_img_temp(wires, campos=campos, seeat=seeat, output_path=output_path)



def main():
    view_num = 2
    shape_name = '00004935'
    test_sketch_render(shape_name, view_num)


if __name__ == "__main__":
    main()

