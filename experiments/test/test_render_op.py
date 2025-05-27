from random import random
import os
from cadlib.Brep_utils import get_BRep_from_file, get_wireframe
import numpy as np
from PIL import Image


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
from OCC.Core.Quantity import Quantity_Color, Quantity_TOC_RGB, Quantity_NOC_BLACK
from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
import trimesh
from OCC.Core.TopLoc import TopLoc_Location
from OCC.Core.BRep import BRep_Tool
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeFace
from OCC.Core.TopoDS import TopoDS_Face, TopoDS_Compound
from OCC.Core.BRep import BRep_Builder
from OCC.Core.TColgp import TColgp_Array1OfPnt
from OCC.Core.Quantity import Quantity_Color, Quantity_TOC_RGB
from OCC.Core.AIS import AIS_Shape
from random import random
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeEdge, BRepBuilderAPI_MakeWire, BRepBuilderAPI_MakeFace
from OCC.Core.V3d import  V3d_DirectionalLight

def create_box():
    p1_1 = gp_Pnt(-0.5, -0.5, -0.5)
    p1_2 = gp_Pnt(0.5, 0.5, 0.5)
    box1 = BRepPrimAPI_MakeBox(p1_1, p1_2)
    shape1 = box1.Shape()

    return shape1


def read_file_shape():
    name = '00902881'
    file_dir = "/home/lkh/siga/dataset/deepcad/data/cad_json/"
    file_dir = os.path.join(file_dir, name[:4])
    file_path = os.path.join(file_dir, name+'.json')
    shape = get_BRep_from_file(file_path)

    return shape


def get_tris(shape):

    vs = []
    ns = []
    for k,fc in enumerate(TopologyExplorer(shape).faces()):
        loc = TopLoc_Location()
        triangulation = BRep_Tool.Triangulation(fc, loc)
        tr = loc.Transformation()
        # nodes = [triangulation.Node(i + 1) for i in range(triangulation.NbNodes())]
        nodes = []  # 每个面的点坐标
        for i in range(1, triangulation.NbNodes() + 1):
            node = triangulation.Node(i)
            # 将局部坐标转换为全局坐标
            transformed_node = node.Transformed(tr)
            nodes.append(transformed_node)

        triangles = triangulation.Triangles()
        for i in range(triangulation.NbTriangles()):
            tri = triangles.Value(i + 1)
            n1, n2, n3 = tri.Get()
            p1, p2, p3 = nodes[n1 - 1], nodes[n2 - 1], nodes[n3 - 1]  # 每个三角形的节点坐标

            v1 = np.array([p1.X(), p1.Y(), p1.Z()])
            v2 = np.array([p2.X(), p2.Y(), p2.Z()])
            v3 = np.array([p3.X(), p3.Y(), p3.Z()])

            normal = np.cross(v3 - v1, v2 - v1)
            normal = normal / np.linalg.norm(normal)  
            vs.append([v1, v2, v3])
            ns.append(normal)

    return vs, ns



def vis_tri_mesh(shape):
    display, start_display, _, _ = init_display()

    for k,fc in enumerate(TopologyExplorer(shape).faces()):
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

            v1 = np.array([p1.X(), p1.Y(), p1.Z()])
            v2 = np.array([p2.X(), p2.Y(), p2.Z()])
            v3 = np.array([p3.X(), p3.Y(), p3.Z()])

            # 计算法线
            normal = np.cross(v3 - v1, v2 - v1)
            m = np.max(np.abs(normal))

            normal = normal / m
            color = normal / 2.0 +0.5
            # 创建顶点
            gp1, gp2, gp3 = gp_Pnt(p1.X(), p1.Y(), p1.Z()), gp_Pnt(p2.X(), p2.Y(), p2.Z()), gp_Pnt(p3.X(), p3.Y(), p3.Z())

            # 创建边和面
            edge1 = BRepBuilderAPI_MakeEdge(p1, p2).Edge()
            edge2 = BRepBuilderAPI_MakeEdge(p2, p3).Edge()
            edge3 = BRepBuilderAPI_MakeEdge(p3, p1).Edge()
            wire = BRepBuilderAPI_MakeWire(edge1, edge2, edge3).Wire()
            mk_face = BRepBuilderAPI_MakeFace(wire).Face()

            display.DisplayShape(mk_face, color=Quantity_Color(128/255, 128/255, 128/255, Quantity_TOC_RGB), update=False)
    display.FitAll()
    display.Repaint()
    start_display()


def look_at(eye, target, up=[0, 1, 0]):
    f = (target - eye)
    f = f / np.linalg.norm(f)
    r = np.cross(f, up)
    r = r / np.linalg.norm(r)
    u = np.cross(r, f)

    view = np.eye(4)
    view[0, :3] = r
    view[1, :3] = u
    view[2, :3] = -f
    view[:3, 3] = -view[:3, :3] @ eye
    return view

def project_vertex(v, view_matrix, img_size):
    v_homo = np.hstack([v, 1])
    v_cam = view_matrix @ v_homo
    x, y = v_cam[0], v_cam[1]
    # 归一化坐标映射到图像像素
    img_w, img_h = img_size
    px = int((x + 1) / 2 * img_w)
    py = int((1 - (y + 1) / 2) * img_h)
    return px, py

def encode_normal_to_rgb(n):
    rgb = ((n + 1) * 0.5 * 255).astype(np.uint8)
    return rgb

camera_pos = np.array([2, 2, 2])
target = np.array([0, 0, 0])
view_matrix = look_at(camera_pos, target)

img_size = (512, 512)
normal_map = np.zeros((img_size[1], img_size[0], 3), dtype=np.uint8)

shape = create_box()
mesh = BRepMesh_IncrementalMesh(shape, 0.001)
mesh.Perform()


vs, ns = get_tris(shape)
all_vertices = np.array([v for tri in vs for v in tri])
unique_vertices, inverse_indices = np.unique(all_vertices, axis=0, return_inverse=True)

# 构建 faces：每个三角形由顶点索引组成
faces = inverse_indices.reshape(-1, 3)

# 创建 trimesh 对象
mesh = trimesh.Trimesh(vertices=unique_vertices, faces=faces)

# 导出为 PLY 文件
mesh.export('/home/lkh/siga/CADIMG/test/output_mesh.ply')



for tri_vertices, normal in zip(vs, ns):
    # 逐个投影三角形顶点
    screen_pts = [project_vertex(v, view_matrix, img_size) for v in tri_vertices]
    
    # 使用 skimage 画三角形（polygon 需要 y 坐标列表和 x 坐标列表）
    from skimage.draw import polygon
    rr, cc = polygon(
        [pt[1] for pt in screen_pts],  # y 坐标
        [pt[0] for pt in screen_pts],  # x 坐标
        normal_map.shape[:2]           # 图像大小作为边界
    )
    
    rgb = encode_normal_to_rgb(normal)
    normal_map[rr, cc] = rgb
img = Image.fromarray(normal_map)
img.save('normal_map.png')
img.show()
# offscreen_renderer = Viewer3d()
# offscreen_renderer.Create(create_default_lights=False, phong_shading=False)

# from OCC.Core.Graphic3d import Graphic3d_LightSet
# # light_set = offscreen_renderer.View.ActiveLights()

# # for l in light_set.begin():
# #     print(1)



# # offscreen_renderer.Create()
# offscreen_renderer.SetModeShaded()
# offscreen_renderer.SetSize(512, 512)
# offscreen_renderer.default_drawer.SetFaceBoundaryDraw(False) 

# bg_color = Quantity_Color(0.0, 0.0, 0, Quantity_TOC_RGB)
# offscreen_renderer.View.SetBgGradientColors(bg_color,bg_color)
# # 随机方向
# # random_dir = gp_Dir(0, 0, -1)
# # light = V3d_DirectionalLight(random_dir, Quantity_Color(0, 0, 0, Quantity_TOC_RGB), True)  # 创建光源
# # light.SetIntensity(1.0)  # 设置光强
# # offscreen_renderer.Viewer.AddLight(light)  # 将光源添加到渲染器
# # offscreen_renderer.Viewer.SetLightOff()  # 打开光源


# view_rot = lookAt()

# for k,fc in enumerate(TopologyExplorer(shape).faces()):
#     mesh = BRepMesh_IncrementalMesh(fc, 0.0001).Perform()
#     loc = TopLoc_Location()
#     triangulation = BRep_Tool.Triangulation(fc, loc)
#     tr = loc.Transformation()
#     # nodes = [triangulation.Node(i + 1) for i in range(triangulation.NbNodes())]
#     nodes = []
#     for i in range(1, triangulation.NbNodes() + 1):
#         node = triangulation.Node(i)
#         # 将局部坐标转换为全局坐标
#         transformed_node = node.Transformed(tr)
#         nodes.append(transformed_node)
#     triangles = triangulation.Triangles()
    
#     for i in range(triangulation.NbTriangles()):
#         tri = triangles.Value(i + 1)
#         n1, n2, n3 = tri.Get()
#         p1, p2, p3 = nodes[n1 - 1], nodes[n2 - 1], nodes[n3 - 1]

#         v1 = np.array([p1.X(), p1.Y(), p1.Z()])
#         v2 = np.array([p2.X(), p2.Y(), p2.Z()])
#         v3 = np.array([p3.X(), p3.Y(), p3.Z()])

#         # 计算法线
#         normal = np.cross(v3 - v1, v2 - v1)
#         m = np.max(np.abs(normal))

#         normal = normal / m
#         # if k == 9:
#         #     print(normal)

#         # normal_view = normalize(view_rot @ normal)
#         color = normal / 2.0 +0.5
#         # 创建顶点
#         gp1, gp2, gp3 = gp_Pnt(p1.X(), p1.Y(), p1.Z()), gp_Pnt(p2.X(), p2.Y(), p2.Z()), gp_Pnt(p3.X(), p3.Y(), p3.Z())

#         # 创建边和面
#         edge1 = BRepBuilderAPI_MakeEdge(p1, p2).Edge()
#         edge2 = BRepBuilderAPI_MakeEdge(p2, p3).Edge()
#         edge3 = BRepBuilderAPI_MakeEdge(p3, p1).Edge()
#         wire = BRepBuilderAPI_MakeWire(edge1, edge2, edge3).Wire()
#         mk_face = BRepBuilderAPI_MakeFace(wire).Face()
#         # ais_shp = AIS_ColoredShape(mk_face)k
#         # ais_shp.SetCustomColor(fc, Quantity_Color(1, 0.5, 0.5, Quantity_TOC_RGB))
#         # display.Context.Display(ais_shp, False)
#         # if k == 9:
#         offscreen_renderer.DisplayShape(mk_face, color=Quantity_Color(128/255, 128/255, 128/255, Quantity_TOC_RGB), update=False)
#         # else:
#         #     offscreen_renderer.DisplayShape(mk_face, color=Quantity_Color(0.2, 0.2, 0.2, Quantity_TOC_RGB), update=False)


# offscreen_renderer.View.SetEye(1, 1, 1)  
# offscreen_renderer.View.SetAt(0, 0, 0)  
# offscreen_renderer.View.SetScale(500)

# offscreen_renderer.Repaint()
# offscreen_renderer.View.Dump('/home/lkh/siga/CADIMG/test/norm.png')


# builder = BRep_Builder()
# compound = TopoDS_Compound()
# builder.MakeCompound(compound)

# exp = TopExp_Explorer(shape, TopAbs_FACE)
# while exp.More():
#     face = exp.Current()
#     triangulation = BRep_Tool.Triangulation(face, TopLoc_Location())
#     if triangulation is None:
#         exp.Next()
#         continue

#     # 获取节点和三角形
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

#         if mk_face.IsDone():
#             builder.Add(compound, mk_face.Face())

#     exp.Next()

# # 现在 compound 是新的 shape，可以显示或保存
# new_shape = compound  # type: TopoDS_Shape
# display.default_drawer.SetFaceBoundaryDraw(False)
# display.DisplayShape(new_shape, update=True)

# display.FitAll()
# display.Repaint()

# start_display()

