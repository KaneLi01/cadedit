from random import random
import os
from cadlib.Brep_utils import get_BRep_from_file, get_wireframe
import numpy as np

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
from  math import *

R_A = np.array([[1, 0, 0],
                [0, 1, 0],
                [0, 0, 1]])  
R_B = np.array([[1/sqrt(6), 1/sqrt(2), 1/sqrt(3)],
                [1/sqrt(6), -1/sqrt(2), 1/sqrt(3)],
                [-2/sqrt(6), 0, 1/sqrt(3)]]) 
R_C = np.array([[1/sqrt(2), -1/sqrt(6), 1/sqrt(3)],
                [-1/sqrt(2), -1/sqrt(6), 1/sqrt(3)],
                [0, 2/sqrt(6), 1/sqrt(3)]]) 
R_D = np.array([[ ((3+sqrt(3))/6),  ((3-sqrt(3))/6), 1/sqrt(3)],
                [-((3-sqrt(3))/6), -((3+sqrt(3))/6), 1/sqrt(3)],
                [-(sqrt(3)/3    ),  (sqrt(3)/3    ), 1/sqrt(3)]]) 
R_E = np.array([[ ((3-sqrt(3))/6),  ((3+sqrt(3))/6), 1/sqrt(3)],
                [-((3+sqrt(3))/6), -((3-sqrt(3))/6), 1/sqrt(3)],
                [-(sqrt(3)/3    ),  (sqrt(3)/3    ), 1/sqrt(3)]]) 

name = '00902881'
file_dir = "/home/lkh/siga/dataset/deepcad/data/cad_json/"
file_dir = os.path.join(file_dir, name[:4])
file_path = os.path.join(file_dir, name+'.json')
shape = get_BRep_from_file(file_path)


offscreen_renderer = Viewer3d()
offscreen_renderer.Create(create_default_lights=False, phong_shading=False)
offscreen_renderer.SetModeShaded()
offscreen_renderer.SetSize(512, 512)
offscreen_renderer.default_drawer.SetFaceBoundaryDraw(False) 

bg_color = Quantity_Color(0.0, 0.0, 0, Quantity_TOC_RGB)
offscreen_renderer.View.SetBgGradientColors(bg_color,bg_color)
offscreen_renderer.DisplayShape(shape, color=Quantity_Color(128/255, 128/255, 255/255, Quantity_TOC_RGB), update=True)
offscreen_renderer.Repaint()
offscreen_renderer.View.Dump('/home/lkh/siga/CADIMG/test/norm.png')

# for k,fc in enumerate(TopologyExplorer(shape).faces()):
#     mesh = BRepMesh_IncrementalMesh(fc, 0.0001).Perform()
#     loc = TopLoc_Location()
#     triangulation = BRep_Tool.Triangulation(fc, loc)
#     tr = loc.Transformation()

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

#         normal = normal / np.linalg.norm(normal)
#         # n_world = R_A @ normal
#         # n_view =   R_E.T @ n_world

#         color = normal / 2.0 +0.5
#         # 创建顶点
#         gp1, gp2, gp3 = gp_Pnt(p1.X(), p1.Y(), p1.Z()), gp_Pnt(p2.X(), p2.Y(), p2.Z()), gp_Pnt(p3.X(), p3.Y(), p3.Z())

#         # 创建边和面
#         edge1 = BRepBuilderAPI_MakeEdge(p1, p2).Edge()
#         edge2 = BRepBuilderAPI_MakeEdge(p2, p3).Edge()
#         edge3 = BRepBuilderAPI_MakeEdge(p3, p1).Edge()
#         wire = BRepBuilderAPI_MakeWire(edge1, edge2, edge3).Wire()
#         mk_face = BRepBuilderAPI_MakeFace(wire).Face()

#         offscreen_renderer.DisplayShape(mk_face, color=Quantity_Color(128/255, 128/255, 255/255, Quantity_TOC_RGB), update=False)



# offscreen_renderer.View.SetEye(10, 10, 10)  
# offscreen_renderer.View.SetAt(0, 0, 0)  
# offscreen_renderer.View.SetScale(500)

# offscreen_renderer.Repaint()
# offscreen_renderer.View.Dump('/home/lkh/siga/CADIMG/test/norm.png')

