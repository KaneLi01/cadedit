import json
import h5py
import random
import numpy as np
from OCC.Display.SimpleGui import init_display
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cadlib.extrude import CADSequence
from cadlib.visualize import vec2CADsolid, create_CAD
from cadlib.Brep_utils import get_BRep_from_file,  get_wires
from OCC.Core.Quantity import Quantity_Color, Quantity_TOC_RGB, Quantity_NOC_BLACK
from OCC.Core.V3d import  V3d_DirectionalLight
from OCC.Core.Graphic3d import  Graphic3d_MaterialAspect
from OCC.Display.OCCViewer import Viewer3d
from OCC.Core.AIS import AIS_Shape, AIS_Line
from OCC.Core.gp import gp_Dir



# 定义全局颜色字典，使用 Quantity_Color 对象
COLORS = {
    "red": Quantity_Color(1.0, 0.0, 0.0, Quantity_TOC_RGB),       # 红色
    "green": Quantity_Color(0.0, 1.0, 0.0, Quantity_TOC_RGB),     # 绿色
    "blue": Quantity_Color(0.0, 0.0, 1.0, Quantity_TOC_RGB),      # 蓝色
    "yellow": Quantity_Color(1.0, 1.0, 0.0, Quantity_TOC_RGB),    # 黄色
    "cyan": Quantity_Color(0.0, 1.0, 1.0, Quantity_TOC_RGB),      # 青色
    "magenta": Quantity_Color(1.0, 0.0, 1.0, Quantity_TOC_RGB),   # 品红
    "white": Quantity_Color(1.0, 1.0, 1.0, Quantity_TOC_RGB),     # 白色
    "black": Quantity_Color(0.0, 0.0, 0.0, Quantity_TOC_RGB),     # 黑色

}


MECHANICAL_COLORS = {
    "copper":      (0.2294, 0.3314, 0.1510),     # 铜色
    "graphite":    (0.3412, 0.1294, 0.3020),       # 石墨色
    "stainless":   (0.5020, 0.5020, 0.4941),    # 不锈钢
    "blackish":    (0.2392, 0.2196, 0.2078),       # 偏黑
    "mahogany":    (0.5451, 0.2706, 0.0745),      # 马棕色
    "cobalt":      (0.2392, 0.3490, 0.6706),      # 钴蓝
}

from OCC.Core.Graphic3d import Graphic3d_NOM_ALUMINIUM, Graphic3d_NOM_STONE, Graphic3d_NOM_OBSIDIAN, Graphic3d_NOM_COPPER, Graphic3d_NOM_PEWTER

MATERIALS = [
    Graphic3d_NOM_ALUMINIUM,
    Graphic3d_NOM_STONE,
    Graphic3d_NOM_OBSIDIAN,
    Graphic3d_NOM_COPPER,
    Graphic3d_NOM_PEWTER,
]



def get_mechanical_color(color_dict=MECHANICAL_COLORS):
    _, (r, g, b) = random.choice(list(color_dict.items()))
    return Quantity_Color(r, g, b, Quantity_TOC_RGB)


def get_random_color(a1, a2):
    r, g, b = random.uniform(a1, a2), random.uniform(a1, a2), random.uniform(a1, a2)
    color = Quantity_Color(r, g, b, Quantity_TOC_RGB)
    return color


def display_BRep(shape=None, wire_list=None):
    """
    用于在线渲染形状或线框，或二者合并。用于调试。
    """

    if shape == None and wire_list == None:
        raise Exception('input at least one item!')

    display, start_display, _, _ = init_display()

    # 设置随机浅色背景
    r, g, b = random.uniform(0.7, 1.0), random.uniform(0.7, 1.0), random.uniform(0.7, 1.0)
    bg_color = Quantity_Color(r, g, b, Quantity_TOC_RGB)
    display.View.SetBgGradientColors(bg_color,bg_color)

    # 设置摄像机
    display.View.SetEye(2, 2, 2)  
    display.View.SetAt(1, 1, 1)  
    display.View.SetScale(500)

    if wire_list is not None:
        for wire in wire_list:
            display.DisplayShape(wire, update=False, color="black")
    if shape is not None:
        ais_out_shape = display.DisplayShape(shape, update=False, color=get_mechanical_color(MECHANICAL_COLORS))
 
    start_display()


def save_BRep(output_path, shape=None, wire_list=None,
              cam_pos=[2,2,2], see_at=[0,0,0], bg_color=1.0):
    '''离线渲染shape'''

    if shape == None and wire_list == None:
        raise Exception('input at least one item!')

    offscreen_renderer = Viewer3d()  
    offscreen_renderer.Create()  
    offscreen_renderer.SetModeShaded()
    offscreen_renderer.SetSize(512, 512)
    offscreen_renderer.SetPerspectiveProjection()

    # 设置摄像机
    offscreen_renderer.View.SetEye(cam_pos[0], cam_pos[1], cam_pos[2])  
    offscreen_renderer.View.SetAt(see_at[0], see_at[1], see_at[2]) 

    # 设置背景颜色
    bg_color = get_random_color(bg_color, bg_color)
    offscreen_renderer.View.SetBgGradientColors(bg_color,bg_color)

    if wire_list is not None:
        for wire in wire_list:
            ais_wire = AIS_Shape(wire)
            ais_wire.SetWidth(2.0)  # 设置线宽
            ais_wire.SetColor(Quantity_Color(Quantity_NOC_BLACK))
            offscreen_renderer.Context.Display(ais_wire, False)
    
    if shape is not None:
        offscreen_renderer.DisplayShape(shape, update=False, color=Quantity_Color(0.1, 0.1, 0.1, Quantity_TOC_RGB))
    
    offscreen_renderer.Repaint()
    # offscreen_renderer.FitAll()
    # c = offscreen_renderer.GetCamera()
    # e = c.Eye()
    # ce = c.Center()
    # print(ce.X(),ce.Y(),ce.Z())
    # print(e.X(),e.Y(),e.Z())
    
    offscreen_renderer.View.Dump(output_path)


class ARGS:
    def __init__(self):
        # self.file_path = r"/data/lkunh/datasets/DeepCAD/data/cad_json/0000/00000007.json"
        self.file_path = r"/data/lkunh/datasets/DeepCAD/data/cad_json/0002/00020718.json"
        self.save = False
        self.show_type = "body"  # wireframe | body


if __name__ == '__main__':
    args = ARGS()
    run(args)


