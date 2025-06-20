import json, copy
from .extrude import CADSequence
from .visualize import create_CAD, create_CAD_by_seq

from dataclasses import dataclass
from collections import namedtuple

from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopAbs import TopAbs_VERTEX, TopAbs_EDGE, TopAbs_WIRE, TopAbs_FACE
from OCC.Core.BRep import BRep_Tool, BRep_Builder
from OCC.Core.TopoDS import TopoDS_Compound, TopoDS_Shape, TopoDS_Wire, TopoDS_Edge, TopoDS_Face
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeWire
from OCC.Core.BRepTools import breptools_OuterWire, breptools
from OCC.Core.Bnd import Bnd_Box
from OCC.Core.BRepBndLib import brepbndlib
from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeBox
from OCC.Core.gp import gp_Pnt
from OCC.Core.GProp import GProp_GProps
from OCC.Core.BRepGProp import brepgprop
from OCC.Core.BRepExtrema import BRepExtrema_DistShapeShape
from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.IFSelect import IFSelect_RetDone


@dataclass
class Point:
    x: float
    y: float
    z: float

@dataclass
class BBox:
    min: Point
    max: Point
    center: Point



def get_BRep_from_seq(sub_seq):
    try:
        out_shape = create_CAD_by_seq(copy.deepcopy(sub_seq))
        return out_shape
    except Exception as e:
        print("create shape from cad_seq fail.", e)


def get_seq_from_json(file_path):
    try:
        with open(file_path, 'r') as fp:
            data = json.load(fp)
        cad_seq = CADSequence.from_dict(data)
        cad_seq.normalize()
        return cad_seq
    except Exception as e:
        print("read json or get seq failed.", e)


def get_BRep_from_file(file_path):
    try:
        cad_seq = get_seq_from_json(file_path)
        out_shape = get_BRep_from_seq(cad_seq.seq)
        return out_shape
    except Exception as e:
        print("load and create failed.", e)


def get_BRep_from_step(file_path):
    """读取STEP文件并返回TopoDS_Shape对象"""
    reader = STEPControl_Reader()
    status = reader.ReadFile(file_path)
    
    if status != IFSelect_RetDone:
        return None
    
    reader.TransferRoots()  # 转换几何实体
    shape = reader.OneShape()  # 获取合并后的形状
    return shape


def get_bbox(shape: TopoDS_Compound):
    from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
    """获取shape的包围盒"""
    bbox = Bnd_Box()
    mesh = BRepMesh_IncrementalMesh(shape, 0.1)
    mesh.Perform()
    brepbndlib.Add(shape, bbox)

    xmin, ymin, zmin, xmax, ymax, zmax = bbox.Get()
    min=Point(xmin, ymin, zmin)
    max=Point(xmax, ymax, zmax)
    center=Point((xmin + xmax) / 2, (ymin + ymax) / 2, (zmin + zmax) / 2)

    return BBox(min, max, center)


def get_vertices(shape):
    """获取shape的顶点"""
    vertices_list = set()  # 存储边的列表
    explorer = TopExp_Explorer(shape, TopAbs_VERTEX)  # 遍历 Compound 中的边
    while explorer.More():
        point = explorer.Current()  # 获取当前边 
        vertices_list.add(point)
        explorer.Next()

    return vertices_list  # 返回生成的线框


def get_wires(shape):
    """获取shape的线框"""
    edge_list = []  # 存储边的列表
    explorer = TopExp_Explorer(shape, TopAbs_EDGE)  # 遍历 Compound 中的边
    while explorer.More():
        edge = explorer.Current()  # 获取当前边 
        edge_list.append(edge)
        explorer.Next()

    return edge_list  # 返回生成的线框


def get_faces(shape):
    """获取shape的面"""
    faces = set()
    explorer = TopExp_Explorer(shape, TopAbs_FACE)
    while explorer.More():
        face = explorer.Current()  # 获取当前面（可选）
        faces.add(face)
        explorer.Next()
    return faces


def get_volume(shape):
    """计算shape的体积"""
    if shape is None or shape.IsNull():
        return 0.0

    props = GProp_GProps()
    brepgprop.VolumeProperties(shape, props)
    return props.Mass()


def get_min_distance(shape1: TopoDS_Shape, shape2: TopoDS_Shape) -> float:
    """计算两个Shape之间的最小距离"""
    dist_calc = BRepExtrema_DistShapeShape(shape1, shape2)
    dist_calc.Perform()
    if not dist_calc.IsDone():
        raise RuntimeError("距离计算失败")
    return dist_calc.Value()


def get_wireframe_cir(shape):
    from OCC.Core.Geom import Geom_Line
    edge_list = []  # 存储边的列表
    explorer = TopExp_Explorer(shape, TopAbs_EDGE)  # 遍历 Compound 中的边
    while explorer.More():
        edge = explorer.Current()  # 获取当前边 
        

        curve_handle, _, _ = BRep_Tool.Curve(edge)
        if curve_handle.DynamicType().Name() == "Geom_Circle":
            edge_list.append(edge)


        explorer.Next()

    return edge_list 


def create_box_from_minmax(min_point: Point, max_point: Point):
    """
    从最小点和最大点创建平行于坐标轴的topoDS长方体
    """
    dx = max_point.x - min_point.x
    dy = max_point.y - min_point.y
    dz = max_point.z - min_point.z
    
    # 创建长方体（从 min_point 出发，沿 XYZ 正方向延伸 dx, dy, dz）
    box = BRepPrimAPI_MakeBox(
        gp_Pnt(min_point.x, min_point.y, min_point.z),  # 起点（最小点）
        dx, dy, dz           # 长、宽、高
    ).Solid()
    
    return box


def test_create_2boxs(mode='intersection'):
    """
    创建不同的box实例，用于测试
    """
    box1 = BRepPrimAPI_MakeBox(gp_Pnt(0, 0, 0), 1, 1, 1).Solid()

    if mode == 'inter':
        p2 = gp_Pnt(0.5, 0.5, 0)
    elif mode == 'tang':
        p2 = gp_Pnt(1, 0, 0)
    elif mode == 'away':
        p2 = gp_Pnt(2, 2, 0)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    box2 = BRepPrimAPI_MakeBox(p2, 1, 1, 1).Solid()
    return box1, box2
    

def explore_shape(shape, level=0):
    from OCC.Core.TopoDS import TopoDS_Compound, TopoDS_Shape
    from OCC.Core.TopAbs import TopAbs_COMPOUND, TopAbs_SOLID, TopAbs_SHELL, TopAbs_FACE
    from OCC.Core.TopExp import TopExp_Explorer


    """仅打印直接子层级（非递归）"""
    shape_type = shape.ShapeType()
    
    # 类型映射
    type_names = {
        TopAbs_COMPOUND: "COMPOUND",
        TopAbs_SOLID: "SOLID",
        TopAbs_SHELL: "SHELL",
        TopAbs_FACE: "FACE"
    }
    
    print(f"父形状: {type_names.get(shape_type, 'UNKNOWN')} (子级数量: {shape.NbChildren()})")
    
    # 初始化探索器（遍历所有直接子级）
    explorer = TopExp_Explorer()
    explorer.Init(shape, TopAbs_COMPOUND)  # 可替换为TopAbs_SOLID等
    
    childs = []

    # 仅遍历直接子级
    while explorer.More():
        child = explorer.Current()
        child_type = child.ShapeType()
        print(f"  └─ {type_names.get(child_type, 'UNKNOWN')} (子级数量: {child.NbChildren()})")
        childs.append(child)
        explorer.Next()
    return childs


def get_first_level_shapes(shape):
    from OCC.Core.TopoDS import TopoDS_Iterator, TopoDS_Shape
    from OCC.Core.TopAbs import TopAbs_VERTEX, TopAbs_EDGE, TopAbs_FACE, TopAbs_SOLID
    it = TopoDS_Iterator(shape)  # 初始化迭代器
    count = 0

    childs = []
    while it.More():  # 检查是否还有子形状
        child = it.Value()  # 获取当前子形状
        count += 1

        # # 打印子形状的类型
        # if child.ShapeType() == TopAbs_VERTEX:
        #     print(f"Child {count}: VERTEX")
        # elif child.ShapeType() == TopAbs_EDGE:
        #     print(f"Child {count}: EDGE")
        # elif child.ShapeType() == TopAbs_FACE:
        #     print(f"Child {count}: FACE")
        # elif child.ShapeType() == TopAbs_SOLID:
        #     print(f"Child {count}: SOLID")
        childs.append(child)
        it.Next()  # 移动到下一个子形状

    # print(f"Total children: {count}")
    return childs