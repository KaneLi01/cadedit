from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopAbs import TopAbs_EDGE
from OCC.Core.BRepAdaptor import BRepAdaptor_Curve
from OCC.Core.GCPnts import GCPnts_UniformAbscissa
from OCC.Core.TopoDS import topods
import numpy as np
import trimesh


'''导出为obj文件后，可以直接使用blender渲染。'''

def make_cylinder(p1, p2, radius=0.1, sections=16):
    """创建两个点之间的圆柱体（三角网格）"""
    vec = np.array(p2) - np.array(p1)
    height = np.linalg.norm(vec)
    if height < 1e-6:
        return None

    direction = vec / height
    cylinder = trimesh.creation.cylinder(radius=radius, height=height, sections=sections)
    cylinder.apply_translation([0, 0, height / 2.0])

    axis = trimesh.geometry.align_vectors([0, 0, 1], direction)
    cylinder.apply_transform(axis)
    cylinder.apply_translation(p1)
    return cylinder


def sample_edge(edge, n_points=20):
    """
    对单条边均匀采样，返回采样点列表[(x,y,z), ...]
    """
    adaptor = BRepAdaptor_Curve(edge)
    discretizer = GCPnts_UniformAbscissa(adaptor, n_points)
    points = []
    if not discretizer.IsDone():
        # 采样失败，取端点
        p1 = adaptor.Value(adaptor.FirstParameter())
        p2 = adaptor.Value(adaptor.LastParameter())
        points = [(p1.X(), p1.Y(), p1.Z()), (p2.X(), p2.Y(), p2.Z())]
    else:
        for i in range(1, discretizer.NbPoints() + 1):
            p = adaptor.Value(discretizer.Parameter(i))
            points.append((p.X(), p.Y(), p.Z()))
    return points

def extract_wireframe(step_path, n_points_per_edge=20):
    """
    从STEP文件中提取线框，将每条边均匀采样重构为直线段
    返回格式：list of line segments, 每条线段为 [(x1,y1,z1), (x2,y2,z2)]
    """
    # 读取STEP文件
    reader = STEPControl_Reader()
    status = reader.ReadFile(step_path)
    if status != 1:
        raise RuntimeError("STEP文件读取失败")

    reader.TransferRoots()
    shape = reader.OneShape()

    exp = TopExp_Explorer(shape, TopAbs_EDGE)

    lines = []

    while exp.More():
        edge = topods.Edge(exp.Current())
        points = sample_edge(edge, n_points_per_edge)
        # 将采样点两两连接成线段
        for i in range(len(points) - 1):
            lines.append([points[i], points[i+1]])
        exp.Next()

    return lines




if __name__ == "__main__":
    step_file = "/home/lkh/siga/output/temp/2256.step"  # 修改为你的STEP文件路径
    lines = extract_wireframe(step_file, n_points_per_edge=50)
    meshes = []
    for p1, p2 in lines:
        cyl = make_cylinder(p1, p2, radius=0.4)  # 你可以调节圆柱半径
        if cyl is not None:
            meshes.append(cyl)

    # 合并所有圆柱网格
    combined_mesh = trimesh.util.concatenate(meshes)

    # 保存为文件，例如STL或OBJ
    combined_mesh.export("wireframe1.stl")
    print("圆柱体网格导出完成：wireframe.stl")
