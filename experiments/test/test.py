import sys, datetime, os, random
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import trimesh
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from utils.cadlib import Brep_utils
from utils.vis import render_cad

from OCC.Core.BRep import BRep_Tool


# def triangulate_shape(shape, deflection=0.1):



def write_obj(filename, verts, tris):
    with open(filename, 'w') as f:
        # 写入顶点
        for v in verts:
            f.write(f"v {v[0]} {v[1]} {v[2]}\n")
        # 写入面（OBJ 的索引从 1 开始）
        for tri in tris:
            f.write(f"f {tri[0] + 1} {tri[1] + 1} {tri[2] + 1}\n")


def render_obj_to_image(obj_path, output_img_path):
    """将单个 OBJ 文件渲染为图片"""
    try:
        # 加载 OBJ 文件
        mesh = trimesh.load(obj_path)
        # 创建一个 3D 场景
        scene = mesh.scene()
        # 保存渲染结果到图片
        data = scene.save_image(resolution=[800, 600], visible=False)
        with open(output_img_path, 'wb') as f:
            f.write(data)
        return True
    except Exception as e:
        print(f"Error rendering {obj_path}: {e}")
        return False

def combine_images(image_paths, output_combined_path, rows=2, cols=5):
    """将多张图片合并为一张（默认 2x5=10 张）"""
    images = [Image.open(img) for img in image_paths]
    width, height = images[0].size
    combined = Image.new('RGB', (cols * width, rows * height))
    
    for i, img in enumerate(images):
        row = i // cols
        col = i % cols
        combined.paste(img, (col * width, row * height))
    
    combined.save(output_combined_path)

def process_directory(root_dir, output_dir="output"):
    """处理目录下的所有 OBJ 文件"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    obj_files = []
    # 遍历所有子目录，收集 OBJ 文件路径
    for subdir, _, files in os.walk(root_dir):
        for file in files:
            if file.lower().endswith('.obj'):
                obj_files.append(os.path.join(subdir, file))

    print(obj_files[4899])
    raise Exception('-')
    
    # 每 10 个 OBJ 文件为一组
    for i in range(0, len(obj_files), 10):
        group = obj_files[i:i+10]
        img_paths = []
        # 渲染每组 OBJ 文件
        for j, obj_path in enumerate(group):
            img_path = os.path.join(output_dir, f"temp_{i+j}.png")
            if render_obj_to_image(obj_path, img_path):
                img_paths.append(img_path)
        # 合并图片
        if img_paths:
            combined_path = os.path.join(output_dir, f"combined_{i//10}.png")
            combine_images(img_paths, combined_path)
            print(f"Saved combined image: {combined_path}")
            # 清理临时文件
            for img_path in img_paths:
                os.remove(img_path)





def get_num(n):
    
    root_dir = "/home/lkh/siga/dataset/ABC/abc_obj/00"  # 替换为你的 OBJ 文件根目录
    
    obj_files = []
    # 遍历所有子目录，收集 OBJ 文件路径
    for subdir, _, files in os.walk(root_dir):
        for file in files:
            if file.lower().endswith('.obj'):
                obj_files.append(os.path.join(subdir, file))

    print(obj_files[n])
    return obj_files[n]


def look(n):
    import copy
    from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
    from OCC.Core.StlAPI import StlAPI_Writer

    file1 = get_num(n=n).replace('obj', 'step').replace('trimesh', 'step')


    shape = copy.deepcopy(Brep_utils.get_BRep_from_step(file1))
    l = Brep_utils.get_first_level_shapes(shape)
    mesh = BRepMesh_IncrementalMesh(l[1], 0.0001)
    mesh.Perform()
    stl_writer = StlAPI_Writer()
    stl_path = "/home/lkh/siga/output/temp/2256_1.stl"
    stl_writer.Write(shape, stl_path)

    mesh = trimesh.load(stl_path)
    mesh.export(stl_path.replace('stl', 'obj')) 

    # render_cad.display_BRep(shape)
    # Brep_utils.explore_shape(shape)
    # l = Brep_utils.get_first_level_shapes(shape)
    # # b1 = Brep_utils.get_bbox(l[0])
    # # b2 = Brep_utils.get_bbox(l[1])
    # # print(b1, b2)
    # # print(l)
    # for ll in l:
    #     render_cad.display_BRep(ll)
    # render_cad.save_BRep(output_path=o1, shape=l[0], see_at=[-198.69615714153213,1.4759205120735075,11.244403690310648], cam_pos=[-150.9623117688493,71.209765884756315,60.97824906299346])
    # render_cad.save_BRep(output_path=o2, shape=l[1], see_at=[0.6146319062223426,0.13443163012419745,18.638946488677888], cam_pos=[56.22102892490321,77.74082864880506,76.24534350735875])


def create_sphere_meshes(points, radius=0.02, subdivisions=3):
    """将所有点变为球体网格"""
    spheres = []
    for pt in points:
        sphere = trimesh.creation.icosphere(radius=radius, subdivisions=subdivisions)
        sphere.apply_translation(pt)
        spheres.append(sphere)
    return trimesh.util.concatenate(spheres)

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




def convert_edges_to_mesh(edges, line_radius=0.05):
    from OCC.Core.BRepAdaptor import BRepAdaptor_Curve
    """将STEP中的边转换为mesh网格：直线为圆柱，弧线为环面"""
    all_meshes = []
    for edge in edges:
        curve_adapt = BRepAdaptor_Curve(edge)
        curve_type = curve_adapt.GetType()
        print(f"处理曲线类型: {curve_type}")
        first = curve_adapt.FirstParameter()
        last = curve_adapt.LastParameter()

        p1 = curve_adapt.Value(first)
        p2 = curve_adapt.Value(last)

        pt1 = np.array([p1.X(), p1.Y(), p1.Z()])

        pt2 = np.array([p2.X(), p2.Y(), p2.Z()])

        if curve_type == 0:  # Line
            cyl = make_cylinder(pt1, pt2, radius=line_radius)
            cyl.apply_translation([0, 0, 0.5])
            if cyl:
                all_meshes.append(cyl)
        elif curve_type == 8:  # Circle (即弧线)
            circ = curve_adapt.Circle()
            center = circ.Location()
            axis = circ.Axis().Direction()
            torus = make_torus(
                center=[center.X(), center.Y(), center.Z()],
                normal=[axis.X(), axis.Y(), axis.Z()],
                radius=circ.Radius(),
                tube_radius=line_radius
            )
            if torus:
                all_meshes.append(torus)
        else:
            print(f"跳过未处理的曲线类型: {curve_type}")
    return trimesh.util.concatenate(all_meshes)



def render_yuan():
    shape = Brep_utils.get_BRep_from_step('/home/lkh/siga/output/temp/wireframe.step')
    from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
    from OCC.Core.StlAPI import StlAPI_Writer
    mesh = BRepMesh_IncrementalMesh(shape, 0.0001)
    mesh.Perform()
    stl_writer = StlAPI_Writer()
    stl_path = "/home/lkh/siga/output/temp/ws.stl"
    stl_writer.Write(shape, stl_path)
    # p1 = Brep_utils.Point(0, 0, 0)
    # p2 = Brep_utils.Point(1, 1, 1)
    # shape = Brep_utils.create_box_from_minmax(p1, p2)
    # wires = Brep_utils.get_wires(shape)
    
    # mesh1 = convert_edges_to_mesh(wires, line_radius=0.2)
    # points = list(Brep_utils.get_vertices(shape))
    # vs = [BRep_Tool.Pnt(p) for p in points]
    # vss = [[v.X(), v.Y(), v.Z()] for v in vs]
    # mesh2 = create_sphere_meshes(vss, radius=0.02)
    # output_obj = '/home/lkh/siga/output/temp/test.obj'
    # # mesh_total = trimesh.util.concatenate([mesh1, mesh2])
    # mesh1.export(output_obj)



def main():

    render_yuan()




if __name__ == "__main__":
    main()

