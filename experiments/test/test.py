import sys, datetime, os, random
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import trimesh
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from utils.cadlib import Brep_utils
from utils.vis import render_cad


def triangulate_shape(shape, deflection=0.1):
    from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
    from OCC.Extend.TopologyUtils import TopologyExplorer
    from OCC.Core.BRep import BRep_Tool
    import numpy as np

    BRepMesh_IncrementalMesh(shape, deflection).Perform()
    verts = []
    tris = []

    for face in TopologyExplorer(shape).faces():
        location = face.Location()
        triangulation = BRep_Tool.Triangulation(face, location)

        if triangulation is None:
            continue

        nodes = triangulation.Node()
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
    file1 = get_num(n=n).replace('obj', 'step').replace('trimesh', 'step')


    shape = copy.deepcopy(Brep_utils.get_BRep_from_step(file1))
    verts, tris = triangulate_shape(shape)
    write_obj("/home/lkh/siga/output/temp/2256_1.obj", verts, tris)
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

def main():

    look(n=4690)




if __name__ == "__main__":
    main()

