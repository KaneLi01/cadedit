import os
import pickle
import h5py
from PIL import Image
import cv2
import numpy as np
import trimesh
import argparse


def compare_sets(set1, set2):
    keys1 = set(set1.keys())
    keys2 = set(set2.keys())

    only_in_pkl1 = keys1 - keys2
    only_in_pkl2 = keys2 - keys1

    print("文件名只在 pkl1 中存在：")
    for name in sorted(only_in_pkl1):
        print("  ", name)

    print("\n文件名只在 pkl2 中存在：")
    for name in sorted(only_in_pkl2):
        print("  ", name)


def get_set_from_pkl(pkl_path):
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    keys = set(data.keys())
    return keys


def get_set_from_dir(dir_path, file_type=None):
    files = os.listdir(dir_path)
    file_names = {f"{f.split('.')[0]}" for f in files}
    return file_names


def get_pkl(pkl_path):
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    any_key = next(iter(data))
    print(data['00020346'])


def compare_keys(pkl1, pkl2):
    with open(pkl1, 'rb') as f:
        data1 = pickle.load(f)
    with open(pkl2, 'rb') as f:
        data2 = pickle.load(f)

    keys1 = set(data1.keys())
    keys2 = set(data2.keys())

    only_in_pkl1 = keys1 - keys2
    only_in_pkl2 = keys2 - keys1

    print("文件名只在 pkl1 中存在：")
    for name in sorted(only_in_pkl1):
        print("  ", name)

    print("\n文件名只在 pkl2 中存在：")
    for name in sorted(only_in_pkl2):
        print("  ", name)


def compare_keys_and_files(pkl_path, dir_path):
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    pkl_keys = set(data.keys())
    dir_files = {f"{f.split('.')[0]}" for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f))}
    only_in_pkl = pkl_keys - dir_files  # pkl比文件多
    only_in_dir = dir_files - pkl_keys

    # print("文件名只在 pkl 中存在：")
    for name in sorted(only_in_pkl):
        # print("  ", name)
        del data[name]

    print(len(data))
    with open('/home/lkh/siga/dataset/deepcad/data/cad_ply/views_correct.pkl', 'wb') as f:
        pickle.dump(data, f)


    # print("\n文件名只在目录中存在：")
    # for name in sorted(only_in_dir):
        # print("  ", name)



def compare_filenames(path_a, path_b):
    # 对比文件夹中的文件
    files_a = set(os.listdir(path_a))
    files_b = set(os.listdir(path_b))

    files_a = {f for f in files_a if os.path.isfile(os.path.join(path_a, f))}
    files_b = {f for f in files_b if os.path.isfile(os.path.join(path_b, f))}

    only_in_a = files_a - files_b
    only_in_b = files_b - files_a

    print(f"仅在 {path_a} 中的文件:")
    for f in sorted(only_in_a):
        print("  ", f)

    print(f"\n仅在 {path_b} 中的文件:")
    for f in sorted(only_in_b):
        print("  ", f)


def read_h5(file):
    with h5py.File(file, 'r') as f:
        print("文件中的对象:")
        def print_attrs(name, obj):
            print(name)
            for key, val in obj.attrs.items():
                print(f"    {key}: {val}")
        f.visititems(print_attrs)


def read_normal_imgs_h5(file):
    with h5py.File(file, 'r') as f:
        normals = f['normals'] 
    
    # 检查是否有附加属性（如单位、描述等）
    if normals.attrs:  # 如果有属性
        print("附加属性:")
        for key, value in normals.attrs.items():
            print(f"  {key}: {value}")


def merge_imgs(imgs_dir):
    imgs_name = sorted(os.listdir(imgs_dir)) 
    print(imgs_name)
    imgs = []
    
    for img_name in imgs_name:
        if img_name.split('.')[0] != 'total':
            img_path = os.path.join(imgs_dir, img_name)
            img = Image.open(img_path).convert("RGB")
            img_np = np.array(img)
            img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)  # ✅ RGB → BGR
            imgs.append(img_np)
    print(len(imgs))
    merged_h = np.hstack(imgs)
    cv2.imwrite(os.path.join(imgs_dir, 'total.png'), merged_h)


def from_ply_get_box(path_dir='/home/lkh/siga/dataset/deepcad/data/cad_ply/body2/result'):
    # 读取ply文件
    center_dir = {}
    ply_files = os.listdir(path_dir)
    for ply_file in ply_files:
        ply_path = os.path.join(path_dir, ply_file)
        mesh = trimesh.load(ply_path)
        aabb = mesh.bounding_box
        min_corner = aabb.bounds[0] 
        max_corner = aabb.bounds[1]
        center = (min_corner + max_corner) /2
        center_dir[ply_file.split('.')[0]] = center
    print('finish')
    with open('/home/lkh/siga/dataset/deepcad/data/cad_ply/centers.pkl', 'wb') as f:
        pickle.dump(center_dir, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("1")
    parser.add_argument('--path', type=str, help='merge imgs path')

    args = parser.parse_args()

    # path_a = "/home/lkh/siga/dataset/deepcad/data/cad_ply/body2/base"
    # path_b = "/home/lkh/siga/dataset/deepcad/data/cad_ply/body2/operate"
    # path_c = "/home/lkh/siga/dataset/deepcad/data/cad_ply/body2/result"
    # path_d = "/home/lkh/siga/dataset/deepcad/data/cad_ply/views_correct.pkl"

    # compare_filenames(path_a, path_c)
    # get_pkl(path_d)
    # compare_keys_and_files(path_d, path_c)

    # h5_dir = '/home/lkh/siga/CADIMG/output'
    # h5_files = os.listdir(h5_dir)
    # for i, file in enumerate(h5_files):
    #     h5_path = os.path.join(h5_dir, file)
    #     read_normal_imgs_h5(h5_path)
    # from_ply_get_box()
    merge_imgs(args.path)
    # get_pkl('/home/lkh/siga/CADIMG/datasets/render_normal/centers.pkl')