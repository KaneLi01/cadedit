import os
from pathlib import Path
import shutil

def get_relative_dirs(dir):
    # 获取子文件夹的名称
    base_path = Path(dir)
    relative_dirs = set()
    
    for root, dirs, _ in os.walk(dir):
        current_path = Path(root)
        for d in dirs:
            rel_path = os.path.relpath(current_path / d, base_path)
            relative_dirs.add(rel_path)
    
    return relative_dirs

def compare_directories(dir1, dir2):
    # 比较子文件夹之前的差异
    dirs1 = get_relative_dirs(dir1)
    dirs2 = get_relative_dirs(dir2)
    
    only_in_dir1 = dirs1 - dirs2
    only_in_dir2 = dirs2 - dirs1

    print("\n只在第一个目录中存在的文件夹:")
    for path in sorted(only_in_dir1):
        print(f"- {path}")

    print("\n只在第二个目录中存在的文件夹:")
    for path in sorted(only_in_dir2):
        print(f"- {path}")

def delete_empty_directories(dir):
    # 删除没有文件的子目录
    subdirs = get_relative_dirs(dir)
    
    for subdir in subdirs:
        # 跳过根目录本身
        subdir_path = os.path.join(dir, subdir)
        if not os.path.isdir(subdir_path):
            return False 

        if not os.listdir(subdir_path):  # 如果目录为空
            os.rmdir(subdir_path)  # 删除空目录
            print(f"已删除空目录: {subdir_path}")

def chech_files_num(dir, n=6):
    # 检查目录下的子目录中的文件个数是否满足要求
    subdirs = get_relative_dirs(dir)
    
    for subdir in subdirs:
        # 跳过根目录本身
        subdir_path = os.path.join(dir, subdir)
        l = len(os.listdir(subdir_path))
        if not os.path.isdir(subdir_path):
            return False 

        if l != n:  
            print(f"{subdir_path}目录下只有{l}个文件")
            # shutil.rmtree(subdir_path)



if __name__ == "__main__":
    
    dir1 = '/home/lkh/siga/dataset/my_dataset/normals_train_dataset/train_dataset_6views/val/base_img'
    dir2 = '/home/lkh/siga/dataset/my_dataset/normals_train_dataset/train_dataset_6views/val/sketch_img'

    compare_directories(dir1, dir2)
    
    
