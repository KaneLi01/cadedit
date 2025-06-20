""" 主要包含：
    路径、文件的操作，如复制移动读写；
"""

import json, os, inspect
from pathlib import Path
import shutil
from typing import Callable


def load_json_file(json_path):
    with open(json_path, 'r') as f:
        return json.load(f)
    

def load_json_get_dir(json_path, get_key='explaination', get_value='sketch single circle'):
    """
    读取一个记录字典列表的json文件，获取其中特定的值
    用于读取需要的数据集形状名称
    """
    dirs = load_json_file(json_path)
    for dir in dirs:
        if dir[get_key] == get_value:
            return dir


def get_sub_items(dir):
    """
    读取某目录下的内容，返回子目录列表或文件列表（相对）
    """
    base_path = Path(dir)
    if not base_path.exists() or not base_path.is_dir():
        raise ValueError(f"路径问题: {dir}")
    
    subdirs = []
    files = []

    for item in base_path.iterdir():
        if item.is_dir():
            subdirs.append(item.relative_to(base_path))
        elif item.is_file():
            files.append(item.name)

    if subdirs and files:
        raise ValueError("该目录下同时存在目录和文件")

    if subdirs:
        return [str(p) for p in sorted(subdirs)]
    elif files:
        return sorted(files)
    else:
        return []  # 空目录也返回空列表
    

def compare_dirs(dir1, dir2):
    """
    比较不同路径下的内容是否相同
    """
    dirs1 = set(get_sub_items(dir1))
    dirs2 = set(get_sub_items(dir2))
    
    only_in_dir1 = dirs1 - dirs2
    only_in_dir2 = dirs2 - dirs1

    print("\n只在第一个目录中存在的内容:")
    for path in sorted(only_in_dir1):
        print(f"- {path}")

    print("\n只在第二个目录中存在的内容:")
    for path in sorted(only_in_dir2):
        print(f"- {path}")

    return only_in_dir1, only_in_dir2


def check_subidrs_num(dir, n=6, mode='check'):
    """
    传入一个父目录，检查其所有子目录，判断每个子目录下的文件数是否满足要求
    如果传入的n=0，则检查其中没有子文件的子目录；
    如果传入的n不等于0，则检查其中文件数不等于n的子目录
    """
    subdirs = get_sub_items(dir)
    
    for subdir in subdirs:
        subdir_path = os.path.join(dir, subdir)
        l = len(os.listdir(subdir_path))
        if not os.path.isdir(subdir_path):
            return False 

        if n == 0:
            if not os.listdir(subdir_path):  # 如果目录为空
                if mode == 'check':
                    print(f"{subdir_path}目录下没有内容")
                elif mode == 'del':
                    os.rmdir(subdir_path)  # 删除空目录
                else: raise Exception('wrong mode')
        else:
            if l != n:  
                if mode == 'check':
                    print(f"{subdir_path}目录下只有{l}个文件")
                elif mode == 'del':
                    shutil.rmtree(subdir_path)
                else: raise Exception('wrong mode')


def process_files_auto(input_root, output_root, file_handler, op_root=None, *, suffix_filter=None):
    """
    自动根据 file_handler 的参数数量处理单输入或双输入。
    """
    num_params = len(inspect.signature(file_handler).parameters)

    if num_params == 2:
        # 单输入文件处理
        for dirpath, _, filenames in os.walk(input_root):
            for filename in filenames:
                if suffix_filter and not filename.endswith(suffix_filter):
                    continue
                input_path = os.path.join(dirpath, filename)
                rel_path = os.path.relpath(input_path, input_root)
                output_path = os.path.join(output_root, rel_path)
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                file_handler(input_path, output_path)

    elif num_params == 3:
        # 双输入文件处理
        assert op_root is not None, "第二输入路径 op_root 不能为空"
        for dirpath, _, filenames in os.walk(input_root):
            for filename in filenames:
                if suffix_filter and not filename.endswith(suffix_filter):
                    continue
                path1 = os.path.join(dirpath, filename)
                rel_path = os.path.relpath(path1, input_root)
                path2 = os.path.join(op_root, rel_path)
                output_path = os.path.join(output_root, rel_path)
                if not os.path.exists(path2):
                    print(f"跳过未对齐文件: {rel_path}")
                    continue
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                file_handler(path1, path2, output_path)

    else:
        raise ValueError("file_handler 参数数量必须为2或3")


def write_filter_json(json_path, filter: str, names: list):
    '''
    将筛选的数据集name写入json文件。
    该文件是字典列表，包含两个键：筛选条件filter和通过筛选的文件名names。
    DEEPCAD数据集的两个键是 explaination 和 file_names
    '''
    dir = {'filter': filter, 'names': names}

    with open(json_path, 'r') as file:
        current_data = json.load(file)  
    current_data.append(dir)

    with open (json_path, "w") as f:
        json.dump(current_data, f, indent=4)
        f.write("\n")



def test():
    dir1 = '/home/lkh/siga/dataset/ABC/temp/step/02'
    dir2 = '/home/lkh/siga/output/temp'
    check_subidrs_num(dir1, n=0, mode='check')



if __name__ == "__main__":
    test()