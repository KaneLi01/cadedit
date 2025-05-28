import pickle
import numpy as np

def convert_ndarrays_to_lists(obj):
    """递归地将 numpy.ndarray 转为 list"""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_ndarrays_to_lists(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_ndarrays_to_lists(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_ndarrays_to_lists(item) for item in obj)
    else:
        return obj

# 输入和输出路径
input_pkl = '/home/lkh/siga/CADIMG/datasets/render_normal/centers.pkl'       # 替换为你的 pkl 路径
output_pkl = '/home/lkh/siga/CADIMG/datasets/render_normal/centers_correct.pkl'

# 读取原始 pkl 文件
with open(input_pkl, 'rb') as f:
    data = pickle.load(f)

# 转换 numpy.ndarray 为 list
converted_data = convert_ndarrays_to_lists(data)

# 保存新的 pkl 文件
with open(output_pkl, 'wb') as f:
    pickle.dump(converted_data, f)

print(f"转换完成，保存至：{output_pkl}")
