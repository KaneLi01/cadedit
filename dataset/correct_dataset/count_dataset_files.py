import os
import json

# 统计某个数据集路径下的train/test/val，并将文件名记录在对应路径下

# 需要统计的数据集路径
root_dir = "/home/lkh/siga/dataset/my_dataset/normals_train_dataset/"  
json_path = "/home/lkh/siga/CADIMG/dataset/correct_dataset/vaild_train_dataset_names.json"  # 写入的字典路径

# 子目录
splits = ['train', 'test', 'val']
# splits = ['normal_img_addbody_6views_temp']

result = {'explaination': 'init dataset', 'file_names': []}

for split in splits:
    split_path = os.path.join(root_dir, split)
    
    # 下一级目录路径
    base_path = os.path.join(split_path, 'base_img')
    sketch_path = os.path.join(split_path, 'sketch_img')
    target_path = os.path.join(split_path, 'target_img')

    # 获取文件名集合（去掉扩展名）
    base_files = sorted([os.path.splitext(f)[0] for f in os.listdir(base_path) if f.endswith('.png')])
    sketch_files = sorted([os.path.splitext(f)[0] for f in os.listdir(sketch_path) if f.endswith('.png')])
    target_files = sorted([os.path.splitext(f)[0] for f in os.listdir(target_path) if f.endswith('.png')])
    print("base/sketch/target文件数量分别为:", len(base_files), len(sketch_files), len(target_files))

    base_names = list(set([n.split('_')[0] for n in base_files]))

    # 判断三个列表是否一致
    if base_files == sketch_files == target_files:
        result['file_names'].extend(base_names)
    else:
        print(f"[警告] 子目录 '{split}' 中的文件名不一致，跳过！")

result['file_names'] = list(set(result['file_names']))
l2 = len(result['file_names'])
print("总文件数为:", l2)


with open(json_path, 'r') as file:
    current_data = json.load(file)  # data 是一个列表，列表的元素是字典
current_data.append(result)

with open (json_path, "w") as f:
    json.dump(current_data, f, indent=4)
    f.write("\n")