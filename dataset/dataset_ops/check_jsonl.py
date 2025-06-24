import json
import os 

def deduplicate_jsonl(jsonl_path, output_path):
    name_dict = {}  # name -> (line_num, dict)
    duplicates = {}  # name -> list of (line_num, dict)

    # 读取并记录所有字典（按行号存储）
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    for line_num, line in enumerate(lines, 1):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
            if not isinstance(obj, dict):
                print(f"[第{line_num}行错误] 不是字典类型")
                continue

            name = obj.get("name")
            if name is None:
                print(f"[第{line_num}行错误] 缺少 name 字段")
                continue

            if name in name_dict:
                # 是重复项：记录所有重复项
                if name not in duplicates:
                    duplicates[name] = [(name_dict[name][0], name_dict[name][1])]  # 加入第一次出现的
                duplicates[name].append((line_num, obj))
            else:
                name_dict[name] = (line_num, obj)

        except json.JSONDecodeError as e:
            print(f"[第{line_num}行错误] JSON解析失败: {e}")

    # 构建新的列表，决定哪些行保留
    keep_lines = set()
    deleted_lines = set()

    for name, items in duplicates.items():
        valid_items = [(ln, d) for ln, d in items if d.get("valid") is True]
        invalid_items = [(ln, d) for ln, d in items if d.get("valid") is False]

        if valid_items:
            # 保留 valid 为 true 的第一个（或所有，如果你愿意）
            keep_lines.add(valid_items[0][0])
        else:
            # 没有 valid 为 true 的，只保留第一项（可以改逻辑）
            keep_lines.add(items[0][0])

        # 其他都标记为删除
        for ln, _ in invalid_items:
            deleted_lines.add(ln)

    # 将其余非重复 name 添加进保留集合
    all_kept_names = {ln for ln, _ in name_dict.values()} - deleted_lines
    keep_lines |= all_kept_names

    # 写入新文件
    with open(output_path, 'w', encoding='utf-8') as f_out:
        for i, line in enumerate(lines, 1):
            if i in keep_lines:
                f_out.write(line.strip() + '\n')

    print(f"✅ 去重完成：保留 {len(keep_lines)} 行，删除 {len(deleted_lines)} 行")


def is_valid_dic(dic, line_num, name_index):
    required_keys = ["name", "valid", "child_num", "face_num", "wire_num", "bbox_min_max", "bbox_center"]
    for key in required_keys:
        if key not in dic:
            raise Exception(f"第{line_num}行缺少键：{key}")
    

    if not (
        isinstance(dic["name"], str)
        and len(dic["name"]) == 8 
        and dic["name"].isdigit()
        and f"00{name_index}0000" <= dic["name"] <= f"00{name_index}9999"
        ):
        raise ValueError(f"第{line_num}行的name格式错误")
    
    # 直接检查序号
    if not dic["name"] == f"00{name_index}{(line_num-1):04d}":
        print(f"00{name_index}{(line_num-1):04d}")
        raise ValueError(f"第{line_num}行的name序号错误")

    if not isinstance(dic["valid"], bool):
        raise ValueError(f"第{line_num}行的vaild格式错误")

    if dic["valid"]:
        child_num = dic["child_num"]
        if not (isinstance(child_num, int)):
            raise ValueError(f"第{line_num}行的child_num应为整数")
        if not (isinstance(dic["face_num"], list) and len(dic["face_num"]) == child_num and all(isinstance(x, int) for x in dic["face_num"])):
            raise ValueError(f"第{line_num}行的face_num有误")
        if not (isinstance(dic["wire_num"], list) and len(dic["wire_num"]) == child_num and all(isinstance(x, int) for x in dic["wire_num"])):
            raise ValueError(f"第{line_num}行的wire_num有误")

        if not isinstance(dic["bbox_min_max"], list):
            raise ValueError(f"第{line_num}行的 bbox_min_max 应为 list 类型")

        if len(dic["bbox_min_max"]) != child_num:
            raise ValueError(f"第{line_num}行的 bbox_min_max 长度应为 child_num={child_num}，实际为 {len(dic['bbox_min_max'])}")

        for sub_idx, sub in enumerate(dic["bbox_min_max"]):
            if not isinstance(sub, list):
                raise ValueError(f"第{line_num}行的 bbox_min_max[{sub_idx}] 应为 list 类型")
            if len(sub) != 6:
                raise ValueError(f"第{line_num}行的 bbox_min_max[{sub_idx}] 应有 6 个元素，实际为 {len(sub)}")
            for val_idx, val in enumerate(sub):
                if not isinstance(val, float):
                    raise ValueError(f"第{line_num}行的 bbox_min_max[{sub_idx}][{val_idx}] 应为 float，实际为 {type(val)}")

        # 检查 bbox_center
        if not isinstance(dic["bbox_center"], list):
            raise ValueError(f"第{line_num}行的 bbox_center 应为 list 类型")

        if len(dic["bbox_center"]) != child_num:
            raise ValueError(f"第{line_num}行的 bbox_center 长度应为 child_num={child_num}，实际为 {len(dic['bbox_center'])}")

        for sub_idx, sub in enumerate(dic["bbox_center"]):
            if not isinstance(sub, list):
                raise ValueError(f"第{line_num}行的 bbox_center[{sub_idx}] 应为 list 类型")
            if len(sub) != 3:
                raise ValueError(f"第{line_num}行的 bbox_center[{sub_idx}] 应有 3 个元素，实际为 {len(sub)}")
            for val_idx, val in enumerate(sub):
                if not isinstance(val, float):
                    raise ValueError(f"第{line_num}行的 bbox_center[{sub_idx}][{val_idx}] 应为 float，实际为 {type(val)}")


    else:
        # valid 为 False，其他字段必须为 None
        for key in ["child_num", "face_num", "wire_num", "bbox_min_max", "bbox_center"]:
            if dic[key] is not None:
                raise ValueError(f"第{line_num}行的{key} 应为 null")


def read_jsonl(jsonl_path):

    name_index = jsonl_path.split('/')[-1].split('.')[0]
    name_set = set()

    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if not isinstance(obj, dict):
                    raise ValueError(f"第{line_num}行不是字典类型：{type(obj)}")
                
                # 检查是否重复
                name = obj.get("name")
                if name in name_set:
                    raise ValueError(f"第{line_num}行的 name 值 '{name}' 重复")
                name_set.add(name)
                is_valid_dic(obj, line_num, name_index)              
            except (json.JSONDecodeError, ValueError) as e:
                print(f"[第{line_num}行错误] {e}")
                return 0


def dedup():
    raw_dir = '/home/lkh/siga/dataset/ABC/temp/filter_log'
    out_dir = '/home/lkh/siga/dataset/ABC/temp/filter_log_dedup'
    files = os.listdir(raw_dir)
    for f in files:
        print(f)
        path = os.path.join(raw_dir, f)
        opath = os.path.join(out_dir, f)
        deduplicate_jsonl(path,opath)


def check():
    raw_dir = '/home/lkh/siga/dataset/ABC/temp/filter_log_dedup'
    files = os.listdir(raw_dir)
    for f in files:
        path = os.path.join(raw_dir, f)
        read_jsonl(path)

def main():
    check()
    # dedup()

    

    # 如果有重复的，删去重复的中为null的那个

if __name__ == "__main__":
    main()

