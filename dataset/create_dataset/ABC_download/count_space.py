from collections import defaultdict

def parse_and_sum(file_path):
    size_map = defaultdict(int)

    with open(file_path, 'r') as f:
        for line in f:
            if ':' not in line:
                continue
            name, size_str = line.strip().split(':')
            size = int(size_str.strip())

            # 提取文件类型，比如 'feat', 'obj', etc.
            parts = name.split('_')
            if len(parts) < 4:
                continue
            file_type = parts[2]  # abc_0000_feat_v00.7z → parts = ['abc', '0000', 'feat', 'v00.7z']

            size_map[file_type] += size

    # 打印总和（单位换算为 GB）
    t = 0
    for file_type in sorted(size_map):
        gb = size_map[file_type] / (1024 ** 2) / 176 * 2.4
        t+=gb
        print(f"{file_type:<8}: {gb:.2f} G")
    print(t)

# 用你的文件名替换 'abc_file_sizes.txt'
parse_and_sum('dx.txt')
