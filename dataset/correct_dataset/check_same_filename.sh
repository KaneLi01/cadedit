#!/bin/bash

# 用于检查某目录下的若干子目录中，不同子目录的文件名是否相同
# 用于测试train/test/val文件夹下的子目录文件名是否相同

DIR_ORI="/home/lkh/siga/dataset/my_dataset/cad_rgb_imgs/cad_controlnet_cube_light/train"  # 待检测的目录
declare -A file_sets
reference_set=""
reference_dir=""
all_match=true

echo "每个子目录中的文件数量："
for subdir in "$DIR_ORI"/*/; do
    [ -d "$subdir" ] || continue
    count=$(find "$subdir" -maxdepth 1 -type f | wc -l)
    echo "$(basename "$subdir")：$count 个文件"

    # 获取文件名（不含路径），排序后组成文件名集合字符串
    files=$(find "$subdir" -maxdepth 1 -type f -exec basename {} \; | sort)
    files_joined=$(echo "$files" | tr '\n' ' ')

    # 保存文件名集合
    file_sets["$subdir"]="$files_joined"

    # 记录第一个子目录的文件名集合作为参考
    if [ -z "$reference_set" ]; then
        reference_set="$files_joined"
        reference_dir="$subdir"
    else
        if [ "$files_joined" != "$reference_set" ]; then
            all_match=false
        fi
    fi
done

echo
if $all_match; then
    echo "所有子目录中的文件名完全相同。"
else
    echo "存在文件名不同的子目录："
    for dir in "${!file_sets[@]}"; do
        if [ "${file_sets[$dir]}" != "$reference_set" ]; then
            echo "$(basename "$dir") 中的不同文件名："

            # 打印不同的文件名（与参考集比较）
            diff <(echo "$reference_set" | tr ' ' '\n' | sort) \
                 <(echo "${file_sets[$dir]}" | tr ' ' '\n' | sort)
            echo
        fi
    done
fi
