#!/bin/bash

path_A="/home/lkh/siga/dataset/my_dataset/normals_train_dataset/normal_img_addbody/result"  
path_B="/home/lkh/siga/dataset/my_dataset/normals_train_dataset/train_dataset/target_img"  # 目标路径


# # 遍历路径A下的所有子文件夹
# for folder in "$path_A"/*/; do
#     # 获取文件夹名（去掉路径和末尾的/）
#     folder_name=$(basename "$folder")
    
#     # 检查文件夹中是否存在0.png到3.png
#     for i in {0..3}; do
#         source_file="$folder/$i.png"
#         if [ -f "$source_file" ]; then
#             # 构建新文件名并复制文件
#             new_name="${folder_name}_${i}.png"
#             cp "$source_file" "$path_B/$new_name"
#             echo "已复制: $source_file -> $path_B/$new_name"
#         else
#             echo "警告: 文件 $source_file 不存在"
#         fi
#     done
# done

# echo "所有文件处理完成"



for folder in "$path_A"/*/; do
    # 获取文件夹名（去掉路径和末尾的/）
    folder_name=$(basename "$folder")
    
    # 检查文件夹中是否存在 0_normals.png 到 3_normals.png
    for i in {0..3}; do
        source_file="$folder/${i}_normals.png"  # 文件名格式如 0_normals.png
        if [ -f "$source_file" ]; then
            # 构建新文件名（去掉 _normals）
            new_name="${folder_name}_${i}.png"  # 例如 A1_0.png
            cp "$source_file" "$path_B/$new_name"
            echo "已复制: $source_file -> $path_B/$new_name"
        else
            echo "警告: 文件 $source_file 不存在"
        fi
    done
done

echo "所有文件处理完成"