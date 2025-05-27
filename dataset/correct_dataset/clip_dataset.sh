#!/bin/bash

src_dir="/home/lkh/siga/dataset/my_dataset/normals_train_dataset/train_dataset/target_img"
dir_a="/home/lkh/siga/dataset/my_dataset/normals_train_dataset/train_dataset/train/target_img"
dir_b="/home/lkh/siga/dataset/my_dataset/normals_train_dataset/train_dataset/test/target_img"
dir_c="/home/lkh/siga/dataset/my_dataset/normals_train_dataset/train_dataset/val/target_img"

mkdir -p "$dir_a" "$dir_b" "$dir_c"

# 移动前29400个文件到A
find "$src_dir" -maxdepth 1 -type f | head -n 29400 | xargs -I {} mv {} "$dir_a"

# 移动接下来的3675个文件(29401-33075)到B
find "$src_dir" -maxdepth 1 -type f | head -n 3675 | xargs -I {} mv {} "$dir_b"

# 移动剩下的3675个文件(33076-36750)到C
find "$src_dir" -maxdepth 1 -type f | head -n 3675 | xargs -I {} mv {} "$dir_c"