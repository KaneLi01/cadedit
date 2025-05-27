#!/bin/bash

# 该脚本用于将某个目录下的文件划分为train/test/val，并移动到对应类型的目录

DIR_ORI="/home/lkh/siga/dataset/my_dataset/cad_rgb_imgs/cad_controlnet01/stroke_img"  # 待处理的路径
DIR_TARGET="/home/lkh/siga/dataset/my_dataset/cad_rgb_imgs/cad_controlnet_cube_light/"  # tarin/test/val的父目录
IMG_TYPE="sketch_img"  # 数据类型

DIR_TRAIN=$DIR_TARGET"train/"$IMG_TYPE
DIR_TEST=$DIR_TARGET"test/"$IMG_TYPE
DIR_VAL=$DIR_TARGET"val/"$IMG_TYPE


mkdir -p "$DIR_TRAIN" "$DIR_TEST" "$DIR_VAL"

# 获取非隐藏文件，排序
files=($(find "$DIR_ORI" -maxdepth 1 -type f ! -name ".*" | sort))

total=${#files[@]}
if [ "$total" -eq 0 ]; then
  echo "没有文件可处理"
  exit 1
fi

# 计算分割索引
split1=$((total * 8 / 10))
split2=$((total * 9 / 10))

# 前 8/10
for ((i=0; i<split1; i++)); do
  mv "${files[i]}" "$DIR_TRAIN/"
done

# 第 8/10 到 9/10
for ((i=split1; i<split2; i++)); do
  mv "${files[i]}" "$DIR_TEST/"
done

# 第 9/10 到 10/10
for ((i=split2; i<total; i++)); do
  mv "${files[i]}" "$DIR_VAL/"
done

echo "文件已移动完毕"
