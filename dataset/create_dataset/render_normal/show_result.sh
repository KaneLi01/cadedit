#!/bin/bash
PLY_DIR='/home/lkh/siga/dataset/deepcad/data/cad_ply/body2/base'
HDF5_DIR='/home/lkh/siga/dataset/deepcad/data/cad_img/normal_body2/'
IMG_OUTPUT_DIR='/home/lkh/siga/dataset/deepcad/data/cad_img/normal_body2_img/'


for file in "$PLY_DIR"/*.ply; do
    file_name=$(basename "$file" .ply)
    echo "Processing file: $file_name"
    for SHAPE_TYPE  in 'base' 'operate' 'result'; do
    # 渲染三种不同形状

        DIR="${HDF5_DIR}${SHAPE_TYPE}/${file_name}/"
        OUTPUT_DIR="${IMG_OUTPUT_DIR}${SHAPE_TYPE}/${file_name}"

        # 渲染h5文件
        blenderproc run render_normal.py --shape_type "${SHAPE_TYPE}" --ply_name "${file_name}"
        # 从h5文件中渲染图片
        # blenderproc vis hdf5 "${DIR}0.hdf5" "${DIR}1.hdf5" "${DIR}2.hdf5"  "${DIR}3.hdf5" --keys "normals" --save "${OUTPUT_DIR}"

        # python /home/lkh/siga/CADIMG/utils/file_util.py --path "${OUTPUT_DIR}"
    done
done