#!/bin/bash
PLY_DIR='/home/kane/code/datasets/cad_ply/addbody/base'
HDF5_DIR='/home/kane/code/output/render_6views/'
IMG_OUTPUT_DIR='/home/kane/code/output/render_6views_png/'

# 本地使用的变量
PLY_DIR_PARENT='/home/kane/code/datasets/cad_ply/addbody'
VIEW_PATH='/home/kane/code/cadedit/dataset/create_dataset/render_normal/views_correct.pkl'
CENTER_PATH='/home/kane/code/cadedit/dataset/create_dataset/render_normal/centers_correct.pkl'
OUT_DIR='/home/kane/code/output/render_6views'
CORRECT_INDEX='/home/kane/code/cadedit/dataset/correct_dataset/vaild_train_dataset_names.json'


for file in "$PLY_DIR"/*.ply; do
    file_name=$(basename "$file" .ply)
    echo "Processing file: $file_name"
    for SHAPE_TYPE  in 'base' 'sketch' 'target'; do
    # for SHAPE_TYPE in 'base'; do
    # 渲染三种不同形状

        DIR="${HDF5_DIR}${SHAPE_TYPE}/${file_name}/"
        OUTPUT_DIR="${IMG_OUTPUT_DIR}${SHAPE_TYPE}/${file_name}"

        # 渲染h5文件
        blenderproc run render_normal.py --shape_type "${SHAPE_TYPE}" --ply_name "${file_name}" \
        --view_path "${VIEW_PATH}" --center_path "${CENTER_PATH}" --ply_dir "${PLY_DIR_PARENT}" --output_dir "${OUT_DIR}" --correct_index "${CORRECT_INDEX}"
        # 从h5文件中渲染图片
        # blenderproc vis hdf5 "${DIR}0.hdf5" "${DIR}1.hdf5" "${DIR}2.hdf5"  "${DIR}3.hdf5" "${DIR}4.hdf5"  "${DIR}5.hdf5" --keys "normals" --save "${OUTPUT_DIR}"

        # python /home/lkh/siga/CADIMG/utils/file_util.py --path "${OUTPUT_DIR}"
    done
done