#!/bin/bash
NAME='00708426'
NUM=4

python test_sketch_compare.py --name ${NAME} --num ${NUM}


# 渲染h5文件
# blenderproc run render_normal.py --shape_type "${SHAPE_TYPE}" --ply_name "${file_name}"
# 从h5文件中渲染图片
# blenderproc vis hdf5 "${DIR}0.hdf5" "${DIR}1.hdf5" "${DIR}2.hdf5"  "${DIR}3.hdf5" --keys "normals" --save "${OUTPUT_DIR}"

# python /home/lkh/siga/CADIMG/utils/file_util.py --path "${OUTPUT_DIR}"
