DIR='/home/kane/code/output/render_6views/base/'
NAME='00006669'
OUTPUT='/home/kane/code/output/render_6views_png/base/'

PATH="${DIR}""${NAME}"
OUTPUT_PATH="${OUTPUT}""${NAME}"

blenderproc vis hdf5 "${PATH}0.hdf5" "${PATH}1.hdf5" "${PATH}2.hdf5"  "${PATH}3.hdf5" "${PATH}4.hdf5" "${PATH}5.hdf5" --keys "normals" --save "${OUTPUT_PATH}"