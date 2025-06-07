

DATA_DIR="/home/lkh/siga/dataset/my_dataset/cad_rgb_imgs/cad_controlnet_cube_light"
# BASE="/home/lkh/siga/dataset/my_dataset/normals_train_dataset/train_dataset_6views/val/base_img/00866880_4.png"
# SKETCH="/home/lkh/siga/dataset/my_dataset/normals_train_dataset/train_dataset_6views/val/sketch_img/00866880_4.png"

BASE='/home/lkh/siga/dataset/my_dataset/cad_rgb_imgs/cad_controlnet_cube_light/test/base_img/004809.png'
SKETCH='/home/lkh/siga/dataset/my_dataset/cad_rgb_imgs/cad_controlnet_cube_light/test/sketch_img/004809.png'

python /home/lkh/siga/CADIMG/experiments/scripts/cad_edit.py \
        --test_img_path "$BASE" \
        --test_sketch_path "$SKETCH" \
        --index "0606_2026" \
        --device "cpu" \
        --torch_dtype "torch32" \
        --res 256