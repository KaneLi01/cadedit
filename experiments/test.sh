# for num in 2e-2 5e-3 
# do
#     python train_cn.py --weight_decay $num --batch_size 12 --lr 1e-5 --num_epochs 15 --torch_dtype float16
# done

DATA_DIR="/home/lkh/siga/dataset/my_dataset/cad_rgb_imgs/cad_controlnet_cube_light"
BASE="/home/lkh/siga/dataset/my_dataset/cad_rgb_imgs/cad_controlnet_cube_light/test/base_img/007200.png"
SKETCH="/home/lkh/siga/dataset/my_dataset/cad_rgb_imgs/cad_controlnet_cube_light/test/sketch_img/007200.png"

python /home/lkh/siga/CADIMG/experiments/scripts/cad_edit.py \
        --test_img_path "$BASE" \
        --test_sketch_path "$SKETCH" \
        --index "0526_2335" \
        --device "cuda"