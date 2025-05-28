# for num in 2e-2 5e-3 
# do
#     python train_cn.py --weight_decay $num --batch_size 12 --lr 1e-5 --num_epochs 15 --torch_dtype float16
# done

DATA_DIR="/home/lkh/siga/dataset/my_dataset/cad_rgb_imgs/cad_controlnet_cube_light"

python /home/lkh/siga/CADIMG/experiments/scripts/train_cn.py\
        --file_path $DATA_DIR\
        --tip test_controlnet_ability\
        --torch_dtype float32\
        --batch_size 32\
        --res 128\
        --num_epochs 10\
        --lr 2e-5
