# for num in 2e-5 4e-5 5e-6 2e-6
# do
#     echo "正在执行 --lr $num"
#     python train_cn.py --lr $num
# done


CP="/home/lkh/siga/ckpt/controlnet_canny"
DD="/home/lkh/siga/dataset/my_dataset/cad_rgb_imgs/cad_controlnet_cube_dark/train"
EP='/home/lkh/siga/ckpt/clip-vit-base-patch32'

python train_cn.py --lam 0.0 --controlnet_path $CP --file_path $DD --batch_size 20 --img_encoder_path $EP