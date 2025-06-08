import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from PIL import Image
import cv2
import numpy as np

from utils import change_bg_img, scale_crop_img, get_contour_img, stack_imgs, process_files


'''将渲染的初始图片进行修正'''

def correct_normal_bg_dataset(init_normal_img_dir, output_dir):
    """
    将blender渲染的图片背景替换为白色
    """
    
    type_name = os.listdir(init_normal_img_dir)  # base
    type_name = ["base"]
    for t in type_name:
        type_path = os.path.join(init_normal_img_dir, t)  
        cad_name = os.listdir(type_path)   #00000001
        os.makedirs(os.path.join(output_dir, t), exist_ok=True)
        for c in cad_name:
            img_dir = os.path.join(type_path, c)
            imgs = os.listdir(img_dir)
            os.makedirs(os.path.join(output_dir, t, c), exist_ok=True)
            for i in imgs:
                if i != 'total.png':
                    img_path = os.path.join(img_dir, i)
                    output_img_path = os.path.join(output_dir, t, c, i)
                    img = Image.open(img_path)
                    change_bg_img(img, output_img_path)


def get_final_sketch_img(img1, img2, output_path):
    """
    将blender和pythonocc渲染的sketch进行处理，得到用于训练的图
    img1: pythonocc
    img2: blender
    """  
    img1 = scale_crop_img(img1)
    img2 = get_contour_img(img2)
    _ = stack_imgs(img1, img2, output_path)


def change_bg_img_fp(img_path, output_path):
    img = Image.open(img_path)
    change_bg_img(img, output_path)


def main():
    init_root = '/home/lkh/siga/dataset/my_dataset/normals_train_dataset/normal_img_addbody_6views_init/operate'
    output_root = '/home/lkh/siga/output/temp'
    process_files(init_root, output_root, change_bg_img_fp)



if __name__ == '__main__':
    main()
    # 00584159