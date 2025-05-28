import sys
sys.path.append("..")
import os
import cv2
import numpy as np
from vis.vis_utils import get_obj_contour_from_img

'''将制作的初始数据集进行修正'''

def correct_sketch_img(sketch_img_path, align_normal_img_path):
    '''将初始渲染的sketch图像 尽可能贴合到渲染的normla图像上'''

    pass

def correct_sketch_dataset():
    '''修正sketch数据集'''
    sketch_img_path = '/home/lkh/siga/dataset/my_dataset/normals_train_dataset/init_sketch/00017021/0.png'
    align_normal_img_path = '/home/lkh/siga/output/test/correct/bg/result.png'
    output_path = '/home/lkh/siga/output/test/correct/sketch'
    correct_sketch_img(sketch_img_path, align_normal_img_path)
    

def correct_normal_bg_img(obj_img_path, output_img_path):
    '''把背景颜色替换'''
    img = cv2.imread(obj_img_path)
    bg_color = img[0, 0].copy()

    result = img.copy()
    mask = np.all(result == bg_color, axis=2)
    result[mask] = [255, 255, 255]

    cv2.imwrite(os.path.join(output_img_path), result)

def correct_normal_bg_dataset():
    init_normal_img_dir = '/home/lkh/siga/dataset/my_dataset/normals_train_dataset/normal_img_addbody_init'
    output_dir = '/home/lkh/siga/dataset/my_dataset/normals_train_dataset/normal_img_addbody'
    type_name = os.listdir(init_normal_img_dir)  # base
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
                    correct_normal_bg_img(img_path, output_img_path)



def main():

    correct_normal_bg_dataset()


if __name__ == '__main__':
    main()