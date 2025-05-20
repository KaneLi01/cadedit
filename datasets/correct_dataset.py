import sys
sys.path.append("..")
import os
import cv2
import numpy as np


'''将制作的初始数据集进行修正'''

def get_obj_paras(img):
    '''需要输入白色背景的图片'''
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 254, 255, cv2.THRESH_BINARY_INV)

    points = cv2.findNonZero(binary)
    if points is not None:
        x, y, w, h = cv2.boundingRect(points)
        return x, y, w, h
    else: return None


def correct_sketch_img(sketch_img_path, align_normal_img_path, output_path):
    '''将初始渲染的sketch图像 尽可能贴合到渲染的normla图像上'''
    sketch_img = cv2.imread(sketch_img_path)  
    align_img = cv2.imread(align_normal_img_path)
    if sketch_img is None or align_img is None:
        raise FileNotFoundError("图片读取失败")
    sx, sy, sw, sh = get_obj_paras(sketch_img)
    ax, ay, aw, ah = get_obj_paras(align_img)

    roi2 = sketch_img[sy:sy+sh, sx:sx+sw]  # 裁减
    resized_roi2 = cv2.resize(roi2, (aw, ah), interpolation=cv2.INTER_LINEAR)  # 缩放
    result = np.full_like(align_img, 255)
    result[ay:ay+ah, ax:ax+aw] = resized_roi2

    cv2.imwrite(output_path, result)




def correct_sketch_dataset():
    '''修正sketch数据集'''
    sketch_dir = '/home/lkh/siga/dataset/my_dataset/normals_train_dataset/init_img_sketch/'
    align_normal_img_dir = '/home/lkh/siga/dataset/my_dataset/normals_train_dataset/normal_img_addbody/operate/'
    output_dir = '/home/lkh/siga/dataset/my_dataset/normals_train_dataset/img_sketch'
    type_name = os.listdir(sketch_dir)
    for t in type_name:
        os.makedirs(os.path.join(output_dir, t), exist_ok=True)
        sketch_img_dir = os.path.join(sketch_dir, t)
        align_img_dir = os.path.join(align_normal_img_dir, t)
        sketch_imgs = os.listdir(sketch_img_dir)  # 0.png
        for sketch_img in sketch_imgs:
            try:
                normal_img_name = sketch_img.split('.')[0] + '_normals.png'
                sketch_img_path = os.path.join(sketch_img_dir, sketch_img)
                align_img_path = os.path.join(align_img_dir, normal_img_name)
                output_path = os.path.join(output_dir, t, sketch_img)
                correct_sketch_img(sketch_img_path, align_img_path, output_path)
            except Exception as e:
                print(f"Error processing {t}: {e}")
                continue
    print('finish')

    


    # correct_sketch_img(sketch_img_path, align_normal_img_path)
    

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
    correct_sketch_dataset()


if __name__ == '__main__':
    main()
    # 00584159