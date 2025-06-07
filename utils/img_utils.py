from PIL import Image
import os, json
import cv2
import numpy as np



def merge_imgs(
    img_list, 
    save_path, 
    mode='horizontal', 
    grid_size=None, 
    bg_color='white'
):
    """
    合并图片并保存到指定路径

    参数：
    - img_list: list[Image.Image]，PIL图片对象列表
    - save_path: str，保存路径，例如 'output.png'
    - mode: str，'horizontal' / 'vertical' / 'grid'
    - grid_size: (rows, cols)，当 mode='grid' 时必填
    - bg_color: str，背景色（默认白色）

    返回：
    - merged PIL.Image 对象
    """
    if not img_list:
        raise ValueError("图片列表为空")

    # 确保所有图片尺寸相同（可以扩展成自动 resize）
    w, h = img_list[0].size

    if mode == 'horizontal':
        merged = Image.new('RGB', (w * len(img_list), h), color=bg_color)
        for i, img in enumerate(img_list):
            merged.paste(img, (i * w, 0))

    elif mode == 'vertical':
        merged = Image.new('RGB', (w, h * len(img_list)), color=bg_color)
        for i, img in enumerate(img_list):
            merged.paste(img, (0, i * h))

    elif mode == 'grid':
        if grid_size is None:
            raise ValueError("grid_size 必须指定（行数, 列数）")
        rows, cols = grid_size
        if len(img_list) > rows * cols:
            raise ValueError("图片数量超过网格容量")

        merged = Image.new('RGB', (w * cols, h * rows), color=bg_color)
        for idx, img in enumerate(img_list):
            row, col = divmod(idx, cols)
            merged.paste(img, (col * w, row * h))

    else:
        raise ValueError("mode 应为 'horizontal', 'vertical', 或 'grid'")

    merged.save(save_path)



def scale_crop_img(img, output_path=None, scale=15/13):
    """
    将图片放大后裁剪
    用于将pythonocc渲染的wireframe对齐到blender渲染的normal上
    """
    original_width, original_height = img.size  # 获取图片的长宽
    
    new_width = int(original_width * scale)
    new_height = int(original_height * scale)  
    
    scaled_img = img.resize((new_width, new_height), Image.NEAREST)  # 缩放图像
    
    # 计算裁剪区域 (中心512x512)
    left = (new_width - original_width) / 2
    top = (new_height - original_height) / 2
    right = left + original_width
    bottom = top + original_height
    
    cropped_img = scaled_img.crop((left, top, right, bottom))
    if output_path:
        cropped_img.save(output_path)
    return cropped_img
    

def get_contour_img(img, output_path=None):
    """
    获取图片中对象形状的最外层轮廓
    用于从blender渲染的图片中获取轮廓，以和pythonocc渲染的wireframe叠加
    """
    img_cv = np.array(img.convert("RGB"))
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY_INV)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    height, width = gray.shape
    contour_image_cv = np.ones((height, width), dtype=np.uint8) * 255
    cv2.drawContours(contour_image_cv, contours, -1, (0, 255, 0), 2)

    contour_image = Image.fromarray(contour_image_cv)
    if output_path:
        contour_image.save(output_path)
    return contour_image


def stack_imgs(img1, img2, output_path=None):
    """
    将图1叠加在图2上。只有黑色部分叠加，其余保留底图。
    用于将pythonocc渲染的圆柱的两个底面的wireframe和blender渲染的normal轮廓进行叠加。
    """
    img1 = np.array(img1.convert("RGB"))
    img2 = np.array(img2.convert("RGB"))

    if img1.shape != img2.shape:
        raise ValueError("A 和 B 图片大小不同，无法逐像素处理")
    
    # 创建一个掩码：检测 A 是否为黑色像素（所有通道都是 0）
    mask_black = np.all(img2 == [0, 0, 0], axis=-1)  # shape: (H, W)，bool数组
    img3 = img1.copy()
    img3[mask_black] = img2[mask_black]
    stacked_img = Image.fromarray(img3)

    if output_path:
        stacked_img.save(output_path)

    return stacked_img


def change_bg_img(img, output_path=None, color='white'):
    """
    替换图片的背景颜色。
    用于将blender渲染的图片的灰色背景改为白色/黑色
    """
    img = np.array(img.convert("RGB"))
    bg_color = img[0, 0].copy()

    result_img = img.copy()
    mask = np.all(result_img == bg_color, axis=2)
    if color == 'white':
        result_img[mask] = [255, 255, 255]
    elif color == 'black':
        result_img[mask] = [0, 0, 0]
    else: raise Exception('wrong color')

    result_img = Image.fromarray(result_img)
    if output_path:
        result_img.save(output_path)
    return result_img


def resize_imgs():
    input_dir = '/home/lkh/siga/dataset/my_dataset/normals_train_dataset/sketch_temp1'
    output_dir = '/home/lkh/siga/dataset/my_dataset/normals_train_dataset/sketch_temp2'

    img_names = os.listdir(input_dir)
    for img_name in img_names:
        img_path = os.path.join(input_dir, img_name)
        output_path = os.path.join(output_dir, img_name)
        process_image(img_path, output_path)



    
def save_contour():
    ref_filter_file = '/home/lkh/siga/CADIMG/dataset/correct_dataset/vaild_train_dataset_names.json'
    explaination = "sketch single circle"
    with open(ref_filter_file, 'r') as f:
        dirs = json.load(f)
    for dir in dirs:
        if dir['explaination'] == explaination:
            d = dir

    input_dir = '/home/lkh/siga/dataset/my_dataset/normals_train_dataset/normal_img_addbody_6views_temp/sketch_img'
    output_dir = '/home/lkh/siga/dataset/my_dataset/normals_train_dataset/sketch_temp3'
    for i, name in enumerate(dir['file_names']):
        for i in range(0,6):
            img_path = os.path.join(input_dir, name+f'_{i}.png')
            output_path = os.path.join(output_dir, name+f'_{i}.png')
            get_contour(img_path, output_path)

    # get_contour(img_path, output_path)





if __name__ == "__main__":
    
    base_dir = '/home/lkh/siga/dataset/my_dataset/normals_train_dataset/sketch_temp3'
    up_dir = '/home/lkh/siga/dataset/my_dataset/normals_train_dataset/sketch_temp2'

    output_dir = '/home/lkh/siga/dataset/my_dataset/normals_train_dataset/sketch_temp4'
    imgs = os.listdir(base_dir)
    for img in imgs:
        img_path = os.path.join(base_dir, img)
        up_img_path = os.path.join(up_dir, img)
        output_path = os.path.join(output_dir, img)
        hb_imgs(img_path, up_img_path, output_path)
