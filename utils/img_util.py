from PIL import Image
import os
import cv2
import numpy as np
import json

def merge_and_save_images(
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
    return merged


def process_image(input_path, output_path):
    # 打开原始图像
    with Image.open(input_path) as img:
        # 计算新的宽度（高度保持512）
        original_width, original_height = img.size
        if original_width != 512 or original_height != 512:
            print(f"警告: 图像尺寸不是512x512，而是{original_width}x{original_height}")
        
        # 计算缩放后的宽度 (512/13*15 ≈ 590.77)
        new_width = int(512 / 13 * 15)
        new_height = int(512 / 13 * 15)  # 保持高度不变
        
        # 缩放图像
        scaled_img = img.resize((new_width, new_height), Image.NEAREST)
        
        # 计算裁剪区域 (中心512x512)
        left = (new_width - 512) / 2
        top = (new_height - 512) / 2
        right = left + 512
        bottom = top + 512
        
        # 裁剪图像
        cropped_img = scaled_img.crop((left, top, right, bottom))
        
        # 保存结果
        cropped_img.save(output_path)


def keep_black_make_transparent(input_path, output_path, threshold=30):
    # 打开原始图像并转换为RGBA模式(确保有透明度通道)
    with Image.open(input_path).convert("RGBA") as img:
        # 获取像素数据
        pixels = img.load()
        
        width, height = img.size
        
        # 处理每个像素
        for y in range(height):
            for x in range(width):
                r, g, b, a = pixels[x, y]
                
                # 判断是否为黑色(或接近黑色)
                if r <= threshold and g <= threshold and b <= threshold:
                    # 保留黑色像素(保持原样)
                    pass
                else:
                    # 其他像素设为完全透明
                    pixels[x, y] = (0, 0, 0, 0)
        
        # 保存为PNG(支持透明度)
        img.save(output_path, "PNG")
        print(f"处理完成: {input_path} -> {output_path}")


def black_white(input_path, output_path, threshold=128):

    with Image.open(input_path).convert("RGB") as img:
        # 获取像素数据
        pixels = img.load()
        width, height = img.size

        for y in range(height):
            for x in range(width):
                r, g, b = pixels[x, y]
                
                if r <= threshold and g <= threshold and b <= threshold:
                    pixels[x, y] = (0, 0, 0)
                else: pixels[x, y] = (255, 255, 255)

        img.save(output_path, "PNG")



def solid_black_with_transparency(input_path, output_path):
    """
    将RGBA图像中所有不透明像素改为纯黑色，保持透明部分不变
    
    参数:
        input_path: 输入图片路径
        output_path: 输出图片路径
    """
    # 打开原始图像并确保是RGBA模式
    with Image.open(input_path).convert("RGBA") as img:
        # 获取像素数据
        pixels = img.load()
        width, height = img.size
        
        # 处理每个像素
        for y in range(height):
            for x in range(width):
                r, g, b, a = pixels[x, y]
                
                # 如果像素不透明（alpha > 0）
                if a > 0:
                    # 设置为纯黑色（保持原始alpha值）
                    pixels[x, y] = (0, 0, 0, a)
                # 完全透明的像素保持不变
        
        # 保存为PNG（保持透明度）
        img.save(output_path, "PNG")


def resize_imgs():
    input_dir = '/home/lkh/siga/dataset/my_dataset/normals_train_dataset/sketch_temp1'
    output_dir = '/home/lkh/siga/dataset/my_dataset/normals_train_dataset/sketch_temp2'

    img_names = os.listdir(input_dir)
    for img_name in img_names:
        img_path = os.path.join(input_dir, img_name)
        output_path = os.path.join(output_dir, img_name)
        process_image(img_path, output_path)


def get_contour(img_path, output_path):
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"未能读取图片：{img_path}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY_INV)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    height, width = gray.shape
    contour_image = np.ones((height, width), dtype=np.uint8) * 255
    cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 2)

    cv2.imwrite(output_path, contour_image)
    
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


def hb_imgs(img1, img2, output_path):
    A = cv2.imread(img1)
    B = cv2.imread(img2)
    if A.shape != B.shape:
        raise ValueError("A 和 B 图片大小不同，无法逐像素处理")
    
    # 创建一个掩码：检测 A 是否为黑色像素（所有通道都是 0）
    mask_black = np.all(B == [0, 0, 0], axis=-1)  # shape: (H, W)，bool数组
    C = A.copy()
    C[mask_black] = B[mask_black]
    cv2.imwrite(output_path, C)


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
