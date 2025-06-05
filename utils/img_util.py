from PIL import Image


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
        print(f"处理完成: {input_path} -> {output_path}")



if __name__ == "__main__":
    img_path = '/home/lkh/siga/CADIMG/experiments/test/output/imgs/00000126_sketch.png'
    output_path = '/home/lkh/siga/CADIMG/experiments/test/output/imgs/00000126_sketch_new.png'
    output_path0 = '/home/lkh/siga/CADIMG/experiments/test/output/imgs/00000126_sketch_new0.png'
    output_path1 = '/home/lkh/siga/CADIMG/experiments/test/output/imgs/00000126_sketch_new1.png'
    process_image(img_path, output_path)
    keep_black_make_transparent(output_path, output_path0)
    solid_black_with_transparency(output_path0, output_path1)