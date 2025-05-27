import os
from PIL import Image

from torch.utils.data import Dataset
from torchvision import transforms

class NormalSketchControlNetDataset(Dataset):
    def __init__(self, root_dir, mode='train', res=256):
        super().__init__()
        self.res = res
        assert mode in ['train', 'val', 'test'], "mode must be 'train', 'val' or 'test'"

        self.dir = os.path.join(root_dir, mode)
        self.base_dir = os.path.join(self.dir, "base_img")
        self.sketch_dir = os.path.join(self.dir, "sketch_img")
        self.target_dir = os.path.join(self.dir, "target_img")

        self.base_files = sorted(os.listdir(self.base_dir))
        self.sketch_files = sorted(os.listdir(self.sketch_dir))
        self.target_files = sorted(os.listdir(self.target_dir))

        self.files_len = len(self.base_files)
        
        # 检查数据集
        assert self.files_len == len(self.sketch_files) == len(self.target_files), \
            f"count error: base({self.files_len}), sketch({len(self.sketch_files)}), target({len(self.target_files)})"

        for b, s, t in zip(self.base_files, self.sketch_files, self.target_files):
            assert os.path.splitext(b)[0] == os.path.splitext(s)[0] == os.path.splitext(t)[0], \
                f"mismatch: {b} / {s} / {t}"
            
        print(f"total {self.files_len} imgs")

        self.normlize = transforms.Normalize([0.5], [0.5])
        self.transform = transforms.Compose([
                transforms.Resize((self.res, self.res)),
                transforms.ToTensor(),
            ])


    def __len__(self):
        return self.files_len

    def __getitem__(self, idx):
        filename = self.base_files[idx]

        base_path = os.path.join(self.base_dir, filename)  
        sketch_path = os.path.join(self.sketch_dir, filename)  
        target_path = os.path.join(self.target_dir, filename)  


        base_image = self.transform(Image.open(base_path).convert("RGB"))
        sketch_image = self.transform(Image.open(sketch_path).convert("RGB"))
        target_image = self.transform(Image.open(target_path).convert("RGB"))


        return {
            "base": base_image,  
            "sketch": self.normlize(sketch_image),     
            "target": self.normlize(target_image)    
        }

class RGBSketchControlNetCUBEDataset(Dataset):
    def __init__(self, root_dir):

        super().__init__()
        self.root_dir = root_dir
        self.img_dir = os.path.join(root_dir, "init_img")
        self.stroke_dir = os.path.join(root_dir, "stroke_img")
        self.target_dir = os.path.join(root_dir, "result_img")
        self.mask_dir = os.path.join(root_dir, "mask_img")
        # 获取所有图片名字
        self.image_filenames = os.listdir(self.img_dir)
        
        # 定义图像
        self.normlize_1 = transforms.Normalize([0.5], [0.5])
        self.transform = transforms.Compose([
                transforms.Resize((128,128)),
                transforms.ToTensor()
            ])


    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        filename = self.image_filenames[idx]

        img_path = os.path.join(self.img_dir, filename)  # 作为stable diffusion的输入
        stroke_path = os.path.join(self.stroke_dir, filename)  # 手绘灰度图
        target_path = os.path.join(self.target_dir, filename)  # 匹配的目标图
        mask_path = os.path.join(self.mask_dir, filename)

        img_image = self.transform(Image.open(img_path).convert("RGB"))
        stroke_image = self.transform(Image.open(stroke_path).convert("RGB"))
        target_image = self.transform(Image.open(target_path).convert("RGB"))
        mask_image = self.transform(Image.open(mask_path).convert("L"))

        return {
            "original": img_image,  
            "sketch": self.normlize_1(stroke_image),     
            "target": self.normlize_1(target_image),
            "mask": mask_image      
        }
