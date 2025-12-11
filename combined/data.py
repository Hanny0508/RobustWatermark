import os
import glob
import random
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import config as c

class StegoDataset(Dataset):
    """隐写术数据集：从同一文件夹自动拆分host和secret（无需单独secret文件夹）"""
    def __init__(self, is_train=True):
        # 数据集路径（直接使用config中的TRAIN_PATH/VAL_PATH，无secret子路径）
        self.root = c.TRAIN_PATH if is_train else c.VAL_PATH
        self.img_format = c.format_train if is_train else c.format_val

        # 步骤1：读取文件夹内所有图像路径
        self.img_paths = glob.glob(os.path.join(self.root, f'*.{self.img_format}'))
        if len(self.img_paths) == 0:
            raise ValueError(f"未找到图像，请检查路径: {self.root}，格式: {self.img_format}")

        # 步骤2：拆分host和secret（两种方案选其一，推荐方案1）
        # ========== 方案1：随机配对（推荐，数量不变，每个host对应随机secret） ==========
        self.host_paths = self.img_paths  # host用所有图像
        self.secret_paths = self.img_paths.copy()  # secret用相同列表，然后随机打乱
        random.shuffle(self.secret_paths)  # 打乱secret列表，实现随机配对

        # ========== 方案2：均分列表（前一半为host，后一半为secret，数量减半） ==========
        # split_idx = len(self.img_paths) // 2
        # self.host_paths = self.img_paths[:split_idx]
        # self.secret_paths = self.img_paths[split_idx:]

        # 确保host和secret数量匹配（方案1必然匹配，方案2也匹配）
        self.length = min(len(self.host_paths), len(self.secret_paths))

        # 数据预处理（与模型输入适配，保持原有逻辑）
        self.transform = transforms.Compose([
            transforms.RandomCrop(c.cropsize if is_train else c.cropsize_val),
            transforms.RandomHorizontalFlip(p=0.5) if is_train else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),  # 转为0~1张量
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # 归一化到-1~1（适配DCGAN的Tanh输出）
        ])

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # 加载宿主图像和秘密图像（从同一文件夹的不同图像中读取）
        host_img = Image.open(self.host_paths[idx]).convert('RGB')
        secret_img = Image.open(self.secret_paths[idx]).convert('RGB')

        # 预处理
        host = self.transform(host_img)
        secret = self.transform(secret_img)

        return host, secret

# 测试数据集加载（可保留，用于验证）
if __name__ == "__main__":
    dataset = StegoDataset(is_train=True)
    host, secret = dataset[0]
    print(f"宿主图像形状: {host.shape}, 秘密图像形状: {secret.shape}")
    print(f"数据集总长度: {len(dataset)}")
