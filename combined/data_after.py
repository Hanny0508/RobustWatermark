import os
import glob
import random
import warnings
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import config as c

# 设置全局随机种子（保证复现性）
random.seed(42)
torch.manual_seed(42)

class StegoDataset(Dataset):
    """
    隐写术数据集：从同一文件夹（支持子文件夹）自动拆分host和secret
    核心特性：
    1. 支持两种配对方式：随机配对（训练）、固定配对（验证，保证复现）
    2. 自动处理小图像（尺寸不足时先缩放再裁剪）
    3. 训练集增强：随机裁剪、水平翻转、可选色彩抖动；验证集：中心裁剪（固定）
    4. 跳过损坏图像，增强鲁棒性
    5. 支持递归查找子文件夹中的图像
    """
    def __init__(self, is_train=True, pair_strategy="random", enable_color_jitter=False):
        """
        Args:
            is_train (bool): 是否为训练集
            pair_strategy (str): 配对策略，可选 "random"（随机配对，训练用）或 "split"（均分列表，固定配对）或 "fixed"（验证集固定随机配对）
            enable_color_jitter (bool): 是否启用训练集的色彩抖动增强
        """
        # 基础配置
        self.is_train = is_train
        self.pair_strategy = pair_strategy.lower()
        self.enable_color_jitter = enable_color_jitter
        self.root = c.TRAIN_PATH if is_train else c.VAL_PATH
        self.img_format = c.format_train if is_train else c.format_val
        self.crop_size = c.cropsize if is_train else c.cropsize_val

        # 步骤1：递归读取所有图像路径（支持子文件夹，跳过损坏图像）
        self.img_paths = self._get_all_image_paths()
        if len(self.img_paths) == 0:
            raise ValueError(f"未找到有效图像，请检查路径: {self.root}，格式: {self.img_format}")
        print(f"📊 加载{'训练' if is_train else '验证'}集：共找到 {len(self.img_paths)} 张有效图像")

        # 步骤2：拆分host和secret（根据配对策略）
        self.host_paths, self.secret_paths = self._split_host_secret()
        self.length = min(len(self.host_paths), len(self.secret_paths))
        print(f"🔍 配对后数据集长度：{self.length}（策略：{self.pair_strategy}）")

        # 步骤3：构建数据预处理管道（适配图像尺寸，区分训练/验证）
        self.transform = self._build_transform_pipeline()

    def _get_all_image_paths(self):
        """
        辅助函数：递归查找所有图像路径，跳过损坏的图像
        Returns:
            list: 有效图像路径列表
        """
        valid_paths = []
        # 递归查找所有匹配格式的图像（** 表示子文件夹）
        search_pattern = os.path.join(self.root, f"**/*.{self.img_format}")
        for img_path in glob.iglob(search_pattern, recursive=True):
            # 跳过隐藏文件/文件夹（如 .DS_Store）
            if os.path.basename(img_path).startswith('.'):
                continue
            # 验证图像是否可正常加载
            try:
                with Image.open(img_path) as img:
                    img.verify()  # 检查图像完整性
                    # 检查图像尺寸（至少1x1，避免空图像）
                    img = Image.open(img_path).convert('RGB')
                    if img.size[0] > 0 and img.size[1] > 0:
                        valid_paths.append(img_path)
            except (IOError, SyntaxError, Exception) as e:
                warnings.warn(f"⚠️ 跳过损坏图像：{img_path}，错误：{str(e)[:50]}")
        return valid_paths

    def _split_host_secret(self):
        """
        辅助函数：根据配对策略拆分host和secret路径
        Returns:
            tuple: (host_paths, secret_paths)
        """
        if self.pair_strategy == "split":
            # 方案1：均分列表（前一半为host，后一半为secret）
            split_idx = len(self.img_paths) // 2
            host_paths = self.img_paths[:split_idx]
            secret_paths = self.img_paths[split_idx:]
        elif self.pair_strategy == "random":
            # 方案2：随机配对（训练用，打乱secret列表）
            host_paths = self.img_paths.copy()
            secret_paths = self.img_paths.copy()
            random.shuffle(secret_paths)  # 打乱后与host一一配对
        elif self.pair_strategy == "fixed":
            # 方案3：固定随机配对（验证用，设置种子保证复现）
            host_paths = self.img_paths.copy()
            secret_paths = self.img_paths.copy()
            # 使用固定种子打乱，确保每次实例化配对结果一致
            random.seed(42)
            random.shuffle(secret_paths)
        else:
            raise ValueError(f"不支持的配对策略：{self.pair_strategy}，可选：random/split/fixed")
        return host_paths, secret_paths

    def _build_transform_pipeline(self):
        """
        辅助函数：构建数据预处理管道（区分训练/验证，处理小图像）
        Returns:
            transforms.Compose: 预处理管道
        """
        transform_steps = []

        # 步骤1：处理小图像（尺寸小于裁剪尺寸时，先缩放到裁剪尺寸的1.1倍再裁剪）
        transform_steps.append(transforms.Lambda(lambda img: self._resize_small_image(img)))

        # 步骤2：训练集/验证集的裁剪策略（训练：随机裁剪，验证：中心裁剪）
        if self.is_train:
            transform_steps.append(transforms.RandomCrop(self.crop_size))
            # 可选：随机水平/垂直翻转（增强多样性）
            transform_steps.append(transforms.RandomHorizontalFlip(p=0.5))
            transform_steps.append(transforms.RandomVerticalFlip(p=0.2))
        else:
            transform_steps.append(transforms.CenterCrop(self.crop_size))

        # 步骤3：训练集可选色彩抖动（增强鲁棒性）
        if self.is_train and self.enable_color_jitter:
            transform_steps.append(transforms.ColorJitter(
                brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05
            ))

        # 步骤4：转为张量 + 归一化到-1~1（适配模型输入）
        transform_steps.extend([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

        return transforms.Compose(transform_steps)

    def _resize_small_image(self, img):
        """
        辅助函数：调整小图像尺寸（宽度/高度小于裁剪尺寸时，缩放到裁剪尺寸的1.1倍）
        Args:
            img (PIL.Image): 输入图像
        Returns:
            PIL.Image: 调整后的图像
        """
        img_w, img_h = img.size
        min_size = self.crop_size
        if img_w < min_size or img_h < min_size:
            # 计算缩放比例（保持宽高比）
            scale = max(min_size / img_w, min_size / img_h) * 1.1  # 1.1倍留有余量
            new_w = int(img_w * scale)
            new_h = int(img_h * scale)
            img = img.resize((new_w, new_h), Image.Resampling.BICUBIC)  # 高分辨率缩放
        return img

    def __len__(self):
        """返回数据集长度"""
        return self.length

    def __getitem__(self, idx):
        """
        获取索引对应的host和secret图像
        Args:
            idx (int): 索引
        Returns:
            tuple: (host_tensor, secret_tensor)
        """
        # 加载图像（再次验证，避免索引越界或图像损坏）
        try:
            host_img = Image.open(self.host_paths[idx]).convert('RGB')
            secret_img = Image.open(self.secret_paths[idx]).convert('RGB')
        except IndexError:
            # 极端情况：索引越界，返回第一个图像（兜底）
            host_img = Image.open(self.host_paths[0]).convert('RGB')
            secret_img = Image.open(self.secret_paths[0]).convert('RGB')
        except (IOError, SyntaxError) as e:
            warnings.warn(f"⚠️ 加载图像失败，使用备用图像：{str(e)[:50]}")
            # 跳过损坏图像，使用下一个索引的图像（兜底）
            new_idx = (idx + 1) % self.length
            host_img = Image.open(self.host_paths[new_idx]).convert('RGB')
            secret_img = Image.open(self.secret_paths[new_idx]).convert('RGB')

        # 预处理
        host = self.transform(host_img)
        secret = self.transform(secret_img)

        return host, secret

# 测试数据集加载（可保留，用于验证功能）
if __name__ == "__main__":
    # 测试训练集（随机配对，启用色彩抖动）
    train_dataset = StegoDataset(
        is_train=True,
        pair_strategy="random",
        enable_color_jitter=True
    )
    host, secret = train_dataset[0]
    print(f"\n训练集 - 宿主图像形状: {host.shape}, 秘密图像形状: {secret.shape}")
    print(f"训练集总长度: {len(train_dataset)}")

    # 测试验证集（固定配对，禁用色彩抖动）
    val_dataset = StegoDataset(
        is_train=False,
        pair_strategy="fixed",
        enable_color_jitter=False
    )
    host_val, secret_val = val_dataset[0]
    print(f"\n验证集 - 宿主图像形状: {host_val.shape}, 秘密图像形状: {secret_val.shape}")
    print(f"验证集总长度: {len(val_dataset)}")
