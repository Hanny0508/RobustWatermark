import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
from torchvision import transforms
import glob
import matplotlib.pyplot as plt
from tqdm import tqdm
import config as c  # 导入config


# 固定随机种子（提升可复现性）
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(42)

# ===================== 设备配置 =====================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {DEVICE}")

# 创建路径（确保所有路径存在）
os.makedirs(c.CHECKPOINT_PATH, exist_ok=True)
os.makedirs(c.LOG_PATH, exist_ok=True)
os.makedirs(c.IMAGE_PATH_host, exist_ok=True)
os.makedirs(c.IMAGE_PATH_container, exist_ok=True)
os.makedirs(c.IMAGE_PATH_secret, exist_ok=True)
os.makedirs(c.IMAGE_PATH_extracted, exist_ok=True)
os.makedirs(c.IMAGE_PATH_combined, exist_ok=True)  # 新增：拼接对比图路径


# ===================== 工具函数 =====================
def PSNR(x, y):
    """计算PSNR（批次级，返回张量）"""
    x = x.clamp(0, 1)
    y = y.clamp(0, 1)
    mse = F.mse_loss(x, y, reduction='none')
    mse = mse.view(x.shape[0], -1).mean(dim=1)  # 每个样本的MSE
    psnr = 10 * torch.log10(1 / (mse + 1e-8))
    return psnr.mean()  # 返回批次均值（张量）


def SSIM(x, y):
    """计算SSIM（批次级，返回张量）"""
    x = x.clamp(0, 1)
    y = y.clamp(0, 1)
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    mu_x = F.avg_pool2d(x, 3, 1, 1)
    mu_y = F.avg_pool2d(y, 3, 1, 1)

    sigma_x = F.avg_pool2d(x ** 2, 3, 1, 1) - mu_x ** 2
    sigma_y = F.avg_pool2d(y ** 2, 3, 1, 1) - mu_y ** 2
    sigma_xy = F.avg_pool2d(x * y, 3, 1, 1) - mu_x * mu_y

    ssim_map = ((2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)) / \
               ((mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2))
    return ssim_map.mean()


# -------------------- 对抗性攻击函数（FGSM/PGD） --------------------
def fgsm_attack(image, epsilon, data_grad):
    """FGSM对抗性攻击：生成对抗样本"""
    sign_data_grad = data_grad.sign()  # 梯度符号
    perturbed_image = image + epsilon * sign_data_grad
    return torch.clamp(perturbed_image, 0, 1)


def pgd_attack(image, epsilon, data_grad, steps=4, alpha=0.01):
    """PGD对抗性攻击：迭代版FGSM"""
    perturbed_image = image.clone()
    for _ in range(steps):
        sign_data_grad = data_grad.sign()
        perturbed_image = perturbed_image + alpha * sign_data_grad
        # 限制扰动范围
        perturbed_image = torch.clamp(perturbed_image, image - epsilon, image + epsilon)
        perturbed_image = torch.clamp(perturbed_image, 0, 1)
    return perturbed_image


# -------------------- 扩展攻击函数（包含对抗性攻击） --------------------
def attack_image(img, attack_type, container_grad=None, epsilon=0.05):
    """
    扩展攻击函数：包含高斯、JPEG、几何、FGSM、PGD攻击
    img: 输入图像（0~1）
    attack_type: 攻击类型（gaussian/jpeg/geometry/fgsm/pgd）
    container_grad: 图像梯度（用于对抗性攻击）
    epsilon: 扰动强度（对抗性攻击）
    """
    img = img.clamp(0, 1)
    if attack_type == "gaussian":
        noise = torch.randn_like(img) * getattr(c, 'gaussian_noise_std', 0.01)
        return (img + noise).clamp(0, 1)
    elif attack_type == "jpeg":
        img = img * 255
        img = torch.round(img / getattr(c, 'jpeg_quant_step', 10)) * getattr(c, 'jpeg_quant_step', 10)
        return (img / 255).clamp(0, 1)
    elif attack_type == "geometry":
        scale = np.random.uniform(0.9, 1.1)
        h, w = img.shape[2], img.shape[3]
        img_scaled = F.interpolate(img, scale_factor=scale, mode="bilinear", align_corners=False)
        # 裁剪/填充回原尺寸
        img_scaled = img_scaled[:, :, :h, :w] if img_scaled.shape[2] >= h else F.pad(img_scaled,
                                                                                     (0, 0, 0, h - img_scaled.shape[2]))
        img_scaled = img_scaled[:, :, :, :w] if img_scaled.shape[3] >= w else F.pad(img_scaled,
                                                                                    (0, w - img_scaled.shape[3], 0, 0))
        return img_scaled.clamp(0, 1)
    elif attack_type == "fgsm":
        if container_grad is None:
            return img  # 无梯度时返回原图像
        return fgsm_attack(img, epsilon, container_grad)
    elif attack_type == "pgd":
        if container_grad is None:
            return img
        # 使用config中的参数
        return pgd_attack(img, epsilon, container_grad, steps=c.pgd_steps, alpha=c.pgd_alpha)
    return img


# ===================== 示例图片保存函数（优化版） =====================
def save_sample_images(model, dataloader, num_samples=50, epoch=None):
    """
    保存示例图片（优化版）：
    1. 保存单独的宿主/容器/秘密/提取图像
    2. 保存拼接的对比图
    3. 支持按epoch保存
    """
    model.eval()
    count = 0
    epoch_suffix = f"_epoch_{epoch}" if epoch is not None else ""

    with torch.no_grad():
        for host_imgs, secret_imgs in tqdm(dataloader, desc="保存示例图片"):
            if count >= num_samples:
                break

            host_imgs = host_imgs.to(DEVICE)
            secret_imgs = secret_imgs.to(DEVICE)

            # 执行隐写操作（适配EnhancedPRIS的输出）
            if hasattr(model, 'embed'):
                container_imgs = model.embed(host_imgs, secret_imgs)
                extracted_imgs = model.extract(container_imgs)
            else:
                # 兼容原模型的前向传播
                container_imgs, _, _ = model(host_imgs, secret_imgs)
                extracted_imgs = model(container_imgs, secret_imgs, rev=True)

            # 转换为0~1范围
            host_imgs_01 = (host_imgs + 1) / 2
            container_imgs_01 = (container_imgs + 1) / 2
            secret_imgs_01 = (secret_imgs + 1) / 2
            extracted_imgs_01 = extracted_imgs.clamp(0, 1)

            # 保存批次中的每张图像
            for i in range(host_imgs.size(0)):
                if count >= num_samples:
                    break

                # 转换为PIL图像
                def tensor_to_pil(tensor):
                    tensor = tensor.cpu().clamp(0, 1)
                    if tensor.shape[0] == 1:
                        return Image.fromarray((tensor.squeeze(0).numpy() * 255).astype(np.uint8), mode='L')
                    else:
                        return Image.fromarray((tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8), mode='RGB')

                host_img = tensor_to_pil(host_imgs_01[i])
                container_img = tensor_to_pil(container_imgs_01[i])
                secret_img = tensor_to_pil(secret_imgs_01[i])
                extracted_img = tensor_to_pil(extracted_imgs_01[i])

                # 保存单独图像
                idx = count + 1
                host_img.save(os.path.join(c.IMAGE_PATH_host, f'host{epoch_suffix}_{idx}.png'))
                container_img.save(os.path.join(c.IMAGE_PATH_container, f'container{epoch_suffix}_{idx}.png'))
                secret_img.save(os.path.join(c.IMAGE_PATH_secret, f'secret{epoch_suffix}_{idx}.png'))
                extracted_img.save(os.path.join(c.IMAGE_PATH_extracted, f'extracted{epoch_suffix}_{idx}.png'))

                # 保存拼接对比图
                fig, axes = plt.subplots(2, 2, figsize=(12, 12))
                axes[0, 0].imshow(host_img)
                axes[0, 0].set_title("Host Image", fontsize=14)
                axes[0, 0].axis('off')

                axes[0, 1].imshow(container_img)
                axes[0, 1].set_title("Container Image", fontsize=14)
                axes[0, 1].axis('off')

                axes[1, 0].imshow(secret_img)
                axes[1, 0].set_title("Original Secret", fontsize=14)
                axes[1, 0].axis('off')

                axes[1, 1].imshow(extracted_img)
                axes[1, 1].set_title("Extracted Secret", fontsize=14)
                axes[1, 1].axis('off')

                plt.tight_layout()
                plt.savefig(os.path.join(c.IMAGE_PATH_combined, f'combined{epoch_suffix}_{idx}.png'), dpi=300,
                            bbox_inches='tight')
                plt.close()

                count += 1

    print(f"\n已保存{count}张示例图片：")
    print(
        f"- 单独图像：{c.IMAGE_PATH_host} / {c.IMAGE_PATH_container} / {c.IMAGE_PATH_secret} / {c.IMAGE_PATH_extracted}")
    print(f"- 对比图像：{c.IMAGE_PATH_combined}")
    model.train()


# ===================== 模型扩展类（优化版） =====================
# -------------------- 1. DCGAN生成器（融合多域特征） --------------------
class DCGANGenerator(nn.Module):
    """DCGAN生成器：融合多域特征，生成全局鲁棒+局部可逆的容器图像"""

    def __init__(self, in_channels=64, out_channels=3):
        super().__init__()
        self.model = nn.Sequential(
            # 输入：融合后的多域特征 (batch, 64, h, w)
            nn.Conv2d(in_channels, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            # 反卷积上采样（适配图像分辨率）
            nn.ConvTranspose2d(128, 64, 4, 2, 1),  # (b,64,2h,2w)
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            # 输出3通道容器图像（-1~1）
            nn.Conv2d(64, out_channels, 3, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        return self.model(x)


# -------------------- 2. 生成式误差预判器（优化版：支持误差修正） --------------------
class GenerativeErrorPredictor(nn.Module):
    """生成式误差预判：预判像素/rounding/量化误差，联动DCGAN生成器进行修正"""

    def __init__(self, in_channels=3):
        super().__init__()
        # 像素误差预判
        self.pixel_err = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, in_channels, 3, padding=1)
        )
        # Rounding误差预判（模拟图像保存时的四舍五入）
        self.round_err = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, in_channels, 3, padding=1)
        )
        # 量化误差预判（模拟8bit量化）
        self.quant_err = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, in_channels, 3, padding=1)
        )

    def forward(self, x, simulate_round=True, simulate_quant=True):
        """
        x: 容器图像（-1~1）
        return: 总误差（-1~1）+ 各分项误差
        """
        # 像素误差
        pixel_err = self.pixel_err(x)
        total_err = pixel_err

        # Rounding误差：模拟0-255整数化的误差
        if simulate_round:
            x_255 = (x + 1) * 127.5  # 转换为0-255
            x_round = torch.round(x_255)
            x_round_norm = (x_round / 127.5) - 1  # 转回-1~1
            round_err = self.round_err(x - x_round_norm)
            total_err += round_err

        # 量化误差：模拟8bit量化的误差
        if simulate_quant:
            x_quant = torch.clamp(torch.floor((x + 1) * 127.5) / 127.5 - 1, -1, 1)
            quant_err = self.quant_err(x - x_quant)
            total_err += quant_err

        return total_err, pixel_err, round_err, quant_err

    def correct_image(self, x, error, correction_weight=0.1):
        """使用预测误差修正图像"""
        corrected_x = x - correction_weight * error
        return torch.clamp(corrected_x, -1, 1)


# -------------------- 3. 判别器（优化版：支持中间特征提取） --------------------
class Discriminator(nn.Module):
    """DCGAN判别器：支持中间特征提取，用于多域闭环融合"""

    def __init__(self, in_channels=3):
        super().__init__()
        # 拆分模块，便于获取中间特征
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(256, 1, 4, 1, 0),
            nn.Sigmoid()
        )

    def forward(self, x):
        """前向传播：输出判别评分"""
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        return x4

    def get_intermediate_features(self, x):
        """获取中间层特征（layer3输出），用于特征匹配损失"""
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        return x3


# -------------------- 4. 判别器辅助修复头（优化版：利用中间特征） --------------------
class DiscriminatorRefineHead(nn.Module):
    """利用DCGAN判别器的中间特征精细化修复提取的秘密图像"""

    def __init__(self, in_channels=3, disc=None):
        super().__init__()
        self.disc = disc  # 传入DCGAN判别器
        self.refine = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 16, 3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(16, in_channels, 3, padding=1),
            nn.Sigmoid()
        )
        # 特征融合层（判别器中间特征 + 提取图像特征）
        self.feat_fusion = nn.Sequential(
            nn.Conv2d(in_channels + 256, 64, 1, 1, 0),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, in_channels, 1, 1, 0)
        )

    def forward(self, x):
        """
        x: 提取的秘密图像（0~1）
        return: 修复后的秘密图像
        """
        # 初步修复
        x_refine = self.refine(x)

        # 获取判别器中间特征（上采样到原尺寸）
        disc_feat = self.disc.get_intermediate_features(x_refine)
        disc_feat = F.interpolate(disc_feat, size=x.shape[2:], mode='bilinear', align_corners=False)

        # 特征融合
        x_concat = torch.cat([x_refine, disc_feat], dim=1)
        x_fused = self.feat_fusion(x_concat)

        # 自适应修复强度
        disc_score = self.disc(x_refine).mean(dim=[1, 2, 3]).view(-1, 1, 1, 1)
        refine_strength = torch.clamp(1 - disc_score, 0.1, 1.0)

        # 最终修复
        x_final = x_refine * refine_strength + x_fused * (1 - refine_strength)
        return x_final.clamp(0, 1)


# -------------------- 5. 动态权重平衡器（优化版：三目标动态平衡+损失反馈） --------------------
class DynamicWeightBalancer:
    """
    三目标动态平衡器：容量-鲁棒性-不可感知性
    优化点：结合纹理复杂度、攻击类型、当前损失值进行动态调整
    """

    def __init__(self, base_weights={'cap': 1.0, 'rob': 1.0, 'imp': 1.0}):
        self.base_weights = base_weights
        self.loss_history = {'cap': [], 'rob': [], 'imp': []}  # 损失历史

    def adjust(self, texture_complexity, attack_type, current_losses=None):
        """
        texture_complexity: PZMs计算的纹理复杂度（batch维度张量）
        attack_type: 攻击类型（gaussian/jpeg/geometry/fgsm/pgd）
        current_losses: 当前损失值（dict: cap/rob/imp）
        return: 动态权重（cap_w, rob_w, imp_w）→ 标量张量
        """
        # 1. 纹理复杂度归一化（0~1）
        tex_min = torch.min(texture_complexity)
        tex_max = torch.max(texture_complexity)
        tex_norm = (texture_complexity - tex_min) / (tex_max - tex_min + 1e-8)
        tex_norm = torch.mean(tex_norm)  # 批次均值

        # 2. 基础权重：纹理复杂度影响
        cap_w = self.base_weights['cap'] * (1 + tex_norm)
        imp_w = self.base_weights['imp'] * (1 - tex_norm)
        rob_w = torch.tensor(self.base_weights['rob'], device=texture_complexity.device)

        # 3. 攻击类型调整鲁棒性权重
        if attack_type in ['geometry', 'fgsm', 'pgd']:
            rob_w *= 2.0
        elif attack_type in ['jpeg']:
            rob_w *= 1.5
        else:
            rob_w *= 1.0

        # 4. 当前损失值反馈调整（核心：损失越大，权重越高）
        if current_losses is not None:
            # 更新损失历史
            for key in ['cap', 'rob', 'imp']:
                self.loss_history[key].append(current_losses[key])
                if len(self.loss_history[key]) > 10:  # 保留最近10轮
                    self.loss_history[key].pop(0)

            # 损失归一化（相对于历史最大值）
            cap_loss_norm = current_losses['cap'] / (max(self.loss_history['cap']) + 1e-8)
            rob_loss_norm = current_losses['rob'] / (max(self.loss_history['rob']) + 1e-8)
            imp_loss_norm = current_losses['imp'] / (max(self.loss_history['imp']) + 1e-8)

            # 调整权重
            cap_w *= (1 + cap_loss_norm)
            rob_w *= (1 + rob_loss_norm)
            imp_w *= (1 + imp_loss_norm)

        # 5. 权重归一化
        total = cap_w + rob_w + imp_w + 1e-8
        return cap_w / total, rob_w / total, imp_w / total


# ===================== 数据集加载（优化版：添加数据增强） =====================
class WatermarkDataset(Dataset):
    def __init__(self, split="train", max_samples=None):
        self.split = split
        self.img_size = (c.channels_in, c.cropsize, c.cropsize) if split == "train" else (
        c.channels_in, c.cropsize_val, c.cropsize_val)

        # 数据增强（仅训练集）
        if split == "train":
            self.transform = transforms.Compose([
                transforms.Resize((self.img_size[1] + 10, self.img_size[2] + 10), Image.BICUBIC),
                transforms.RandomCrop((self.img_size[1], self.img_size[2])),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((self.img_size[1], self.img_size[2]), Image.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])

        # 从config获取路径
        if split == "train":
            self.img_paths = sorted(glob.glob(os.path.join(c.TRAIN_PATH, f"*.{c.format_train}")))
        else:
            self.img_paths = sorted(glob.glob(os.path.join(c.VAL_PATH, f"*.{c.format_val}")))

        # 检查数据集是否为空
        if len(self.img_paths) == 0:
            raise ValueError(f"数据集路径下未找到图像文件: {c.TRAIN_PATH if split == 'train' else c.VAL_PATH}")

        if max_samples is not None:
            self.img_paths = self.img_paths[:max_samples]

        # 核心：拆分host和secret（随机配对）
        self.host_paths = self.img_paths
        self.secret_paths = self.img_paths.copy()
        random.shuffle(self.secret_paths)

    def __len__(self):
        return len(self.host_paths)

    def __getitem__(self, idx):
        host_path = self.host_paths[idx]
        host_img = Image.open(host_path).convert("RGB")
        host = self.transform(host_img).clamp(-1, 1)

        secret_path = self.secret_paths[idx]
        secret_img = Image.open(secret_path).convert("RGB")
        secret = self.transform(secret_img).clamp(-1, 1)

        return host, secret


# ===================== 训练函数（优化版） =====================
def train_epoch(net, disc, error_predictor, loader, optim_gen, optim_disc, epoch, writer, mode="train",
                weight_balancer=None):
    """
    单轮训练/验证函数（优化版）：
    1. 添加特征匹配损失（多域闭环融合）
    2. 优化梯度计算
    3. 完善损失记录
    4. 批次级进度条
    """
    net.train() if mode == "train" else net.eval()
    disc.train() if mode == "train" else disc.eval()
    error_predictor.train() if mode == "train" else error_predictor.eval()

    total_loss = 0
    psnr_list = []
    ssim_list = []
    epsilon = c.epsilon  # 从config读取攻击强度

    pbar = tqdm(loader, desc=f"{mode} Epoch {epoch + 1}", leave=False)
    for batch_idx, (host, secret) in enumerate(pbar):
        host, secret = host.to(DEVICE), secret.to(DEVICE)
        batch_size = host.shape[0]
        # 扩展攻击类型：从config读取支持的攻击
        attack_type = np.random.choice(c.supported_attacks)

        with torch.set_grad_enabled(mode == "train"):
            # -------------------- 正向嵌入：生成容器图像 --------------------
            if hasattr(net, 'embed'):
                container = net.embed(host, secret)
            else:
                container, pred_error, pzms_feat = net(host, secret)

            # 生成式误差预判 + 图像修正（核心创新点：误差预判用于修正）
            pred_error_total, _, _, _ = error_predictor(container)
            container_corrected = error_predictor.correct_image(container, pred_error_total, c.error_correction_weight)
            container_corrected = torch.clamp(container_corrected, -1, 1)

            # 转换为0~1范围（用于攻击和评估）
            container_01 = (container_corrected + 1) / 2
            host_01 = (host + 1) / 2
            container_01.requires_grad = True if mode == "train" else False

            # -------------------- 生成对抗性攻击所需的梯度 --------------------
            container_grad = None
            if mode == "train" and attack_type in ["fgsm", "pgd"]:
                # 计算判别器对容器图像的预测梯度（更高效）
                pred = disc(container_01)
                loss_temp = F.binary_cross_entropy(pred, torch.ones_like(pred))
                loss_temp.backward(retain_graph=True)
                container_grad = container_01.grad.data
                container_01.grad.zero_()

            # -------------------- 施加攻击 --------------------
            attacked_container = attack_image(container_01, attack_type, container_grad, epsilon)
            actual_error = container_01 - attacked_container.detach()

            # -------------------- 逆向提取：秘密图像 --------------------
            if hasattr(net, 'extract'):
                extracted = net.extract(container_corrected)
            else:
                extracted = net(attacked_container, secret, rev=True)
            extracted = extracted.clamp(0, 1)

            # -------------------- 动态权重计算 --------------------
            # 假设net有pzms_extractor属性（适配原模型）
            if hasattr(net, 'pzms_extractor'):
                pzms_complexity = net.pzms_extractor.get_texture_complexity(host)
            else:
                pzms_complexity = torch.rand(batch_size, device=DEVICE)  # 模拟纹理复杂度

            # 计算基础损失（用于动态权重）
            cap_loss = F.mse_loss(extracted, (secret + 1) / 2)
            rob_loss = F.mse_loss(container_01, attacked_container)
            imp_loss = 1 - SSIM(container_01, host_01)

            # 收集当前损失值
            current_losses = {
                'cap': cap_loss.item(),
                'rob': rob_loss.item(),
                'imp': imp_loss.item()
            }

            # 动态调整权重（三目标动态平衡）
            cap_weight, rob_weight, imp_weight = weight_balancer.adjust(pzms_complexity, attack_type, current_losses)

            # -------------------- 损失计算 --------------------
            # 三目标损失
            content_loss = cap_weight * cap_loss + rob_weight * rob_loss + imp_weight * imp_loss

            # 误差预判损失（生成式误差补偿）
            pred_error_01 = (pred_error_total + 1) / 2
            error_pred_loss = F.mse_loss(pred_error_01, actual_error)

            # 对抗性损失（泛化增强）
            adv_loss = torch.tensor(0.0, device=DEVICE)
            if attack_type in ["fgsm", "pgd"]:
                adv_loss = F.mse_loss(extracted, (secret + 1) / 2) * c.adv_loss_weight

            # 特征匹配损失（多域闭环融合：核心创新点）
            feat_match_loss = torch.tensor(0.0, device=DEVICE)
            if mode == "train":
                real_feat = disc.get_intermediate_features(host_01)
                fake_feat = disc.get_intermediate_features(container_01)
                feat_match_loss = F.mse_loss(fake_feat, real_feat.detach()) * c.feat_match_weight

            current_loss = 0
            if mode == "train":
                # -------------------- 判别器训练 --------------------
                optim_disc.zero_grad()
                real_pred = disc(host_01).view(-1)
                real_loss = F.binary_cross_entropy(real_pred, torch.ones_like(real_pred) * 0.9)  # 标签平滑
                fake_pred = disc(container_01.detach()).view(-1)
                fake_loss = F.binary_cross_entropy(fake_pred, torch.zeros_like(fake_pred))
                disc_loss = (real_loss + fake_loss) / 2
                disc_loss.backward(retain_graph=True)
                torch.nn.utils.clip_grad_norm_(disc.parameters(), max_norm=1.0)  # 梯度裁剪
                optim_disc.step()

                # -------------------- 生成器训练 --------------------
                optim_gen.zero_grad()
                # GAN对抗损失
                gan_pred = disc(container_01).view(-1)
                gan_loss = F.binary_cross_entropy(gan_pred, torch.ones_like(gan_pred))
                # 总损失：融合所有损失项
                total_gen_loss = (
                        content_loss +
                        c.gan_weight * gan_loss +
                        c.error_pred_weight * error_pred_loss +
                        adv_loss +
                        feat_match_loss
                )
                total_gen_loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)  # 梯度裁剪
                optim_gen.step()
                current_loss = total_gen_loss.item()

                # 更新进度条
                pbar.set_postfix({
                    'Loss': f'{current_loss:.4f}',
                    'CapW': f'{cap_weight.item():.2f}',
                    'RobW': f'{rob_weight.item():.2f}',
                    'ImpW': f'{imp_weight.item():.2f}'
                })
            else:
                current_loss = (content_loss + error_pred_loss + adv_loss).item()

            # -------------------- 评估指标收集 --------------------
            total_loss += current_loss
            psnr_list.append(PSNR(extracted, (secret + 1) / 2).item())
            ssim_list.append(SSIM(extracted, (secret + 1) / 2).item())

    # -------------------- 批次平均计算 --------------------
    avg_loss = total_loss / len(loader)
    avg_psnr = np.mean(psnr_list)
    avg_ssim = np.mean(ssim_list)
    print(f"\n{mode} Epoch {epoch + 1}: Loss={avg_loss:.4f}, PSNR={avg_psnr:.2f} dB, SSIM={avg_ssim:.4f}")

    # -------------------- 张量板日志 --------------------
    if writer is not None:
        writer.add_scalar(f"{mode}/Loss", avg_loss, epoch)
        writer.add_scalar(f"{mode}/PSNR", avg_psnr, epoch)
        writer.add_scalar(f"{mode}/SSIM", avg_ssim, epoch)
        writer.add_scalar(f"{mode}/Cap_Weight", cap_weight.item(), epoch)
        writer.add_scalar(f"{mode}/Rob_Weight", rob_weight.item(), epoch)
        writer.add_scalar(f"{mode}/Imp_Weight", imp_weight.item(), epoch)
        writer.add_scalar(f"{mode}/Feat_Match_Loss", feat_match_loss.item(), epoch)
        writer.add_scalar(f"{mode}/Error_Pred_Loss", error_pred_loss.item(), epoch)

    return avg_loss, avg_psnr, avg_ssim


# ===================== 加载/保存最佳模型 =====================
def load_best_model(net, disc, error_predictor, optim_gen, optim_disc, checkpoint_path):
    """加载最佳模型（支持断点续训）"""
    start_epoch = 0
    best_psnr = 0.0
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
        net.load_state_dict(checkpoint['net_state_dict'])
        disc.load_state_dict(checkpoint['disc_state_dict'])
        error_predictor.load_state_dict(checkpoint['error_predictor_state_dict'])
        optim_gen.load_state_dict(checkpoint['optim_gen_state_dict'])
        optim_disc.load_state_dict(checkpoint['optim_disc_state_dict'])
        start_epoch = checkpoint.get('epoch', 0)
        best_psnr = checkpoint.get('best_psnr', 0.0)
        print(f"✅ 加载最佳模型成功：{checkpoint_path}")
        print(f"  - 起始Epoch: {start_epoch + 1}, 历史最佳PSNR: {best_psnr:.2f} dB")
    else:
        print(f"❌ 未找到最佳模型：{checkpoint_path}，将从头开始训练")
    return start_epoch, best_psnr


def save_best_model(net, disc, error_predictor, optim_gen, optim_disc, epoch, psnr, checkpoint_path):
    """保存最佳模型"""
    torch.save({
        'epoch': epoch,
        'net_state_dict': net.state_dict(),
        'disc_state_dict': disc.state_dict(),
        'error_predictor_state_dict': error_predictor.state_dict(),
        'optim_gen_state_dict': optim_gen.state_dict(),
        'optim_disc_state_dict': optim_disc.state_dict(),
        'best_psnr': psnr
    }, checkpoint_path)
    print(f"✅ 保存最佳模型（PSNR: {psnr:.2f} dB）：{checkpoint_path}")


# ===================== 主训练函数（优化版） =====================
def train():
    # -------------------- 初始化组件 --------------------
    # 动态权重平衡器
    base_weights = {
        'cap': getattr(c, 'base_capacity_weight', 1.0),
        'rob': getattr(c, 'base_robustness_weight', 1.0),
        'imp': getattr(c, 'base_imperceptibility_weight', 1.0)
    }
    weight_balancer = DynamicWeightBalancer(base_weights=base_weights)

    # 模型初始化（适配原model.py的EnhancedPRIS）
    try:
        from model import EnhancedPRIS
        disc = Discriminator(in_channels=c.channels_in).to(DEVICE)
        net = EnhancedPRIS(
            in_channels=c.channels_in,
            target_channels=c.target_channels,
            disc=disc
        ).to(DEVICE)
        # 生成式误差预判器
        error_predictor = GenerativeErrorPredictor(in_channels=c.channels_in).to(DEVICE)
    except ImportError as e:
        print(f"⚠️  未找到model.py，使用占位模型：{e}")

        # 占位模型（仅用于测试）
        class DummyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.embed = lambda h, s: h
                self.extract = lambda c: torch.rand_like(c)

        net = DummyModel().to(DEVICE)
        disc = Discriminator(in_channels=c.channels_in).to(DEVICE)
        error_predictor = GenerativeErrorPredictor(in_channels=c.channels_in).to(DEVICE)

    # 优化器
    optim_gen = optim.Adam(
        list(net.parameters()) + list(error_predictor.parameters()),
        lr=c.lr_gen, betas=c.betas, weight_decay=1e-5  # 添加权重衰减
    )
    optim_disc = optim.Adam(
        disc.parameters(),
        lr=c.lr_disc, betas=c.betas, weight_decay=1e-5
    )

    # 学习率调度器
    scheduler_gen = optim.lr_scheduler.StepLR(optim_gen, step_size=c.lr_step_size, gamma=c.lr_gamma)
    scheduler_disc = optim.lr_scheduler.StepLR(optim_disc, step_size=c.lr_step_size, gamma=c.lr_gamma)

    # 数据集
    train_dataset = WatermarkDataset(split="train", max_samples=c.max_train_samples)
    val_dataset = WatermarkDataset(split="val", max_samples=c.max_val_samples)
    train_loader = DataLoader(train_dataset, batch_size=c.batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=c.batchsize_val, shuffle=False, num_workers=0, pin_memory=True)

    # 日志
    writer = SummaryWriter(log_dir=c.LOG_PATH)

    # 加载最佳模型
    best_model_path = os.path.join(c.CHECKPOINT_PATH, "best_model.pt")
    start_epoch, best_psnr = load_best_model(net, disc, error_predictor, optim_gen, optim_disc, best_model_path)

    # 早停参数
    early_stop_patience = c.early_stop_patience
    no_improve_epochs = 0

    # -------------------- 训练循环 --------------------
    print(f"\n开始训练（总Epoch: {c.epochs}，起始Epoch: {start_epoch + 1}）")
    for epoch in range(start_epoch, c.epochs):
        # 训练轮
        train_loss, train_psnr, train_ssim = train_epoch(
            net, disc, error_predictor, train_loader, optim_gen, optim_disc,
            epoch, writer, mode="train", weight_balancer=weight_balancer
        )

        # 验证轮
        if (epoch + 1) % c.val_freq == 0:
            with torch.no_grad():
                val_loss, val_psnr, val_ssim = train_epoch(
                    net, disc, error_predictor, val_loader, optim_gen, optim_disc,
                    epoch, writer, mode="val", weight_balancer=weight_balancer
                )

            # 保存最佳模型
            if val_psnr > best_psnr:
                best_psnr = val_psnr
                save_best_model(net, disc, error_predictor, optim_gen, optim_disc, epoch, val_psnr, best_model_path)
                no_improve_epochs = 0
                # 保存示例图片（最佳模型）
                save_sample_images(net, val_loader, num_samples=c.num_sample_images, epoch=epoch + 1)
            else:
                no_improve_epochs += 1
                print(f"⚠️  验证PSNR连续{no_improve_epochs}轮未提升（耐心：{early_stop_patience}）")

                # 早停机制
                if no_improve_epochs >= early_stop_patience:
                    print(f"🛑 早停触发，停止训练（Epoch: {epoch + 1}）")
                    break

        # 保存定期检查点
        if (epoch + 1) % c.save_freq == 0:
            checkpoint_path = os.path.join(c.CHECKPOINT_PATH, f"checkpoint_{epoch + 1}.pt")
            torch.save({
                'epoch': epoch,
                'net_state_dict': net.state_dict(),
                'disc_state_dict': disc.state_dict(),
                'error_predictor_state_dict': error_predictor.state_dict(),
                'optim_gen_state_dict': optim_gen.state_dict(),
                'optim_disc_state_dict': optim_disc.state_dict(),
                'best_psnr': best_psnr
            }, checkpoint_path)
            print(f"💾 保存检查点：{checkpoint_path}")

        # 更新学习率
        scheduler_gen.step()
        scheduler_disc.step()

    # 训练完成：保存最终示例图片
    print("\n🎉 训练完成，保存最终示例图片...")
    save_sample_images(net, val_loader, num_samples=c.num_sample_images)

    writer.close()
    print(f"\n训练结果：最佳PSNR = {best_psnr:.2f} dB")


if __name__ == "__main__":
    train()
