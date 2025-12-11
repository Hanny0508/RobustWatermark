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
import config as c  # 导入config
# 注意：若model.py中的EnhancedPRIS和Discriminator与以下改造冲突，可注释原有导入，使用下方重新定义的版本
# from model import EnhancedPRIS, Discriminator

# ===================== 配置参数 =====================
DEVICE = torch.device("cuda")
print(f"使用设备: {DEVICE}")

# 创建路径
os.makedirs(c.CHECKPOINT_PATH, exist_ok=True)
os.makedirs(c.LOG_PATH, exist_ok=True)

# ===================== 工具函数 =====================
def PSNR(x, y):
    x = x.clamp(0, 1)
    y = y.clamp(0, 1)
    mse = F.mse_loss(x, y)
    return 10 * torch.log10(1 / (mse + 1e-8)).item()

def SSIM(x, y):
    x = x.clamp(0, 1)
    y = y.clamp(0, 1)
    mu_x = F.avg_pool2d(x, 3, 1, 1)
    mu_y = F.avg_pool2d(y, 3, 1, 1)
    sigma_x = F.avg_pool2d(x**2, 3, 1, 1) - mu_x**2 + 1e-8
    sigma_y = F.avg_pool2d(y**2, 3, 1, 1) - mu_y**2 + 1e-8
    sigma_xy = F.avg_pool2d(x*y, 3, 1, 1) - mu_x*mu_y + 1e-8
    # 计算 SSIM 映射并返回均值（保留张量类型）
    ssim_map = ((2*mu_x*mu_y + 0.01) * (2*sigma_xy + 0.03)) / \
               ((mu_x**2 + mu_y**2 + 0.01) * (sigma_x + sigma_y + 0.03))
    return ssim_map.mean()  # 移除 .item()，返回标量张量

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
        noise = torch.randn_like(img) * 0.01
        return (img + noise).clamp(0, 1)
    elif attack_type == "jpeg":
        img = img * 255
        img = torch.round(img / 10) * 10
        return (img / 255).clamp(0, 1)
    elif attack_type == "geometry":
        scale = np.random.uniform(0.9, 1.1)
        h, w = img.shape[2], img.shape[3]
        img_scaled = F.interpolate(img, scale_factor=scale, mode="bilinear", align_corners=False)
        # 裁剪/填充回原尺寸
        img_scaled = img_scaled[:, :, :h, :w] if img_scaled.shape[2] >= h else F.pad(img_scaled, (0,0,0,h-img_scaled.shape[2]))
        img_scaled = img_scaled[:, :, :, :w] if img_scaled.shape[3] >= w else F.pad(img_scaled, (0,w-img_scaled.shape[3],0,0))
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


# ===================== 示例图片保存函数 =====================
def save_sample_images(model, dataloader, num_samples=50):
    """保存宿主图像、容器图像、秘密图像和提取的秘密图像示例（使用config.py路径）"""
    # 确保保存目录存在
    os.makedirs(c.IMAGE_PATH_host, exist_ok=True)
    os.makedirs(c.IMAGE_PATH_container, exist_ok=True)
    os.makedirs(c.IMAGE_PATH_secret, exist_ok=True)
    os.makedirs(c.IMAGE_PATH_extracted, exist_ok=True)

    model.eval()
    count = 0

    with torch.no_grad():
        # 同时迭代宿主图像和秘密图像
        for host_imgs, secret_imgs in dataloader:  # 直接迭代val_loader，每次返回(host, secret)
            if count >= num_samples:
                break

            host_imgs = host_imgs.to(DEVICE)
            secret_imgs = secret_imgs.to(DEVICE)

            # 执行隐写操作（根据模型实际方法调整）
            container_imgs = model.embed(host_imgs, secret_imgs)
            extracted_imgs = model.extract(container_imgs)

            # 保存批次中的每张图像
            for i in range(host_imgs.size(0)):
                if count >= num_samples:
                    break

                # 转换为PIL图像（注意：若模型输出为[-1,1]范围需转换为[0,1]）
                host_img = transforms.ToPILImage()(host_imgs[i].cpu().clamp(0, 1))
                container_img = transforms.ToPILImage()(container_imgs[i].cpu().clamp(0, 1))
                secret_img = transforms.ToPILImage()(secret_imgs[i].cpu().clamp(0, 1))
                extracted_img = transforms.ToPILImage()(extracted_imgs[i].cpu().clamp(0, 1))

                # 保存图像（直接使用config.py中的路径）
                idx = count + 1
                host_img.save(os.path.join(c.IMAGE_PATH_host, f'host_{idx}.png'))
                container_img.save(os.path.join(c.IMAGE_PATH_container, f'container_{idx}.png'))
                secret_img.save(os.path.join(c.IMAGE_PATH_secret, f'secret_{idx}.png'))
                extracted_img.save(os.path.join(c.IMAGE_PATH_extracted, f'extracted_{idx}.png'))

                count += 1

    print(f"已保存{count}张示例图片到以下路径：")
    print(f"宿主图像: {c.IMAGE_PATH_host}")
    print(f"容器图像: {c.IMAGE_PATH_container}")
    print(f"秘密图像: {c.IMAGE_PATH_secret}")
    print(f"提取图像: {c.IMAGE_PATH_extracted}")
    model.train()


# ===================== 模型扩展类（替换model.py中的对应模块） =====================
# -------------------- 1. DCGAN生成器（替换原gen_head） --------------------
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

# -------------------- 2. 生成式误差预判器（替换原ErrorPredictionModule） --------------------
class GenerativeErrorPredictor(nn.Module):
    """生成式误差预判：预判像素/rounding/量化误差，联动DCGAN生成器"""
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
        """x: 容器图像（-1~1），返回总误差"""
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

        return total_err

# -------------------- 3. 判别器辅助修复头（替换原refine_head） --------------------
class DiscriminatorRefineHead(nn.Module):
    """利用DCGAN判别器的反馈精细化修复提取的秘密图像"""
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

    def forward(self, x):
        # 第一步：初步修复
        x_refine = self.refine(x)
        # 第二步：获取判别器的评分（评分越低，修复力度越大）
        disc_score = self.disc(x_refine)
        refine_strength = torch.clamp(1 - disc_score, 0.1, 1.0)
        # 广播到图像维度
        refine_strength = refine_strength.view(-1, 1, 1, 1).repeat(1, x.shape[1], x.shape[2], x.shape[3])
        # 第三步：根据评分调整修复
        x_final = x_refine * refine_strength + x * (1 - refine_strength)
        return x_final

# -------------------- 4. 动态权重平衡器（替换原get_dynamic_weights函数） --------------------
class DynamicWeightBalancer:
    """三目标动态平衡器：容量-鲁棒性-不可感知性"""
    def __init__(self, base_weights={'cap': 1.0, 'rob': 0.5, 'imp': 0.3}):
        self.base_weights = base_weights

    def adjust(self, texture_complexity, attack_type):
        """
        texture_complexity: PZMs计算的纹理复杂度（batch维度张量）
        attack_type: 攻击类型（gaussian/jpeg/geometry/fgsm/pgd）
        return: 动态权重（cap_w, rob_w, imp_w）→ 标量张量
        """
        # 1. 纹理复杂度归一化（0~1）
        tex_min = torch.min(texture_complexity)
        tex_max = torch.max(texture_complexity)
        tex_norm = (texture_complexity - tex_min) / (tex_max - tex_min + 1e-8)
        # 纹理越复杂，容量权重越高，不可感知性权重越低
        cap_w = self.base_weights['cap'] * (1 + tex_norm)
        imp_w = self.base_weights['imp'] * (1 - tex_norm)

        # 2. 根据攻击类型调整鲁棒性权重
        if attack_type in ['geometry', 'fgsm', 'pgd']:  # 几何/对抗性攻击，鲁棒性权重翻倍
            rob_w = torch.tensor(self.base_weights['rob'] * 2.0).to(texture_complexity.device)
        elif attack_type in ['jpeg']:  # JPEG压缩，权重提升50%
            rob_w = torch.tensor(self.base_weights['rob'] * 1.5).to(texture_complexity.device)
        else:  # 高斯噪声，权重不变
            rob_w = torch.tensor(self.base_weights['rob'] * 1.0).to(texture_complexity.device)

        # 3. 权重归一化（batch维度取均值，转为标量）
        cap_w = torch.mean(cap_w)
        rob_w = rob_w  # 标量
        imp_w = torch.mean(imp_w)
        total = cap_w + rob_w + imp_w + 1e-8
        return cap_w/total, rob_w/total, imp_w/total

# ===================== 数据集加载（使用config路径） =====================
# 在main.py的WatermarkDataset类中修改
class WatermarkDataset(Dataset):
    def __init__(self, split="train", max_samples=None):
        self.img_size = (c.channels_in, c.cropsize, c.cropsize) if split == "train" else (c.channels_in, c.cropsize_val, c.cropsize_val)
        self.transform = transforms.Compose([
            transforms.Resize((self.img_size[1], self.img_size[2]), Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

        # 从config获取路径（仅读取train/val，无secret子路径）
        if split == "train":
            self.img_paths = sorted(glob.glob(os.path.join(c.TRAIN_PATH, f"*.{c.format_train}")))
        else:
            self.img_paths = sorted(glob.glob(os.path.join(c.VAL_PATH, f"*.{c.format_val}")))

        # 检查数据集是否为空
        if len(self.img_paths) == 0:
            raise ValueError(f"数据集路径下未找到图像文件: {c.TRAIN_PATH if split == 'train' else c.VAL_PATH}")

        if max_samples is not None:
            self.img_paths = self.img_paths[:max_samples]

        # 核心修改：拆分host和secret（随机配对）
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

# ===================== 训练函数 =====================
def train_epoch(net, disc, loader, optim_gen, optim_disc, epoch, writer, mode="train", weight_balancer=None):
    net.train() if mode == "train" else net.eval()
    disc.train() if mode == "train" else disc.eval()

    total_loss = 0
    psnr_list = []
    ssim_list = []
    epsilon = 0.05  # 对抗性攻击扰动强度

    for batch_idx, (host, secret) in enumerate(loader):
        host, secret = host.to(DEVICE), secret.to(DEVICE)
        # 扩展攻击类型：加入FGSM、PGD
        attack_type = np.random.choice(c.supported_attacks)

        with torch.set_grad_enabled(mode == "train"):
            # 正向嵌入：生成容器图像
            container, pred_error, pzms_feat = net(host, secret)
            # 转换为0~1范围（用于攻击和评估）
            container_01 = (container + 1) / 2
            container_01.requires_grad = True if mode == "train" else False  # 对抗性攻击需要梯度

            # -------------------- 生成对抗性攻击所需的梯度 --------------------
            container_grad = None
            if mode == "train" and attack_type in ["fgsm", "pgd"]:
                # 计算容器图像的梯度（相对于宿主图像的MSE损失）
                with torch.enable_grad():  # 显式启用梯度
                    temp_loss = F.mse_loss(container_01, (host + 1) / 2)
                temp_loss.backward(retain_graph=True)  # 仅保留必要的计算图
                container_grad = container_01.grad.data
                container_01.grad.zero_()  # 清除梯度，避免影响后续计算

            # -------------------- 施加攻击（包含对抗性攻击） --------------------
            attacked_container = attack_image(container_01, attack_type, container_grad, epsilon)
            actual_error = container_01 - attacked_container.detach()

            # -------------------- 逆向提取：秘密图像（使用判别器修复头） --------------------
            extracted = net(attacked_container, secret, rev=True)
            extracted = extracted.clamp(0, 1)

            # -------------------- 动态权重计算（纹理复杂度+攻击类型） --------------------
            pzms_complexity = net.pzms_extractor.get_texture_complexity(host)
            cap_weight, rob_weight, imp_weight = weight_balancer.adjust(pzms_complexity, attack_type)

            # -------------------- 损失计算 --------------------
            # 三目标损失：容量、鲁棒性、不可感知性
            cap_loss = F.mse_loss(extracted, (secret + 1) / 2)  # secret是-1~1，转换为0~1
            rob_loss = F.mse_loss(container_01, attacked_container)
            imp_loss = 1 - SSIM(container_01, (host + 1) / 2)  # host转换为0~1
            # 误差预判损失（生成式误差补偿）
            pred_error_01 = (pred_error + 1) / 2  # 转换为0~1范围匹配actual_error
            error_pred_loss = F.mse_loss(pred_error_01, actual_error)
            # 三目标加权损失
            content_loss = cap_weight * cap_loss + rob_weight * rob_loss + imp_weight * imp_loss

            # -------------------- 对抗性损失（泛化增强） --------------------
            adv_loss = torch.tensor(0.0).to(DEVICE)
            if attack_type in ["fgsm", "pgd"]:
                # 对抗样本的提取损失：增强泛化性
                adv_loss = F.mse_loss(extracted, (secret + 1) / 2) * 0.2  # 权重0.2

            current_loss = 0
            if mode == "train":
                # -------------------- 判别器训练 --------------------
                optim_disc.zero_grad()
                real_pred = disc((host + 1) / 2).view(-1)  # host转换为0~1
                real_loss = F.binary_cross_entropy(real_pred, torch.ones_like(real_pred)*0.9)  # 标签平滑
                fake_pred = disc(container_01.detach()).view(-1)
                fake_loss = F.binary_cross_entropy(fake_pred, torch.zeros_like(fake_pred))
                disc_loss = (real_loss + fake_loss) / 2
                disc_loss.backward(retain_graph=True)
                optim_disc.step()

                # -------------------- 生成器训练 --------------------
                optim_gen.zero_grad()
                # GAN对抗损失（容器图像欺骗判别器）
                gan_pred = disc(container_01).view(-1)
                gan_loss = F.binary_cross_entropy(gan_pred, torch.ones_like(gan_pred))
                # 总损失：三目标+GAN+误差预判+对抗性损失
                total_gen_loss = content_loss + c.gan_weight * gan_loss + c.error_pred_weight * error_pred_loss + adv_loss
                total_gen_loss.backward()
                optim_gen.step()
                current_loss = total_gen_loss.item()
            else:
                current_loss = (content_loss + error_pred_loss + adv_loss).item()

            # -------------------- 评估指标收集 --------------------
            total_loss += current_loss
            psnr_list.append(PSNR(extracted, (secret + 1) / 2))
            ssim_list.append(SSIM(extracted, (secret + 1) / 2).item())  # 转换为数值

    # -------------------- 批次平均计算 --------------------
    avg_loss = total_loss / len(loader)
    avg_psnr = np.mean(psnr_list)
    avg_ssim = np.mean(ssim_list)
    print(f"{mode} Epoch {epoch}: Loss={avg_loss:.4f}, PSNR={avg_psnr:.2f}, SSIM={avg_ssim:.4f}")

    # -------------------- 张量板日志 --------------------
    if writer is not None:
        writer.add_scalar(f"{mode}/Loss", avg_loss, epoch)
        writer.add_scalar(f"{mode}/PSNR", avg_psnr, epoch)
        writer.add_scalar(f"{mode}/SSIM", avg_ssim, epoch)
        writer.add_scalar(f"{mode}/Cap_Weight", cap_weight.item(), epoch)
        writer.add_scalar(f"{mode}/Rob_Weight", rob_weight.item(), epoch)
        writer.add_scalar(f"{mode}/Imp_Weight", imp_weight.item(), epoch)
    return avg_loss

# ===================== 主训练函数 =====================
def train():
    # -------------------- 初始化动态权重平衡器 --------------------
    base_weights = {
        'cap': getattr(c, 'base_capacity_weight', 1.0),
        'rob': getattr(c, 'base_robustness_weight', 1.0),  # 原0.5→改为配置中的1.0
        'imp': getattr(c, 'base_imperceptibility_weight', 1.0)  # 原0.3→改为配置中的1.0
    }

    weight_balancer = DynamicWeightBalancer(base_weights=base_weights)

    from model import EnhancedPRIS, Discriminator
    disc = Discriminator(in_channels=c.channels_in).to(DEVICE)
    net = EnhancedPRIS(
        in_channels=c.channels_in,
        target_channels=c.target_channels,
        disc=disc  # 传入判别器到修复头
    ).to(DEVICE)

    # -------------------- 优化器 --------------------
    optim_gen = optim.Adam(net.parameters(), lr=c.lr_gen, betas=c.betas)
    optim_disc = optim.Adam(disc.parameters(), lr=c.lr_disc, betas=c.betas)

    # -------------------- 数据集 --------------------
    train_dataset = WatermarkDataset(split="train")
    val_dataset = WatermarkDataset(split="val")
    train_loader = DataLoader(train_dataset, batch_size=c.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=c.batchsize_val, shuffle=False, num_workers=0)

    # -------------------- 日志 --------------------
    writer = SummaryWriter(log_dir=c.LOG_PATH)

    # -------------------- 训练循环 --------------------
    for epoch in range(c.epochs):
        train_loss = train_epoch(net, disc, train_loader, optim_gen, optim_disc, epoch, writer, mode="train", weight_balancer=weight_balancer)
        # 验证阶段
        if (epoch + 1) % c.val_freq == 0:
            with torch.no_grad():
                val_loss = train_epoch(net, disc, val_loader, optim_gen, optim_disc, epoch, writer, mode="val", weight_balancer=weight_balancer)
        # 保存检查点
        if (epoch + 1) % c.save_freq == 0:
            torch.save({
                "net": net.state_dict(),
                "disc": disc.state_dict(),
                "optim_gen": optim_gen.state_dict(),
                "optim_disc": optim_disc.state_dict(),
                "weight_balancer": base_weights
            }, os.path.join(c.CHECKPOINT_PATH, f"checkpoint_{epoch+1}.pt"))
    print("训练完成，保存示例图像...")
    save_sample_images(net, val_loader, num_samples=50)  # 使用验证集加载器val_loader

    writer.close()

    writer.close()

if __name__ == "__main__":
    train()
