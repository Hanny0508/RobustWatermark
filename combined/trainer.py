import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm
import config as c

from data import StegoDataset
from Main import PSNR, SSIM, attack_image, DCGANGenerator, GenerativeErrorPredictor, DiscriminatorRefineHead, \
    DynamicWeightBalancer
from model import EnhancedPRIS, PZMsFeatureExtractor, ReversibleBlock, FeatureFusionModule


# 定义判别器（DCGAN风格）
class Discriminator(nn.Module):
    """判别器：区分正常图像和容器图像，辅助生成器优化"""

    def __init__(self, in_channels=3):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, 64, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 1, 4, 1, 0),  # 输出单通道评分
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x).mean(dim=[1, 2, 3])  # 全局平均评分


class Trainer:
    def __init__(self):
        # 设备配置
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"使用设备: {self.device}")

        # 模型初始化
        self.gen = EnhancedPRIS().to(self.device)  # 隐写生成器
        self.disc = Discriminator().to(self.device)  # 判别器
        self.error_predictor = GenerativeErrorPredictor().to(self.device)  # 误差预判器
        self.refine_head = DiscriminatorRefineHead(disc=self.disc).to(self.device)  # 修复头
        self.pzms_extractor = PZMsFeatureExtractor(max_order=c.pzms_max_order).to(self.device)  # PZMs提取器
        # 修正：从config读取基础权重初始化动态权重平衡器（可选，根据需求）
        self.weight_balancer = DynamicWeightBalancer(
            base_weights={
                'cap': c.base_capacity_weight,
                'rob': c.base_robustness_weight,
                'imp': c.base_imperceptibility_weight
            }
        )

        # 优化器
        self.opt_gen = optim.Adam(self.gen.parameters(), lr=c.lr_gen, betas=c.betas)
        self.opt_disc = optim.Adam(self.disc.parameters(), lr=c.lr_disc, betas=c.betas)

        # 损失函数
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()

        # 数据集和加载器：Windows下num_workers改为0（避免多进程数据加载错误）
        self.train_dataset = StegoDataset(is_train=True)
        self.val_dataset = StegoDataset(is_train=False)
        self.train_loader = DataLoader(self.train_dataset, batch_size=c.batch_size, shuffle=True, num_workers=0)
        self.val_loader = DataLoader(self.val_dataset, batch_size=c.batchsize_val, shuffle=c.shuffle_val, num_workers=0)

        # 日志和检查点
        self.writer = SummaryWriter(c.LOG_PATH)
        os.makedirs(c.CHECKPOINT_PATH, exist_ok=True)
        self.best_psnr = 0.0  # 最佳验证PSNR

    def train_one_epoch(self, epoch):
        self.gen.train()
        self.disc.train()
        total_loss_gen = 0.0
        total_loss_disc = 0.0

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch + 1}/{c.epochs}")
        for host, secret in pbar:
            host = host.to(self.device)
            secret = secret.to(self.device)
            batch_size = host.shape[0]

            # -------------------- 训练判别器 --------------------
            self.opt_disc.zero_grad()
            # 生成容器图像（-1~1范围）
            container = self.gen.embed(host, secret)
            # 关键1：将张量从[-1,1]转换为[0,1]（适配攻击、判别器、SSIM/PSNR）
            container_01 = (container + 1) / 2
            host_01 = (host + 1) / 2  # 宿主图像同步转换
            # 关键2：为非叶子张量保留梯度（解决grad为None的问题）
            container_01.retain_grad()

            # 真实图像（宿主图像0~1）vs 生成图像（容器图像0~1）
            real_label = torch.ones(batch_size, device=self.device)
            fake_label = torch.zeros(batch_size, device=self.device)
            # 判别器损失：输入0~1范围的张量
            loss_real = self.bce_loss(self.disc(host_01), real_label)
            loss_fake = self.bce_loss(self.disc(container_01.detach()), fake_label)
            loss_disc = (loss_real + loss_fake) * 0.5
            loss_disc.backward()
            self.opt_disc.step()
            total_loss_disc += loss_disc.item()

            # -------------------- 训练生成器 --------------------
            self.opt_gen.zero_grad()
            # 提取特征（用于动态权重）
            texture_complexity = self.pzms_extractor.get_texture_complexity(host)
            # 随机选择攻击类型
            attack_type = np.random.choice(c.supported_attacks)
            # 生成带攻击的容器图像（对抗性攻击需要梯度）
            if attack_type in ['fgsm', 'pgd']:
                # 计算判别器对容器图像的预测（用于获取梯度）
                pred = self.disc(container_01)
                loss_gan_temp = self.bce_loss(pred, real_label)
                # 反向传播获取container_01的梯度（保留计算图，后续还要用）
                loss_gan_temp.backward(retain_graph=True)
                # 安全获取梯度：若梯度为None则用0张量
                container_grad = container_01.grad.detach() if container_01.grad is not None else torch.zeros_like(container_01)
                # 执行攻击（传入0~1的张量和梯度）
                container_attacked_01 = attack_image(container_01, attack_type, container_grad, c.epsilon)
            else:
                # 非对抗攻击，直接处理0~1的张量
                container_attacked_01 = attack_image(container_01, attack_type)

            # 关键：将攻击后的图像从[0,1]转换回[-1,1]（适配模型提取方法）
            container_attacked = (container_attacked_01 * 2) - 1

            # 提取秘密图像并修复
            extracted_secret = self.gen.extract(container_attacked)
            extracted_secret_refined = self.refine_head(extracted_secret)
            # 秘密图像转换为0~1范围（用于损失计算）
            secret_01 = (secret + 1) / 2
            extracted_secret_refined_01 = (extracted_secret_refined + 1) / 2
            extracted_secret_01 = (extracted_secret + 1) / 2

            # 动态权重调整
            cap_w, rob_w, imp_w = self.weight_balancer.adjust(texture_complexity, attack_type)

            # 三目标损失计算（全部使用0~1范围的张量）
            loss_capacity = self.mse_loss(extracted_secret_refined_01, secret_01)  # 容量损失
            loss_imperceptible = 1 - SSIM(container_01, host_01)  # 不可感知性损失（0~1范围计算SSIM）
            loss_robustness = self.mse_loss(extracted_secret_01, extracted_secret_refined_01)  # 鲁棒性损失
            # 误差预判损失：容器图像转换为0~1后计算
            loss_error = self.mse_loss((self.error_predictor(container) + 1) / 2, torch.zeros_like(container_01))

            # GAN损失（生成器希望骗过判别器，输入0~1的张量）
            loss_gan = self.bce_loss(self.disc(container_01), real_label)

            # 总损失（加权求和）
            loss_gen = (cap_w * loss_capacity +
                        rob_w * loss_robustness +
                        imp_w * loss_imperceptible +
                        c.gan_weight * loss_gan +
                        c.error_pred_weight * loss_error)

            loss_gen.backward()
            self.opt_gen.step()
            total_loss_gen += loss_gen.item()

            pbar.set_postfix({"Gen Loss": loss_gen.item(), "Disc Loss": loss_disc.item()})

        # 记录日志
        avg_loss_gen = total_loss_gen / len(self.train_loader)
        avg_loss_disc = total_loss_disc / len(self.train_loader)
        self.writer.add_scalar("Train/Gen_Loss", avg_loss_gen, epoch)
        self.writer.add_scalar("Train/Disc_Loss", avg_loss_disc, epoch)
        print(f"Epoch {epoch + 1} | Gen Loss: {avg_loss_gen:.4f} | Disc Loss: {avg_loss_disc:.4f}")

    def validate(self, epoch):
        self.gen.eval()
        total_psnr = 0.0
        total_ssim = 0.0

        with torch.no_grad():
            for host, secret in self.val_loader:
                host = host.to(self.device)
                secret = secret.to(self.device)

                # 生成容器并提取秘密
                container = self.gen.embed(host, secret)
                # 转换为0~1范围
                container_01 = (container + 1) / 2
                host_01 = (host + 1) / 2
                secret_01 = (secret + 1) / 2

                extracted_secret = self.gen.extract(container)
                extracted_refined = self.refine_head(extracted_secret)
                # 转换为0~1范围用于评估
                extracted_refined_01 = (extracted_refined + 1) / 2

                # 计算指标（使用0~1范围的张量）
                total_psnr += PSNR(extracted_refined_01, secret_01)
                total_ssim += SSIM(extracted_refined_01, secret_01).item()

        avg_psnr = total_psnr / len(self.val_loader)
        avg_ssim = total_ssim / len(self.val_loader)
        self.writer.add_scalar("Val/PSNR", avg_psnr, epoch)
        self.writer.add_scalar("Val/SSIM", avg_ssim, epoch)
        print(f"Validation | PSNR: {avg_psnr:.2f} dB | SSIM: {avg_ssim:.4f}")

        # 保存最佳模型
        if avg_psnr > self.best_psnr:
            self.best_psnr = avg_psnr
            torch.save({
                "gen_state_dict": self.gen.state_dict(),
                "disc_state_dict": self.disc.state_dict(),
                "epoch": epoch
            }, os.path.join(c.CHECKPOINT_PATH, "best_model.pth"))
            print(f"保存最佳模型（PSNR: {avg_psnr:.2f}）")

    def run(self):
        """启动训练流程"""
        for epoch in range(c.epochs):
            self.train_one_epoch(epoch)
            if (epoch + 1) % c.val_freq == 0:
                self.validate(epoch)
            if (epoch + 1) % c.save_freq == 0:
                torch.save({
                    "gen_state_dict": self.gen.state_dict(),
                    "disc_state_dict": self.disc.state_dict(),
                    "epoch": epoch
                }, os.path.join(c.CHECKPOINT_PATH, f"model_epoch_{epoch + 1}.pth"))


if __name__ == "__main__":
    trainer = Trainer()
    trainer.run()
