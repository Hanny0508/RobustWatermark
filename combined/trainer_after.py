import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.utils as vutils

# 导入自定义模块
import config as c
from data import StegoDataset
from Main import (
    PSNR, SSIM, attack_image, DCGANGenerator,
    GenerativeErrorPredictor, DiscriminatorRefineHead,
    DynamicWeightBalancer, Discriminator  # 从Main导入Discriminator，不再自定义
)
from model import (
    EnhancedPRIS, PZMsFeatureExtractor,
    ReversibleBlock, FeatureFusionModule
)

class Trainer:
    def __init__(self):
        # 设备配置
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"使用设备: {self.device}")

        # 模型初始化
        self.gen = EnhancedPRIS().to(self.device)  # 隐写生成器
        self.disc = Discriminator(in_channels=3).to(self.device)  # 使用Main的Discriminator
        self.error_predictor = GenerativeErrorPredictor(in_channels=3).to(self.device)  # 传入in_channels匹配
        self.refine_head = DiscriminatorRefineHead(disc=self.disc).to(self.device)  # 修复头
        self.pzms_extractor = PZMsFeatureExtractor(max_order=c.pzms_max_order).to(self.device)  # PZMs提取器
        # 动态权重平衡器（支持传入当前损失值）
        self.weight_balancer = DynamicWeightBalancer(
            base_weights={
                'cap': c.base_capacity_weight,
                'rob': c.base_robustness_weight,
                'imp': c.base_imperceptibility_weight
            }
        )
        self.training = True
        # 优化器：提高判别器学习率（解决判别器无效问题）
        self.opt_gen = optim.Adam(self.gen.parameters(), lr=c.lr_gen, betas=c.betas)
        self.opt_disc = optim.Adam(self.disc.parameters(), lr=c.lr_disc * 2, betas=c.betas)  # 判别器学习率×2

        # 学习率调度器（解决学习率固定问题）
        self.scheduler_gen = optim.lr_scheduler.StepLR(
            self.opt_gen, step_size=c.lr_step_size, gamma=c.lr_gamma
        )
        self.scheduler_disc = optim.lr_scheduler.StepLR(
            self.opt_disc, step_size=c.lr_step_size, gamma=c.lr_gamma
        )

        # 损失函数
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()

        # 数据集和加载器（Windows下num_workers=0避免多进程错误）
        self.train_dataset = StegoDataset(is_train=True)
        self.val_dataset = StegoDataset(is_train=False)
        self.train_loader = DataLoader(
            self.train_dataset, batch_size=c.batch_size, shuffle=True, num_workers=0
        )
        self.val_loader = DataLoader(
            self.val_dataset, batch_size=c.batchsize_val, shuffle=c.shuffle_val, num_workers=0
        )

        # 日志和检查点
        self.writer = SummaryWriter(c.LOG_PATH)
        os.makedirs(c.CHECKPOINT_PATH, exist_ok=True)
        os.makedirs(c.IMAGE_PATH, exist_ok=True)  # 示例图片保存目录
        self.best_psnr = 0.0  # 最佳验证PSNR（秘密提取）
        self.best_psnr_container = 0.0  # 最佳验证PSNR（容器与宿主）
        # 新增：PSNR提升阈值（用于早停和权重调整）
        self.psnr_improve_threshold = 1.02  # 至少提升2%才视为有效提升

    def get_feature_matching_loss(self, real_feats, fake_feats):
        """计算特征匹配损失（解决判别器无效问题）"""
        loss = 0.0
        for r_feat, f_feat in zip(real_feats, fake_feats):
            loss += torch.mean(torch.abs(r_feat - f_feat))
        return loss / len(real_feats)

    def load_checkpoint(self, checkpoint_path):
        """加载检查点（最佳模型或断点续训）"""
        if os.path.exists(checkpoint_path):
            # 加载检查点并映射到当前设备
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.gen.load_state_dict(checkpoint['gen_state_dict'])
            self.disc.load_state_dict(checkpoint['disc_state_dict'])
            # 可选：加载优化器状态（断点续训需要）
            if 'opt_gen_state_dict' in checkpoint:
                self.opt_gen.load_state_dict(checkpoint['opt_gen_state_dict'])
            if 'opt_disc_state_dict' in checkpoint:
                self.opt_disc.load_state_dict(checkpoint['opt_disc_state_dict'])
            # 恢复最佳PSNR和起始epoch
            self.best_psnr = checkpoint.get('best_psnr', self.best_psnr)
            start_epoch = checkpoint.get('epoch', 0) + 1  # 从下一个epoch开始
            print(f"✅ 加载检查点成功：{checkpoint_path}")
            print(f"  - 恢复到Epoch: {start_epoch}")
            print(f"  - 历史最佳PSNR（秘密）：{self.best_psnr:.2f} dB")
            return start_epoch
        else:
            print(f"❌ 未找到检查点：{checkpoint_path}，将从头开始训练")
            return 0

    def save_sample_images(self, epoch=None, num_samples=5):
        """保存示例图片：宿主、容器、原秘密、提取的秘密"""
        self.gen.eval()
        save_dir = os.path.join(c.SAMPLE_IMAGE_PATH, f"epoch_{epoch}" if epoch else "final")
        os.makedirs(save_dir, exist_ok=True)

        with torch.no_grad():
            for idx, (host, secret) in enumerate(self.val_loader):
                if idx >= num_samples:
                    break  # 只保存指定数量的样本

                # 前向传播
                host = host.to(self.device)
                secret = secret.to(self.device)
                container = self.gen.embed(host, secret)
                extracted_secret = self.gen.extract(container)
                # 转换为[0,1]范围后传入refine_head
                extracted_secret_01 = (extracted_secret + 1) / 2
                extracted_refined = self.refine_head(extracted_secret_01)

                # 转换为[0, 1]范围（适配图像保存）
                def tensor_to_01(tensor):
                    return (tensor.cpu() + 1) / 2  # 从[-1,1]转[0,1]

                # 取batch中的第一个样本（维度：C×H×W）
                host_01 = tensor_to_01(host[0])
                container_01 = tensor_to_01(container[0])
                secret_01 = tensor_to_01(secret[0])
                # refined是[0,1]范围，直接转换
                extracted_refined_01 = extracted_refined[0].cpu().clamp(0, 1)

                # 转换为PIL图像（处理单通道/三通道）
                def pil_convert(tensor):
                    if tensor.shape[0] == 1:  # 单通道（灰度图）
                        return Image.fromarray((tensor.squeeze(0).numpy() * 255).astype(np.uint8), mode='L')
                    else:  # 三通道（RGB图）
                        return Image.fromarray((tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8), mode='RGB')

                # 保存单个图像
                pil_convert(host_01).save(os.path.join(save_dir, f"sample_{idx+1}_host.png"))
                pil_convert(container_01).save(os.path.join(save_dir, f"sample_{idx+1}_container.png"))
                pil_convert(secret_01).save(os.path.join(save_dir, f"sample_{idx+1}_secret_ori.png"))
                pil_convert(extracted_refined_01).save(os.path.join(save_dir, f"sample_{idx+1}_secret_ext.png"))

                # 拼接图像并保存（更直观对比）
                fig, axes = plt.subplots(2, 2, figsize=(12, 12))
                axes[0, 0].imshow(pil_convert(host_01))
                axes[0, 0].set_title("Host Image", fontsize=12)
                axes[0, 0].axis('off')

                axes[0, 1].imshow(pil_convert(container_01))
                axes[0, 1].set_title("Container Image", fontsize=12)
                axes[0, 1].axis('off')

                axes[1, 0].imshow(pil_convert(secret_01))
                axes[1, 0].set_title("Original Secret", fontsize=12)
                axes[1, 0].axis('off')

                axes[1, 1].imshow(pil_convert(extracted_refined_01))
                axes[1, 1].set_title("Refined Extracted Secret", fontsize=12)
                axes[1, 1].axis('off')

                plt.tight_layout()
                plt.savefig(os.path.join(save_dir, f"sample_{idx+1}_combined.png"), dpi=300, bbox_inches='tight')
                plt.close()

        print(f"📸 示例图片已保存至：{save_dir}")

    def train_one_epoch(self, epoch):
        """训练单个Epoch"""
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
            # 转换为[0,1]范围（适配攻击、判别器、指标计算）
            container_01 = (container + 1) / 2
            host_01 = (host + 1) / 2
            # 保留梯度（攻击需要）
            container_01.retain_grad()

            # 判别器标签（真实=1，生成=0）
            real_label = torch.ones(batch_size, device=self.device)
            fake_label = torch.zeros(batch_size, device=self.device)

            # 判别器损失：使用Main.Discriminator的get_score方法
            loss_real = self.bce_loss(self.disc.get_score(host_01), real_label)
            loss_fake = self.bce_loss(self.disc.get_score(container_01.detach()), fake_label)
            loss_disc = (loss_real + loss_fake) * 0.5

            # 新增：判别器特征匹配损失（增强判别器学习）
            if hasattr(self.disc, 'get_features'):
                real_feats = self.disc.get_features(host_01)
                fake_feats = self.disc.get_features(container_01.detach())
                loss_disc_feat = self.get_feature_matching_loss(real_feats, fake_feats)
                loss_disc += loss_disc_feat * c.feat_match_weight  # 特征匹配损失权重

            # 反向传播+优化
            loss_disc.backward()
            self.opt_disc.step()
            total_loss_disc += loss_disc.item()

            # -------------------- 训练生成器 --------------------
            self.opt_gen.zero_grad()
            # 1. 生成式误差预判：用预测误差修正容器图像（优化创新点3）
            pred_error_total = None
            if self.training:
                # 解包返回的元组，只取总误差（第一个值）
                pred_error_total, _, _, _ = self.error_predictor(container)
                # 用总误差进行修正
                container = container - c.error_correction_weight * pred_error_total
                container = torch.clamp(container, -1.0, 1.0)  # 限制范围
                # 重新转换为[0,1]（修正后）
                container_01 = (container + 1) / 2
                container_01.retain_grad()  # 重新保留梯度

            # 2. 对抗攻击（随机选择攻击类型）
            attack_type = np.random.choice(c.supported_attacks)
            container_attacked_01 = None
            if attack_type in ['fgsm', 'pgd']:
                # 对抗攻击需要4维张量的梯度，调用disc的forward（返回4维）
                pred_4d = self.disc(container_01)  # 4维张量：(batch, 1, h, w)
                # 计算损失时用全局平均（1维）
                loss_gan_temp = self.bce_loss(pred_4d.mean(dim=[1, 2, 3]), real_label)
                loss_gan_temp.backward(retain_graph=True)
                # 安全获取梯度
                container_grad = container_01.grad.detach() if container_01.grad is not None else torch.zeros_like(container_01)
                # 执行攻击
                container_attacked_01 = attack_image(container_01, attack_type, container_grad, c.epsilon)
                # 清空梯度，避免残留
                container_01.grad.zero_()
            else:
                # 非对抗攻击
                container_attacked_01 = attack_image(container_01, attack_type)

            # 转换回[-1,1]范围
            container_attacked = (container_attacked_01 * 2) - 1

            # 3. 提取并修复秘密
            extracted_secret = self.gen.extract(container_attacked)
            # 转换为[0,1]范围后传入refine_head（匹配Main的refine_head预期）
            extracted_secret_01 = (extracted_secret + 1) / 2
            extracted_secret_refined = self.refine_head(extracted_secret_01)  # refine_head输出[0,1]

            # 4. 动态权重调整（修复键不匹配问题+强制提升容量权重）
            texture_complexity = self.pzms_extractor.get_texture_complexity(host)
            # 转换为[0,1]范围（损失计算）
            secret_01 = (secret + 1) / 2
            # 计算基础损失（键改为cap/rob/imp，匹配weight_balancer）
            # 核心改进：提高容量损失的基础权重（×5），优先保证秘密提取
            loss_capacity = self.mse_loss(extracted_secret_refined, secret_01) * 5.0  # 容量损失×5
            loss_imperceptible = 1 - SSIM(container_01, host_01)
            loss_robustness = self.mse_loss(extracted_secret_01, extracted_secret_refined)
            # 收集当前损失值（键匹配）
            current_losses = {
                'cap': loss_capacity.item(),
                'rob': loss_robustness.item(),
                'imp': loss_imperceptible.item()
            }
            # 动态调整权重
            cap_w, rob_w, imp_w = self.weight_balancer.adjust(texture_complexity, attack_type, current_losses)

            # 新增：当秘密提取损失过高时，强制提升容量权重
            if loss_capacity.item() > 0.1:  # 阈值可根据实际情况调整
                cap_w *= 2.0  # 容量权重×2

            # 5. 损失计算
            # 修复error_predictor的元组问题：使用之前保存的pred_error_total
            if pred_error_total is None:
                pred_error_total, _, _, _ = self.error_predictor(container)
            pred_error_01 = (pred_error_total + 1) / 2  # 转换为[0,1]
            loss_error = self.mse_loss(pred_error_01, torch.zeros_like(container_01))

            # GAN损失：使用get_score方法（返回1维，匹配BCE损失）
            loss_gan = self.bce_loss(self.disc.get_score(container_01), real_label)

            # 新增：生成器特征匹配损失（增强生成器学习）
            loss_feat_match = 0.0
            if hasattr(self.disc, 'get_features'):
                real_feats = self.disc.get_features(host_01)
                fake_feats = self.disc.get_features(container_01)
                loss_feat_match = self.get_feature_matching_loss(real_feats, fake_feats)

            # 总损失（加权求和）：添加特征匹配损失
            loss_gen = (
                cap_w * loss_capacity +
                rob_w * loss_robustness +
                imp_w * loss_imperceptible +
                c.gan_weight * loss_gan +
                c.error_pred_weight * loss_error +
                c.feat_match_weight * loss_feat_match  # 特征匹配损失
            )

            # 反向传播+优化
            loss_gen.backward()
            self.opt_gen.step()
            total_loss_gen += loss_gen.item()

            # 更新进度条
            pbar.set_postfix({
                "Gen Loss": f"{loss_gen.item():.4f}",
                "Disc Loss": f"{loss_disc.item():.4f}"
            })

        # 平均损失
        avg_loss_gen = total_loss_gen / len(self.train_loader)
        avg_loss_disc = total_loss_disc / len(self.train_loader)
        # 记录日志
        self.writer.add_scalar("Train/Gen_Loss", avg_loss_gen, epoch)
        self.writer.add_scalar("Train/Disc_Loss", avg_loss_disc, epoch)
        print(f"\nEpoch {epoch + 1} | Gen Loss: {avg_loss_gen:.4f} | Disc Loss: {avg_loss_disc:.4f}")

        # 更新学习率
        self.scheduler_gen.step()
        self.scheduler_disc.step()

    def validate(self, epoch):
        """验证阶段：计算PSNR/SSIM并保存最佳模型"""
        self.gen.eval()
        total_psnr_secret = 0.0  # 秘密提取的PSNR
        total_ssim_secret = 0.0  # 秘密提取的SSIM
        total_psnr_container = 0.0  # 容器与宿主的PSNR（不可感知性）
        total_ssim_container = 0.0  # 容器与宿主的SSIM（不可感知性）

        with torch.no_grad():
            for host, secret in self.val_loader:
                host = host.to(self.device)
                secret = secret.to(self.device)

                # 生成容器并提取秘密
                container = self.gen.embed(host, secret)
                # 转换为[0,1]范围
                container_01 = (container + 1) / 2
                host_01 = (host + 1) / 2
                secret_01 = (secret + 1) / 2

                # 提取并修复秘密
                extracted_secret = self.gen.extract(container)
                extracted_secret_01 = (extracted_secret + 1) / 2
                extracted_refined = self.refine_head(extracted_secret_01)

                # 计算指标：秘密提取（原逻辑）
                total_psnr_secret += PSNR(extracted_refined, secret_01)
                total_ssim_secret += SSIM(extracted_refined, secret_01).item()
                # 计算指标：容器与宿主（不可感知性，新增）
                total_psnr_container += PSNR(container_01, host_01)
                total_ssim_container += SSIM(container_01, host_01).item()

        # 平均指标
        avg_psnr_secret = total_psnr_secret / len(self.val_loader)
        avg_ssim_secret = total_ssim_secret / len(self.val_loader)
        avg_psnr_container = total_psnr_container / len(self.val_loader)
        avg_ssim_container = total_ssim_container / len(self.val_loader)

        # 记录日志
        self.writer.add_scalar("Val/PSNR_Secret", avg_psnr_secret, epoch)
        self.writer.add_scalar("Val/SSIM_Secret", avg_ssim_secret, epoch)
        self.writer.add_scalar("Val/PSNR_Container", avg_psnr_container, epoch)
        self.writer.add_scalar("Val/SSIM_Container", avg_ssim_container, epoch)

        # 打印结果
        print(f"\n📊 Validation Results:")
        print(f"  - Secret: PSNR = {avg_psnr_secret:.2f} dB | SSIM = {avg_ssim_secret:.4f}")
        print(f"  - Container: PSNR = {avg_psnr_container:.2f} dB | SSIM = {avg_ssim_container:.4f}")

        # 保存最佳模型（以秘密提取的PSNR为指标，要求至少提升2%）
        if avg_psnr_secret > self.best_psnr * self.psnr_improve_threshold:
            # 重置学习率调度器（核心改进：验证指标提升时重置学习率）
            self.scheduler_gen.last_epoch = -1
            self.scheduler_disc.last_epoch = -1
            print(f"🔄 验证PSNR提升超过{self.psnr_improve_threshold-1:.0%}，重置学习率调度器")

            self.best_psnr = avg_psnr_secret
            self.best_psnr_container = avg_psnr_container
            # 保存检查点（包含优化器状态，支持断点续训）
            torch.save({
                "gen_state_dict": self.gen.state_dict(),
                "disc_state_dict": self.disc.state_dict(),
                "opt_gen_state_dict": self.opt_gen.state_dict(),
                "opt_disc_state_dict": self.opt_disc.state_dict(),
                "epoch": epoch,
                "best_psnr": self.best_psnr,
                "best_psnr_container": self.best_psnr_container
            }, os.path.join(c.CHECKPOINT_PATH, "best_model.pth"))
            print(f"✅ 保存最佳模型（Secret PSNR: {avg_psnr_secret:.2f} dB）")

        return avg_psnr_secret  # 返回PSNR用于早停判断

    def run(self):
        """启动训练流程（含早停、断点续训、图片保存）"""
        # 步骤1：加载最佳模型（第二次训练时自动使用）
        start_epoch = self.load_checkpoint(os.path.join(c.CHECKPOINT_PATH, "best_model.pth"))

        # 步骤2：初始化早停参数
        no_improve_epochs = 0
        early_stop_triggered = False

        # 步骤3：开始训练
        for epoch in range(start_epoch, c.epochs):
            # 训练单个epoch
            self.train_one_epoch(epoch)

            # 验证+保存示例图片
            if (epoch + 1) % c.val_freq == 0:
                avg_psnr = self.validate(epoch)
                # 保存示例图片
                self.save_sample_images(epoch=epoch+1)

                # 改进早停判断：动态耐心值+比例提升判断
                # 动态耐心值：训练后期（超过一半epoch）耐心值减半
                current_patience = c.early_stop_patience if epoch < c.epochs//2 else c.early_stop_patience // 2

                if avg_psnr > self.best_psnr * self.psnr_improve_threshold:
                    no_improve_epochs = 0  # 重置计数器（仅当有效提升时）
                else:
                    no_improve_epochs += 1
                    print(f"⚠️  验证PSNR连续{no_improve_epochs}轮未显著提升（耐心：{current_patience}）")
                    if no_improve_epochs >= current_patience:
                        print(f"🛑 早停触发，停止训练（Epoch: {epoch+1}）")
                        early_stop_triggered = True
                        break

            # 保存断点
            if (epoch + 1) % c.save_freq == 0 and not early_stop_triggered:
                torch.save({
                    "gen_state_dict": self.gen.state_dict(),
                    "disc_state_dict": self.disc.state_dict(),
                    "opt_gen_state_dict": self.opt_gen.state_dict(),
                    "opt_disc_state_dict": self.opt_disc.state_dict(),
                    "epoch": epoch,
                    "best_psnr": self.best_psnr
                }, os.path.join(c.CHECKPOINT_PATH, f"model_epoch_{epoch + 1}.pth"))
                print(f"💾 保存断点模型：model_epoch_{epoch+1}.pth")

        # 训练结束后保存最终示例图片
        if not early_stop_triggered:
            self.save_sample_images()  # 保存final版本

        # 关闭日志写入器
        self.writer.close()
        print("\n🎉 训练流程完成！")

if __name__ == "__main__":
    # 实例化并启动训练
    trainer = Trainer()
    trainer.run()
