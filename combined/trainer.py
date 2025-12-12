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
        self.training = True  # 控制误差预测器的使用

        # 优化器
        self.opt_gen = optim.Adam(self.gen.parameters(), lr=c.lr_gen, betas=c.betas)
        self.opt_disc = optim.Adam(self.disc.parameters(), lr=c.lr_disc, betas=c.betas)

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
        os.makedirs(c.SAMPLE_IMAGE_PATH, exist_ok=True)  # 确保样本图片目录存在
        self.best_psnr = 0.0  # 最佳验证PSNR（秘密提取）
        self.best_psnr_container = 0.0  # 最佳验证PSNR（容器与宿主）

    def load_checkpoint(self, checkpoint_path):
        """加载检查点（最佳模型或断点续训）"""
        if os.path.exists(checkpoint_path):
            try:
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
            except Exception as e:
                raise RuntimeError(f"❌ 加载检查点失败：{str(e)}")
        else:
            print(f"❌ 未找到检查点：{checkpoint_path}，将从头开始训练")
            return 0

    def save_sample_images(self, epoch=None, num_samples=5):
        """保存示例图片：宿主、容器、原秘密、提取的秘密"""
        self.gen.eval()
        save_dir = os.path.join(c.SAMPLE_IMAGE_PATH, f"epoch_{epoch}" if epoch else "final")
        os.makedirs(save_dir, exist_ok=True)

        with torch.no_grad():
            # 修正：StegoDataset返回的是字典，需用key取值
            for idx, batch in enumerate(self.val_loader):
                if idx >= num_samples:
                    break  # 只保存指定数量的样本

                # 从batch字典中获取宿主和秘密图像
                host = batch['host'].to(self.device)
                secret = batch['secret'].to(self.device)

                # 前向传播
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
                    tensor_np = tensor.numpy()
                    if tensor.shape[0] == 1:  # 单通道（灰度图）
                        return Image.fromarray((tensor_np.squeeze(0) * 255).astype(np.uint8), mode='L')
                    else:  # 三通道（RGB图）
                        # 转换维度：C×H×W → H×W×C
                        return Image.fromarray((tensor_np.transpose(1, 2, 0) * 255).astype(np.uint8), mode='RGB')

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
        # 修正：StegoDataset返回的是字典，需遍历batch字典
        for batch in pbar:
            # 从batch字典中获取宿主和秘密图像
            host = batch['host'].to(self.device)
            secret = batch['secret'].to(self.device)
            batch_size = host.shape[0]

            # -------------------- 训练判别器（多轮训练增强对抗压力） --------------------
            # 判别器训练2次，生成器1次（解决判别器训练不足的问题）
            for _ in range(2):
                self.opt_disc.zero_grad()
                # 生成容器图像（-1~1范围）
                with torch.no_grad():  # 判别器训练时，生成器不计算梯度
                    container = self.gen.embed(host, secret)
                # 转换为[0,1]范围（适配攻击、判别器、指标计算）
                container_01 = (container + 1) / 2
                host_01 = (host + 1) / 2

                # 判别器标签（真实=1，生成=0）
                real_label = torch.ones(batch_size, device=self.device)
                fake_label = torch.zeros(batch_size, device=self.device)

                # 判别器损失：使用Main.Discriminator的get_score方法（返回1维）
                loss_real = self.bce_loss(self.disc.get_score(host_01), real_label)
                loss_fake = self.bce_loss(self.disc.get_score(container_01.detach()), fake_label)
                loss_disc = (loss_real + loss_fake) * 0.5

                # 反向传播+优化
                loss_disc.backward()
                self.opt_disc.step()
                total_loss_disc += loss_disc.item() / 2  # 平均到2次训练

            # -------------------- 训练生成器 --------------------
            self.opt_gen.zero_grad()
            # 重新生成容器（生成器训练时，需要计算梯度）
            container = self.gen.embed(host, secret)
            # 转换为[0,1]范围（适配攻击、判别器、指标计算）
            container_01 = (container + 1) / 2
            host_01 = (host + 1) / 2
            # 保留梯度（攻击需要）
            container_01.requires_grad_(True)  # 显式开启梯度

            # 1. 生成式误差预判：用预测误差修正容器图像（优化创新点3）
            pred_error_total = None
            if self.training:
                # 解包返回的元组，只取总误差（第一个值）
                pred_error_total, _, _, _ = self.error_predictor(container)
                # 用总误差进行修正（保持-1~1范围）
                container = container - c.error_correction_weight * pred_error_total
                container = torch.clamp(container, -1.0, 1.0)  # 限制范围，避免溢出
                # 重新转换为[0,1]（修正后）
                container_01 = (container + 1) / 2
                container_01.requires_grad_(True)  # 重新开启梯度

            # 2. 对抗攻击（随机选择攻击类型）
            attack_type = np.random.choice(c.supported_attacks)
            container_attacked_01 = None
            if attack_type in ['fgsm', 'pgd']:
                # 对抗攻击需要梯度，先计算判别器的预测值
                pred_4d = self.disc(container_01)  # 4维张量：(batch, 1, h, w)
                # 计算临时GAN损失以获取梯度
                loss_gan_temp = self.bce_loss(pred_4d.mean(dim=[1, 2, 3]), torch.ones(batch_size, device=self.device))
                # 反向传播获取梯度（仅计算container_01的梯度）
                self.gen.zero_grad()
                self.disc.zero_grad()
                loss_gan_temp.backward(retain_graph=True)
                # 安全获取梯度（避免梯度为None）
                container_grad = container_01.grad.detach() if container_01.grad is not None else torch.zeros_like(container_01)
                # 执行攻击
                container_attacked_01 = attack_image(container_01, attack_type, container_grad, c.epsilon)
                # 清空梯度，避免残留
                container_01.grad.zero_()
            else:
                # 非对抗攻击，直接调用
                container_attacked_01 = attack_image(container_01, attack_type)

            # 转换回[-1,1]范围，适配生成器提取
            container_attacked = (container_attacked_01 * 2) - 1

            # 3. 提取并修复秘密
            extracted_secret = self.gen.extract(container_attacked)
            # 转换为[0,1]范围后传入refine_head（匹配Main的refine_head预期）
            extracted_secret_01 = (extracted_secret + 1) / 2
            extracted_secret_refined = self.refine_head(extracted_secret_01)  # refine_head输出[0,1]

            # 4. 动态权重调整（修复键不匹配问题）
            texture_complexity = self.pzms_extractor.get_texture_complexity(host)
            # 转换为[0,1]范围（损失计算）
            secret_01 = (secret + 1) / 2
            # 计算基础损失（键改为cap/rob/imp，匹配weight_balancer）
            loss_capacity = self.mse_loss(extracted_secret_refined, secret_01)  # refined已是[0,1]
            loss_imperceptible = 1 - SSIM(container_01, host_01)  # 不可感知性损失（SSIM越小，损失越大）
            loss_robustness = self.mse_loss(extracted_secret_01, extracted_secret_refined)  # 鲁棒性损失

            # 收集当前损失值（键匹配）
            current_losses = {
                'cap': loss_capacity.item(),
                'rob': loss_robustness.item(),
                'imp': loss_imperceptible.item()
            }
            # 动态调整权重
            cap_w, rob_w, imp_w = self.weight_balancer.adjust(texture_complexity, attack_type, current_losses)

            # 5. 损失计算
            # 误差预测损失（确保pred_error_total不为None）
            if pred_error_total is None:
                pred_error_total, _, _, _ = self.error_predictor(container)
            pred_error_01 = (pred_error_total + 1) / 2  # 转换为[0,1]，匹配容器范围
            loss_error = self.mse_loss(pred_error_01, torch.zeros_like(container_01))  # 预测误差越小越好

            # GAN损失：使用get_score方法（返回1维，匹配BCE损失）
            loss_gan = self.bce_loss(self.disc.get_score(container_01), torch.ones(batch_size, device=self.device))

            # 特征匹配损失（增强多域闭环融合）
            disc_features_real = self.disc.get_intermediate_features(host_01)
            disc_features_fake = self.disc.get_intermediate_features(container_01)
            loss_feat_match = self.mse_loss(disc_features_fake, disc_features_real.detach())

            # 总损失（加权求和）
            loss_gen = (
                cap_w * loss_capacity +
                rob_w * loss_robustness +
                imp_w * loss_imperceptible +
                c.gan_weight * loss_gan +
                c.error_pred_weight * loss_error +
                c.feat_match_weight * loss_feat_match
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

        # 平均损失（除以迭代次数）
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
            # 修正：StegoDataset返回的是字典，需遍历batch字典
            for batch in self.val_loader:
                host = batch['host'].to(self.device)
                secret = batch['secret'].to(self.device)

                # 生成容器并提取秘密
                container = self.gen.embed(host, secret)
                # 转换为[0,1]范围（适配指标计算）
                container_01 = (container + 1) / 2
                host_01 = (host + 1) / 2
                secret_01 = (secret + 1) / 2

                # 提取并修复秘密
                extracted_secret = self.gen.extract(container)
                extracted_secret_01 = (extracted_secret + 1) / 2
                extracted_refined = self.refine_head(extracted_secret_01)  # 修复后的秘密

                # 计算指标：秘密提取（核心指标）
                total_psnr_secret += PSNR(extracted_refined, secret_01)
                total_ssim_secret += SSIM(extracted_refined, secret_01).item()
                # 计算指标：容器与宿主（不可感知性，新增）
                total_psnr_container += PSNR(container_01, host_01)
                total_ssim_container += SSIM(container_01, host_01).item()

        # 平均指标（除以验证集迭代次数）
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

        # 保存最佳模型（以秘密提取的PSNR为核心指标）
        if avg_psnr_secret > self.best_psnr:
            self.best_psnr = avg_psnr_secret
            self.best_psnr_container = avg_psnr_container
            # 保存检查点（包含优化器状态，支持断点续训）
            checkpoint_path = os.path。join(c.CHECKPOINT_PATH, "best_model.pth")
            torch.save({
                "gen_state_dict": self.gen。state_dict()，
                "disc_state_dict": self.disc。state_dict()，
                "opt_gen_state_dict": self.opt_gen.state_dict(),
                "opt_disc_state_dict": self.opt_disc.state_dict(),
                "epoch": epoch,
                "best_psnr": self.best_psnr，
                "best_psnr_container": self.best_psnr_container
            }, checkpoint_path)
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

            # 验证+保存示例图片（按验证频率执行）
            if (epoch + 1) % c.val_freq == 0:
                avg_psnr = self.validate(epoch)
                # 保存示例图片
                self.save_sample_images(epoch=epoch+1)

                # 早停判断（核心指标：秘密提取的PSNR）
                if avg_psnr > self.best_psnr:
                    no_improve_epochs = 0  # 重置计数器
                else:
                    no_improve_epochs += 1
                    print(f"⚠️  验证PSNR连续{no_improve_epochs}轮未提升（耐心：{c.early_stop_patience}）")
                    if no_improve_epochs >= c.early_stop_patience:
                        print(f"🛑 早停触发，停止训练（Epoch: {epoch+1}）")
                        early_stop_triggered = True
                        break

            # 保存断点（按保存频率执行，且未触发早停）
            if (epoch + 1) % c.save_freq == 0 and not early_stop_triggered:
                checkpoint_path = os.path。join(c.CHECKPOINT_PATH, f"model_epoch_{epoch + 1}.pth")
                torch.save({
                    "gen_state_dict": self.gen.state_dict(),
                    "disc_state_dict": self.disc.state_dict(),
                    "opt_gen_state_dict": self.opt_gen.state_dict(),
                    "opt_disc_state_dict": self.opt_disc.state_dict(),
                    "epoch": epoch,
                    "best_psnr": self.best_psnr
                }, checkpoint_path)
                print(f"💾 保存断点模型：model_epoch_{epoch+1}.pth")

        # 训练结束后保存最终示例图片
        if not early_stop_triggered:
            self.save_sample_images()  # 保存final版本

        # 关闭日志写入器（修正中文标点）
        self.writer.close()
        print("\n🎉 训练流程完成！")

if __name__ == "__main__":
    # 实例化并启动训练
    trainer = Trainer()
    trainer.run()
