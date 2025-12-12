import os
import torch
from PIL import Image
import numpy as np
from torchvision import transforms
import config as c
from model import EnhancedPRIS
from Main import attack_image

# 移除重复的transforms导入

class StegoInference:
    """隐写术推理工具：实现图像隐写（嵌入秘密）和秘密提取"""

    def __init__(self, checkpoint_path):
        # 设备配置
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"推理使用设备: {self.device}")

        # 初始化模型
        self.gen = EnhancedPRIS().to(self.device)  # 修正：中文句号→英文句号，to方法正确调用
        self._load_checkpoint(checkpoint_path)
        self.gen.eval()  # 推理模式，关闭BN/Dropout

        # 图像预处理（与训练一致：转为张量+归一化到[-1, 1]，添加尺寸适配）
        self.transform = transforms.Compose([
            transforms.Resize((c.cropsize, c.cropsize)),  # 适配训练时的裁剪尺寸（从config读取）
            transforms.ToTensor(),  # 修正：中文逗号→英文逗号
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # 修正：中文逗号→英文逗号
        ])

        # 图像后处理（将张量从[-1,1]转回PIL图像[0,1]，添加范围限制）
        self.postprocess = self._build_postprocess()

    def _build_postprocess(self):
        """构建后处理流水线，确保张量范围正确"""
        def clamp_tensor(tensor):
            """限制张量在[0, 1]范围，避免PIL图像生成异常"""
            return tensor.clamp(0, 1)

        return transforms.Compose([
            transforms.Normalize(mean=[-1, -1, -1], std=[2, 2, 2]),  # 逆归一化：(-1~1)→(0~1)
            transforms.Lambda(clamp_tensor),  # 关键：限制范围，防止像素值溢出
            transforms.ToPILImage()
        ])

    def _load_checkpoint(self, checkpoint_path):
        """加载模型权重（增强鲁棒性：处理权重键不匹配、路径不存在）"""
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"❌ 模型文件不存在: {checkpoint_path}")
        
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            # 处理两种权重保存格式：gen_state_dict 或直接的state_dict
            if "gen_state_dict" in checkpoint:
                self.gen.load_state_dict(checkpoint["gen_state_dict"])
            elif "state_dict" in checkpoint:
                self.gen.load_state_dict(checkpoint["state_dict"])
            else:
                self.gen.load_state_dict(checkpoint)
            print(f"✅ 成功加载模型: {checkpoint_path}")
        except RuntimeError as e:
            raise RuntimeError(f"❌ 模型权重加载失败（可能是模型结构不匹配或权重键错误）: {str(e)}")
        except Exception as e:
            raise Exception(f"❌ 加载模型时发生未知错误: {str(e)}")

    def embed(self, host_path, secret_path, attack_type=None, save_container=False, save_path=None):
        """
        隐写：将秘密图像嵌入宿主图像生成容器图像
        Args:
            host_path: 宿主图像路径
            secret_path: 秘密图像路径
            attack_type: 攻击类型（可选，如"fgsm"，需在c.supported_attacks中定义）
            save_container: 是否保存容器图像
            save_path: 容器图像保存路径
        Returns:
            容器图像（PIL格式）
        """
        # 加载并预处理图像（添加异常处理）
        try:
            host = Image.open(host_path).convert('RGB')
            secret = Image.open(secret_path).convert('RGB')
        except FileNotFoundError as e:
            raise FileNotFoundError(f"❌ 图像文件不存在: {str(e)}")
        except Exception as e:
            raise Exception(f"❌ 加载图像失败: {str(e)}")

        # 转为张量并添加batch维度
        host_tensor = self.transform(host).unsqueeze(0).to(self.device)  # [1, 3, H, W]
        secret_tensor = self.transform(secret).unsqueeze(0).to(self.device)  # 修正：中文句号→英文句号

        # 生成容器图像（禁用梯度计算，加速推理）
        with torch.no_grad():
            container_tensor = self.gen.embed(host_tensor, secret_tensor)

        # 施加攻击（可选：处理张量范围，适配attack_image的输入要求）
        if attack_type and attack_type in c.supported_attacks:
            # 攻击函数通常需要[0,1]的张量，因此先转换范围
            container_tensor_01 = (container_tensor + 1) / 2  # [-1,1] → [0,1]
            container_tensor_01 = attack_image(container_tensor_01, attack_type)  # 施加攻击
            container_tensor = container_tensor_01 * 2 - 1  # [0,1] → [-1,1]，转回原范围

        # 后处理为PIL图像
        container = self.postprocess(container_tensor.squeeze(0)。cpu())

        # 保存容器图像
        if save_container 和 save_path:
            os.makedirs(os.path。dirname(save_path), exist_ok=True)
            container.save(save_path)
            print(f"📸 容器图像已保存至: {save_path}")

        return container

    def extract(self, container_path, save_secret=False, save_path=None):
        """
        提取：从容器图像中提取秘密图像
        Args:
            container_path: 容器图像路径
            save_secret: 是否保存提取的秘密图像
            save_path: 秘密图像保存路径
        Returns:
            提取的秘密图像（PIL格式）
        """
        # 加载并预处理容器图像（添加异常处理）
        try:
            container = Image.open(container_path)。convert('RGB')
        except FileNotFoundError as e:
            raise FileNotFoundError(f"❌ 容器图像文件不存在: {str(e)}")
        except Exception as e:
            raise Exception(f"❌ 加载容器图像失败: {str(e)}")

        container_tensor = self.transform(container)。unsqueeze(0)。到(self.device)

        # 提取秘密图像（禁用梯度计算）
        with torch.no_grad():
            secret_tensor = self.gen。extract(container_tensor)

        # 后处理为PIL图像
        extracted_secret = self.postprocess(secret_tensor.squeeze(0)。cpu())

        # 保存提取的秘密图像
        if save_secret 和 save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            extracted_secret.save(save_path)
            print(f"📸 提取的秘密图像已保存至: {save_path}")

        return extracted_secret


# 示例用法
if __name__ == "__main__":
    # 初始化推理器（使用最佳模型）
    try:
        checkpoint = os.path.join(c.CHECKPOINT_PATH, "best_model.pth")
        stego = StegoInference(checkpoint)

        # 隐写示例
        # 确保config中定义了对应的图像路径
        host_img = os.path.join(c.IMAGE_PATH_host， "test_host.png")
        secret_img = os.path.join(c.IMAGE_PATH_secret, "test_secret.png")
        container = stego.embed(
            host_path=host_img,
            secret_path=secret_img,
            attack_type=无，  # 不施加攻击，可改为"fgsm"/"pgd"等
            save_container=True,
            save_path=os.path.join(c.IMAGE_PATH_container, "test_container.png")
        )

        # 提取示例
        extracted = stego.extract(
            container_path=os.path.join(c.IMAGE_PATH_container, "test_container.png"),
            save_secret=True,
            save_path=os.path.join(c.IMAGE_PATH_extracted, "test_extracted.png")
        )
        print("✅ 隐写和提取流程完成！")
    except Exception as e:
        print(f"❌ 执行失败: {str(e)}")
