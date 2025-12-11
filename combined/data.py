import os
import torch
from PIL import Image
import numpy as np
from torchvision import transforms
import config as c
from model import EnhancedPRIS
from Main import attack_image


class StegoInference:
    """隐写术推理工具：实现图像隐写和秘密提取"""

    def __init__(self, checkpoint_path):
        # 设备配置
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 初始化模型
        self.gen = EnhancedPRIS()。到(self.device)
        self._load_checkpoint(checkpoint_path)
        self.gen.eval()

        # 图像预处理（与训练一致）
        self.transform = transforms.Compose([
            transforms.ToTensor()，
            transforms.Normalize(mean=[0.5， 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

        # 图像后处理（将张量转为PIL图像）
        self.postprocess = transforms.Compose([
            transforms.Normalize(mean=[-1, -1, -1], std=[2, 2, 2]),  # 从-1~1转回0~1
            transforms.ToPILImage()
        ])

    def _load_checkpoint(self, checkpoint_path):
        """加载模型权重"""
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"模型文件不存在: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.gen.load_state_dict(checkpoint["gen_state_dict"])
        print(f"成功加载模型: {checkpoint_path}")

    def embed(self, host_path, secret_path, attack_type=None, save_container=False, save_path=None):
        """
        隐写：将秘密图像嵌入宿主图像生成容器图像
        Args:
            host_path: 宿主图像路径
            secret_path: 秘密图像路径
            attack_type: 攻击类型（可选，如"fgsm"）
            save_container: 是否保存容器图像
            save_path: 容器图像保存路径
        Returns:
            容器图像（PIL格式）
        """
        # 加载并预处理图像
        host = Image.open(host_path).convert('RGB')
        secret = Image.open(secret_path).convert('RGB')
        host_tensor = self.transform(host).unsqueeze(0).to(self.device)
        secret_tensor = self.transform(secret)。unsqueeze(0).to(self.device)

        # 生成容器图像
        with torch.no_grad():
            container_tensor = self.gen.embed(host_tensor, secret_tensor)

        # 施加攻击（可选）
        if attack_type in c.supported_attacks:
            container_tensor = attack_image(container_tensor, attack_type)

        # 后处理为PIL图像
        container = self.postprocess(container_tensor.squeeze(0).cpu())

        # 保存容器图像
        if save_container and save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            container.save(save_path)
            print(f"容器图像已保存至: {save_path}")

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
        # 加载并预处理容器图像
        container = Image.open(container_path).convert('RGB')
        container_tensor = self.transform(container).unsqueeze(0).to(self.device)

        # 提取秘密图像
        with torch.no_grad():
            secret_tensor = self.gen.extract(container_tensor)

        # 后处理为PIL图像
        extracted_secret = self.postprocess(secret_tensor.squeeze(0).cpu())

        # 保存提取的秘密图像
        if save_secret and save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            extracted_secret.save(save_path)
            print(f"提取的秘密图像已保存至: {save_path}")

        return extracted_secret


# 示例用法
if __name__ == "__main__":
    # 初始化推理器（使用最佳模型）
    checkpoint = os.path.join(c.CHECKPOINT_PATH, "best_model.pth")
    stego = StegoInference(checkpoint)

    # 隐写示例
    host_img = os.path.join(c.IMAGE_PATH_host, "test_host.png")
    secret_img = os.path.join(c.IMAGE_PATH_secret, "test_secret.png")
    container = stego.embed(
        host_path=host_img,
        secret_path=secret_img,
        attack_type=None,  # 不施加攻击
        save_container=True,
        save_path=os.path.join(c.IMAGE_PATH_container, "test_container.png")
    )

    # 提取示例
    extracted = stego.extract(
        container_path=os.path.join(c.IMAGE_PATH_container, "test_container.png"),
        save_secret=True,
        save_path=os.path.join(c.IMAGE_PATH_extracted, "test_extracted.png")
    )
