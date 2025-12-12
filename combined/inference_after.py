import os
import torch
from PIL import Image
import numpy as np
from torchvision import transforms
import config as c
from model import EnhancedPRIS
# 注意：若 attack_image 不在 Main.py 顶层，需调整导入路径，或直接复制函数到此处
from Main import attack_image  # 若报错，可将 Main.py 中的 attack_image 及依赖函数复制到当前文件


class StegoInference:
    """隐写术推理工具：实现图像隐写和秘密提取（优化版）"""

    def __init__(self, checkpoint_path):
        # 设备配置
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"使用设备: {self.device}")

        # 初始化模型
        self.gen = EnhancedPRIS().to(self.device)
        self._load_checkpoint(checkpoint_path)
        self.gen.eval()  # 推理模式

        # 图像预处理（与训练完全一致：包含裁剪/缩放，匹配训练的cropsize）
        self.transform = transforms.Compose([
            transforms.Resize((c.cropsize, c.cropsize), transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # 转-1~1
        ])

        # 图像后处理（将-1~1张量转回0~1的PIL图像）
        self.postprocess = self._get_postprocess()

    def _get_postprocess(self):
        """构建后处理管道（修复原代码的归一化顺序问题）"""

        def process_tensor(tensor):
            # 步骤1：从-1~1转回0~1
            tensor = (tensor + 1) / 2  # 更直观的转换，避免归一化参数错误
            # 步骤2：限制范围（防止数值溢出）
            tensor = torch.clamp(tensor, 0.0, 1.0)
            # 步骤3：转为PIL图像
            if tensor.ndim == 3:
                return Image.fromarray((tensor.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8))
            else:
                raise ValueError("张量维度必须为3（C, H, W）")

        return process_tensor

    def _load_checkpoint(self, checkpoint_path):
        """加载模型权重（修复键不匹配问题，兼容训练时的保存格式）"""
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"模型文件不存在: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        # 兼容两种键名：训练时保存的是net_state_dict，若自定义保存为gen_state_dict也支持
        state_dict_key = "gen_state_dict" if "gen_state_dict" in checkpoint else "net_state_dict"
        try:
            self.gen.load_state_dict(checkpoint[state_dict_key], strict=False)  # 忽略不匹配的层（如多GPU训练的前缀）
        except RuntimeError as e:
            print(f"⚠️ 模型权重加载时出现不匹配，尝试移除模块前缀：{e}")
            # 移除DataParallel的module.前缀（多GPU训练的模型）
            new_state_dict = {k.replace("module.", ""): v for k, v in checkpoint[state_dict_key].items()}
            self.gen.load_state_dict(new_state_dict, strict=False)
        print(f"✅ 成功加载模型: {checkpoint_path}")

    def _preprocess_image(self, image_path):
        """辅助函数：加载并预处理图像，返回张量（1, C, H, W）"""
        try:
            img = Image.open(image_path).convert('RGB')
        except Exception as e:
            raise ValueError(f"加载图像失败: {image_path}，错误：{e}")
        img_tensor = self.transform(img).unsqueeze(0).to(self.device)  # 添加batch维度
        return img_tensor

    def embed(self, host_path, secret_path, attack_type=None, save_container=False, save_path=None):
        """
        隐写：将秘密图像嵌入宿主图像生成容器图像
        Args:
            host_path: 宿主图像路径
            secret_path: 秘密图像路径
            attack_type: 攻击类型（可选，如"gaussian"，对抗攻击FGSM/PGD在推理阶段不支持）
            save_container: 是否保存容器图像
            save_path: 容器图像保存路径（需包含文件名，如xxx.png）
        Returns:
            容器图像（PIL格式）
        """
        # 加载并预处理图像
        host_tensor = self._preprocess_image(host_path)
        secret_tensor = self._preprocess_image(secret_path)

        # 生成容器图像（推理阶段无梯度）
        with torch.no_grad():
            container_tensor = self.gen.embed(host_tensor, secret_tensor)  # (1, C, H, W)

        # 施加攻击（修复：转换为0~1范围，匹配attack_image的输入要求）
        if attack_type is not None:
            if attack_type in ["fgsm", "pgd"]:
                print(f"⚠️ 推理阶段无法计算梯度，跳过对抗攻击：{attack_type}，建议使用非对抗攻击（如gaussian/jpeg）")
            elif attack_type in c.supported_attacks:
                # 转换为0~1范围（attack_image处理的是0~1的张量）
                container_01 = (container_tensor + 1) / 2  # -1~1 → 0~1
                # 攻击函数不需要梯度（对抗攻击除外），container_grad设为None
                attacked_01 = attack_image(container_01, attack_type, container_grad=None, epsilon=c.epsilon)
                # 转换回-1~1范围
                container_tensor = (attacked_01 * 2) - 1  # 0~1 → -1~1
            else:
                print(f"⚠️ 不支持的攻击类型：{attack_type}，支持的类型：{c.supported_attacks}")

        # 后处理为PIL图像（移除batch维度）
        container = self.postprocess(container_tensor.squeeze(0))

        # 保存容器图像
        if save_container and save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            container.save(save_path)
            print(f"📸 容器图像已保存至: {save_path}")

        return container

    def extract(self, container_path, save_secret=False, save_path=None):
        """
        提取：从容器图像中提取秘密图像
        Args:
            container_path: 容器图像路径
            save_secret: 是否保存提取的秘密图像
            save_path: 秘密图像保存路径（需包含文件名，如xxx.png）
        Returns:
            提取的秘密图像（PIL格式）
        """
        # 加载并预处理容器图像
        container_tensor = self._preprocess_image(container_path)

        # 提取秘密图像（推理阶段无梯度）
        with torch.no_grad():
            secret_tensor = self.gen.extract(container_tensor)  # (1, C, H, W)

        # 后处理为PIL图像（移除batch维度）
        extracted_secret = self.postprocess(secret_tensor.squeeze(0))

        # 保存提取的秘密图像
        if save_secret and save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            extracted_secret.save(save_path)
            print(f"📸 提取的秘密图像已保存至: {save_path}")

        return extracted_secret


# 示例用法
if __name__ == "__main__":
    # ===================== 配置项（根据实际情况调整） =====================
    # 模型路径：注意训练时保存的是best_model.pt（不是pth）
    checkpoint = os.path.join(c.CHECKPOINT_PATH, "best_model.pt")
    # 测试图像路径（需提前准备，或替换为自己的图像路径）
    host_img = os.path.join(c.IMAGE_PATH_host, "test_host.png")  # 宿主图像
    secret_img = os.path.join(c.IMAGE_PATH_secret, "test_secret.png")  # 秘密图像
    container_img = os.path.join(c.IMAGE_PATH_container, "test_container.png")  # 生成的容器图像
    extracted_img = os.path.join(c.IMAGE_PATH_extracted, "test_extracted.png")  # 提取的秘密图像

    # ===================== 推理流程 =====================
    # 初始化推理器
    stego = StegoInference(checkpoint)

    # 1. 隐写：嵌入秘密图像，可选施加高斯噪声攻击
    print("\n--- 开始隐写 ---")
    container = stego.embed(
        host_path=host_img,
        secret_path=secret_img,
        attack_type="gaussian",  # 可选：None, "gaussian", "jpeg", "geometry"
        save_container=True,
        save_path=container_img
    )

    # 2. 提取：从容器图像中提取秘密图像
    print("\n--- 开始提取 ---")
    extracted = stego.extract(
        container_path=container_img,
        save_secret=True,
        save_path=extracted_img
    )

    print("\n🎉 推理完成！")
