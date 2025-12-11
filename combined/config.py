# ===================== 超参数配置（全局） =====================
# 基础数值参数
clamp = 2.0  # 梯度裁剪阈值（备用）
channels_in = 3  # 输入图像通道数（RGB=3）
log10_lr = -4.5  # 备用学习率（实际使用lr_gen/lr_disc）
lr = 10 ** log10_lr  # 备用学习率
weight_decay = 1e-5  # 权重衰减（正则化）
init_scale = 0.01  # 模型初始化缩放因子

# ===================== 模型配置 =====================
device_ids = [0]  # 多GPU训练时的设备ID（单GPU用[0]）
target_channels = 64  # 模型中间层目标通道数（可逆块/融合模块）
pzms_max_order = 4  # PZMs最大阶数（原8改为4，提升稳定性，匹配model.py）

# ===================== 训练配置 =====================
# 基础训练参数
batch_size = 2  # 训练批次大小
cropsize = 256  # 训练图像裁剪尺寸
betas = (0.5, 0.999)  # Adam优化器的beta参数
weight_step = 200  # 学习率衰减步长（备用）
gamma = 0.5  # 学习率衰减系数（备用）
epochs = 100  # 训练总轮数
lr_gen = 2e-4  # 生成器（EnhancedPRIS）学习率
lr_disc = 1e-4  # 判别器（DCGAN Discriminator）学习率

# 模型保存配置
save_freq = 20  # 模型检查点保存频率（每20轮保存一次）
checkpoint_on_error = True  # 出错时是否保存检查点

# ===================== 验证配置 =====================
cropsize_val = 256  # 验证图像裁剪尺寸
batchsize_val = 2  # 验证批次大小
shuffle_val = False  # 验证集是否打乱
val_freq = 10  # 验证频率（每10轮验证一次）

# ===================== GAN相关配置（创新点1/2/4） =====================
gan_lr = 1e-4  # 判别器学习率（与lr_disc保持一致，备用）
gan_weight = 0.1  # GAN对抗损失权重
error_pred_weight = 0.1  # 生成式误差预判损失权重
repair_weight = 0.05  # 判别器辅助修复损失权重（备用）

# ===================== 三目标动态平衡配置（创新点3） =====================
base_capacity_weight = 1.0  # 容量损失基础权重
base_robustness_weight = 1.0  # 鲁棒性损失基础权重
base_imperceptibility_weight = 1.0  # 不可感知性损失基础权重

# ===================== 攻击配置（创新点4：对抗性攻击泛化） =====================
# 支持的攻击类型（新增FGSM/PGD对抗性攻击）
supported_attacks = ["gaussian", "jpeg", "geometry", "fgsm", "pgd"]
# 攻击类型说明
attack_types = {
    'gaussian': '高斯噪声攻击',
    'jpeg': 'JPEG压缩攻击',
    'geometry': '几何变换（缩放+旋转）攻击',
    'fgsm': 'FGSM快速梯度符号对抗攻击',
    'pgd': 'PGD投影梯度下降对抗攻击'
}
# 对抗性攻击参数（FGSM/PGD）
epsilon = 0.05  # 扰动强度（对抗攻击）
pgd_steps = 4  # PGD攻击迭代步数
pgd_alpha = 0.01  # PGD攻击每步扰动步长

# ===================== 数据集配置 =====================
# 数据集路径（Windows系统适配，使用绝对路径）
TRAIN_PATH = r'D:\pythonProject\Watermarking\DIV2K\train'
VAL_PATH = r'D:\pythonProject\Watermarking\DIV2K\valid'
# 图像格式
format_train = 'png'
format_val = 'png'

# ===================== 路径配置（Windows系统适配，修正路径错误） =====================
# 模型保存路径
MODEL_PATH = r'D:\pythonProject\Watermarking\Code\combined\model'
# 图像保存路径（宿主/秘密/容器/提取图像）
IMAGE_PATH = r'D:\pythonProject\Watermarking\Code\combined\image'
IMAGE_PATH_host = fr'{IMAGE_PATH}\host'
IMAGE_PATH_secret = fr'{IMAGE_PATH}\secret'
IMAGE_PATH_container = fr'{IMAGE_PATH}\container'
IMAGE_PATH_extracted = fr'{IMAGE_PATH}\extracted'

# 检查点和日志路径（与main_GAN.py匹配）
CHECKPOINT_PATH = r'D:\pythonProject\Watermarking\Code\combined\checkpoints'
LOG_PATH = r'D:\pythonProject\Watermarking\Code\combined\logs'

# ===================== 日志与可视化配置 =====================
loss_display_cutoff = 2.0  # 损失显示阈值（超过则截断）
loss_names = ['train loss', 'val loss', 'lr', 'attack method']  # 损失日志名称
silent = False  # 是否静默训练（不打印日志）
live_visualization = False  # 是否实时可视化（需TensorBoard）
progress_bar = True  # 是否显示进度条
