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
pzms_max_order = 4  # PZMs最大阶数（匹配model.py）

# ===================== 训练配置 =====================
# 基础训练参数
batch_size = 2  # 训练批次大小（根据GPU显存调整，2适合256x256）
cropsize = 256  # 训练图像裁剪尺寸
betas = (0.5, 0.999)  # Adam优化器的beta参数
weight_step = 200  # 学习率衰减步长（备用）
gamma = 0.5  # 学习率衰减系数（备用）
epochs = 6000  # 训练总轮数
lr_step_size = 100  # 学习率衰减步长（覆盖备用的weight_step）
lr_gamma = 0.9  # 学习率衰减系数（覆盖备用的gamma）
lr_gen = 2e-4  # 生成器（EnhancedPRIS）学习率
lr_disc = 1e-4  # 判别器（DCGAN Discriminator）学习率
early_stop_patience = 100  # 早停机制：多少轮无提升则停止（最终生效值）

# 模型保存配置
save_freq = 20  # 模型检查点保存频率（每20轮保存一次）
checkpoint_on_error = True  # 出错时是否保存检查点

# ===================== 验证配置 =====================
cropsize_val = 256  # 验证图像裁剪尺寸
batchsize_val = 2  # 验证批次大小
shuffle_val = False  # 验证集是否打乱
val_freq = 10  # 验证频率（每10轮验证一次）

# ===================== GAN相关配置 =====================
gan_lr = 1e-4  # 判别器学习率（与lr_disc保持一致，备用）
gan_weight = 0.1  # GAN对抗损失权重
error_pred_weight = 0.1  # 生成式误差预判损失权重
error_correction_weight = 0.05  # 误差修正权重（用于容器图像误差修正）
repair_weight = 0.05  # 判别器辅助修复损失权重（备用）
feat_match_weight = 0.2  # 特征匹配损失权重
adv_loss_weight = 0.5  # 对抗性攻击损失权重（补充：main.py中使用）

# ===================== 三目标动态平衡配置 =====================
base_capacity_weight = 1.0  # 容量损失基础权重
base_robustness_weight = 1.0  # 鲁棒性损失基础权重
base_imperceptibility_weight = 1.0  # 不可感知性损失基础权重

# ===================== 攻击配置 =====================
# 支持的攻击类型
supported_attacks = ["gaussian", "jpeg", "geometry", "fgsm", "pgd"]
# 攻击类型说明（备用）
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
# 非对抗攻击参数（补充：main.py中使用）
gaussian_noise_std = 0.01  # 高斯噪声标准差
jpeg_quant_step = 10  # JPEG压缩量化步长

# ===================== 数据集配置 =====================
# 数据集路径（Windows系统适配，使用绝对路径）
TRAIN_PATH = r'D:\pythonProject\Watermarking\DIV2K\train'
VAL_PATH = r'D:\pythonProject\Watermarking\DIV2K\valid'
# 图像格式
format_train = 'png'
format_val = 'png'
# 数据集样本限制（补充：main.py中使用，可根据需求调整）
max_train_samples = None  # 不限制训练样本数
max_val_samples = None  # 不限制验证样本数

# ===================== 路径配置（Windows系统适配） =====================
# 模型保存路径
MODEL_PATH = r'D:\pythonProject\Watermarking\Code\combined\model'
# 图像保存路径（宿主/秘密/容器/提取图像）
IMAGE_PATH = r'D:\pythonProject\Watermarking\Code\combined\image'
IMAGE_PATH_host = fr'{IMAGE_PATH}\host'
IMAGE_PATH_secret = fr'{IMAGE_PATH}\secret'
IMAGE_PATH_container = fr'{IMAGE_PATH}\container'
IMAGE_PATH_extracted = fr'{IMAGE_PATH}\extracted'
IMAGE_PATH_combined = fr'{IMAGE_PATH}\combined'
SAMPLE_IMAGE_PATH = fr'{IMAGE_PATH}\samples'  # 示例图片保存路径（备用）

# 检查点和日志路径
CHECKPOINT_PATH = r'D:\pythonProject\Watermarking\Code\combined\checkpoints'
LOG_PATH = r'D:\pythonProject\Watermarking\Code\combined\logs'

# ===================== 日志与可视化配置 =====================
loss_display_cutoff = 2.0  # 损失显示阈值（超过则截断）
loss_names = ['train loss', 'val loss', 'lr', 'attack method']  # 损失日志名称
silent = False  # 是否静默训练（不打印日志）
live_visualization = False  # 是否实时可视化（需TensorBoard）
progress_bar = True  # 是否显示进度条

# ===================== 示例图片配置（补充：main.py中使用） =====================
num_sample_images = 10  # 每次保存的示例图片数量
