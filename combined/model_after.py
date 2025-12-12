import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from scipy import ndimage
import config as c


# ===================== 创新点1：PZMs抗几何特征提取模块（稳定版+效率优化） =====================
class PZMsFeatureExtractor(nn.Module):
    """PZMs（Phase-Zernike Moments）特征提取器，抗几何攻击的全局特征"""

    def __init__(self, max_order=8):
        super(PZMsFeatureExtractor, self).__init__()
        self.max_order = max_order
        self.radial_poly = self._precompute_radial_polynomials()

    def _precompute_radial_polynomials(self):
        """预计算Zernike多项式径向部分"""
        radial = {}
        for n in range(self.max_order + 1):
            for m in range(-n, n + 1, 2):
                if (n - abs(m)) % 2 != 0:
                    continue
                radial[(n, m)] = self._radial(n, abs(m))
        return radial

    def _radial(self, n, m):
        """径向多项式R_n^m(r)计算"""

        def func(r):
            r = np.clip(r, 0, 1)
            result = 0
            for s in range((n - m) // 2 + 1):
                coeff = ((-1) ** s) * math.factorial(n - s) / (
                        math.factorial(s) * math.factorial((n + m) // 2 - s) *
                        math.factorial((n - m) // 2 - s)
                )
                result += coeff * (r ** (n - 2 * s))
            return result

        return func

    def forward(self, x):
        """提取PZMs全局特征（batch, features）"""
        batch_size, channels, h, w = x.shape
        features = []

        # 转灰度图
        if channels == 3:
            gray = 0.299 * x[:, 0] + 0.587 * x[:, 1] + 0.114 * x[:, 2]
        else:
            gray = x[:, 0]

        for i in range(batch_size):
            img = gray[i].cpu().detach().numpy()
            img = (img - np.mean(img)) / (np.std(img) + 1e-8)  # 归一化

            # 质心校正（抗平移）- 处理空图像边界情况
            img_abs = np.abs(img)
            if np.sum(img_abs) < 1e-8:  # 空图像/全黑图像
                cy, cx = h / 2, w / 2
            else:
                cy, cx = ndimage.center_of_mass(img_abs)

            # 主轴方向校正（抗旋转）- 修复类型错误：索引转float64
            y, x_grid = np.indices(img.shape, dtype=np.float64)  # 关键修复：指定float64
            y -= cy
            x_grid -= cx

            # 惯性矩阵计算（避免除以0）
            inertia = np.array([[np.sum(x_grid ** 2 * img), np.sum(x_grid * y * img)],
                                [np.sum(x_grid * y * img), np.sum(y ** 2 * img)]])
            eigvals, eigvecs = np.linalg.eigh(inertia)
            angle = np.arctan2(eigvecs[1, 0], eigvecs[0, 0])

            # 计算PZMs特征（简化版，避免复数异常）
            pzm_feat = []
            for n in range(min(self.max_order, 4)):  # 降低阶数，提升稳定性
                for m in range(-n, n + 1, 2):
                    if (n - abs(m)) % 2 != 0:
                        continue
                    moment = self._compute_zernike_moment(img, n, m, cy, cx, angle)
                    pzm_feat.append(moment)
            features.append(np.array(pzm_feat, dtype=np.float32))  # 指定float32，适配torch

        # 统一特征长度（填充或截断到20维）
        max_len = 20  # 固定为20，与后续线性层输入维度一致
        for i in range(len(features)):
            if len(features[i]) < max_len:
                features[i] = np.pad(features[i], (0, max_len - len(features[i])), mode='constant')
            else:
                features[i] = features[i][:max_len]  # 超过20维则截断

        # 转换为torch tensor，避免类型不匹配
        return torch.tensor(np.array(features), dtype=torch.float32).to(x.device)

    def _compute_zernike_moment(self, img, n, m, cy, cx, angle):
        """计算单阶Zernike矩（纯实数版，避免复数错误）"""
        h, w = img.shape
        # 索引转float64，避免类型错误
        y, x = np.indices((h, w), dtype=np.float64)
        y -= cy
        x -= cx

        # 旋转校正
        cos_ang = np.cos(-angle)
        sin_ang = np.sin(-angle)
        x_rot = x * cos_ang - y * sin_ang
        y_rot = x * sin_ang + y * cos_ang

        # 尺度归一化（避免除以0）
        max_r = np.sqrt(np.max(x_rot ** 2 + y_rot ** 2)) + 1e-8
        r = np.sqrt(x_rot ** 2 + y_rot ** 2) / max_r
        theta = np.arctan2(y_rot, x_rot)

        # 实数版Zernike多项式（避免复数）
        radial = self.radial_poly[(n, m)](r)
        # 用三角函数替代复数指数，避免1j相关错误
        zernike_real = radial * np.cos(m * theta)
        zernike_imag = radial * np.sin(m * theta)
        moment_real = np.sum(img * zernike_real) / (max_r ** 2 + 1e-8)
        moment_imag = np.sum(img * zernike_imag) / (max_r ** 2 + 1e-8)
        moment = np.sqrt(moment_real ** 2 + moment_imag ** 2)  # 模长

        return moment.astype(np.float32)  # 转float32

    def get_texture_complexity(self, x):
        """计算图像纹理复杂度（用于创新点3：动态权重调整）"""
        feat = self.forward(x)
        return torch.var(feat, dim=1)  # 特征方差作为复杂度指标


# ===================== 基础可逆块模块（通道数适配+维度稳定） =====================
class ReversibleBlock(nn.Module):
    """可逆块（局部特征提取，适配64通道输入，维度稳定优化）"""

    def __init__(self, channels=64):
        super(ReversibleBlock, self).__init__()
        self.channels = channels
        self.half_channels = channels // 2
        # 确保输入通道数是偶数（提前处理，避免动态pad）
        if self.channels % 2 != 0:
            self.channels += 1
            self.half_channels = self.channels // 2

        # 卷积层初始化（使用kaiming初始化，提升收敛速度）
        self.conv1 = nn.Conv2d(self.half_channels, self.half_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(self.half_channels, self.half_channels, 3, padding=1)
        self.norm = nn.InstanceNorm2d(self.half_channels)

        # 初始化卷积层权重
        nn.init.kaiming_normal_(self.conv1.weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.conv2.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x, rev=False):
        """可逆变换：正向嵌入/逆向提取（维度稳定版）"""
        # 提前pad到指定通道数（避免动态pad导致的维度不一致）
        if x.shape[1] < self.channels:
            x = F.pad(x, (0, 0, 0, 0, 0, self.channels - x.shape[1]))  # 通道数补到指定值
        elif x.shape[1] > self.channels:
            x = x[:, :self.channels, :, :]  # 截断多余通道

        x1, x2 = x.chunk(2, dim=1)  # 此时x1/x2的通道数都是half_channels

        if not rev:
            y1 = x1 + self.norm(F.relu(self.conv1(x2)))
            y2 = x2 + self.norm(F.relu(self.conv2(y1)))
        else:
            y2 = x2 - self.norm(F.relu(self.conv2(x1)))
            y1 = x1 - self.norm(F.relu(self.conv1(y2)))

        out = torch.cat([y1, y2], dim=1)
        return out


# ===================== 创新点1：多域特征融合模块（注意力增强版） =====================
class FeatureFusionModule(nn.Module):
    """融合PZMs全局特征 + 可逆块局部特征，解决特征冲突（通道+空间注意力增强）"""

    def __init__(self, local_dim=64, global_dim=20, output_dim=64):
        super(FeatureFusionModule, self).__init__()
        # 局部特征投影（64→64）
        self.local_proj = nn.Conv2d(local_dim, output_dim, kernel_size=1)
        # 全局特征投影（20→64）
        self.global_proj = nn.Sequential(
            nn.Linear(global_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.LeakyReLU(0.2)
        )
        # 通道注意力（增强特征选择）
        self.channel_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(output_dim * 2, output_dim // 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(output_dim // 2, output_dim, 1),
            nn.Sigmoid()
        )
        # 空间注意力（增强位置特征）
        self.spatial_attn = nn.Sequential(
            nn.Conv2d(output_dim * 2, 1, 7, padding=3),
            nn.Sigmoid()
        )
        # 融合后卷积（增加残差连接）
        self.fuse_conv = nn.Sequential(
            nn.Conv2d(output_dim, output_dim, 3, padding=1),
            nn.BatchNorm2d(output_dim),
            nn.LeakyReLU(0.2)
        )
        # 残差连接
        self.residual = nn.Conv2d(output_dim, output_dim, 1)

    def forward(self, local_feat, global_feat):
        """
        local_feat: 可逆块输出 (batch, local_dim, h, w)
        global_feat: PZMs特征 (batch, global_dim)
        """
        b, c, h, w = local_feat.shape
        # 局部特征投影
        local_proj = self.local_proj(local_feat)  # (b, output_dim, h, w)
        # 全局特征投影并广播
        global_proj = self.global_proj(global_feat)  # (b, output_dim)
        global_proj = global_proj.view(b, -1, 1, 1).repeat(1, 1, h, w)  # (b, output_dim, h, w)

        # 注意力融合（通道+空间）
        concat_feat = torch.cat([local_proj, global_proj], dim=1)  # (b, output_dim*2, h, w)
        # 通道注意力
        channel_weight = self.channel_attn(concat_feat)  # (b, output_dim, 1, 1)
        # 空间注意力
        spatial_weight = self.spatial_attn(concat_feat)  # (b, 1, h, w)
        # 融合权重
        attn_weight = channel_weight * spatial_weight  # (b, output_dim, h, w)
        fused_feat = local_proj * attn_weight + global_proj * (1 - attn_weight)

        # 残差连接+卷积
        out = self.fuse_conv(fused_feat) + self.residual(local_proj)
        return out


# ===================== 创新点2：生成式误差预判模块（修复返回值不匹配问题） =====================
class GenerativeErrorPredictor(nn.Module):
    """生成式误差预判：预判像素/rounding/量化误差，联动DCGAN生成器（返回4值适配trainer）"""

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
        """x: 容器图像（-1~1），返回(total_err, pixel_err, round_err, quant_err)适配trainer"""
        # 像素误差
        pixel_err = self.pixel_err(x)
        total_err = pixel_err.clone()

        # Rounding误差：模拟0-255整数化的误差
        round_err = torch.zeros_like(x)
        if simulate_round:
            x_255 = (x + 1) * 127.5  # 转换为0-255
            x_round = torch.round(x_255)
            x_round_norm = (x_round / 127.5) - 1  # 转回-1~1
            round_err = self.round_err(x - x_round_norm)
            total_err += round_err

        # 量化误差：模拟8bit量化的误差
        quant_err = torch.zeros_like(x)
        if simulate_quant:
            x_quant = torch.clamp(torch.floor((x + 1) * 127.5) / 127.5 - 1, -1, 1)
            quant_err = self.quant_err(x - x_quant)
            total_err += quant_err

        # 返回4个值，适配trainer中的解包操作
        return total_err, pixel_err, round_err, quant_err


# ===================== 创新点2：判别器辅助修复头（修复get_score方法缺失问题） =====================
class DiscriminatorRefineHead(nn.Module):
    """利用DCGAN判别器的反馈精细化修复提取的秘密图像（适配disc的get_score方法）"""

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
        # 第二步：获取判别器的评分（兼容disc的get_score方法或直接forward）
        if hasattr(self.disc, 'get_score'):
            disc_score = self.disc.get_score(x_refine)
        else:
            # 若没有get_score，直接调用forward并平均为1维评分
            disc_pred = self.disc(x_refine)  # (b,1,h,w)
            disc_score = disc_pred.mean(dim=[1, 2, 3])  # (b,)

        # 修复力度：评分越低，修复力度越大
        refine_strength = torch.clamp(1 - disc_score, 0.1, 1.0)
        # 广播到图像维度
        refine_strength = refine_strength.view(-1, 1, 1, 1).repeat(1, x.shape[1], x.shape[2], x.shape[3])
        # 第三步：根据评分调整修复
        x_final = x_refine * refine_strength + x * (1 - refine_strength)
        return x_final


# ===================== 解码器模块（通道数适配） =====================
class Decoder(nn.Module):
    def __init__(self, target_channels):
        super(Decoder, self).__init__()
        # 设计对称的通道流转：64→32→32→3（正向）；3→32→32→64（反向）
        self.forward_layers = nn.ModuleList([
            # 正向：64→32
            nn.Conv2d(target_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            # 正向：32→32
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            # 正向：32→3（输出秘密图像）
            nn.Conv2d(32, 3, kernel_size=3, padding=1),
            nn.Tanh()  # 匹配输入图像的归一化范围（如[-1,1]）
        ])

        self.rev_layers = nn.ModuleList([
            # 反向：3→32
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            # 反向：32→32
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            # 反向：32→64（输出提取的特征）
            nn.Conv2d(32, target_channels, kernel_size=3, padding=1)
        ])

    def forward(self, x, rev=False):
        if rev:
            # 反向模式（提取水印）：3→64
            for layer in self.rev_layers:
                x = layer(x)
        else:
            # 正向模式（嵌入水印）：64→3
            for layer in self.forward_layers:
                x = layer(x)
        return x


# ===================== 创新点1：DCGAN生成器（多域特征融合生成+残差连接） =====================
class DCGANGenerator(nn.Module):
    """DCGAN生成器：融合多域特征，生成全局鲁棒+局部可逆的容器图像（残差连接优化）"""

    def __init__(self, in_channels=64, out_channels=3):
        super().__init__()
        self.model = nn.Sequential(
            # 输入：融合后的多域特征 (batch, 64, h, w)
            nn.Conv2d(in_channels, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            # 反卷积上采样（适配图像分辨率，添加尺寸匹配处理）
            nn.ConvTranspose2d(128, 64, 4, 2, 1),  # (b,64,2h,2w)
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            # 输出3通道容器图像（-1~1）
            nn.Conv2d(64, out_channels, 3, padding=1),
            nn.Tanh()
        )
        # 残差连接（当输入和输出尺寸相同时）
        self.residual = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x):
        out = self.model(x)
        # 裁剪回原输入尺寸（解决反卷积上采样后的尺寸不匹配）
        if out.shape[2:] != x.shape[2:]:
            out = F.interpolate(out, size=x.shape[2:], mode='bilinear', align_corners=False)
        # 残差连接（增强特征保留）
        if x.shape[2:] == out.shape[2:]:
            out = out + self.residual(x)
            out = torch.tanh(out)  # 保持范围在[-1,1]
        return out


# ===================== 核心网络：EnhancedPRIS（完整整合创新点+特征增强） =====================
class EnhancedPRIS(nn.Module):
    """增强型PRIS隐写模型：整合PZMs特征+可逆块+生成式误差预判（特征增强版）"""

    def __init__(self):
        super(EnhancedPRIS, self).__init__()
        # 子模块初始化（保留原有逻辑，增强特征提取）
        self.pzms = PZMsFeatureExtractor(max_order=c.pzms_max_order)  # PZMs特征提取
        self.reversible = ReversibleBlock(channels=c.target_channels)  # 可逆块（局部特征）
        self.fusion = FeatureFusionModule(
            local_dim=c.target_channels,
            global_dim=20,  # 与PZMs输出维度一致
            output_dim=c.target_channels
        )
        self.gen = DCGANGenerator(  # 容器生成器（保留原有DCGANGenerator类）
            in_channels=c.target_channels,
            out_channels=3
        )
        # 秘密提取分支（可逆块逆向+多尺度卷积，增强提取能力）
        self.extract_conv = nn.Sequential(
            # 多尺度卷积1
            nn.Conv2d(c.target_channels, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            # 多尺度卷积2（空洞卷积）
            nn.Conv2d(64, 32, 3, padding=2, dilation=2),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            # 输出3通道秘密图像
            nn.Conv2d(32, 3, 3, padding=1),
            nn.Tanh()  # 输出-1~1（与输入秘密图像格式一致）
        )
        # 秘密特征增强层（提升提取精度）
        self.secret_enhance = nn.Sequential(
            nn.Conv2d(3, 3, 3, padding=1),
            nn.BatchNorm2d(3),
            nn.LeakyReLU(0.2),
            nn.Conv2d(3, 3, 3, padding=1),
            nn.Tanh()
        )

    def embed(self, host, secret):
        """
        隐写过程：宿主图像 + 秘密图像 → 容器图像（特征增强版）
        Args:
            host: 宿主图像 (batch, 3, h, w)
            secret: 秘密图像 (batch, 3, h, w)
        Returns:
            container: 容器图像 (batch, 3, h, w)
        """
        # 1. 提取宿主和秘密的PZMs全局特征（融合双特征，增强鲁棒性）
        host_global = self.pzms(host)  # (batch, 20)
        secret_global = self.pzms(secret)  # (batch, 20)
        fused_global = (host_global + secret_global) / 2  # 融合全局特征

        # 2. 融合宿主和秘密的局部特征（通过可逆块）
        host_secret = torch.cat([host, secret], dim=1)  # (batch, 6, h, w)
        local_feat = self.reversible(host_secret)  # (batch, 64, h, w)

        # 3. 多域特征融合（全局+局部）
        fused_feat = self.fusion(local_feat, fused_global)  # (batch, 64, h, w)

        # 4. 生成容器图像
        container = self.gen(fused_feat)  # (batch, 3, h, w)

        return container

    def extract(self, container):
        """
        提取过程：容器图像 → 秘密图像（多尺度卷积+增强层，提升精度）
        Args:
            container: 容器图像 (batch, 3, h, w)
        Returns:
            secret: 提取的秘密图像 (batch, 3, h, w)
        """
        # 1. 容器图像通过可逆块逆向操作（提取局部特征）
        local_feat_rev = self.reversible(container, rev=True)  # (batch, 64, h, w)

        # 2. 多尺度卷积提取秘密图像
        secret_extracted = self.extract_conv(local_feat_rev)  # (batch, 3, h, w)

        # 3. 秘密特征增强（提升提取精度）
        secret_enhanced = self.secret_enhance(secret_extracted)  # (batch, 3, h, w)

        return secret_enhanced

    # 可选：保留forward方法（兼容原有调用方式，可选删除）
    def forward(self, x, y=None, rev=False):
        if not rev and y is not None:
            return self.embed(x, y)
        else:
            return self.extract(x)


# ===================== DCGAN判别器（新增get_score/get_features方法，适配trainer） =====================
class Discriminator(nn.Module):
    """DCGAN判别器：区分真实/生成图像，辅助修复（通道数适配+特征提取方法）"""

    def __init__(self, in_channels=3):
        super(Discriminator, self).__init__()
        # 主干网络（保存中间层特征，用于特征匹配）
        self.features = nn.ModuleList([
            # 层1：3→64
            nn.Conv2d(in_channels, 64, 4, 2, 1),
            nn.LeakyReLU(0.2),
            # 层2：64→128
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            # 层3：128→256
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            # 层4：256→1
            nn.Conv2d(256, 1, 4, 1, 0),
            nn.Sigmoid()
        ])
        # 保存中间层输出的钩子
        self.intermediate_features = []

    def forward(self, x):
        """前向传播，返回4维判别结果"""
        out = x
        for layer in self.features:
            out = layer(out)
        return out

    def get_score(self, x):
        """获取1维判别评分（适配BCE损失）"""
        pred = self.forward(x)  # (b,1,h,w)
        score = pred.mean(dim=[1, 2, 3])  # (b,)
        return score

    def get_features(self, x):
        """获取中间层特征（用于特征匹配损失）"""
        features = []
        out = x
        for i, layer in enumerate(self.features[:-2]):  # 取前3层特征
            out = layer(out)
            features.append(out)
        return features

    def get_intermediate_features(self, x):
        """兼容trainer中的旧方法名，获取中间层特征"""
        return self.get_features(x)
