"""
改进与优化后的完整训练脚本（用于散射介质中的可训练鬼成像 + UNet 重建）

主要改动与优化点：
- 将 Spi 类改为 raw_params -> patterns -> (可选二值 STE / sigmoid 约束) -> reshape 为 (M,1,H,W)
  并对每个 pattern 与外部提供的 PSF 做卷积（grouped conv）得到退化后的 pattern；
  用退化后的 pattern 进行测量和线性重构。
- CombinedLoss 中的协方差正则化修正为基向量间 Gram 矩阵正则 (params.T @ params)，这样可鼓励基向量互不冗余。
- SSIMLoss 的常数改为适配 [0,1] 数据范围（C1=(0.01)^2, C2=(0.03)^2）。
- 数据不再一次性搬到 GPU，DataLoader 在 batch 级别移动张量到 device（节省显存）。
- 使用混合精度训练（torch.cuda.amp）以提高速度和降低显存占用（如果 GPU 可用）。
- 添加 patterns 的可视化函数 visualize_degraded_pattern，并在训练开始前保存示例对比图（原始 pattern vs 卷积退化后 pattern）。
- 修复若干细节（避免 next(parameters()) 只取第一个参数、PSF 归一化、数值稳定性、EarlyStopping 目录处理等）。
- 增加 column normalization 的可选项（防止某些 pattern 主导能量）。
- 更合理的参数初始化幅度（raw_params 乘以小常数）。

使用说明：
- 请确保安装依赖：torch, torchvision, numpy, scikit-image, scipy, matplotlib, sklearn, tensorboard
- 将你的 PSF mat 文件路径替换为 psf_path 变量（脚本中已有默认路径）。
- 如果要把 patterns 约束为二值，可在 Spi 初始化时使用 constrain='ste_binary'（会使用 STE 近似梯度）。
- 训练时会把模型与 TensorBoard 日志、检查点文件等保存到指定目录。
"""

import os
import time
import random
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
import argparse

# --------------------------- 默认配置 ---------------------------
# 这些默认值可通过命令行参数覆盖
DEFAULT_CONFIG = {
    'psf_path': './PSF_129_1.0.mat',
    'data_bin_path': './stl10/unlabeled_X.bin',
    'result_dir': './outputs/model',
    'tb_logdir': './outputs/logs',
    'img_size_original': 96,
    'img_size': 128,
    'base_nums': 128,  # M
    'batch_size': 64,
    'num_epochs': 100,
    'lr': 1e-3,
}

# --------------------------- 实用函数 ---------------------------
def load_psf_from_mat(mat_path):
    """加载 .mat 文件，返回 numpy array PSF"""
    mat = sio.loadmat(mat_path)
    keys = [k for k in mat.keys() if not k.startswith('__')]
    if len(keys) == 0:
        raise ValueError("No variable found in mat file")
    if 'PSF' in mat:
        arr = mat['PSF']
    else:
        arr = mat[keys[0]]
    arr = np.asarray(arr, dtype=np.float32)
    # 如果 PSF 是二维矩阵但以 (H,W) 或 (W,H) 存储，直接返回
    return arr

def load_dataset(file_path, img_size_original=96, img_size_resized=128):
    """从二进制 STL-10 文件载入图像，转灰度并 resize 到 img_size_resized"""
    from skimage.transform import resize
    from skimage.color import rgb2gray

    with open(file_path, 'rb') as f:
        data = np.fromfile(f, dtype=np.uint8)
    x = data.reshape(-1, 3, img_size_original, img_size_original).astype(np.float32).transpose(0, 2, 3, 1)
    x_gray = np.array([rgb2gray(img) for img in x], dtype=np.float32)
    x_resized = np.zeros((x_gray.shape[0], img_size_resized, img_size_resized), dtype=np.float32)
    for i in range(x_gray.shape[0]):
        x_resized[i] = resize(x_gray[i], (img_size_resized, img_size_resized), anti_aliasing=True)
    return x_resized

def calculate_psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 255.0
    return 20 * np.log10(max_pixel / np.sqrt(mse))

def calculate_ssim(img1, img2, data_range=255):
    from skimage.metrics import structural_similarity as ssim
    return ssim(img1, img2, data_range=data_range)

# --------------------------- 模型组件 ---------------------------
class Spi(nn.Module):
    """
    可训练的单像素成像模块（params 每一列视为一个 pattern）。
    支持：
    - constrain: 'sigmoid' / 'ste_binary' / 'none'
    - psf_kernel: numpy array 或 torch tensor，卷积核（会归一化）
    - optional: column normalization（在 degraded_flat 上进行）
    """
    def __init__(self, img_size=128, img_pixels=16384, base_nums=1024,
                 psf_kernel=None, constrain='sigmoid', normalize_columns=False,
                 debug_save_dir=None):
        super(Spi, self).__init__()
        self.img_size = img_size
        self.img_pixels = img_pixels
        self.base_nums = base_nums
        # raw params: 较小的初始化幅度，避免数值问题
        self.raw_params = nn.Parameter(torch.randn(img_pixels, base_nums) * 0.01)
        self.constrain = constrain
        self.normalize_columns = normalize_columns
        self.sigmoid = nn.Sigmoid()
        self.debug_save_dir = debug_save_dir

        # PSF 处理
        if psf_kernel is None:
            self.register_buffer('psf', torch.zeros(1,1,1,1))
            self.has_psf = False
        else:
            if isinstance(psf_kernel, np.ndarray):
                psf_t = torch.from_numpy(psf_kernel.astype(np.float32))
            elif isinstance(psf_kernel, torch.Tensor):
                psf_t = psf_kernel.float()
            else:
                raise ValueError("psf_kernel must be numpy.ndarray or torch.Tensor")
            # 归一化并注册 buffer
            psf_t = psf_t / (psf_t.sum() + 1e-12)
            self.register_buffer('psf', psf_t.unsqueeze(0).unsqueeze(0))
            self.has_psf = True

    def _patterns_from_raw(self):
        if self.constrain == 'sigmoid':
            patterns = torch.sigmoid(self.raw_params)
        elif self.constrain == 'ste_binary':
            prob = torch.sigmoid(self.raw_params)
            hard = (prob > 0.5).float()
            patterns = hard.detach() - prob.detach() + prob  # STE
        else:
            patterns = self.raw_params
        return patterns

    def forward(self, x):
        """
        x: [B, H, W] or [B, P]
        输出: [B,1,H,W]
        """
        B = x.shape[0]
        P = self.img_pixels
        M = self.base_nums
        H = W = self.img_size

        x_flat = x.view(-1, P)  # [B, P]
        patterns = self._patterns_from_raw()  # (P, M)

        # reshape (M,1,H,W)
        patterns_imgs = patterns.t().contiguous().view(M, 1, H, W)

        if self.has_psf:
            # grouped conv: 每个 pattern 卷积 psf
            kH, kW = self.psf.shape[-2], self.psf.shape[-1]
            pad_h, pad_w = kH // 2, kW // 2
            weight = self.psf.repeat(M, 1, 1, 1)  # (M,1,kH,kW)
            patterns_grouped = patterns_imgs.permute(1, 0, 2, 3).contiguous()  # (1,M,H,W)
            patterns_padded = F.pad(patterns_grouped, (pad_w, pad_w, pad_h, pad_h), mode='reflect')
            degraded = F.conv2d(patterns_padded, weight, bias=None, stride=1, groups=M)  # (1,M,H,W)
            degraded = degraded.permute(1, 0, 2, 3).contiguous()  # (M,1,H,W)
            # ============ 保存原始 pattern 与退化后 pattern（只保存一次，防止每次 forward 都写文件） ============
            if not hasattr(self, '_saved_patterns_debug') and self.debug_save_dir is not None:
                try:
                    K = min(4, M)  # 保存前 K 个 pattern，可按需修改
                    save_dir = os.path.join(self.debug_save_dir, 'pattern_image')
                    os.makedirs(save_dir, exist_ok=True)
                    # patterns_imgs: (M,1,H,W)  degraded: (M,1,H,W)
                    orig_imgs = patterns_imgs[:K, 0].detach().cpu().numpy()
                    deg_imgs = degraded[:K, 0].detach().cpu().numpy()
                    for idx in range(K):
                        fig, axs = plt.subplots(1, 2, figsize=(6, 3))
                        axs[0].imshow(orig_imgs[idx], cmap='gray', vmin=orig_imgs.min(), vmax=orig_imgs.max())
                        axs[0].set_title(f'orig_{idx}')
                        axs[0].axis('off')
                        axs[1].imshow(deg_imgs[idx], cmap='gray', vmin=deg_imgs.min(), vmax=deg_imgs.max())
                        axs[1].set_title(f'deg_{idx}')
                        axs[1].axis('off')
                        plt.tight_layout()
                        plt.savefig(os.path.join(save_dir, f'pattern_compare_{idx}.png'), dpi=150, bbox_inches='tight')
                        plt.close(fig)
                    print(f"Saved {K} original vs degraded patterns to {save_dir}")
                except Exception as e:
                    print("Failed to save pattern debug images:", e)
                # 标记已保存，避免后续 forward 再次写文件。如果想每次保存/按 epoch 保存，删掉或改写此标志逻辑。
                self._saved_patterns_debug = True
            # =======================================================================================
        else:
            degraded = patterns_imgs

        # 将 degraded reshape 回 (P, M)
        degraded_flat = degraded.view(M, -1).transpose(0, 1).contiguous()  # (P, M)

        # 可选列归一化（L2）
        if self.normalize_columns:
            norms = degraded_flat.norm(dim=0, keepdim=True) + 1e-12
            degraded_flat = degraded_flat / norms

        # 测量与线性重建
        i = torch.matmul(x_flat, degraded_flat)  # [B, M]
        denom = (torch.mean(torch.sum(degraded_flat, 0)) + 1e-12)
        out = (1.0 / M) * torch.matmul(i, degraded_flat.t()) - \
              (torch.mean(i, 1, keepdim=True) / denom) * \
              ((1.0 / M) * torch.matmul(torch.sum(degraded_flat, 0).unsqueeze(0), degraded_flat.t()))
        output = out.view(-1, H, W)
        return output.unsqueeze(1)  # [B,1,H,W]

# --------------------------- UNet 及辅助模块（与原代码类似，略小幅精简） ---------------------------
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv1 = ResidualBlock(in_channels, out_channels)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.conv1(x)
        return self.dropout(x)

class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class AttentionBlock(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(AttentionBlock, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.psi(g1 + x1)
        return x * psi

class DFF(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_atten = nn.Sequential(
            nn.Conv2d(dim * 2, dim * 2, kernel_size=1, bias=False),
            nn.Sigmoid()
        )
        self.conv_redu = nn.Conv2d(dim * 2, dim, kernel_size=1, bias=False)
        self.conv1 = nn.Conv2d(dim, 1, kernel_size=1, stride=1, bias=True)
        self.conv2 = nn.Conv2d(dim, 1, kernel_size=1, stride=1, bias=True)
        self.nonlin = nn.Sigmoid()

    def forward(self, x, skip):
        output = torch.cat([x, skip], dim=1)
        att = self.conv_atten(self.avg_pool(output))
        output = output * att
        output = self.conv_redu(output)
        att2 = self.conv1(x) + self.conv2(skip)
        att2 = self.nonlin(att2)
        output = output * att2
        return output

class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Up, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = DoubleConv(out_channels + in_channels, out_channels)
        self.attention = AttentionBlock(out_channels, out_channels, out_channels // 2)
        self.dff = DFF(out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        x = self.attention(x1, x)
        x = self.dff(x, x1)
        return x

class UNet(nn.Module):
    def __init__(self, n_channels, out_channels=1):
        super(UNet, self).__init__()
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 512)
        self.up1 = Up(512, 256)
        self.up2 = Up(256, 128)
        self.up3 = Up(128, 64)
        self.up4 = Up(64, 32)
        self.outc = nn.Conv2d(32, out_channels, kernel_size=1)
        self.out_channels = out_channels

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        if self.out_channels == 1:
            return x.squeeze(1)
        return x

class SpiUNet(nn.Module):
    def __init__(self, spi_model, unet_model, img_size=128):
        super(SpiUNet, self).__init__()
        self.spi = spi_model
        self.unet = unet_model
        self.sigmoid = nn.Sigmoid()
        self.img_size = img_size

    def forward(self, x):
        x = self.spi(x)                # [B,1,H,W]
        x = x.view(-1, 1, self.img_size, self.img_size)
        x = self.unet(x)               # [B, H, W]
        x = self.sigmoid(x)
        return x

# --------------------------- 损失函数 ---------------------------
class SSIMLoss(nn.Module):
    """适用于输入数据范围为 [0,1] 的 SSIM 损失实现"""
    def __init__(self):
        super(SSIMLoss, self).__init__()

    def forward(self, output, target):
        # 使用 [0,1] 范围的常数
        C1 = (0.01) ** 2
        C2 = (0.03) ** 2
        # 使用 avg_pool2d 计算局部统计量（kernel=11, stride=1）
        mu1 = F.avg_pool2d(output, 11, stride=1, padding=0)
        mu2 = F.avg_pool2d(target, 11, stride=1, padding=0)
        sigma1_sq = F.avg_pool2d(output * output, 11, stride=1, padding=0) - mu1 * mu1
        sigma2_sq = F.avg_pool2d(target * target, 11, stride=1, padding=0) - mu2 * mu2
        sigma12 = F.avg_pool2d(output * target, 11, stride=1, padding=0) - mu1 * mu2
        ssim_map = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / ((mu1 * mu1 + mu2 * mu2 + C1) * (sigma1_sq + sigma2_sq + C2) + 1e-12)
        return 1 - ssim_map.mean()

class CombinedLoss(nn.Module):
    """
    loss = MSE + SSIM + reg * (0.1 * var(diag(Gram)) + mean_off_diag_abs(Gram))
    Gram = patterns.T @ patterns (MxM)
    这里的 patterns 由 net.spi._patterns_from_raw() 生成（在 [0,1] 或 raw 空间的 pattern）
    """
    def __init__(self, reg=0.01):
        super(CombinedLoss, self).__init__()
        self.mse_loss = nn.MSELoss()
        self.ssim_loss = SSIMLoss()
        self.reg = reg

    def forward(self, output, target, net):
        mse = self.mse_loss(output, target)
        ssim = self.ssim_loss(output.unsqueeze(1), target.unsqueeze(1)) if output.dim() == 3 else self.ssim_loss(output, target)
        # 获取 patterns（在当前 net.spi 的设备上）
        patterns = net.spi._patterns_from_raw()  # (P, M)
        # 使用 Gram = patterns.T @ patterns
        Gram = torch.matmul(patterns.t(), patterns)  # (M, M)
        diag = torch.diagonal(Gram, 0)
        # off-diagonal absolute mean
        off = (torch.sum(torch.abs(Gram)) - torch.sum(torch.abs(diag))) / (Gram.numel() - diag.numel() + 1e-12)
        diag_var = torch.var(diag)
        reg_term = self.reg * (0.1 * diag_var + off)
        return mse + ssim + reg_term

# --------------------------- EarlyStopping ---------------------------
class EarlyStopping:
    def __init__(self, patience=15, verbose=False, delta=0, path=None, trace_func=print):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta
        self.path = path or os.path.join(result_dir, 'best_model.pth')
        self.trace_func = trace_func
        d = os.path.dirname(self.path)
        if d:
            os.makedirs(d, exist_ok=True)

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

    def load_best_weights(self, model, device=None):
        if device is None:
            # Try to get device from model parameters, fallback to CPU
            try:
                device = next(model.parameters()).device
            except StopIteration:
                device = torch.device('cpu')
        model.load_state_dict(torch.load(self.path, map_location=device))
        if self.verbose:
            self.trace_func(f'Loaded best model weights from {self.path}')

# --------------------------- 可视化函数 ---------------------------
def visualize_degraded_pattern(spi_model, psf_array=None, idx=None, save_path='pattern_degraded_compare.png', 
                               writer=None, tb_step=0, target_device=None):
    """
    在 CPU 上把 spi_model 当前 raw_params 的第 idx 列（或随机）可视化：原始 pattern vs degraded pattern。
    """
    if target_device is None:
        target_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    spi_model_cpu = spi_model.cpu()
    raw = spi_model_cpu.raw_params.detach().cpu()
    H = spi_model_cpu.img_size
    P, M = raw.shape
    if idx is None:
        idx = random.randint(0, M-1)
    if spi_model_cpu.constrain == 'sigmoid':
        patterns = torch.sigmoid(raw)
    elif spi_model_cpu.constrain == 'ste_binary':
        prob = torch.sigmoid(raw)
        hard = (prob > 0.5).float()
        patterns = hard.detach() - prob.detach() + prob
    else:
        patterns = raw
    pattern_orig = patterns[:, idx].view(H, H).numpy()
    # 归一化显示
    pmin, pmax = pattern_orig.min(), pattern_orig.max()
    if pmax - pmin > 1e-8:
        pattern_orig_vis = (pattern_orig - pmin) / (pmax - pmin)
    else:
        pattern_orig_vis = pattern_orig - pmin

    # PSF
    if spi_model_cpu.has_psf:
        psf_t = spi_model_cpu.psf.detach().cpu().numpy().squeeze()
    else:
        psf_t = psf_array

    pattern_img = torch.from_numpy(pattern_orig_vis.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    if psf_t is not None:
        psf_t = psf_t / (psf_t.sum() + 1e-12)
        kH, kW = psf_t.shape
        pad_h, pad_w = kH // 2, kW // 2
        weight = torch.from_numpy(psf_t).unsqueeze(0).unsqueeze(0).float()
        pattern_padded = F.pad(pattern_img, (pad_w, pad_w, pad_h, pad_h), mode='reflect')
        degraded = F.conv2d(pattern_padded, weight, bias=None, stride=1)
        degraded = degraded.squeeze().numpy()
        dmin, dmax = degraded.min(), degraded.max()
        if dmax - dmin > 1e-8:
            degraded_vis = (degraded - dmin) / (dmax - dmin)
        else:
            degraded_vis = degraded - dmin
    else:
        degraded_vis = pattern_orig_vis.copy()

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(pattern_orig_vis, cmap='gray', vmin=0, vmax=1)
    axes[0].set_title(f'Original pattern #{idx}')
    axes[0].axis('off')
    axes[1].imshow(degraded_vis, cmap='gray', vmin=0, vmax=1)
    axes[1].set_title('Degraded (convolved)')
    axes[1].axis('off')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)

    if writer is not None:
        orig_chw = pattern_orig_vis[np.newaxis, :, :]
        deg_chw = degraded_vis[np.newaxis, :, :]
        combined = np.concatenate([orig_chw, deg_chw], axis=2)  # side-by-side
        writer.add_image('Pattern/Orig_vs_Degraded', combined, tb_step, dataformats='CHW')
    print(f"Saved pattern compare image to: {save_path} (idx {idx})")
    spi_model.to(target_device)  # 把模型移回训练设备

# --------------------------- 主训练流程 ---------------------------
def parse_args():
    parser = argparse.ArgumentParser(description='Ghost Imaging with UNet Reconstruction Training Script')
    parser.add_argument('--psf_path', type=str, default=DEFAULT_CONFIG['psf_path'],
                        help='Path to PSF .mat file')
    parser.add_argument('--data_bin_path', type=str, default=DEFAULT_CONFIG['data_bin_path'],
                        help='Path to STL-10 binary data file')
    parser.add_argument('--result_dir', type=str, default=DEFAULT_CONFIG['result_dir'],
                        help='Directory to save model checkpoints')
    parser.add_argument('--tb_logdir', type=str, default=DEFAULT_CONFIG['tb_logdir'],
                        help='TensorBoard log directory')
    parser.add_argument('--img_size_original', type=int, default=DEFAULT_CONFIG['img_size_original'],
                        help='Original image size (STL-10 default is 96)')
    parser.add_argument('--img_size', type=int, default=DEFAULT_CONFIG['img_size'],
                        help='Target image size after resize')
    parser.add_argument('--base_nums', type=int, default=DEFAULT_CONFIG['base_nums'],
                        help='Number of patterns (M)')
    parser.add_argument('--batch_size', type=int, default=DEFAULT_CONFIG['batch_size'],
                        help='Training batch size')
    parser.add_argument('--num_epochs', type=int, default=DEFAULT_CONFIG['num_epochs'],
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=DEFAULT_CONFIG['lr'],
                        help='Learning rate')
    parser.add_argument('--constrain', type=str, default='sigmoid', choices=['sigmoid', 'ste_binary', 'none'],
                        help='Pattern constraint type')
    parser.add_argument('--no_amp', action='store_true',
                        help='Disable mixed precision training')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Configuration from arguments
    psf_path = args.psf_path
    data_bin_path = args.data_bin_path
    result_dir = args.result_dir
    tb_logdir = args.tb_logdir
    img_size_original = args.img_size_original
    img_size = args.img_size
    img_pixels = img_size * img_size
    base_nums = args.base_nums
    batch_size = args.batch_size
    num_epochs = args.num_epochs
    lr = args.lr
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = torch.cuda.is_available() and not args.no_amp
    
    print("Device:", device, "AMP:", use_amp)
    
    # Create directories
    os.makedirs(result_dir, exist_ok=True)
    os.makedirs(tb_logdir, exist_ok=True)
    
    # TensorBoard writer
    writer = SummaryWriter(log_dir=tb_logdir)
    
    # 加载数据（在 CPU 上）
    print("Loading dataset ...")
    x_all = load_dataset(data_bin_path, img_size_original=img_size_original, img_size_resized=img_size)
    x_train, x_test = x_all[10000:], x_all[:10000]
    print(f"Train {x_train.shape[0]}, Test {x_test.shape[0]}")

    # MinMax 缩放到 [0,1]（在 CPU 上）
    scaler = MinMaxScaler(feature_range=(0, 1))
    x_train_flat = x_train.reshape(-1, img_pixels)
    x_test_flat = x_test.reshape(-1, img_pixels)
    x_train_scaled = scaler.fit_transform(x_train_flat).reshape(-1, img_size, img_size).astype(np.float32)
    x_test_scaled = scaler.transform(x_test_flat).reshape(-1, img_size, img_size).astype(np.float32)

    # 转为 TensorDataset（保留在 CPU，批内再移动到 device）
    x_train_t = torch.from_numpy(x_train_scaled)
    x_test_t = torch.from_numpy(x_test_scaled)
    train_set = TensorDataset(x_train_t, x_train_t)
    test_set = TensorDataset(x_test_t, x_test_t)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # 加载 PSF
    print("Loading PSF ...")
    psf_arr = load_psf_from_mat(psf_path)
    print("PSF shape:", psf_arr.shape)

    # 初始化模型
    spi_model = Spi(img_size=img_size, img_pixels=img_pixels, base_nums=base_nums,
                    psf_kernel=psf_arr, constrain=args.constrain, normalize_columns=True,
                    debug_save_dir=result_dir).to(device)
    unet_model = UNet(n_channels=1, out_channels=1).to(device)
    spi = SpiUNet(spi_model, unet_model, img_size=img_size).to(device)

    # 可视化一个 pattern 的原始 vs degraded（写到文件和 TensorBoard）
    visualize_degraded_pattern(spi_model, psf_array=psf_arr, idx=None,
                               save_path=os.path.join(result_dir, 'pattern_compare_initial.png'),
                               writer=writer, tb_step=0, target_device=device)

    # 损失、优化器、scheduler、early stopping
    loss_fn = CombinedLoss(reg=0.01)
    optimizer = torch.optim.Adam(spi.parameters(), lr=lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-4)
    early_stopping = EarlyStopping(patience=10, verbose=True,
                                   delta=0.001,
                                   path=os.path.join(result_dir, 'best_spi_model_optimized.pth'),
                                   trace_func=print)

    scaler_amp = torch.cuda.amp.GradScaler(enabled=use_amp)

    train_losses = []
    test_losses = []
    best_val_loss = float('inf')
    start_time = time.time()

    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        spi.train()
        running_loss = 0.0
        pbar = tqdm(train_loader, desc='Train', leave=False)
        for imgs_cpu, targets_cpu in pbar:
            imgs = imgs_cpu.to(device, non_blocking=True)
            targets = targets_cpu.to(device, non_blocking=True)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=use_amp):
                outputs = spi(imgs)         # [B,1,H,W]
                outputs = outputs.view(-1, img_size, img_size)  # [B,H,W]
                loss = loss_fn(outputs, targets, spi)
            scaler_amp.scale(loss).backward()
            scaler_amp.step(optimizer)
            scaler_amp.update()
            running_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})

        scheduler.step()
        train_loss = running_loss / len(train_loader)
        train_losses.append(train_loss)
        writer.add_scalar('Loss/train', train_loss, epoch)
        torch.save(spi.state_dict(), os.path.join(result_dir, f'spi_epoch_{epoch+1}.pth'))

        # 验证
        spi.eval()
        total_test_loss = 0.0
        with torch.no_grad():
            for imgs_cpu, targets_cpu in tqdm(test_loader, desc='Test', leave=False):
                imgs = imgs_cpu.to(device, non_blocking=True)
                targets = targets_cpu.to(device, non_blocking=True)
                with torch.cuda.amp.autocast(enabled=use_amp):
                    outputs = spi(imgs)
                    outputs = outputs.view(-1, img_size, img_size)
                    loss_val = loss_fn(outputs, targets, spi).item()
                total_test_loss += loss_val
        total_test_loss /= len(test_loader)
        test_losses.append(total_test_loss)
        writer.add_scalar('Loss/test', total_test_loss, epoch)
        print(f"Train loss: {train_loss:.6f}  Test loss: {total_test_loss:.6f}")

        # 保存最佳
        if total_test_loss < best_val_loss:
            best_val_loss = total_test_loss
            torch.save(spi.state_dict(), os.path.join(result_dir, 'best_so_far.pth'))

        # EarlyStopping
        early_stopping(total_test_loss, spi)
        if early_stopping.early_stop:
            print("Early stopping triggered. Loading best weights.")
            early_stopping.load_best_weights(spi, device=device)
            break

        # 记录一个测试样本的可视化和度量
        example_img_cpu, example_target_cpu = test_set[19]
        example_img = example_img_cpu.unsqueeze(0).to(device)
        example_target = example_target_cpu.to(device)
        spi.eval()
        with torch.no_grad():
            out_example = spi(example_img).squeeze(0).view(img_size, img_size).cpu().numpy()
        target_np = example_target.cpu().numpy()
        # 写入 TensorBoard（CHW）
        writer.add_image('Generated_Image', np.expand_dims(out_example, axis=0), epoch, dataformats='CHW')
        writer.add_image('Target_Image', np.expand_dims(target_np, axis=0), epoch, dataformats='CHW')
        # 计算 PSNR/SSIM（uint8）
        out_uint8 = (np.clip(out_example, 0, 1) * 255).astype(np.uint8)
        tgt_uint8 = (np.clip(target_np, 0, 1) * 255).astype(np.uint8)
        psnr_val = calculate_psnr(out_uint8, tgt_uint8)
        try:
            ssim_val = calculate_ssim(out_uint8, tgt_uint8, data_range=255)
        except Exception:
            ssim_val = 0.0
        writer.add_scalar('Metric/PSNR', psnr_val, epoch)
        writer.add_scalar('Metric/SSIM', ssim_val, epoch)

        # 每隔若干 epoch 写入 pattern 可视化（退化前后）
        if epoch % 5 == 0:
            vis_path = os.path.join(result_dir, f'pattern_compare_epoch_{epoch+1}.png')
            visualize_degraded_pattern(spi.spi, psf_array=psf_arr, idx=None, save_path=vis_path, 
                                      writer=writer, tb_step=epoch, target_device=device)

    # 保存 losses
    import csv
    losses_path = os.path.join(result_dir, 'losses.csv')
    with open(losses_path, 'w', newline='') as f:
        writer_csv = csv.writer(f)
        writer_csv.writerow(['train_loss', 'test_loss'])
        for train_l, test_l in zip(train_losses, test_losses):
            writer_csv.writerow([train_l, test_l])

    end_time = time.time()
    print(f"Training finished. Best val loss: {best_val_loss:.6f}")
    print(f"Total time: {end_time - start_time:.2f} s")
    writer.close()

if __name__ == "__main__":
    main()
