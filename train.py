# train.py
# 训练脚本（已更新为使用 PaperRecon 网络）
# 使用说明：
#   python train.py --A_path measurement/A_409x4096.npz --data_root ./data --out_dir outputs \
#       --image_size 64 --base_channels 32 --n_resblocks 18 --batch_size 64 --epochs 80

import os
import argparse
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from dataset import CompressiveGhostDataset
from model import get_model
from utils import compute_psnr, compute_ssim, save_comparison_grid
import numpy as np

def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # 加载测量矩阵 A
    A_path = args.A_path
    data_root = args.data_root

    # dataset
    train_ds = CompressiveGhostDataset(root=data_root, split="train",
                                      A_path=A_path, image_size=args.image_size,
                                      noise_sigma=args.noise_sigma)
    val_ds = CompressiveGhostDataset(root=data_root, split="test",
                                    A_path=A_path, image_size=args.image_size,
                                    noise_sigma=args.noise_sigma)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)

    M = train_ds.A.shape[0]
    # 使用你提供的 PaperRecon 网络实现（final_size=64）
    model = get_model(M, final_size=args.image_size, device=device,
                      out_channels=1, base_channels=args.base_channels, n_resblocks=args.n_resblocks)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=30, gamma=0.5)
    criterion = torch.nn.MSELoss()

    best_val_loss = 1e9
    os.makedirs(args.out_dir, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        for i, (y, x) in enumerate(train_loader):
            y = y.to(device)
            x = x.to(device)
            optimizer.zero_grad()
            pred = model(y)
            loss = criterion(pred, x)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * y.size(0)
        epoch_loss = running_loss / len(train_loader.dataset)

        # 验证
        model.eval()
        val_loss = 0.0
        psnr_list = []
        ssim_list = []
        with torch.no_grad():
            for j, (y, x) in enumerate(val_loader):
                y = y.to(device)
                x = x.to(device)
                pred = model(y)
                loss = criterion(pred, x)
                val_loss += loss.item() * y.size(0)
                # 计算指标 (逐项)
                pred_np = pred.cpu().numpy()
                x_np = x.cpu().numpy()
                for k in range(pred_np.shape[0]):
                    p = pred_np[k,0]
                    t = x_np[k,0]
                    psnr_list.append(compute_psnr(p, t))
                    ssim_list.append(compute_ssim(p, t))
        val_loss = val_loss / len(val_loader.dataset)
        mean_psnr = float(np.mean(psnr_list)) if len(psnr_list)>0 else 0.0
        mean_ssim = float(np.mean(ssim_list)) if len(ssim_list)>0 else 0.0

        print(f"Epoch {epoch:03d} TrainLoss {epoch_loss:.6f} ValLoss {val_loss:.6f} PSNR {mean_psnr:.3f} SSIM {mean_ssim:.4f}")

        # 保存 checkpoint
        ckpt_path = os.path.join(args.out_dir, f"model_epoch{epoch:03d}.pt")
        torch.save({
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "A_path": A_path,
            "image_size": args.image_size,
        }, ckpt_path)
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(args.out_dir, "best_model.pt"))

        scheduler.step()

        # 每几个 epoch 保存示例对比图
        if epoch % args.save_every == 0:
            # 获取一个 batch 做展示
            y_show, x_show = next(iter(val_loader))
            model.eval()
            with torch.no_grad():
                pred_show = model(y_show.to(device)).cpu()
            save_comparison_grid(pred_show, x_show, os.path.join(args.out_dir, f"comparison_epoch{epoch:03d}.png"), nrow=min(8, x_show.shape[0]//2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="./data", help="STL-10 data root")
    parser.add_argument("--A_path", type=str, default="measurement/A_409x4096.npz", help="measurement matrix path")
    parser.add_argument("--image_size", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--noise_sigma", type=float, default=0.0)
    parser.add_argument("--out_dir", type=str, default="outputs")
    parser.add_argument("--save_every", type=int, default=5)
    parser.add_argument("--base_channels", type=int, default=32, help="base feature channels used in PaperRecon FC output")
    parser.add_argument("--n_resblocks", type=int, default=18, help="number of residual blocks (default 18 as paper)")
    args = parser.parse_args()
    train(args)