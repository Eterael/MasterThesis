#!/usr/bin/env python3
"""
Train an autoencoder on peptide images, extract encoder features for train/test,
and generate saliency heatmaps + reconstructions to see which input regions
drive reconstruction.

Outputs:
- <ALLELE>/folds/28tot_fold_<fold>_train.pt      (encoder features/labels/names)
- <ALLELE>/folds/28tot_fold_<fold>_test.pt       (encoder features/labels/names)
- <ALLELE>/results_cl/models/autoencoder.pt      (model state_dict)
- <ALLELE>/results_cl/recon/*.png                (decoder reconstructions)
- <ALLELE>/results_cl/saliency/*_overlay.png     (saliency overlaid on input)
- <ALLELE>/results_cl/saliency/*_saliency_gray.png (raw saliency map)
"""

import os
import argparse
import random
import copy
import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

from torchvision import transforms
from torchvision.utils import save_image

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt


# ----------------------------
# Args
# ----------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--fold", type=str, default="pn", help="cluster/fold identifier")
parser.add_argument("--test", action="store_true",
                    help="activate test mode with corresponding data")

# Visualization options
parser.add_argument("--n_viz", type=int, default=64,
                    help="Number of examples to visualize")
parser.add_argument("--viz_split", type=str, choices=["train", "test"], default="train",
                    help="Which split to visualize")
parser.add_argument("--smoothgrad", action="store_true",
                    help="Use SmoothGrad for saliency")
parser.add_argument("--smoothgrad_samples", type=int, default=25,
                    help="Number of noisy samples for SmoothGrad")
parser.add_argument("--smoothgrad_noise", type=float, default=0.1,
                    help="Std of noise as fraction of pixel range (0-1)")

# Training options
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--epochs", type=int, default=10)
parser.add_argument("--lr", type=float, default=5e-4)
parser.add_argument("--seed", type=int, default=49)
parser.add_argument("--num_workers", type=int, default=1)

args = parser.parse_args()


# ----------------------------
# Environment / Paths
# ----------------------------
allele = os.environ.get("ALLELE")
if allele is None:
    raise ValueError("Environment variable ALLELE not set!")

depictions_dir = '../data/28struc/'  # not used below but kept from original
indir = '../data/28pep_seq/'         # directory containing <seq>.png

pn_ts = allele.replace("_normal", "")

if args.test:
    train_file = f'../data/test/folds/fold_{args.fold}/train.txt'
    test_file  = f'../data/test/folds/fold_{args.fold}/test.txt'
else:
    train_file = f"pn_split/{pn_ts}_normal.txt"
    test_file  = f"pn_split/{pn_ts}_phospho.txt"

allele_root = os.path.join(allele)
folds_dir   = os.path.join(allele_root, "folds")
results_dir = os.path.join(allele_root, "results_cl")
models_dir  = os.path.join(results_dir, "models")

os.makedirs(allele_root, exist_ok=True)
os.makedirs(folds_dir,   exist_ok=True)
os.makedirs(results_dir, exist_ok=True)
os.makedirs(models_dir,  exist_ok=True)

# feature output paths
train_feat_out = f"{folds_dir}/28tot_fold_{args.fold}_train.pt"
test_feat_out  = f"{folds_dir}/28tot_fold_{args.fold}_test.pt"

# visualization output dirs
split_tag   = args.viz_split  # "train" or "test"
saliency_dir = os.path.join(results_dir, "saliency", split_tag)
recon_dir    = os.path.join(results_dir, "recon", split_tag)
os.makedirs(saliency_dir, exist_ok=True)
os.makedirs(recon_dir,    exist_ok=True)


# ----------------------------
# Repro
# ----------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
set_seed(args.seed)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ----------------------------
# Dataset
# ----------------------------
class ProteinImageDataset(Dataset):
    """
    If label_map is provided, returns (image, label, seq);
    otherwise returns (image, seq).
    """
    def __init__(self, seq_list, indir, transform=None, label_map=None):
        self.seq_list = seq_list
        self.indir = indir
        self.transform = transform
        self.label_map = label_map

    def __len__(self):
        return len(self.seq_list)

    def __getitem__(self, idx):
        seq = self.seq_list[idx]
        img_path = os.path.join(self.indir, f"{seq}.png")
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        if self.label_map is None:
            return image, seq
        else:
            label = self.label_map[seq]
            return image, label, seq


# ----------------------------
# Data
# ----------------------------
transform = transforms.ToTensor()

# training list (no labels needed to train AE)
df_train = pd.read_csv(train_file, sep=" ", header=None, names=["seq", "label", "origin"])
train_seqs = df_train["seq"].tolist()
del df_train

train_dataset = ProteinImageDataset(train_seqs, indir, transform=transform, label_map=None)
train_loader  = DataLoader(train_dataset, batch_size=args.batch_size,
                           shuffle=False, num_workers=args.num_workers, pin_memory=True)

print(f'number of training samples: {len(train_dataset)}')
for images, _ in train_loader:
    print('Image batch dimensions:', images.size())
    break


# ----------------------------
# Model
# ----------------------------
class Reshape(nn.Module):
    def __init__(self, *shape):
        super().__init__()
        self.shape = shape
    def forward(self, x):
        return x.view(self.shape)

class Trim(nn.Module):
    def __init__(self, target_height, target_width):
        super().__init__()
        self.target_height = target_height
        self.target_width = target_width
    def forward(self, x):
        return x[:, :, :self.target_height, :self.target_width]

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

class AutoEncoderV2(nn.Module):
    def __init__(self):
        super(AutoEncoderV2, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2),
            SEBlock(16),

            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2),
            SEBlock(32),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2),
            SEBlock(64),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2),
            SEBlock(128),

            nn.Flatten(),
            nn.Linear(128 * 1 * 15, 256)  # Bottleneck
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(256, 128 * 1 * 15),
            Reshape(-1, 128, 1, 15),

            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1),

            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1),

            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.1),

            nn.ConvTranspose2d(16, 3, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
            
            nn.Upsample(size=(28, 252), mode='bilinear', align_corners=False),
            Trim(28, 252)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

model = AutoEncoderV2().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
loss_fn = nn.MSELoss()


# ----------------------------
# Train
# ----------------------------
def compute_epoch_loss_autoencoder(model, dataloader, loss_fn):
    loss = 0.0
    predictions = []
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            if len(batch) == 2:
                features, _seq = batch
            else:
                features, _lbl, _seq = batch
            features = features.to(device)
            logits = model(features)
            loss += loss_fn(logits, features)
            predictions.append(logits.detach())
    return loss / len(dataloader), predictions

def train_model(num_epochs, model, optimizer, train_loader, loss_fn,
                logging_interval=250, save_model_path=None):
    log_dict = {'train_loss_per_batch': [], 'train_loss_per_epoch': []}
    for epoch in range(num_epochs):
        model.train()
        for batch_idx, batch in enumerate(train_loader):
            if len(batch) == 2:
                features, _seq = batch
            else:
                features, _lbl, _seq = batch
            features = features.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(features)
            loss = loss_fn(logits, features)
            loss.backward()
            optimizer.step()

            log_dict['train_loss_per_batch'].append(loss.item())

            if not batch_idx % logging_interval:
                print('Epoch: %03d/%03d | Batch %04d/%04d | Loss: %.4f'
                      % (epoch+1, num_epochs, batch_idx, len(train_loader), loss.item()))

        # (optional) per-epoch stats; skipped originally
        # train_loss, _ = compute_epoch_loss_autoencoder(model, train_loader, loss_fn)
        # log_dict['train_loss_per_epoch'].append(train_loss.item())
        # print('***Epoch: %03d/%03d | Loss: %.3f' % (epoch+1, num_epochs, train_loss))

    if save_model_path is not None:
        torch.save(model.state_dict(), save_model_path)
    return log_dict

model_ckpt_path = os.path.join(models_dir, "autoencoder.pt")
_ = train_model(args.epochs, model, optimizer, train_loader, loss_fn,
                logging_interval=250, save_model_path=model_ckpt_path)


# ----------------------------
# Feature extraction (encoder)
# ----------------------------
def extract_features(seq_file, indir, transform, encoder, batch_size):
    df = pd.read_csv(seq_file, sep=" ", header=None, names=["seq", "label", "origin"])
    df["label"] = df["label"].astype(int)
    peptides = df["seq"].tolist()
    label_map = dict(zip(df["seq"], df["label"]))
    ds = ProteinImageDataset(peptides, indir, transform=transform, label_map=label_map)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=args.num_workers)
    feats, labels, names = [], [], []
    encoder.eval()
    encoder.to("cpu")
    with torch.no_grad():
        for imgs, lbls, ids in loader:
            encoded = encoder(imgs)
            feats.append(encoded)
            labels.append(lbls)
            names.extend(ids)
    feats = torch.cat(feats)
    labels = torch.cat(labels)
    return feats, labels, names

# move to CPU for feature extraction consistency
model.eval()
model.to("cpu")
encoder = model.encoder

X_train, y_train, names_train = extract_features(train_file, indir, transform, encoder, args.batch_size)
torch.save({'features': X_train, 'labels': y_train, 'peptides': names_train}, train_feat_out)
del X_train, y_train, names_train

X_test, y_test, names_test = extract_features(test_file, indir, transform, encoder, args.batch_size)
torch.save({'features': X_test, 'labels': y_test, 'peptides': names_test}, test_feat_out)
del X_test, y_test, names_test


# ----------------------------
# Saliency & Recon Visualization
# ----------------------------
def to_numpy_img(t):
    """
    t: (3, H, W), values in [0,1]
    returns: (H, W, 3) float32 in [0,1]
    """
    t = t.detach().cpu().clamp(0, 1)
    return t.permute(1, 2, 0).numpy()

def normalize_01(x):
    x = x - x.min()
    denom = (x.max() + 1e-8)
    return x / denom

def save_overlay(input_img_t, heat_t, out_path, cmap="jet", alpha=0.45):
    """Save original image with heatmap overlay."""
    img_np  = to_numpy_img(input_img_t)  # (H,W,3) in [0,1]
    heat_np = heat_t.squeeze().detach().cpu().numpy()  # (H,W)
    heat_np = normalize_01(heat_np)

    plt.figure(figsize=(6, 2.5))  # wide aspect for (28 x 252)
    plt.axis("off")
    plt.imshow(img_np)
    plt.imshow(heat_np, cmap=cmap, alpha=alpha)
    plt.tight_layout(pad=0)
    plt.savefig(out_path, dpi=200, bbox_inches="tight", pad_inches=0)
    plt.close()

def compute_saliency_single(model, img, loss_fn=F.mse_loss,
                            smoothgrad=False, sg_samples=25, sg_noise=0.1):
    """
    img: (1, 3, H, W) in [0,1]
    returns: dict with 'saliency' (1,H,W) and 'recon' (3,H,W)
    """
    model.eval()
    img = img.clone().detach().requires_grad_(True)

    if not smoothgrad:
        recon = model(img)
        loss  = loss_fn(recon, img)
        model.zero_grad(set_to_none=True)
        loss.backward()
        grad = img.grad.detach()                 # (1,3,H,W)
        sal  = grad.abs().mean(dim=1, keepdim=True)  # (1,1,H,W)
        return {"saliency": sal[0], "recon": recon.detach()[0]}

    # SmoothGrad
    grads = []
    recon = None
    for _ in range(sg_samples):
        noisy = img + sg_noise * torch.randn_like(img)
        noisy = noisy.clamp(0, 1).requires_grad_(True)
        out   = model(noisy)
        loss  = loss_fn(out, noisy)
        model.zero_grad(set_to_none=True)
        loss.backward()
        grads.append(noisy.grad.detach())
        if recon is None:
            recon = out.detach()
    grads = torch.stack(grads, dim=0)  # (S,1,3,H,W)
    sal   = grads.abs().mean(dim=0).mean(dim=1, keepdim=True)  # (1,1,H,W)
    return {"saliency": sal[0], "recon": recon[0]}

def make_viz_dataset(split="train"):
    df_v = pd.read_csv(train_file if split=="train" else test_file,
                       sep=" ", header=None, names=["seq", "label", "origin"])
    seqs = df_v["seq"].tolist()
    ds   = ProteinImageDataset(seqs, indir, transform=transform, label_map=None)
    return ds

# Use CPU for input-gradient autograd
model.eval()
model.to("cpu")

viz_ds = make_viz_dataset(args.viz_split)
viz_loader = DataLoader(viz_ds, batch_size=1, shuffle=False, num_workers=0)

num_done = 0
for batch in viz_loader:
    # dataset returns (image, seq) here
    if len(batch) == 2:
        img, seq = batch
    else:
        img, _lbl, seq = batch

    res = compute_saliency_single(
        model,
        img,  # (1,3,H,W)
        loss_fn=F.mse_loss,
        smoothgrad=args.smoothgrad,
        sg_samples=args.smoothgrad_samples,
        sg_noise=args.smoothgrad_noise
    )

    sal   = res["saliency"]   # (1,H,W)
    recon = res["recon"]      # (3,H,W)
    seq_id = seq[0] if isinstance(seq, (list, tuple)) else seq

    # Save reconstruction
    recon_path = os.path.join(recon_dir, f"{seq_id}_recon.png")
    save_image(recon, recon_path)

    # Save heatmap (grayscale)
    heat_gray_path = os.path.join(saliency_dir, f"{seq_id}_saliency_gray.png")
    plt.figure(figsize=(6, 2.5))
    plt.axis("off")
    plt.imshow(sal.squeeze().cpu().numpy(), cmap="gray")
    plt.tight_layout(pad=0)
    plt.savefig(heat_gray_path, dpi=200, bbox_inches="tight", pad_inches=0)
    plt.close()

    # Save overlay
    overlay_path = os.path.join(saliency_dir, f"{seq_id}_overlay.png")
    save_overlay(img[0], sal, overlay_path, cmap="jet", alpha=0.45)

    num_done += 1
    if num_done >= args.n_viz:
        break

print(f"[viz] Saved {num_done} reconstructions to: {recon_dir}")
print(f"[viz] Saved {num_done} saliency maps and overlays to: {saliency_dir}")
print(f"[model] Saved AE state_dict to: {model_ckpt_path}")
print(f"[features] Train features: {train_feat_out}")
print(f"[features] Test  features: {test_feat_out}")
