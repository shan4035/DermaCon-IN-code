import os
from collections import Counter

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

import torch
torch.seed = 42
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.metrics import precision_score, recall_score, f1_score
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm

from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix
)
import torch.nn.functional as F
import numpy as np

from torchvision import transforms
from timm import create_model


# ─── Config & Paths ────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_csv = "./METADATA/train_split.csv"
val_csv = "./METADATA/test_split.csv"
image_folder = "./DATASET"


# ─── Label Mapping & Sampler ──────────────────────────────────────────────────
train_df = pd.read_csv(train_csv)
unique_labels = sorted(train_df['Main_class'].dropna().unique().tolist())
label_to_idx = {lbl: i for i, lbl in enumerate(unique_labels)}

class_counts = train_df['Main_class'].value_counts()
sample_weights = 1.0 / class_counts.loc[train_df['Main_class']].values
sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)


# ─── Transforms ───────────────────────────────────────────────────────────────
mean = [0.53749797, 0.45875554, 0.40382471]
std = [0.21629889, 0.20366619, 0.20136241]

train_transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])

val_transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])

def pad_collate_fn(batch):
    imgs, labs = zip(*batch)
    return torch.stack(imgs), torch.tensor(labs, dtype=torch.long)


class ImageDataset(Dataset):
    def __init__(self, csv_file, image_folder, transform=None):
        self.df = pd.read_csv(csv_file)
        self.folder = image_folder
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.folder, row['Image_name'])
        label_idx = label_to_idx[row['Main_class']]

        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label_idx

train_ds = ImageDataset(train_csv, image_folder, transform=train_transform)
val_ds = ImageDataset(val_csv, image_folder, transform=val_transform)

train_loader = DataLoader(train_ds, batch_size=84, sampler=sampler, num_workers=4, collate_fn=pad_collate_fn)
val_loader = DataLoader(val_ds, batch_size=84, shuffle=False, num_workers=4, collate_fn=pad_collate_fn)


# ─── Training Function for Swin Transformer ────────────────────────────────────
def train_swin(train_loader, val_loader, num_classes, save_dir='./checkpoints/swin', num_epochs=100, lr=1e-4, device='cuda', accumulation_steps=1):
    os.makedirs(save_dir, exist_ok=True)

    model = create_model(
        'swin_base_patch4_window12_384',
        pretrained=True,
        num_classes=num_classes,
        img_size=512 
    )

    model = model.to(device)
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs.")
        model = nn.DataParallel(model)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30, eta_min=1e-6)

    best_val_acc = 0.0

    for epoch in range(1, num_epochs + 1):
        print(f"\n=== Epoch {epoch}/{num_epochs} ===")
        model.train()
        optimizer.zero_grad()

        running_loss, correct, total = 0.0, 0, 0
        for i, (imgs, labs) in enumerate(tqdm(train_loader, desc="Train")):
            imgs, labs = imgs.to(device), labs.to(device)
            outs = model(imgs)
            loss = criterion(outs, labs) / accumulation_steps
            loss.backward()

            if (i + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            bs = labs.size(0)
            running_loss += loss.item() * accumulation_steps * bs
            correct += (outs.argmax(1) == labs).sum().item()
            total += bs

        scheduler.step()
        train_acc = correct / total
        train_loss = running_loss / total
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")

        model.eval()
        val_loss, correct, total = 0.0, 0, 0
        all_preds, all_tgts = [], []

        # Store softmax probabilities
        all_preds = []
        all_probs = []
        all_tgts = []

        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for imgs, labs in tqdm(val_loader, desc="Validate"):
                imgs, labs = imgs.to(device), labs.to(device)
                outs = model(imgs)
                loss = criterion(outs, labs)

                val_loss += loss.item() * labs.size(0)
                probs = F.softmax(outs, dim=1)

                pred_labels = probs.argmax(dim=1)
                correct += (pred_labels == labs).sum().item()
                total += labs.size(0)

                all_preds.extend(pred_labels.cpu().tolist())
                all_probs.extend(probs.cpu().tolist())
                all_tgts.extend(labs.cpu().tolist())

        # Overall metrics
        val_acc = correct / total
        val_loss = val_loss / total
        precision = precision_score(all_tgts, all_preds, average='weighted', zero_division=0)
        recall = recall_score(all_tgts, all_preds, average='weighted', zero_division=0)
        f1 = f1_score(all_tgts, all_preds, average='weighted', zero_division=0)
        # Compute AUC
        try:
            auc = roc_auc_score(all_tgts, all_probs, multi_class='ovr')
        except ValueError:
            auc = 0.0  # fallback if only one class is present

        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Prec: {precision:.4f}, Rec: {recall:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}")

        # Compute per-class AUC
        y_true = np.array(all_tgts)
        y_score = np.array(all_probs)
        num_classes = y_score.shape[1]

        try:
            aucs = roc_auc_score(
                y_true=np.eye(num_classes)[y_true],
                y_score=y_score,
                average=None,
                multi_class='ovr'
            )
            for i, auc in enumerate(aucs):
                print(f"Class {i} AUC: {auc:.4f}")
        except ValueError as e:
            print("⚠️ AUC Error:", e)

        # Per-class accuracy using confusion matrix
        cm = confusion_matrix(y_true, all_preds)
        class_accuracies = cm.diagonal() / cm.sum(axis=1)
        for i, acc in enumerate(class_accuracies):
            print(f"Class {i} Accuracy: {acc:.4f}")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            fname = os.path.join(save_dir, f"best_epoch{epoch}.pth")
            state = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
            torch.save(state, fname)
            print(f"✔️  Saved best model: {fname}")


# Execution
trained_model = train_swin(
    train_loader, val_loader,
    num_classes=len(label_to_idx),
    save_dir="./checkpoints/swin_base_patch4_window12_384",
    num_epochs=40,
    lr=1e-4,
    device=device,
    accumulation_steps=1
)
