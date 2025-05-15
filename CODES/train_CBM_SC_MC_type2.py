import os

# === Config ===
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

import pandas as pd
import numpy as np
from PIL import Image
from collections import Counter
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from tqdm import tqdm
import random
import timm


# ============================
# GPU Setup
# ============================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
# ──────────────────────────────────────────────────────────────
# 1. determinism helper
# ──────────────────────────────────────────────────────────────

def set_seed(seed: int = 42):
    """
    Guarantees (as much as PyTorch currently allows) bit-for-bit reproducibility.
    """
    # Python & NumPy
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)

    # Torch (CPU + CUDA)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # cuDNN & CUDA algorithms
    torch.backends.cudnn.deterministic = True   # forbid nondeterministic kernels
    torch.backends.cudnn.benchmark     = False  # disable autotuner (random)
    # torch.use_deterministic_algorithms(True)   # throws if an op is inherently nondet.

    print(f"✅ Global seed set to {seed}")

set_seed(0)          # <- do this before you build models/datasets/loaders


# ============================
# Label mapping (from CSV in actual use)
# ============================
def get_label_to_idx_mainClass(csv_file):
    df = pd.read_csv(csv_file)
    unique_labels = sorted(df['Main_class'].dropna().unique().tolist())
    return {label: idx for idx, label in enumerate(unique_labels)}

label_to_idx_mainClass = get_label_to_idx_mainClass("train_split.csv")

def get_label_to_idx_subClass(csv_file):
    train_df = pd.read_csv(csv_file)
    unique_labels = sorted(train_df['Sub_class'].dropna().unique().tolist())
    return {lbl: i for i, lbl in enumerate(unique_labels)}

label_to_idx_subClass = get_label_to_idx_subClass("train_split.csv")
# ============================
# Transforms
# ============================
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

# ============================
# Dataset with Concept Labels
# ============================
class ImageDataset(Dataset):
    def __init__(self, csv_file, image_folder, concept_columns, transform):
        self.df = pd.read_csv(csv_file)
        self.image_folder = image_folder
        self.concept_cols = concept_columns
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_name = row['Image_name']
        label = label_to_idx_mainClass[row['Main_class']]
        sub_label = label_to_idx_subClass[row['Sub_class']]
        img_path = os.path.join(self.image_folder, img_name)

        img = Image.open(img_path).convert('RGB')
        W, H = img.size
        if max(H, W) > 512:
            scale = 512 / max(H, W)
            new_size = (int(W * scale), int(H * scale))
            img = img.resize(new_size, resample=Image.BILINEAR)

        if self.transform:
            img = self.transform(img)

        concept_vec = torch.tensor(row[self.concept_cols].values.astype(np.float32))
        return img, concept_vec, label, sub_label

# ============================
# Collate Function
# ============================
def pad_collate_fn(batch):
    imgs, concepts, labels, sublabels = zip(*batch)
    max_h = max(img.shape[1] for img in imgs)
    max_w = max(img.shape[2] for img in imgs)
    padded_imgs = [F.pad(img, (0, max_w - img.shape[2], 0, max_h - img.shape[1]), value=0) for img in imgs]
    return torch.stack(padded_imgs), torch.stack(concepts), torch.tensor(labels), torch.tensor(sublabels, dtype=torch.long)


# ============================
# Model Definitions
# ============================


# ────────────────────────────────────────────────────────────────────────────────
# 1. Concept predictor  (image → 96-d concept logits)
# ────────────────────────────────────────────────────────────────────────────────
class ConceptPredictor(nn.Module):
    """
    MaxViT backbone  → 96-dim concept logits
    Returns both σ(logits) (probabilities) and raw logits, exactly like your
    original code so you don’t need to change the training loop.
    """
    def __init__(
        self,
        concept_dim: int = 96,
        pretrained: bool = True,
        model_name: str = "swin_base_patch4_window12_384",  # 512×512 default
        dropout_p: float = 0.3,
    ):
        super().__init__()

        # MaxViT without its classifier head
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,       # remove original head
            global_pool="avg",   # (B, E) regardless of input H×W
            img_size=512
        )
        in_dim = self.backbone.num_features 

        # Light two-layer head → concept logits
        self.fc = nn.Sequential(
            nn.Dropout(dropout_p),
            nn.Linear(in_dim, concept_dim, bias=True),  # E → 96
        )

        # (optional) weight init — helps early stability
        nn.init.trunc_normal_(self.fc[-1].weight, std=0.02)
        nn.init.zeros_(self.fc[-1].bias)

    def forward(self, x: torch.Tensor):
        features = self.backbone(x)      # (B, E)
        concept_logits = self.fc(features)   # (B, 96)
        return torch.sigmoid(concept_logits), concept_logits

class SUBLabelPredictor(nn.Module):
    """
    Simple linear classifier over concept logits.
    """
    def __init__(self, concept_dim: int = 96, num_subClasses: int = 19):
        super().__init__()
        self.fc = nn.Linear(concept_dim, num_subClasses)

        nn.init.trunc_normal_(self.fc.weight, std=0.02)
        nn.init.zeros_(self.fc.bias)

    def forward(self, concept_logits: torch.Tensor):
        """
        Args
        ----
        concept_logits : (B, 96)  – **raw** logits from ConceptPredictor
        Returns
        -------
        class_logits   : (B, 8)
        """
        return self.fc(concept_logits)
# ────────────────────────────────────────────────────────────────────────────────
# 2. Label predictor  (96 concepts → 8 class logits)
# ────────────────────────────────────────────────────────────────────────────────
class LabelPredictor(nn.Module):
    """
    Simple linear classifier over concept logits.
    """
    def __init__(self, num_subClasses: int = 96, num_classes: int = 8):
        super().__init__()
        self.fc = nn.Linear(num_subClasses, num_classes)

        nn.init.trunc_normal_(self.fc.weight, std=0.02)
        nn.init.zeros_(self.fc.bias)

    def forward(self, concept_logits: torch.Tensor):
        """
        Args
        ----
        concept_logits : (B, 96)  – **raw** logits from ConceptPredictor
        Returns
        -------
        class_logits   : (B, 8)
        """
        return self.fc(concept_logits)


# ============================
# Epoch Runner
# ============================

from sklearn.metrics import f1_score, precision_score, recall_score

def run_epoch(loader, concept_model, label_model, subLabel_model, criterion_concept, criterion_label, criterion_sublabel, optimizer=None, desc='train'):
    torch.backends.cudnn.enabled = False
    train = optimizer is not None
    concept_model.train() if train else concept_model.eval()
    label_model.train() if train else label_model.eval()
    subLabel_model.train() if train else subLabel_model.eval()

    total_loss, correct,correct_sub, total = 0, 0, 0, 0
    all_preds, all_targets = [], []
    all_subpreds, all_subtargets = [], []
    
    total_l1, total_imgs = 0, 0
    # ---- L1 running sum ----

    #---------------AUC
    all_label_logits, all_label_targets = [], []
    all_sublabel_logits, all_sublabel_targets = [], []
    #------------------------

    for imgs, concepts, labels, subLabels in tqdm(loader, desc=desc):
        imgs, concepts, labels, subLabels = imgs.to(device), concepts.to(device), labels.to(device), subLabels.to(device)

        if train:
            optimizer.zero_grad()

        pred_concepts, logits = concept_model(imgs)
        k = concepts.sum(dim=1).int()
        topk_concepts = torch.topk(pred_concepts, k=k.max().item(), dim=1).indices

        pred_sublabels = subLabel_model(logits)
        pred_labels = label_model(logits)

        l1_lambda = 1e-4  # adjust this based on sparsity-performance trade-off
        l1_loss = l1_lambda * logits.abs().mean()
        loss = criterion_sublabel(pred_sublabels, subLabels) + criterion_concept(logits, concepts) + criterion_label(pred_labels, labels) + l1_loss

        # loss = criterion_concept(pred_concepts, concepts) + criterion_label(pred_labels, labels)
        
        # ---- update L1 running sum ----
        total_l1   += logits.abs().sum().item()      # sum over batch & concepts
        total_imgs += imgs.size(0)
        # --------------------------------

        if train:
            loss.backward()
            optimizer.step()

         # ───── store everything we need for AUC ─────
        if not train:                            # only need this for val / test
            all_label_logits.append(pred_labels.detach().cpu())
            all_label_targets.append(labels.detach().cpu())
            all_sublabel_logits.append(pred_sublabels.detach().cpu())
            all_sublabel_targets.append(subLabels.detach().cpu())

        total_loss += loss.item() * imgs.size(0)
        # ──────────────────────────────────────────────
        preds = pred_labels.argmax(dim=1)
        correct += (preds == labels).sum().item()
        all_preds.extend(preds.cpu().tolist())
        all_targets.extend(labels.cpu().tolist())

        preds_sub = pred_sublabels.argmax(dim=1)
        correct_sub += (preds_sub == subLabels).sum().item()
        all_subpreds.extend(preds_sub.cpu().tolist())
        all_subtargets.extend(subLabels.cpu().tolist())
        # ──────────────────────────────────────────────
        total += labels.size(0)

    label_auc  = float("nan")
    sublabel_auc = float("nan")

    if desc == 'val':
        pred_counts = Counter(all_preds)
        print("\nValidation Predicted Label Counts:")
        for label_idx in range(len(label_to_idx_mainClass)):
            print(f"Label {label_idx}: {pred_counts.get(label_idx, 0)}", end=", ")
        print()
        
        with torch.no_grad():
            # Get ground-truth number of concepts per sample
            k_per_sample = concepts.sum(dim=1).long()  # shape [B]
            # Ensure k >= 1 to avoid zero-topk errors
            k_per_sample = torch.clamp(k_per_sample, min=1)

            # For each sample, keep top-k indices in predicted probabilities
            batch_size, num_concepts = pred_concepts.shape
            topk_preds = torch.zeros_like(pred_concepts)
            print("topk_pred concepts logits:", pred_concepts.max(), end=", ")

            for i in range(batch_size):
                k = k_per_sample[i].item()
                topk_indices = torch.topk(pred_concepts[i], k=k).indices
                topk_preds[i, topk_indices] = 1.0

            # Compute multi-label accuracy (per sample average)
            correct_per_sample = (topk_preds == concepts).float().mean(dim=1)
            concept_acc = correct_per_sample.mean().item()
            print("topk:", topk_preds)

            y_true = concepts.cpu().numpy()
            y_pred = topk_preds.cpu().numpy()

            concept_f1_micro = f1_score(y_true, y_pred, average='micro', zero_division=0)
            concept_f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
            print(f"Concept F1 (micro): {concept_f1_micro:.4f}, F1 (macro): {concept_f1_macro:.4f}")
            avg_k_pred = topk_preds.sum(dim=1).float().mean().item()
            print(f"Avg predicted concepts per image: {avg_k_pred:.2f}")

            # ─── stack & prep ──────────────────────────────────────────
            lbl_logits = torch.cat(all_label_logits)          # (N, 8)
            lbl_probs  = torch.softmax(lbl_logits, dim=1).numpy()
            lbl_true   = torch.cat(all_label_targets).numpy()
            lbl_true_1hot = np.eye(lbl_probs.shape[1])[lbl_true]

            sublbl_logits = torch.cat(all_sublabel_logits)          # (N, 19)
            sublbl_probs  = torch.softmax(sublbl_logits, dim=1).numpy()
            sublbl_true   = torch.cat(all_sublabel_targets).numpy()
            sublbl_true_1hot = np.eye(sublbl_probs.shape[1])[sublbl_true]

        
            # ─── compute safely (might fail if a class absent) ─────────
            try:
                label_auc = roc_auc_score(lbl_true_1hot, lbl_probs,
                                        multi_class='ovr', average='macro')
            except ValueError:
                pass   # leave as NaN – not enough positive samples
            
            try:
                sublabel_auc = roc_auc_score(sublbl_true_1hot, sublbl_probs,
                                        multi_class='ovr', average='macro')
            except ValueError:  
                pass
           




    precision = precision_score(all_targets, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_targets, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_targets, all_preds, average='weighted', zero_division=0)

    subprecision = precision_score(all_subtargets, all_subpreds, average='weighted', zero_division=0)
    subrecall = recall_score(all_subtargets, all_subpreds, average='weighted', zero_division=0)
    subf1 = f1_score(all_subtargets, all_subpreds, average='weighted', zero_division=0)

    acc = correct / total
    subacc = correct_sub / total
    loss_avg = total_loss / total


    avg_l1 = total_l1 / total_imgs
    print(f"{desc.capitalize()} avg L1 (|logits| sum per image): {avg_l1:.2f}")

    return loss_avg, acc, subacc, precision, subprecision, recall, subrecall,  f1, subf1, concept_acc if desc == 'val' else None, label_auc, sublabel_auc

# ============================
# Dataloader Setup
# ============================
def create_cbm_dataloaders(train_csv, val_csv, image_folder, concept_columns, batch_size=130, num_workers=8):
    df = pd.read_csv(train_csv)
    main_counts  = df['Main_class'].value_counts()
    w_main = 1.0 / main_counts.loc[df['Main_class']].values           # shape = (N,)

    # ------------------------------------------------------------------
    # 2. per-row weights for Sub_class   (inverse frequency)
    # ------------------------------------------------------------------
    sub_counts   = df['Sub_class'].value_counts()
    w_sub  = 1.0 / sub_counts.loc[df['Sub_class']].values             # shape = (N,)

    # ------------------------------------------------------------------
    # 3. fuse the two weight vectors  → final sample_weights
    #    – here: arithmetic mean; alternatives:
    #         • (w_main * w_sub)**0.5           geometric mean
    #         • 0.7*w_main + 0.3*w_sub          weighted mix
    #         • np.maximum(w_main, w_sub)       strictest balancing
    # ------------------------------------------------------------------
    sample_weights = (w_main + w_sub) / 2.0

    # turn into a PyTorch tensor (float64 is required by WeightedRandomSampler)
    sample_weights = torch.as_tensor(sample_weights, dtype=torch.double)

    sampler = torch.utils.data.WeightedRandomSampler(
        weights      = sample_weights,
        num_samples  = len(sample_weights),   # full epoch
        replacement  = True                   # classic oversampling
    )


    train_dataset = ImageDataset(train_csv, image_folder, concept_columns, transform=train_transform)
    val_dataset = ImageDataset(val_csv, image_folder, concept_columns, transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, num_workers=num_workers, collate_fn=pad_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=40, shuffle=False, num_workers=num_workers, collate_fn=pad_collate_fn)
    return train_loader, val_loader

# ============================
# Main Trainer
# ============================
def main(train_csv, val_csv, image_folder,arch_name):
    df = pd.read_csv(train_csv)
    # concept_columns = [col for col in df.columns if 'Body_part' in col or 'Concept' in col or 'Monk_skin_tone_MST' in col]
    concept_columns = [col for col in df.columns if 'Body_part' in col or 'Descriptor' in col]
    print("Concept columns length :", len(concept_columns))
    concept_dim = len(concept_columns)
    num_classes = len(label_to_idx_mainClass)
    num_subClasses = len(label_to_idx_subClass)
    print()
    print("------------------------------------------------------------------------")
    print("Number of classes:", num_classes)
    print("Number of subclasses:", num_subClasses)
    print("------------------------------------------------------------------------")


    train_loader, val_loader = create_cbm_dataloaders(train_csv, val_csv, image_folder, concept_columns)

    concept_model = nn.DataParallel(ConceptPredictor(concept_dim)).to(device)
    subLabel_model = nn.DataParallel(SUBLabelPredictor(concept_dim, num_subClasses)).to(device)
    label_model = nn.DataParallel(LabelPredictor(concept_dim, num_classes)).to(device)

    optimizer = torch.optim.AdamW(list(concept_model.parameters()) + list(label_model.parameters()) + list(subLabel_model.parameters()), lr=3e-5, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30, eta_min=1e-6)

    criterion_concept = nn.BCEWithLogitsLoss()

    criterion_label = nn.CrossEntropyLoss(label_smoothing=0.1)
    criterion_sublabel = nn.CrossEntropyLoss(label_smoothing=0.1)

    best_val_acc = 0

    for epoch in range(200):
        print(f"\nEpoch {epoch+1}/200")
        train_loss, train_acc, train_subacc, train_prec, train_subprec, train_rec, train_subrec, train_f1, train_subf1, n_p, label_auc, label_subacc = run_epoch(train_loader, concept_model, label_model, subLabel_model, criterion_concept, criterion_label, criterion_sublabel, optimizer, desc='train')
        val_loss, val_acc, val_subacc,val_prec, val_subprec, val_rec, val_subrec, val_f1, val_subf1, concept_acc, label_auc, label_subacc = run_epoch(val_loader, concept_model, label_model, subLabel_model, criterion_concept, criterion_label, criterion_sublabel, None,desc='val')
        scheduler.step()

        print(f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, Precision: {train_prec:.4f}, Recall: {train_rec:.4f}, F1: {train_f1:.4f}")
        print(f"Train Sub Acc: {train_subacc:.4f}, Precision: {train_subprec:.4f}, Recall: {train_subrec:.4f}, F1: {train_subf1:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, Concept Acc: {concept_acc:.4f}, Precision: {val_prec:.4f}, Recall: {val_rec:.4f}, F1: {val_f1:.4f}")
        print(f"Val Sub Acc: {val_subacc:.4f}, Precision: {val_subprec:.4f}, Recall: {val_subrec:.4f}, F1: {val_subf1:.4f}")
        print(f"AUC(lbl) of label: {label_auc:.4f}")
        print(f"AUC(sub) of sublabel: {label_subacc:.4f}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            path = f"./checkpoints/{arch_name}/best_model_epoch{epoch+1}.pth"
            torch.save({
                'concept_model': concept_model.module.state_dict() if isinstance(concept_model, nn.DataParallel) else concept_model.state_dict(),
                'label_model': label_model.module.state_dict() if isinstance(label_model, nn.DataParallel) else label_model.state_dict()
            }, path)
            print(f"✅ Saved best model: {path}")

    print(f"\nFinished. Best Val Acc: {best_val_acc:.4f}")

# ============================
# Launch Training
# ============================
train_csv = './METADATA/train_split.csv'
val_csv = './METADATA/test_split.csv'
image_folder = './DATASET'
arch_name = 'swin_base_patch4_window12_384_CBM_SC_MC_type2'
main(train_csv, val_csv, image_folder, arch_name)