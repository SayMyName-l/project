#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, random, csv, argparse, math, time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler, Subset
from torchvision import transforms, datasets, models
from torchvision.datasets import CIFAR10
from PIL import Image
import numpy as np

# --------------- Utils ---------------
def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False  # 为速度，保持False
    torch.backends.cudnn.benchmark = True

def num_workers_suggested():
    try:
        import multiprocessing as mp
        return max(2, mp.cpu_count() // 2)
    except:
        return 2

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --------------- Focal Loss (for imbalance expt) ---------------
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, weight=None):
        super().__init__()
        self.gamma = gamma
        self.ce = nn.CrossEntropyLoss(weight=weight)
    def forward(self, logits, target):
        logpt = -self.ce(logits, target)
        pt = torch.exp(logpt)
        loss = ((1 - pt) ** self.gamma) * (-logpt)
        return loss.mean()

# --------------- Dataset helpers ---------------
class TestFolder(torch.utils.data.Dataset):
    """For Kaggle-like test folder without labels; outputs (img, id)."""
    def __init__(self, root, transform):
        self.root = Path(root)
        self.files = sorted([p for p in self.root.glob("*") if p.suffix.lower() in [".jpg",".jpeg",".png"]])
        self.transform = transform
        if not self.files:
            raise FileNotFoundError(f"No images found under {self.root}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fp = self.files[idx]
        img = Image.open(fp).convert("RGB")
        if self.transform:
            img = self.transform(img)
        # id: filename stem or numeric part; PDF允许按样例submission来（此处用stem）
        return img, fp.stem


class KaggleDogsDataset(torch.utils.data.Dataset):
    """Dataset helper for Kaggle Dogs vs Cats style folders without sub-directories."""

    IMG_EXTS = {".jpg", ".jpeg", ".png"}

    def __init__(self, files, labels, transform=None):
        self.files = files
        self.labels = labels
        self.transform = transform

    @classmethod
    def from_directory(cls, root, transform=None):
        root = Path(root)
        files = sorted([p for p in root.glob("*") if p.suffix.lower() in cls.IMG_EXTS])
        if not files:
            raise FileNotFoundError(f"No images found under {root}")
        labels = []
        label_mapping = {"cat": 0, "dog": 1}
        for fp in files:
            prefix = fp.stem.split('.')[0].lower()
            if prefix not in label_mapping:
                raise ValueError(f"Cannot infer label from filename: {fp.name}")
            labels.append(label_mapping[prefix])
        return cls(files, labels, transform)

    def subset(self, indices, transform=None):
        files = [self.files[i] for i in indices]
        labels = [self.labels[i] for i in indices]
        return KaggleDogsDataset(files, labels, transform if transform is not None else self.transform)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fp = self.files[idx]
        img = Image.open(fp).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]

    @property
    def targets(self):
        return self.labels

# --------------- Model factory ---------------
def build_model(backbone: str, num_classes: int, pretrained: bool=True, freeze_backbone: bool=False):
    if backbone.lower() == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        in_f = model.fc.in_features
        model.fc = nn.Linear(in_f, num_classes)
    elif backbone.lower() == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None)
        in_f = model.fc.in_features
        model.fc = nn.Linear(in_f, num_classes)
    else:
        raise ValueError("Unsupported backbone. Choose resnet18/resnet50")

    if freeze_backbone:
        for n, p in model.named_parameters():
            if not n.startswith("fc."):
                p.requires_grad = False
    return model

# --------------- Train / Eval ---------------
def accuracy(logits, y):
    pred = logits.argmax(1)
    return (pred == y).float().mean().item()

def train_one_epoch(model, loader, optimizer, scheduler, criterion, device, scaler=None):
    model.train()
    running_loss, running_acc, n = 0.0, 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=(scaler is not None)):
            logits = model(x)
            loss = criterion(logits, y)
        if scaler:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward(); optimizer.step()
        if scheduler: scheduler.step()
        bs = x.size(0)
        running_loss += loss.item() * bs
        running_acc  += accuracy(logits, y) * bs
        n += bs
    return running_loss / n, running_acc / n

@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss, running_acc, n = 0.0, 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        bs = x.size(0)
        running_loss += loss.item() * bs
        running_acc  += accuracy(logits, y) * bs
        n += bs
    return running_loss / n, running_acc / n

@torch.no_grad()
def predict_to_csv(model, loader, device, out_csv, label_map=None):
    model.eval()
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "label"])
        for x, img_id in loader:
            x = x.to(device)
            logits = model(x)
            pred = logits.argmax(1).cpu().numpy()
            for i, p in enumerate(pred):
                # Dogs vs Cats: 1=dog, 0=cat
                if label_map is None:
                    lab = int(p)
                else:
                    lab = int(label_map[p])
                w.writerow([img_id[i], lab])

# --------------- Data pipelines ---------------
def build_transforms(img_size=224, dataset="dogs"):
    if dataset == "dogs":
        mean, std = [0.485,0.456,0.406], [0.229,0.224,0.225]
        train_tf = transforms.Compose([
            transforms.RandomResizedCrop(img_size, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.2,0.2,0.2,0.05),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        test_tf = transforms.Compose([
            transforms.Resize(int(img_size*1.15)),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    else:  # cifar10
        mean, std = [0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]
        train_tf = transforms.Compose([
            transforms.Resize(img_size),
            transforms.RandomCrop(img_size, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        test_tf = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    return train_tf, test_tf

def _stratified_split_indices(labels, val_ratio, seed):
    labels = np.array(labels)
    classes = np.unique(labels)
    rng = np.random.default_rng(seed)
    train_idx, val_idx = [], []
    for cls in classes:
        cls_indices = np.where(labels == cls)[0]
        rng.shuffle(cls_indices)
        val_count = max(1, int(round(len(cls_indices) * val_ratio)))
        val_count = min(val_count, len(cls_indices) - 1) if len(cls_indices) > 1 else val_count
        val_samples = cls_indices[:val_count]
        train_samples = cls_indices[val_count:]
        if len(train_samples) == 0:
            train_samples = val_samples[:1]
            val_samples = val_samples[1:]
        val_idx.extend(val_samples.tolist())
        train_idx.extend(train_samples.tolist())
    if len(val_idx) == 0 and len(train_idx) > 1:
        val_idx.append(train_idx.pop())
    return train_idx, val_idx


def _dataset_targets(dataset):
    if isinstance(dataset, KaggleDogsDataset):
        return np.array(dataset.targets)
    if isinstance(dataset, Subset):
        base_targets = _dataset_targets(dataset.dataset)
        return base_targets[dataset.indices]
    if hasattr(dataset, "targets"):
        return np.array(dataset.targets)
    raise AttributeError("Dataset does not expose targets for weighted sampling")


def prepare_dataloaders(args):
    img_size = args.img_size
    train_tf, test_tf = build_transforms(img_size, "dogs" if args.dataset=="dogs" else "cifar10")

    if args.dataset == "dogs":
        data_root = Path(args.data_root)
        train_dir = data_root / "train"
        test_dir_candidates = [data_root / name for name in ("test", "test1", "testset", "testing")]
        test_dir = next((d for d in test_dir_candidates if d.exists()), None)

        if not train_dir.exists():
            raise FileNotFoundError(f"Expected training images under {train_dir}")
        if test_dir is None:
            raise FileNotFoundError(f"Could not find test directory under {data_root}. Checked: {[p.name for p in test_dir_candidates]}")

        # 如果 train 下存在类别子目录，则仍然使用 ImageFolder
        has_subdirs = any(p.is_dir() for p in train_dir.iterdir())
        val_dir = data_root / "val"
        if has_subdirs:
            if val_dir.exists() and any(val_dir.iterdir()):
                train_set = datasets.ImageFolder(train_dir, transform=train_tf)
                val_set   = datasets.ImageFolder(val_dir,   transform(test_tf))
            else:
                base_dataset = datasets.ImageFolder(train_dir)
                train_idx, val_idx = _stratified_split_indices(base_dataset.targets, args.val_ratio, args.seed)
                train_set = Subset(datasets.ImageFolder(train_dir, transform=train_tf), train_idx)
                val_set   = Subset(datasets.ImageFolder(train_dir, transform=test_tf), val_idx)
        else:
            full_dataset = KaggleDogsDataset.from_directory(train_dir)
            train_idx, val_idx = _stratified_split_indices(full_dataset.targets, args.val_ratio, args.seed)
            train_set = full_dataset.subset(train_idx, transform=train_tf)
            val_set   = full_dataset.subset(val_idx,   transform=test_tf)

        test_set = TestFolder(test_dir, transform=test_tf)

        sampler = None
        if args.use_weighted_sampler:
            train_targets = _dataset_targets(train_set)
            counts = np.bincount(train_targets, minlength=train_targets.max()+1)
            class_weights = 1.0 / np.maximum(counts, 1)
            sample_weights = class_weights[train_targets]
            sampler = WeightedRandomSampler(weights=torch.DoubleTensor(sample_weights),
                                            num_samples=len(sample_weights),
                                            replacement=True)

        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=(sampler is None),
                                  sampler=sampler, num_workers=num_workers_suggested(), pin_memory=True)
        val_loader   = DataLoader(val_set,   batch_size=args.batch_size, shuffle=False,
                                  num_workers=num_workers_suggested(), pin_memory=True)
        test_loader  = DataLoader(test_set,  batch_size=args.batch_size, shuffle=False,
                                  num_workers=num_workers_suggested(), pin_memory=True)

        num_classes = 2
        label_map = None  # 直接 0=cat,1=dog 对应 ImageFolder 的文件夹顺序（按字母：cat->0, dog->1）
        return train_loader, val_loader, test_loader, num_classes, label_map

    else:  # CIFAR-10
        train_set = CIFAR10(args.data_root, train=True,  download=True, transform=train_tf)
        val_set   = CIFAR10(args.data_root, train=False, download=True, transform=test_tf)

        # 构造“测试提交”同构接口（保存预测成 csv 做记录）
        class TestLike(torch.utils.data.Dataset):
            def __init__(self, base):
                self.base = base
            def __len__(self): return len(self.base)
            def __getitem__(self, i):
                x, _ = self.base[i]
                return x, str(i)  # id 用索引
        test_like = TestLike(val_set)

        sampler = None
        if args.use_weighted_sampler:
            targets = np.array(train_set.targets)
            counts = np.bincount(targets, minlength=10)
            class_weights = 1.0 / np.maximum(counts, 1)
            sample_weights = class_weights[targets]
            sampler = WeightedRandomSampler(weights=torch.DoubleTensor(sample_weights),
                                            num_samples=len(sample_weights),
                                            replacement=True)

        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=(sampler is None),
                                  sampler=sampler, num_workers=num_workers_suggested(), pin_memory=True)
        val_loader   = DataLoader(val_set,   batch_size=args.batch_size, shuffle=False,
                                  num_workers=num_workers_suggested(), pin_memory=True)
        test_loader  = DataLoader(test_like, batch_size=args.batch_size, shuffle=False,
                                  num_workers=num_workers_suggested(), pin_memory=True)
        num_classes = 10
        label_map = None
        return train_loader, val_loader, test_loader, num_classes, label_map

# --------------- Main train / eval loop ---------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="dogs", choices=["dogs","cifar10"])
    parser.add_argument("--data_root", type=str, default="data")
    parser.add_argument("--backbone", type=str, default="resnet18", choices=["resnet18","resnet50"])
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--val_ratio", type=float, default=0.1,
                        help="validation split ratio when a dedicated val folder is absent")
    parser.add_argument("--freeze_backbone", action="store_true", help="freeze all but final FC")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_amp", action="store_true")
    parser.add_argument("--use_focal", action="store_true", help="use FocalLoss instead of CE")
    parser.add_argument("--use_weighted_sampler", action="store_true")
    parser.add_argument("--predict", action="store_true", help="skip training, only predict with outputs/best.pt")
    parser.add_argument("--model_path", type=str, default="outputs/best.pt")
    args = parser.parse_args()

    set_seed(args.seed)
    if not 0 < args.val_ratio < 1:
        raise ValueError("--val_ratio must be between 0 and 1")
    device = get_device()
    Path("outputs").mkdir(exist_ok=True)

    train_loader, val_loader, test_loader, num_classes, label_map = prepare_dataloaders(args)
    model = build_model(args.backbone, num_classes, pretrained=True, freeze_backbone=args.freeze_backbone).to(device)

    if args.predict:
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        print(f"[Predict] Loaded {args.model_path}")
        predict_to_csv(model, test_loader, device, "outputs/submission.csv", label_map)
        print("Saved outputs/submission.csv")
        return

    # 损失函数：CE or Focal；CIFAR-10 可进一步加 class weights（若需）
    if args.use_focal:
        criterion = FocalLoss(gamma=2.0).to(device)
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1).to(device)

    optimizer = optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=args.lr, weight_decay=args.weight_decay)
    # 每个 step 调度（与 epoch 数无关的 Cosine 变体）
    total_steps = args.epochs * max(1, len(train_loader))
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

    scaler = torch.cuda.amp.GradScaler() if (args.use_amp and device.type=="cuda") else None

    best_acc, patience, best_epoch = 0.0, 5, -1
    epochs_no_improve = 0

    global_step = 0
    for epoch in range(1, args.epochs+1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, scheduler, criterion, device, scaler)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        print(f"Epoch {epoch:02d}/{args.epochs} | "
              f"train_loss={train_loss:.4f} acc={train_acc:.4f} | "
              f"val_loss={val_loss:.4f} acc={val_acc:.4f} | "
              f"time={time.time()-t0:.1f}s")

        if val_acc > best_acc:
            best_acc, best_epoch = val_acc, epoch
            torch.save(model.state_dict(), args.model_path)
            print(f"  * New best acc {best_acc:.4f} at epoch {epoch}, saved to {args.model_path}")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print("  * Early stopping triggered.")
                break

    print(f"[Summary] Best val acc={best_acc:.4f} at epoch={best_epoch}")
    # 加载最佳权重，生成提交
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    predict_to_csv(model, test_loader, device, "outputs/submission.csv", label_map)
    print("Saved outputs/submission.csv")

if __name__ == "__main__":
    main()
