import os
import shutil
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import numpy as np
import random
import matplotlib.pyplot as plt
from torch.nn import functional as F
from tqdm import tqdm
from datetime import datetime
import time
import multiprocessing
from collections import defaultdict

torch.backends.cudnn.benchmark = True


# =========================================================
# 訓練資料夾：原始 ImageNet-100 train.X1~train.X4
# =========================================================
train_roots = [
    r"C:\Users\tlhorng\Desktop\ImageNet_100\train.X1",
    r"C:\Users\tlhorng\Desktop\ImageNet_100\train.X2",
    r"C:\Users\tlhorng\Desktop\ImageNet_100\train.X3",
    r"C:\Users\tlhorng\Desktop\ImageNet_100\train.X4"
]

# =========================================================
# 排除資料夾：由 train.X1~train.X4 每類抽 20% 建立
# 這些照片不參與訓練，也不作為驗證集
# =========================================================
split_root = r"C:\Users\tlhorng\Desktop\謝彥德\論文程式\ch3_模擬失智症\80訓練20測試_模型\20_VGG_FC=4096_notrain"

# =========================================================
# 驗證資料夾：ImageNet-100 原始 val.X
# =========================================================
val_roots = [
    r"C:\Users\tlhorng\Desktop\ImageNet_100\val.X"
]


# =========================================================
# 從 train.X1~train.X4 每類隨機抽 20% 複製到 split_root
# 若 split_root 已存在，則不重複建立
# =========================================================
def build_20_percent_val_from_train(train_roots, split_root, ratio=0.2, seed=42):
    random.seed(seed)

    if os.path.exists(split_root):
        print(f"排除資料夾已存在，略過重新抽樣：{split_root}")
        return

    print("開始建立 20% 排除資料 ...")
    os.makedirs(split_root, exist_ok=True)

    class_files = defaultdict(list)

    for root in train_roots:
        for cls in os.listdir(root):
            cls_path = os.path.join(root, cls)

            if not os.path.isdir(cls_path):
                continue

            fnames = [
                f for f in os.listdir(cls_path)
                if f.lower().endswith((".jpg", ".jpeg"))
            ]

            for fname in fnames:
                full_path = os.path.join(cls_path, fname)
                class_files[cls].append(full_path)

    for cls, files in class_files.items():
        random.shuffle(files)

        n_val = int(len(files) * ratio)
        val_files = files[:n_val]

        save_cls_dir = os.path.join(split_root, cls)
        os.makedirs(save_cls_dir, exist_ok=True)

        for src_path in tqdm(val_files, desc=f"Copy {cls}", ncols=100):
            fname = os.path.basename(src_path)
            dst_path = os.path.join(save_cls_dir, fname)
            shutil.copy2(src_path, dst_path)

        print(f"{cls}: total={len(files)}, val={len(val_files)}, train={len(files)-len(val_files)}")

    print("20% 排除資料建立完成")


# =========================================================
# Dataset
# =========================================================
class ImageNet100(Dataset):
    def __init__(
        self,
        root_list,
        transform=None,
        is_train=True,
        train_class_to_idx=None,
        max_per_class=None,
        exclude_keys=None
    ):
        self.transform = transform
        self.is_train = is_train

        if is_train:
            class_names = []
            for root in root_list:
                class_names += [
                    d for d in os.listdir(root)
                    if os.path.isdir(os.path.join(root, d))
                ]

            class_names = sorted(set(class_names))
            self.class_to_idx = {cls: idx for idx, cls in enumerate(class_names)}

        else:
            if train_class_to_idx is None:
                raise ValueError("Validation dataset must provide train_class_to_idx")

            self.class_to_idx = train_class_to_idx

        self.samples = []

        for root in root_list:
            for cls in self.class_to_idx.keys():
                cls_path = os.path.join(root, cls)

                if not os.path.isdir(cls_path):
                    continue

                fnames = [
                    f for f in os.listdir(cls_path)
                    if f.lower().endswith((".jpg", ".jpeg"))
                ]

                if max_per_class:
                    fnames = fnames[:max_per_class]

                for fname in fnames:
                    if exclude_keys is not None and (cls, fname) in exclude_keys:
                        continue

                    self.samples.append(
                        (os.path.join(cls_path, fname), self.class_to_idx[cls])
                    )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]
        img = Image.open(path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        return img, label


# =========================================================
# Data Augmentation + Preprocessing
# =========================================================
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

train_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.2,
        hue=0.02
    ),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
])

val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
])


# =========================================================
# 參數（不改）
# =========================================================
EPOCHS = 30
lr = 0.01
batch_size = 100
weight_decay = 5e-4
topk = (1, 5)


# =========================================================
# 計算 Top-K 準確率函數（不改）
# =========================================================
def accuracy_topk(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []

    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size).item())

    return res


def main():
    # =========================================================
    # 建立 20% 排除資料；若資料夾已存在則沿用原有排除資料
    # =========================================================
    build_20_percent_val_from_train(
        train_roots=train_roots,
        split_root=split_root,
        ratio=0.2,
        seed=42
    )

    # =========================================================
    # 收集排除照片檔名；排除照片不參與訓練，也不參與 val.X 驗證
    # =========================================================
    exclude_keys = set()

    for cls in os.listdir(split_root):
        cls_dir = os.path.join(split_root, cls)

        if not os.path.isdir(cls_dir):
            continue

        for fname in os.listdir(cls_dir):
            if fname.lower().endswith((".jpg", ".jpeg")):
                exclude_keys.add((cls, fname))

    print(f"訓練排除圖片數量：{len(exclude_keys)}")

    # =========================================================
    # Dataset & DataLoader：train 排除 20% + 原始 val.X
    # =========================================================
    train_dataset = ImageNet100(
        train_roots,
        transform=train_transform,
        is_train=True,
        max_per_class=1300,
        exclude_keys=exclude_keys
    )

    val_dataset = ImageNet100(
        val_roots,
        transform=val_transform,
        is_train=False,
        train_class_to_idx=train_dataset.class_to_idx,
        max_per_class=50
    )

    print(f"num_classes = {len(train_dataset.class_to_idx)}")
    print(f"train samples = {len(train_dataset)}")
    print(f"val samples = {len(val_dataset)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}, GPU count: {torch.cuda.device_count()}")

    train_loader = DataLoader(
        train_dataset,
        batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True
    )

    # =========================================================
    # 使用 torchvision 已完整實作的 VGG-16（FC=4096，不含 BN）
    # =========================================================
    num_classes = len(train_dataset.class_to_idx)

    model = models.vgg16(weights=models.VGG16_Weights.DEFAULT)

    # ===== 修改 FC 結構（維持原始 4096）=====
    model.classifier[0] = nn.Linear(512 * 7 * 7, 4096)
    model.classifier[3] = nn.Linear(4096, 4096)
    model.classifier[6] = nn.Linear(4096, num_classes)

    model = nn.DataParallel(model).to(device)
    model = model.to(memory_format=torch.channels_last)

    # =========================================================
    # Loss / Optimizer / Scheduler
    # =========================================================
    criterion = nn.CrossEntropyLoss()

    optimizer = optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=0.9,
        weight_decay=weight_decay
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=0.1,
        patience=5
    )

    # =========================================================
    # 訓練迴圈 + 驗證 Top-1 & Top-5（不改）
    # =========================================================
    loss_list = []
    train_acc1_list = []
    train_acc5_list = []
    val_acc1_list = []
    val_acc5_list = []

    program_start_time = datetime.now()
    program_start_perf = time.time()

    print(f"程式開始時間：{program_start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=== 開始訓練 ===")

    for epoch in range(EPOCHS):
        # -------------------------
        # 訓練階段
        # -------------------------
        model.train()

        running_loss = 0.0
        correct_train1 = 0
        correct_train5 = 0
        total_train = 0

        train_start_perf = time.time()

        train_pbar = tqdm(
            train_loader,
            desc=f"Train Epoch {epoch+1}/{EPOCHS}",
            ncols=120
        )

        for images, labels in train_pbar:
            images = images.to(device, memory_format=torch.channels_last)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            loss_list.append(loss.item())
            running_loss += loss.item()

            acc1, acc5 = accuracy_topk(outputs, labels, topk=topk)

            correct_train1 += acc1 * images.size(0) / 100
            correct_train5 += acc5 * images.size(0) / 100
            total_train += labels.size(0)

            elapsed = time.time() - train_start_perf
            img_per_sec = total_train / elapsed if elapsed > 0 else 0

            train_pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "Top1": f"{(100. * correct_train1 / total_train):.2f}%",
                "Top5": f"{(100. * correct_train5 / total_train):.2f}%",
                "img/s": f"{img_per_sec:.2f}"
            })

        epoch_loss = running_loss / len(train_loader)
        train_acc1 = 100. * correct_train1 / total_train
        train_acc5 = 100. * correct_train5 / total_train

        train_acc1_list.append(train_acc1)
        train_acc5_list.append(train_acc5)

        print(
            f"[Train] Epoch [{epoch+1}/{EPOCHS}] "
            f"Loss: {epoch_loss:.4f} "
            f"Top-1 Acc: {train_acc1:.2f}% "
            f"Top-5 Acc: {train_acc5:.2f}%"
        )

        # -------------------------
        # 驗證階段
        # -------------------------
        model.eval()

        correct_val1 = 0
        correct_val5 = 0
        total_val = 0
        val_loss = 0.0

        val_start_perf = time.time()

        val_pbar = tqdm(
            val_loader,
            desc=f"Val   Epoch {epoch+1}/{EPOCHS}",
            ncols=120
        )

        with torch.no_grad():
            for images, labels in val_pbar:
                images = images.to(device, memory_format=torch.channels_last)
                labels = labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item()

                acc1, acc5 = accuracy_topk(outputs, labels, topk=topk)

                correct_val1 += acc1 * images.size(0) / 100
                correct_val5 += acc5 * images.size(0) / 100
                total_val += labels.size(0)

                elapsed = time.time() - val_start_perf
                img_per_sec = total_val / elapsed if elapsed > 0 else 0

                val_pbar.set_postfix({
                    "loss": f"{loss.item():.4f}",
                    "Top1": f"{(100. * correct_val1 / total_val):.2f}%",
                    "Top5": f"{(100. * correct_val5 / total_val):.2f}%",
                    "img/s": f"{img_per_sec:.2f}"
                })

        val_loss /= len(val_loader)
        val_acc1 = 100. * correct_val1 / total_val
        val_acc5 = 100. * correct_val5 / total_val

        val_acc1_list.append(val_acc1)
        val_acc5_list.append(val_acc5)

        print(
            f"[Val]   Epoch [{epoch+1}/{EPOCHS}] "
            f"Loss: {val_loss:.4f} "
            f"Top-1 Acc: {val_acc1:.2f}% "
            f"Top-5 Acc: {val_acc5:.2f}%"
        )

        # -------------------------
        # 驗證完畢後，用 Top-1 更新學習率
        # -------------------------
        scheduler.step(val_acc1)

    # =========================================================
    # 儲存模型：自動建立資料夾
    # =========================================================
    save_path = r"C:\Users\tlhorng\Desktop\謝彥德\論文程式\ch3_模擬失智症\80訓練20測試_模型\VGG-16_FC=4096_imagenet100_80%train.pth"

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    if isinstance(model, nn.DataParallel):
        torch.save(model.module.state_dict(), save_path)
    else:
        torch.save(model.state_dict(), save_path)

    print(f"模型已儲存至：{save_path}")

    program_end_time = datetime.now()
    program_total_sec = time.time() - program_start_perf

    print(f"程式結束時間：{program_end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"程式總耗時：{program_total_sec/60:.2f} 分鐘")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
