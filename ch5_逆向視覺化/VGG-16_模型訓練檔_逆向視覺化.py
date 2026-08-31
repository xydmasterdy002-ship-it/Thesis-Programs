
import os, csv, time, json, random
from pathlib import Path
from typing import List

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from torchvision.models import VGG16_Weights
from PIL import Image


# =========================================================
# 0. 路徑設定
# =========================================================
train_roots = [
    r"C:\Users\tlhorng\Desktop\ImageNet_100\train.X1",
    r"C:\Users\tlhorng\Desktop\ImageNet_100\train.X2",
    r"C:\Users\tlhorng\Desktop\ImageNet_100\train.X3",
    r"C:\Users\tlhorng\Desktop\ImageNet_100\train.X4",
]
val_roots = [r"C:\Users\tlhorng\Desktop\ImageNet_100\val.X"]

labels_json_path = r"C:\Users\tlhorng\Desktop\ImageNet_100\Labels.json"

# 所有輸出改放到 C:\Users\tlhorng\Desktop\謝彥德\論文程式\ch5_逆向視覺化\ImageNet100_VGG16_FINAL_output 底下
output_root = Path(r"C:\Users\tlhorng\Desktop\謝彥德\論文程式\ch5_逆向視覺化\ImageNet100_VGG16_FINAL_output")
ckpt_dir = output_root / "checkpoints"


# =========================================================
# 1. 訓練參數
# =========================================================
SEED = 42
EPOCHS = 30
BATCH_SIZE = 128
NUM_WORKERS = 8
LR = 1e-3
WEIGHT_DECAY = 5e-4
MOMENTUM = 0.9
IMAGE_SIZE = 224
AMP = True
SAVE_EVERY = 5

# 與您原本訓練程式一致：每個 root、每個 class 最多取多少張
MAX_TRAIN_PER_CLASS = 1300
MAX_VAL_PER_CLASS = 50


# =========================================================
# 2. 自訂 ImageNet-100 Dataset
# =========================================================
class ImageNet100(Dataset):
    def __init__(self, root_list, transform=None, is_train=True, train_class_to_idx=None, max_per_class=None):
        self.transform = transform
        self.is_train = is_train

        if is_train:
            class_names = []
            for root in root_list:
                if not os.path.isdir(root):
                    raise FileNotFoundError(f"找不到資料夾：{root}")
                class_names += [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]
            class_names = sorted(set(class_names))
            self.class_to_idx = {cls: idx for idx, cls in enumerate(class_names)}
        else:
            if train_class_to_idx is None:
                raise ValueError("Validation dataset must provide train_class_to_idx")
            self.class_to_idx = train_class_to_idx

        self.samples = []
        valid_exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
        for root in root_list:
            if not os.path.isdir(root):
                raise FileNotFoundError(f"找不到資料夾：{root}")
            for cls in self.class_to_idx.keys():
                cls_path = os.path.join(root, cls)
                if not os.path.isdir(cls_path):
                    continue
                fnames = [f for f in os.listdir(cls_path) if f.lower().endswith(valid_exts)]
                fnames = sorted(fnames)
                if max_per_class:
                    fnames = fnames[:max_per_class]
                for fname in fnames:
                    self.samples.append((os.path.join(cls_path, fname), self.class_to_idx[cls]))

        if len(self.samples) == 0:
            raise RuntimeError("Dataset 沒有讀到任何圖片，請檢查 root_list 或副檔名")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


# =========================================================
# 3. 工具函式
# =========================================================
def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_vgg16(num_classes: int):
    model = models.vgg16(weights=VGG16_Weights.DEFAULT)
    model.classifier[6] = nn.Linear(4096, num_classes)
    return model


@torch.no_grad()
def accuracy_topk(output, target, topk=(1, 5)):
    maxk = max(topk)
    _, pred = output.topk(maxk, dim=1)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append((correct_k / target.size(0)).item() * 100.0)
    return res


def main():
    # =========================================================
    # 4. 基本設定
    # =========================================================
    output_root.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    print("使用裝置：", device)
    if torch.cuda.is_available():
        print("GPU：", torch.cuda.get_device_name(0))

    # =========================================================
    # 5. 讀取 Labels.json
    # =========================================================
    if not os.path.isfile(labels_json_path):
        raise FileNotFoundError(f"找不到 Labels.json：{labels_json_path}")

    with open(labels_json_path, "r", encoding="utf-8") as f:
        wnid_to_name = json.load(f)

    print(f"Labels.json 類別數：{len(wnid_to_name)}")

    # =========================================================
    # 6. Transform
    # =========================================================
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    train_tf = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(IMAGE_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(imagenet_mean, imagenet_std),
    ])

    val_tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(imagenet_mean, imagenet_std),
    ])

    # =========================================================
    # 7. Dataset / DataLoader
    # =========================================================
    train_dataset = ImageNet100(train_roots, transform=train_tf, is_train=True, max_per_class=MAX_TRAIN_PER_CLASS)
    val_dataset = ImageNet100(val_roots, transform=val_tf, is_train=False, train_class_to_idx=train_dataset.class_to_idx, max_per_class=MAX_VAL_PER_CLASS)

    num_classes = len(train_dataset.class_to_idx)
    idx_to_wnid = {v: k for k, v in train_dataset.class_to_idx.items()}
    idx_to_name = {idx: wnid_to_name.get(wnid, wnid) for idx, wnid in idx_to_wnid.items()}

    print(f"Train images：{len(train_dataset)}")
    print(f"Val images：{len(val_dataset)}")
    print(f"Classes：{num_classes}")

    if num_classes != len(wnid_to_name):
        print(f"提醒：訓練資料實際類別數={num_classes}，Labels.json 類別數={len(wnid_to_name)}")
        print("這不一定是錯誤；代表 train.X1~X4 合併後的資料夾類別數與 Labels.json 數量不同。")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=torch.cuda.is_available(), persistent_workers=(NUM_WORKERS > 0))
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=torch.cuda.is_available(), persistent_workers=(NUM_WORKERS > 0))

    # =========================================================
    # 8. 輸出類別對照表與設定檔
    # =========================================================
    mapping_csv_path = output_root / "class_index_mapping.csv"
    with open(mapping_csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["index", "wnid", "name"])
        for idx in range(num_classes):
            wnid = idx_to_wnid[idx]
            writer.writerow([idx, wnid, wnid_to_name.get(wnid, wnid)])

    config_json_path = output_root / "train_config.json"
    with open(config_json_path, "w", encoding="utf-8") as f:
        json.dump({
            "num_classes": num_classes,
            "image_size": IMAGE_SIZE,
            "labels_json_path": labels_json_path,
            "output_root": str(output_root),
            "checkpoint_best": str(ckpt_dir / "best_vgg16_imagenet100.pth"),
            "class_index_mapping": str(mapping_csv_path),
            "training_log": str(output_root / "training_log.csv")
        }, f, ensure_ascii=False, indent=2)

    # =========================================================
    # 9. 建立模型
    # =========================================================
    model = build_vgg16(num_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.1, patience=5 )
    scaler = torch.cuda.amp.GradScaler(enabled=(AMP and torch.cuda.is_available()))

    # =========================================================
    # 10. 單一 epoch
    # =========================================================
    def run_one_epoch(loader, train: bool):
        model.train(train)
        total_loss, total_top1, total_top5, total_n = 0.0, 0.0, 0.0, 0

        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            if train:
                optimizer.zero_grad(set_to_none=True)

            with torch.set_grad_enabled(train):
                with torch.cuda.amp.autocast(enabled=(AMP and torch.cuda.is_available())):
                    outputs = model(images)
                    loss = criterion(outputs, labels)

                if train:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()

            top1, top5 = accuracy_topk(outputs.detach(), labels, topk=(1, 5))
            bs = labels.size(0)
            total_loss += loss.item() * bs
            total_top1 += top1 * bs
            total_top5 += top5 * bs
            total_n += bs

        return total_loss / total_n, total_top1 / total_n, total_top5 / total_n

    # =========================================================
    # 11. 正式訓練
    # =========================================================
    log_path = output_root / "training_log.csv"
    best_top1 = 0.0

    with open(log_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "train_top1", "train_top5", "val_loss", "val_top1", "val_top5", "lr", "epoch_time_sec"])

        for epoch in range(1, EPOCHS + 1):
            t0 = time.time()

            train_loss, train_top1, train_top5 = run_one_epoch(train_loader, train=True)
            val_loss, val_top1, val_top5 = run_one_epoch(val_loader, train=False)

            scheduler.step(val_top1)
            epoch_time = time.time() - t0
            lr_now = optimizer.param_groups[0]["lr"]

            print(f"Epoch {epoch:03d}/{EPOCHS} | train Top1 {train_top1:.2f} Top5 {train_top5:.2f} | val Top1 {val_top1:.2f} Top5 {val_top5:.2f} | lr={lr_now:.6g} | {epoch_time:.1f}s")
            writer.writerow([epoch, train_loss, train_top1, train_top5, val_loss, val_top1, val_top5, lr_now, epoch_time])
            f.flush()

            save_dict = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "class_to_idx": train_dataset.class_to_idx,
                "wnid_to_name": wnid_to_name,
                "idx_to_wnid": idx_to_wnid,
                "best_top1": max(best_top1, val_top1),
                "num_classes": num_classes,
                "image_size": IMAGE_SIZE
            }

            if val_top1 > best_top1:
                best_top1 = val_top1
                save_dict["best_top1"] = best_top1
                torch.save(save_dict, ckpt_dir / "best_vgg16_imagenet100.pth")
                print(f"已更新 best model：Top1={best_top1:.2f}")

            if epoch % SAVE_EVERY == 0:
                torch.save(save_dict, ckpt_dir / f"checkpoint_epoch_{epoch:03d}.pth")

    print("訓練完成")
    print("best val Top-1 =", best_top1)
    print("輸出資料夾 =", output_root)
    print("Notebook 請讀取 =", ckpt_dir / "best_vgg16_imagenet100.pth")


if __name__ == "__main__":
    main()
