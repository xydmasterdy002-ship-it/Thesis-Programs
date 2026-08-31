import os
import glob
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import numpy as np
from tqdm import tqdm
from datetime import datetime
import time
import multiprocessing

torch.backends.cudnn.benchmark = True


# =========================================================
# 訓練 / 驗證資料夾（不改）
# =========================================================
train_roots = [
    r"C:\Users\user\Desktop\ImageNet_100\train.X1",
    r"C:\Users\user\Desktop\ImageNet_100\train.X2",
    r"C:\Users\user\Desktop\ImageNet_100\train.X3",
    r"C:\Users\user\Desktop\ImageNet_100\train.X4"]

val_roots = [
    r"C:\Users\user\Desktop\ImageNet_100\val.X"]

# =========================================================
# AlexNet 訓練參數
# 原論文：SGD, batch=128, momentum=0.9, weight_decay=5e-4, initial LR=0.01, 約90 epochs
# 本程式使用 ImageNet-1K pretrained AlexNet，因此 fine-tuning LR 改為 0.001
# =========================================================
EPOCHS = 90
lr = 0.001
batch_size = 128
weight_decay = 5e-4
topk = (1, 5)

num_workers = 12
use_amp = True


# =========================================================
# 存檔設定（已加入時間戳記，防止新舊實驗互相覆蓋）
# =========================================================
base_save_dir = r"C:\Users\user\Desktop\謝彥德\論文程式\ch2_各類影像辨識網路\AlexNet"

MODEL_NAME = "AlexNet"

dataset_name = os.path.basename(os.path.dirname(train_roots[0]))

# 建立獨一無二的時間標記 (例如: 20260529_1400)
current_run_time = datetime.now().strftime("%Y%m%d_%H%M")
file_prefix = f"{MODEL_NAME}_ep={EPOCHS}_lr={lr}_bs={batch_size}_{current_run_time}"

record_path = os.path.join(
    base_save_dir,
    f"{file_prefix}_training_record__{dataset_name}.pth"
)

weight_best_path = os.path.join(
    base_save_dir,
    f"{file_prefix}_best_{dataset_name}.pth"
)

weight_final_path = os.path.join(
    base_save_dir,
    f"{file_prefix}_final_{dataset_name}.pth"
)


# =========================================================
# Dataset
# =========================================================
class ImageNet100(Dataset):
    def __init__(self, root_list, transform=None, is_train=True, train_class_to_idx=None,
                 max_per_class=None, split_mode="all", test_ratio=0.2, split_seed=42):
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

        if split_mode not in ("all", "train", "test"):
            raise ValueError("split_mode must be 'all', 'train', or 'test'")

        self.samples = []

        # 先把同一類別在所有 train roots 中的影像收集起來，
        # 再以固定亂數種子做 80/20 切分，確保 Train/Test 不重疊且每次執行一致。
        for cls, cls_idx in self.class_to_idx.items():
            class_paths = []
            for root in root_list:
                cls_path = os.path.join(root, cls)
                if not os.path.isdir(cls_path):
                    continue

                fnames = sorted([
                    f for f in os.listdir(cls_path)
                    if f.lower().endswith((".jpg", ".jpeg"))
                ])
                class_paths.extend(os.path.join(cls_path, fname) for fname in fnames)

            class_paths = sorted(class_paths)

            if max_per_class:
                class_paths = class_paths[:max_per_class]

            if split_mode in ("train", "test"):
                rng = np.random.default_rng(split_seed + cls_idx)
                order = rng.permutation(len(class_paths))
                test_count = int(round(len(class_paths) * test_ratio))
                test_idx = order[:test_count]
                train_idx = order[test_count:]

                selected_idx = train_idx if split_mode == "train" else test_idx
                class_paths = [class_paths[i] for i in selected_idx]

            for path in class_paths:
                self.samples.append((path, cls_idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]
        img = Image.open(path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        return img, label


# =========================================================
# AlexNet Data Augmentation + Preprocessing
# =========================================================
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

train_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
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
# Top-K Accuracy
# =========================================================
def accuracy_topk(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size_now = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()

    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size_now).item())

    return res


def save_model_weight(model, path):
    if isinstance(model, nn.DataParallel):
        torch.save(model.module.state_dict(), path)
    else:
        torch.save(model.state_dict(), path)


def main():
    os.makedirs(base_save_dir, exist_ok=True)

    best_val_acc1 = -1.0
    best_val_acc5 = -1.0
    best_epoch = -1
    best_weight_path = ""

    # =========================================================
    # Dataset & DataLoader
    # =========================================================
    # 原始 train 每類最多 1300 張，固定隨機切分：80% Train / 20% Test
    split_seed = 42
    test_ratio = 0.20

    train_dataset = ImageNet100(
        train_roots,
        transform=train_transform,
        is_train=True,
        max_per_class=1300,
        split_mode="train",
        test_ratio=test_ratio,
        split_seed=split_seed
    )

    test_dataset = ImageNet100(
        train_roots,
        transform=val_transform,
        is_train=False,
        train_class_to_idx=train_dataset.class_to_idx,
        max_per_class=1300,
        split_mode="test",
        test_ratio=test_ratio,
        split_seed=split_seed
    )

    val_dataset = ImageNet100(
        val_roots,
        transform=val_transform,
        is_train=False,
        train_class_to_idx=train_dataset.class_to_idx,
        max_per_class=50
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}, GPU count: {torch.cuda.device_count()}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4
    )

    print(f"num_classes = {len(train_dataset.class_to_idx)}")
    print(f"train samples = {len(train_dataset)}")
    print(f"val samples = {len(val_dataset)}")
    print(f"test samples = {len(test_dataset)}")

    # =========================================================
    # AlexNet 模型
    # =========================================================
    num_classes = len(train_dataset.class_to_idx)

    model = models.alexnet(weights=models.AlexNet_Weights.DEFAULT)
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
        mode="min",
        factor=0.1,
        patience=5,
        threshold=1e-4,
        threshold_mode="rel",
        min_lr=1e-7
    )

    # 新版 PyTorch 建議使用 torch.amp.GradScaler 代替舊的 torch.cuda.amp.GradScaler
    scaler = torch.amp.GradScaler(device=device.type, enabled=(use_amp and device.type == "cuda"))

    # =========================================================
    # 訓練紀錄 list
    # =========================================================
    loss_list = []

    train_acc1_list = []
    train_acc5_list = []

    val_acc1_list = []
    val_acc5_list = []

    train_loss_list = []
    val_loss_list = []

    lr_list = []

    epoch_time_list = []
    epoch_weight_path_list = []

    # =========================================================
    # 訓練迴圈
    # =========================================================
    program_start_time = datetime.now()
    program_start_perf = time.time()

    print(f"程式開始時間：{program_start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=== 開始訓練 AlexNet ===")

    for epoch in range(EPOCHS):

        epoch_start_time = time.time()

        # =====================================================
        # Train
        # =====================================================
        model.train()

        running_loss = 0.0
        correct_train1 = 0
        correct_train5 = 0
        total_train = 0

        train_start_perf = time.time()

        current_lr = optimizer.param_groups[0]["lr"]
        lr_list.append(current_lr)

        train_pbar = tqdm(
            train_loader,
            desc=f"Train Epoch {epoch + 1}/{EPOCHS}",
            ncols=120
        )

        # 這裡加入 enumerate 取得當前 step 數
        for step, (images, labels) in enumerate(train_pbar):
            images = images.to(device, memory_format=torch.channels_last, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            # 新版 PyTorch 建議使用 torch.amp.autocast 代替舊的 torch.cuda.amp.autocast
            with torch.amp.autocast(device_type=device.type, enabled=(use_amp and device.type == "cuda")):
                outputs = model(images)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            loss_list.append(loss.item())
            running_loss += loss.item()

            acc1, acc5 = accuracy_topk(outputs, labels, topk=topk)

            correct_train1 += acc1 * images.size(0) / 100
            correct_train5 += acc5 * images.size(0) / 100
            total_train += labels.size(0)

            elapsed = time.time() - train_start_perf
            img_per_sec = total_train / elapsed if elapsed > 0 else 0

            # tqdm 即時狀態欄更新
            train_pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "Top1": f"{(100. * correct_train1 / total_train):.2f}%",
                "Top5": f"{(100. * correct_train5 / total_train):.2f}%",
                "img/s": f"{img_per_sec:.2f}",
                "lr": f"{current_lr:.1e}"
            })

            # ✨ 每一階段換行輸出：每 50 個 step，或者在該 Epoch 的最後一個 step 時，強制把進度推上新行！
            if (step + 1) % 50 == 0 or (step + 1) == len(train_loader):
                current_top1 = 100. * correct_train1 / total_train
                current_top5 = 100. * correct_train5 / total_train
                train_pbar.write(
                    f"-> Epoch [{epoch + 1}/{EPOCHS}] Step [{step + 1}/{len(train_loader)}] | "
                    f"Loss: {loss.item():.4f} | Top1 Acc: {current_top1:.2f}% | img/s: {img_per_sec:.2f}"
                )

        epoch_loss = running_loss / len(train_loader)
        train_acc1 = 100. * correct_train1 / total_train
        train_acc5 = 100. * correct_train5 / total_train

        train_loss_list.append(epoch_loss)
        train_acc1_list.append(train_acc1)
        train_acc5_list.append(train_acc5)

        print(
            f"[Train Summary] Epoch [{epoch + 1}/{EPOCHS}] "
            f"Loss: {epoch_loss:.4f} "
            f"Top-1 Acc: {train_acc1:.2f}% "
            f"Top-5 Acc: {train_acc5:.2f}% "
            f"lr: {current_lr:.1e}"
        )

        # =====================================================
        # Val
        # =====================================================
        model.eval()

        correct_val1 = 0
        correct_val5 = 0
        total_val = 0
        val_loss = 0.0

        val_start_perf = time.time()

        val_pbar = tqdm(
            val_loader,
            desc=f"Val   Epoch {epoch + 1}/{EPOCHS}",
            ncols=120
        )

        with torch.no_grad():
            for images, labels in val_pbar:
                images = images.to(device, memory_format=torch.channels_last, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                with torch.amp.autocast(device_type=device.type, enabled=(use_amp and device.type == "cuda")):
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

        val_loss_list.append(val_loss)
        val_acc1_list.append(val_acc1)
        val_acc5_list.append(val_acc5)

        print(
            f"[Val Summary]   Epoch [{epoch + 1}/{EPOCHS}] "
            f"Loss: {val_loss:.4f} "
            f"Top-1 Acc: {val_acc1:.2f}% "
            f"Top-5 Acc: {val_acc5:.2f}%"
        )

        # =====================================================
        # Epoch 權重：只保留最新一個（已使用一致的 save_model_weight）
        # =====================================================
        epoch_weight_path = os.path.join(
            base_save_dir,
            f"{file_prefix}_epstep={epoch + 1}_{dataset_name}.pth"
        )

        save_model_weight(model, epoch_weight_path)
        epoch_weight_path_list.append(epoch_weight_path)

        print(f"Epoch {epoch + 1} 權重檔已儲存：{epoch_weight_path}")

        old_epoch_pattern = os.path.join(
            base_save_dir,
            f"{file_prefix}_epstep=*_{dataset_name}.pth"
        )

        for old_path in glob.glob(old_epoch_pattern):
            if os.path.abspath(old_path) != os.path.abspath(epoch_weight_path):
                try:
                    os.remove(old_path)
                    print(f"已刪除舊 epoch 權重檔：{old_path}")
                except Exception as e:
                    print(f"刪除舊 epoch 權重檔失敗：{old_path} 原因：{e}")

        # =====================================================
        # Best 權重
        # =====================================================
        if val_acc1 > best_val_acc1:
            best_val_acc1 = val_acc1
            best_val_acc5 = val_acc5
            best_epoch = epoch + 1
            best_weight_path = weight_best_path

            save_model_weight(model, weight_best_path)

            print(
                f"🔥 目前最佳權重已更新：Epoch {best_epoch}, "
                f"Val Top-1 = {best_val_acc1:.2f}%, "
                f"Val Top-5 = {best_val_acc5:.2f}%"
            )
            print(f"最佳權重檔儲存至：{weight_best_path}")

        scheduler.step(val_loss)

        epoch_time_sec = time.time() - epoch_start_time
        epoch_time_list.append(epoch_time_sec)

        print(f"Epoch [{epoch + 1}/{EPOCHS}] Time: {epoch_time_sec / 60:.2f} min\n" + "-"*60)

    # =========================================================
    # Final 權重
    # =========================================================
    save_model_weight(model, weight_final_path)
    print(f"Final 權重檔已儲存至：{weight_final_path}")

    # =========================================================
    # Test：只在全部訓練完成後，以 Best Validation Top-1 權重評估一次
    # =========================================================
    if isinstance(model, nn.DataParallel):
        model.module.load_state_dict(torch.load(weight_best_path, map_location=device, weights_only=True))
    else:
        model.load_state_dict(torch.load(weight_best_path, map_location=device, weights_only=True))

    model.eval()
    correct_test1 = 0
    correct_test5 = 0
    total_test = 0
    test_loss = 0.0
    test_start_perf = time.time()

    test_pbar = tqdm(test_loader, desc="Test Best Model", ncols=120)

    with torch.no_grad():
        for images, labels in test_pbar:
            images = images.to(device, memory_format=torch.channels_last, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with torch.amp.autocast(device_type=device.type, enabled=(use_amp and device.type == "cuda")):
                outputs = model(images)
                loss = criterion(outputs, labels)

            test_loss += loss.item()
            acc1, acc5 = accuracy_topk(outputs, labels, topk=topk)
            correct_test1 += acc1 * images.size(0) / 100
            correct_test5 += acc5 * images.size(0) / 100
            total_test += labels.size(0)

            test_pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "Top1": f"{(100. * correct_test1 / total_test):.2f}%",
                "Top5": f"{(100. * correct_test5 / total_test):.2f}%"
            })

    test_loss /= len(test_loader)
    test_acc1 = 100. * correct_test1 / total_test
    test_acc5 = 100. * correct_test5 / total_test
    test_time_sec = time.time() - test_start_perf

    print(
        f"[Test Summary - Best Epoch {best_epoch}] "
        f"Loss: {test_loss:.4f} "
        f"Top-1 Acc: {test_acc1:.2f}% "
        f"Top-5 Acc: {test_acc5:.2f}% "
        f"Time: {test_time_sec / 60:.2f} min"
    )

    # =========================================================
    # 時間統計
    # =========================================================
    avg_epoch_time_sec = float(np.mean(epoch_time_list)) if len(epoch_time_list) > 0 else 0.0
    avg_epoch_time_min = avg_epoch_time_sec / 60

    total_epoch_time_sec = float(np.sum(epoch_time_list)) if len(epoch_time_list) > 0 else 0.0
    total_epoch_time_min = total_epoch_time_sec / 60

    # =========================================================
    # Training Record
    # =========================================================
    training_record = {
        "model_name": MODEL_NAME,
        "num_classes": num_classes,
        "class_to_idx": train_dataset.class_to_idx,

        "EPOCHS": EPOCHS,
        "lr": lr,
        "batch_size": batch_size,
        "weight_decay": weight_decay,
        "optimizer": "SGD",
        "momentum": 0.9,
        "paper_initial_lr_from_scratch": 0.01,
        "fine_tuning_initial_lr": lr,
        "paper_batch_size": 128,
        "paper_approx_epochs": 90,
        "scheduler": "ReduceLROnPlateau(mode=min, factor=0.1, patience=5)",
        "scheduler_note": "AlexNet paper: divide LR by 10 when validation error stops improving",
        "pretrained": True,
        "pretrained_source": "torchvision AlexNet_Weights.DEFAULT (ImageNet-1K)",
        "topk": topk,
        "num_workers": num_workers,
        "use_amp": use_amp,

        "dataset_name": dataset_name,
        "train_roots": train_roots,
        "val_roots": val_roots,
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
        "test_samples": len(test_dataset),
        "original_train_max_per_class": 1300,
        "train_ratio": 1.0 - test_ratio,
        "test_ratio": test_ratio,
        "split_seed": split_seed,
        "max_val_per_class": 50,

        "loss_list": loss_list,
        "train_loss_list": train_loss_list,
        "val_loss_list": val_loss_list,

        "train_acc1_list": train_acc1_list,
        "train_acc5_list": train_acc5_list,
        "val_acc1_list": val_acc1_list,
        "val_acc5_list": val_acc5_list,

        "lr_list": lr_list,

        "epoch_time_list_sec": epoch_time_list,
        "epoch_time_list_min": [x / 60 for x in epoch_time_list],

        "avg_epoch_time_sec": avg_epoch_time_sec,
        "avg_epoch_time_min": avg_epoch_time_min,

        "total_epoch_time_sec": total_epoch_time_sec,
        "total_epoch_time_min": total_epoch_time_min,

        "base_save_dir": base_save_dir,

        "latest_epoch_weight_path": epoch_weight_path_list[-1] if len(epoch_weight_path_list) > 0 else "",
        "epoch_weight_path_list": epoch_weight_path_list,

        "weight_best_path": weight_best_path,
        "weight_final_path": weight_final_path,

        "best_epoch": best_epoch,
        "best_val_acc1": best_val_acc1,
        "best_val_acc5": best_val_acc5,
        "best_weight_path": best_weight_path,

        "test_loss": test_loss,
        "test_acc1": test_acc1,
        "test_acc5": test_acc5,
        "test_time_sec": test_time_sec,
        "test_time_min": test_time_sec / 60,
        "test_evaluated_weight": weight_best_path,

        "record_path": record_path,
        "program_start_time": program_start_time.strftime("%Y-%m-%d %H:%M:%S"),
        "program_end_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "program_total_sec": time.time() - program_start_perf,
        "program_total_min": (time.time() - program_start_perf) / 60
    }

    torch.save(training_record, record_path)

    print(f"訓練紀錄檔已儲存至：{record_path}")
    print(f"最佳權重 Epoch：{best_epoch}")
    print(f"最佳 Val Top-1：{best_val_acc1:.2f}%")
    print(f"最佳 Val Top-5：{best_val_acc5:.2f}%")
    print(f"Test Top-1：{test_acc1:.2f}%")
    print(f"Test Top-5：{test_acc5:.2f}%")
    print(f"Test 評估時間：{test_time_sec / 60:.2f} 分鐘")
    print(f"平均每個 Epoch 時間：{avg_epoch_time_min:.2f} 分鐘")
    print(f"全部 Epoch 總時間：{total_epoch_time_min:.2f} 分鐘")

    program_end_time = datetime.now()
    program_total_sec = time.time() - program_start_perf

    print(f"程式結束時間：{program_end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"程式總耗時：{program_total_sec / 60:.2f} 分鐘")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()