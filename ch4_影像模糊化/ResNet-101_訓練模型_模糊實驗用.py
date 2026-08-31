import os
import multiprocessing

import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models


# =========================================================
# 路徑設定
# =========================================================
dataset_root = r"C:\Users\tlhorng\Desktop\ImageNet_100"
train_roots = [
    os.path.join(dataset_root, "train.X1"),
    os.path.join(dataset_root, "train.X2"),
    os.path.join(dataset_root, "train.X3"),
    os.path.join(dataset_root, "train.X4")
]
val_roots = [os.path.join(dataset_root, "val.X")]

output_pth = r"C:\Users\tlhorng\Desktop\謝彥德\論文程式\ch4_影像模糊化\模糊方式準確下降率_ResNet-101\Resnet-101_imagenet100.pth"


# =========================================================
# 參數
# =========================================================
EPOCHS = 30
lr = 0.001
batch_size = 50
weight_decay = 5e-4
topk = (1, 5)
num_workers = 8


# =========================================================
# Dataset
# =========================================================
class ImageNet100(Dataset):
    def __init__(self, root_list, transform=None, is_train=True, train_class_to_idx=None, max_per_class=None):
        self.transform = transform

        if is_train:
            class_names = []
            for root in root_list:
                class_names += [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]
            class_names = sorted(set(class_names))
            self.class_to_idx = {cls: idx for idx, cls in enumerate(class_names)}
        else:
            if train_class_to_idx is None:
                raise ValueError("Validation dataset must provide train_class_to_idx")
            self.class_to_idx = train_class_to_idx

        self.samples = []
        for root in root_list:
            for cls in self.class_to_idx:
                cls_path = os.path.join(root, cls)
                if not os.path.isdir(cls_path):
                    continue
                fnames = [f for f in os.listdir(cls_path) if f.lower().endswith((".jpg", ".jpeg"))]
                if max_per_class:
                    fnames = fnames[:max_per_class]
                for fname in fnames:
                    self.samples.append((os.path.join(cls_path, fname), self.class_to_idx[cls]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


# =========================================================
# ResNet-101 模型
# =========================================================
class ResNet101Custom(nn.Module):
    def __init__(self, num_classes=100, include_top=True, pooling=None, pretrained=True):
        super().__init__()
        self.include_top = include_top
        self.pooling = pooling
        self.base_model = models.resnet101(weights=models.ResNet101_Weights.DEFAULT if pretrained else None)

        if include_top:
            self.base_model.fc = nn.Linear(self.base_model.fc.in_features, num_classes)
        else:
            self.base_model.fc = nn.Identity()

    def forward(self, x):
        x = self.base_model.conv1(x)
        x = self.base_model.bn1(x)
        x = self.base_model.relu(x)
        x = self.base_model.maxpool(x)
        x = self.base_model.layer1(x)
        x = self.base_model.layer2(x)
        x = self.base_model.layer3(x)
        x = self.base_model.layer4(x)
        x = self.base_model.avgpool(x)
        x = torch.flatten(x, 1)

        if self.include_top:
            x = self.base_model.fc(x)
        elif self.pooling == "avg":
            x = torch.mean(x, dim=-1, keepdim=True)
        elif self.pooling == "max":
            x, _ = torch.max(x, dim=-1, keepdim=True)
        return x


# =========================================================
# Top-K Accuracy
# =========================================================
def accuracy_topk(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size_now = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    results = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        results.append(correct_k.mul_(100.0 / batch_size_now).item())
    return results


# =========================================================
# 訓練並輸出 pth
# =========================================================
def main():
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    train_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(imagenet_mean, imagenet_std)
    ])
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(imagenet_mean, imagenet_std)
    ])

    train_dataset = ImageNet100(train_roots, transform=train_transform, is_train=True, max_per_class=1300)
    val_dataset = ImageNet100(val_roots, transform=val_transform, is_train=False, train_class_to_idx=train_dataset.class_to_idx, max_per_class=50)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}, GPU count: {torch.cuda.device_count()}")
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}, Classes: {len(train_dataset.class_to_idx)}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    model = ResNet101Custom(num_classes=len(train_dataset.class_to_idx), include_top=True, pooling=None, pretrained=True)
    model = nn.DataParallel(model).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    print("=== 開始訓練 ResNet-101 ===")
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        correct_train1 = 0.0
        correct_train5 = 0.0
        total_train = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            acc1, acc5 = accuracy_topk(outputs, labels, topk=topk)
            correct_train1 += acc1 * images.size(0) / 100.0
            correct_train5 += acc5 * images.size(0) / 100.0
            total_train += labels.size(0)

        train_loss = running_loss / len(train_loader)
        train_acc1 = 100.0 * correct_train1 / total_train
        train_acc5 = 100.0 * correct_train5 / total_train
        print(f"[Train] Epoch [{epoch + 1}/{EPOCHS}] Loss: {train_loss:.4f} Top-1 Acc: {train_acc1:.2f}% Top-5 Acc: {train_acc5:.2f}%")

        model.eval()
        val_loss = 0.0
        correct_val1 = 0.0
        correct_val5 = 0.0
        total_val = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                acc1, acc5 = accuracy_topk(outputs, labels, topk=topk)
                correct_val1 += acc1 * images.size(0) / 100.0
                correct_val5 += acc5 * images.size(0) / 100.0
                total_val += labels.size(0)

        val_loss /= len(val_loader)
        val_acc1 = 100.0 * correct_val1 / total_val
        val_acc5 = 100.0 * correct_val5 / total_val
        print(f"[Val]   Epoch [{epoch + 1}/{EPOCHS}] Loss: {val_loss:.4f} Top-1 Acc: {val_acc1:.2f}% Top-5 Acc: {val_acc5:.2f}%")
        scheduler.step()

    os.makedirs(os.path.dirname(output_pth), exist_ok=True)
    torch.save(model.module.state_dict(), output_pth)
    print(f"ResNet-101 pth saved to: {output_pth}")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
