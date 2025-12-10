"""
Simple CNN example (PyTorch).

- Defines a small CNN `SimpleCNN` suitable for CIFAR-sized images.
- Shows model summary and runs one training epoch on CIFAR-10 (if torchvision available).
- Intended as an educational starting point; adjust hyperparameters for real experiments.

Run:
    cd f:\githubzyuu\conv\简单神经网络
    python simple_cnn.py

Requires:
    torch, torchvision (optional for running dataset)
"""

import os
import time

import torch
import torch.nn as nn
import torch.optim as optim

try:
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader
    _HAS_TORCHVISION = True
except Exception:
    _HAS_TORCHVISION = False


# --- 简单示例：卷积、激活、损失（最基础） ---
# 目的：在文件最前面展示所使用的基本构件的最简单用法，便于初学者理解。
try:
    # 简单卷积示例：3->8 通道，3x3 核，padding 保持尺寸
    _demo_x = torch.randn(1, 3, 32, 32)
    _demo_conv = nn.Conv2d(3, 8, kernel_size=3, padding=1)
    _demo_act = nn.ReLU()
    _demo_y = _demo_act(_demo_conv(_demo_x))
    print('\n[示例] 卷积 -> 激活: input', tuple(_demo_x.shape), '-> output', tuple(_demo_y.shape))

    # 简单损失示例：CrossEntropy 用法（logits -> 目标索引）
    _logits = torch.randn(4, 10)  # batch_size=4, num_classes=10
    _targets = torch.randint(0, 10, (4,))
    _criterion = nn.CrossEntropyLoss()
    _loss_val = _criterion(_logits, _targets)
    print('[示例] CrossEntropyLoss: loss=', float(_loss_val))
except Exception:
    # 在极少数环境（如 torch 未安装）下，跳过示例
    pass



class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 16x16

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 8x8
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def get_dataloaders(batch_size=64, num_workers=0):
    if not _HAS_TORCHVISION:
        raise RuntimeError('torchvision required to load datasets')

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
    ])

    train_set = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    test_set = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, test_loader


def train_one_epoch(model, device, loader, optimizer, criterion, max_batches=None):
    """Train for one epoch.

    If `max_batches` is provided (int), only that many batches are processed
    — useful for quick demos. If `None`, the entire loader is used.
    Returns: (avg_loss, accuracy_percent, elapsed_seconds)
    """
    model.train()
    running_loss = 0.0
    total = 0
    correct = 0
    start = time.time()
    for i, (images, targets) in enumerate(loader):
        images = images.to(device)
        targets = targets.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = outputs.max(1)
        total += targets.size(0)
        correct += preds.eq(targets).sum().item()
        if (max_batches is not None) and (i + 1 >= max_batches):
            break
    elapsed = time.time() - start
    return running_loss / max(1, total), 100.0 * correct / max(1, total), elapsed


def evaluate(model, device, loader, criterion):
    """Evaluate model on given loader (full pass)."""
    model.eval()
    running_loss = 0.0
    total = 0
    correct = 0
    start = time.time()
    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            targets = targets.to(device)
            outputs = model(images)
            loss = criterion(outputs, targets)
            running_loss += loss.item() * images.size(0)
            _, preds = outputs.max(1)
            total += targets.size(0)
            correct += preds.eq(targets).sum().item()
    elapsed = time.time() - start
    return running_loss / max(1, total), 100.0 * correct / max(1, total), elapsed


if __name__ == '__main__':
    print('\n=== SimpleCNN demo ===')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Using device:', device)

    model = SimpleCNN(num_classes=10).to(device)
    print(model)

    # If torchvision available, run a full training loop (10 epochs)
    if _HAS_TORCHVISION:
        nw = 0 if os.name == 'nt' else 2
        train_loader, test_loader = get_dataloaders(batch_size=128, num_workers=nw)
        optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
        criterion = nn.CrossEntropyLoss()

        epochs = 10
        best_acc = 0.0
        save_path = 'simple_cnn_best.pth'
        print(f"Start training for {epochs} epochs (will save best model to {save_path})")
        for epoch in range(1, epochs + 1):
            train_loss, train_acc, train_time = train_one_epoch(model, device, train_loader, optimizer, criterion, max_batches=None)
            val_loss, val_acc, val_time = evaluate(model, device, test_loader, criterion)
            print(f'Epoch {epoch:02d}/{epochs} - train loss: {train_loss:.4f}, acc: {train_acc:.2f}%, time: {train_time:.2f}s | '
                  f'val loss: {val_loss:.4f}, acc: {val_acc:.2f}%, time: {val_time:.2f}s')
            # save best
            if val_acc > best_acc:
                best_acc = val_acc
                try:
                    torch.save({'epoch': epoch, 'model_state': model.state_dict(), 'optimizer_state': optimizer.state_dict(), 'val_acc': val_acc}, save_path)
                    print(f'  Saved new best model (val_acc={val_acc:.2f}%)')
                except Exception as e:
                    print('  Warning: failed to save model:', e)
        print(f'Training finished. Best val acc: {best_acc:.2f}%')
    else:
        print('torchvision not available; install torchvision to run the demo training loop.')
