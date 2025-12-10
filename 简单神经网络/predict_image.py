#!/usr/bin/env python3
"""
Predict using SimpleCNN on a single image.

改进点：
1. 增加了 Matplotlib 可视化支持（显示图片+概率柱状图）。
2. 增强了模型加载的兼容性（自动处理 strict 匹配和 device 映射）。
3. 增强了图片预处理（处理 RGBA 和灰度图）。
4. 输出格式更美观。

用法：
    python predict_image.py -i ./test_cat.jpg
    python predict_image.py -i ./test_dog.png -m ./checkpoints/best.pth --gui
"""

import argparse
import os
import sys
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

# 尝试导入 matplotlib 用于可视化
try:
    import matplotlib.pyplot as plt
    _HAS_MATPLOTLIB = True
except ImportError:
    _HAS_MATPLOTLIB = False

# ==========================================
# 1. 模型定义 (Fallback)
# ==========================================
try:
    # 优先尝试从 sibling 文件导入
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from simple_cnn import SimpleCNN
except ImportError:
    # 如果找不到 simple_cnn.py，则使用此处的备用定义
    # 注意：必须与训练时的结构完全一致
    class SimpleCNN(nn.Module):
        def __init__(self, num_classes=10):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 32, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 64, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
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

# CIFAR-10 类别名称
CIFAR10_CLASSES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]

# ==========================================
# 2. 辅助函数
# ==========================================

def get_transforms():
    """获取推理时的预处理变换"""
    # 必须与训练时的 normalization 保持一致
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2470, 0.2430, 0.2610)
    return transforms.Compose([
        transforms.Resize((32, 32)), # 强制缩放到模型输入大小
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

def load_image(img_path):
    """加载图片并确保转换为 RGB"""
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Image not found: {img_path}")
    
    img = Image.open(img_path)
    # 处理 PNG 的透明通道 (RGBA -> RGB) 或 灰度图 (L -> RGB)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    return img

def load_checkpoint(model, checkpoint_path, device):
    """加载权重，处理各种保存格式"""
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")

    print(f"Loading model from {checkpoint_path}...")
    # map_location 确保在没有 GPU 的机器上也能加载 GPU 训练的模型
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # 提取 state_dict
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    # 处理 DataParallel 保存时带有的 "module." 前缀
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v

    # 加载权重
    try:
        model.load_state_dict(new_state_dict, strict=True)
    except RuntimeError as e:
        print(f"[Warning] Strict loading failed: {e}")
        print("Retrying with strict=False...")
        model.load_state_dict(new_state_dict, strict=False)
    
    model.to(device)
    model.eval()
    return model

def visualize_results(pil_img, results):
    """使用 Matplotlib 显示图片和预测概率"""
    if not _HAS_MATPLOTLIB:
        print("[Info] Matplotlib not installed, skipping GUI visualization.")
        return

    # 准备数据
    labels = [r[0] for r in results][::-1] # 逆序以适配 barh 从下往上画
    probs = [r[1] * 100 for r in results][::-1]
    colors = ['gray'] * len(labels)
    colors[-1] = 'crimson' # Top-1 标红

    plt.figure(figsize=(10, 4))

    # 子图1：原图
    plt.subplot(1, 2, 1)
    plt.imshow(pil_img)
    plt.axis('off')
    plt.title("Input Image")

    # 子图2：概率条形图
    plt.subplot(1, 2, 2)
    bars = plt.barh(labels, probs, color=colors)
    plt.xlim(0, 100)
    plt.xlabel('Probability (%)')
    plt.title('Top Predictions')
    
    # 在柱状图上标数字
    for bar in bars:
        width = bar.get_width()
        plt.text(width + 1, bar.get_y() + bar.get_height()/2, 
                 f'{width:.1f}%', va='center', fontsize=9)

    plt.tight_layout()
    plt.show()

# ==========================================
# 3. 主逻辑
# ==========================================

def predict(args):
    # 1. 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 2. 准备图片
    try:
        pil_img = load_image(args.image)
    except Exception as e:
        print(f"Error loading image: {e}")
        return

    # 3. 初始化并加载模型
    model = SimpleCNN(num_classes=10)
    try:
        # 如果未指定模型路径，默认寻找当前目录下的 .pth
        if args.model is None:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            args.model = os.path.join(script_dir, 'simple_cnn_best.pth')
        
        load_checkpoint(model, args.model, device)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # 4. 推理
    preprocess = get_transforms()
    img_tensor = preprocess(pil_img).unsqueeze(0).to(device) # [1, 3, 32, 32]

    with torch.no_grad():
        logits = model(img_tensor)
        probs = F.softmax(logits, dim=1)
        
        # 获取 Top-K
        topk_probs, topk_indices = torch.topk(probs, args.topk)
        
        topk_probs = topk_probs.cpu().numpy()[0]
        topk_indices = topk_indices.cpu().numpy()[0]

    # 5. 整理结果
    results = []
    print(f"\nPrediction Results for '{os.path.basename(args.image)}':")
    print("-" * 40)
    print(f"{'Class':<15} | {'Probability':<10}")
    print("-" * 40)
    
    for i in range(len(topk_indices)):
        idx = topk_indices[i]
        prob = topk_probs[i]
        class_name = CIFAR10_CLASSES[idx]
        results.append((class_name, prob))
        
        # 终端打印
        print(f"{class_name:<15} | {prob*100:6.2f}%")
    print("-" * 40)

    # 6. 可视化
    if args.gui:
        visualize_results(pil_img, results)

def parse_args():
    parser = argparse.ArgumentParser(description='SimpleCNN Image Inference')
    parser.add_argument('--image', '-i', type=str, required=True,
                        help='Path to the input image file.')
    parser.add_argument('--model', '-m', type=str, default="simple_cnn_best.pth",
                        help='Path to the saved model checkpoint (default: simple_cnn_best.pth).')
    parser.add_argument('--topk', type=int, default=5, 
                        help='Number of top predictions to display (default: 5).')
    parser.add_argument('--gui', action='store_true', 
                        help='Show graphical visualization (requires matplotlib).')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    predict(args)