import torch
import torch.nn as nn
import torch.nn.functional as F

def show(name, t):
    # 对于标量（loss通常是标量），打印 .item() 会更直观，但为了保持格式一致，这里保留原样
    print(f"{name}: shape={tuple(t.shape)}\n", t)
    # 如果想看具体数值，可以额外打印：
    # print(f"Value: {t.item():.4f}")


def example_mse():
    """均方误差（MSELoss）：(预测值 - 真实值)^2 的平均值。回归问题最常用。"""
    # 假设 batch_size=2, 特征维度=3
    pred = torch.randn(2, 3, requires_grad=True)
    target = torch.randn(2, 3)
    
    criterion = nn.MSELoss()
    loss = criterion(pred, target)
    
    show('MSELoss output', loss)


def example_l1():
    """平均绝对误差（L1Loss / MAE）：|预测值 - 真实值| 的平均值。对异常值更鲁棒。"""
    pred = torch.randn(2, 3, requires_grad=True)
    target = torch.randn(2, 3)
    
    criterion = nn.L1Loss()
    loss = criterion(pred, target)
    
    show('L1Loss output', loss)


def example_crossentropy():
    """交叉熵损失（CrossEntropyLoss）：用于多分类问题。
    输入 logits (未经过 softmax)，目标是类别索引 (Class Index)。
    """
    # 假设 batch_size=2, 共有 3 个类别
    # pred 不需要做 softmax，PyTorch 内部会做
    pred_logits = torch.randn(2, 3, requires_grad=True) 
    # target 是类别的索引，比如第一个样本属于类别 0，第二个属于类别 2
    target = torch.tensor([0, 2], dtype=torch.long)
    
    criterion = nn.CrossEntropyLoss()
    loss = criterion(pred_logits, target)
    
    show('CrossEntropyLoss output', loss)


def example_bce():
    """二元交叉熵（BCELoss）：用于二分类问题。
    输入必须是概率值 (0~1之间)，通常需先经过 Sigmoid。
    """
    # 假设 batch_size=2，输出为单个概率值
    # 先生成随机数，再通过 Sigmoid 压缩到 0-1 之间
    pred_logits = torch.randn(2, 1)
    pred_probs = torch.sigmoid(pred_logits) 
    
    # target 必须是 float 类型，且值为 0 或 1
    target = torch.tensor([[1.0], [0.0]])
    
    criterion = nn.BCELoss()
    loss = criterion(pred_probs, target)
    
    show('BCELoss output', loss)


def example_bce_with_logits():
    """带 Logits 的二元交叉熵（BCEWithLogitsLoss）：
    结合了 Sigmoid + BCELoss。数值更稳定，推荐代替 BCELoss 使用。
    """
    pred_logits = torch.randn(2, 1, requires_grad=True)
    target = torch.tensor([[1.0], [0.0]])
    
    criterion = nn.BCEWithLogitsLoss()
    loss = criterion(pred_logits, target)
    
    show('BCEWithLogitsLoss output', loss)


def example_smooth_l1():
    """Smooth L1 Loss (Huber Loss)：
    结合了 MSE 和 L1 的优点。误差小的时候是 MSE，误差大的时候是 L1。
    常用于目标检测（回归边界框）。
    """
    pred = torch.randn(2, 4, requires_grad=True)
    target = torch.randn(2, 4)
    
    criterion = nn.SmoothL1Loss()
    loss = criterion(pred, target)
    
    show('SmoothL1Loss output', loss)


if __name__ == '__main__':
    print('\n=== 均方误差 (MSE) 示例 ===')
    example_mse()

    print('\n=== 平均绝对误差 (L1) 示例 ===')
    example_l1()

    print('\n=== 交叉熵损失 (多分类) 示例 ===')
    example_crossentropy()

    print('\n=== 二元交叉熵 (BCELoss) 示例 ===')
    example_bce()

    print('\n=== 带 Logits 的二元交叉熵 (推荐) 示例 ===')
    example_bce_with_logits()

    print('\n=== Smooth L1 Loss (目标检测常用) 示例 ===')
    example_smooth_l1()