import torch
import torch.nn as nn


def show(name, t):
    print(f"{name}: shape={tuple(t.shape)}\n", t)


def example_standard():
    """标准卷积：滑动窗口相乘求和；torch -> Conv2d"""
    x = torch.randn(1, 1, 5, 5)
    conv = nn.Conv2d(1, 1, kernel_size=3, padding=1)
    out = conv(x)
    show('Standard Conv2d output', out)


def example_dilated():
    """空洞卷积（dilated）：扩大感受野，不增加核参数。"""
    x = torch.randn(1, 1, 7, 7)
    conv = nn.Conv2d(1, 1, kernel_size=3, dilation=2, padding=2)
    out = conv(x)
    show('Dilated Conv2d output', out)


def example_depthwise():
    """深度卷积（groups=in_channels）：每通道独立卷积（常用于 MobileNet）"""
    C = 3
    x = torch.randn(1, C, 5, 5)
    conv = nn.Conv2d(C, C, kernel_size=3, padding=1, groups=C)
    out = conv(x)
    show('Depthwise Conv2d output', out)


def example_separable():
    """可分离卷积 = depthwise + pointwise (1x1)。用 sequential 演示。"""
    C_in, C_out = 3, 6
    x = torch.randn(1, C_in, 5, 5)
    depthwise = nn.Conv2d(C_in, C_in, kernel_size=3, padding=1, groups=C_in)
    pointwise = nn.Conv2d(C_in, C_out, kernel_size=1)
    model = nn.Sequential(depthwise, pointwise)
    out = model(x)
    show('Separable (depthwise+pointwise) output', out)


def example_grouped():
    """分组卷积：将通道分成 G 组并行卷积。"""
    in_ch, out_ch, groups = 8, 8, 4
    x = torch.randn(1, in_ch, 6, 6)
    conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, groups=groups)
    out = conv(x)
    show('Grouped Conv2d output', out)


def example_transpose():
    """转置卷积（上采样）：ConvTranspose2d 示例"""
    x = torch.randn(1, 1, 3, 3)
    deconv = nn.ConvTranspose2d(1, 1, kernel_size=4, stride=2, padding=1)
    out = deconv(x)
    show('ConvTranspose2d output', out)


if __name__ == '__main__':
    print('\n=== 标准卷积示例 ===')
    example_standard()

    print('\n=== 空洞卷积示例 ===')
    example_dilated()

    print('\n=== 深度卷积示例 ===')
    example_depthwise()

    print('\n=== 可分离卷积示例 ===')
    example_separable()

    print('\n=== 分组卷积示例 ===')
    example_grouped()

    print('\n=== 转置卷积示例 ===')
    example_transpose()
