# 入门：简单卷积与最小 CNN 示例

本项目为初学者准备，重点是用最简单、可读的代码帮助理解基本卷积与小型卷积网络的工作原理。

快速总结（项目概览）

本仓库面向入门读者，包含教学性质的最小示例与参考实现，目的是让初学者快速理解常见卷积变体与卷积网络的训练/推理流程。代码以清晰、可运行为目标，而非性能优化或工程化部署。

项目结构与关键文件：

- `基础知识/`：包含概念示例和小脚本
	- `conv.py` / `卷积.py`：PyTorch 对应的卷积示例（标准、空洞、深度可分离、分组、转置等）。
	- `activation_functions.py`：常用激活函数的 numpy 实现与说明。
	- `损失函数实例.py`：常见损失（MSE、MAE、BCE、CrossEntropy 等）的示例。
	- `dataset_cifar100.py`：CIFAR-100 数据加载与示例显示工具。

- `简单神经网络/`：最小可训练示例与推理脚本
	- `simple_cnn.py`：小型卷积神经网络（适配 CIFAR 大小），包含训练循环（10 轮）与验证，训练期间会将最佳模型保存为 `简单神经网络/simple_cnn_best.pth`。
	- `predict_image.py`：载入保存的模型对单张图片进行推理（支持 Top-K、可视化）。

快速开始（Windows PowerShell）

```powershell
cd f:\githubzyuu\conv\简单神经网络
python -m pip install --upgrade pip
# 安装 CPU 版或根据你的 CUDA 版本安装对应的 PyTorch（见 https://pytorch.org/ ）
pip install torch torchvision pillow
pip install matplotlib   # 可选：用于可视化推理结果
```

训练（示例）

```powershell
python simple_cnn.py
```

说明：脚本会在首次运行时下载 CIFAR-10 数据集，训练 10 个 epoch，并在验证集精度提升时把最佳模型写入 `simple_cnn_best.pth`。

推理（对单张图片）

```powershell
python predict_image.py -i f:\path\to\your_image.jpg
# 指定模型、Top-K 和可视化：
python predict_image.py -i f:\path\to\your_image.jpg -m f:\models\my_best.pth --topk 3 --gui
```
💡 总结：神经网络就像搭积木
其实，深度学习的本质非常有意思，它就像是在搭积木。

在这个项目中，你所接触到的每一个概念都是一块基础的积木：

卷积 (Convolution) 是提取图像特征的“积木块”。

激活函数 (Activation) 是给网络注入灵性（非线性能力）的“连接件”。

损失函数 (Loss) 是一把告诉我们搭得直不直的“尺子”。

我们写代码的过程，就是先将这些基础的算子（Operator）搭建成一个个小的模块（Module），再将这些模块按照设计图纸拼装起来，最终就变成了一个功能强大的神经网络。

