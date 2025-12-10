import torch
import torch.nn.functional as F


def show(name, t):
	print(f"{name}: shape={tuple(t.shape)}\n", t)


def example_relu():
	"""ReLU: max(0, x)"""
	x = torch.tensor([[-2.0, -1.0, 0.0, 1.0, 2.0]])
	out = F.relu(x)
	show('ReLU output', out)


def example_leaky_relu():
	"""LeakyReLU: x if x>0 else 0.01*x"""
	x = torch.tensor([[-2.0, -1.0, 0.0, 1.0, 2.0]])
	out = F.leaky_relu(x, negative_slope=0.01)
	show('LeakyReLU output', out)


def example_sigmoid():
	"""Sigmoid: 1/(1+e^-x)"""
	x = torch.tensor([[-2.0, -1.0, 0.0, 1.0, 2.0]])
	out = torch.sigmoid(x)
	show('Sigmoid output', out)


def example_tanh():
	"""Tanh: 双曲正切"""
	x = torch.tensor([[-2.0, -1.0, 0.0, 1.0, 2.0]])
	out = torch.tanh(x)
	show('Tanh output', out)


def example_softmax():
	"""Softmax across last dim"""
	x = torch.tensor([[1.0, 2.0, 3.0]])
	out = F.softmax(x, dim=1)
	show('Softmax output', out)


def example_gelu():
	"""GELU：近似实现/调用"""
	x = torch.tensor([[-2.0, -1.0, 0.0, 1.0, 2.0]])
	try:
		out = F.gelu(x)
	except Exception:
		# 兼容旧版 torch
		out = 0.5 * x * (1 + torch.tanh(torch.sqrt(torch.tensor(2.0 / 3.141592653589793)) * (x + 0.044715 * x ** 3)))
	show('GELU output', out)


def example_silu_swish():
	"""SiLU/Swish: x * sigmoid(x)"""
	x = torch.tensor([[-2.0, -1.0, 0.0, 1.0, 2.0]])
	try:
		out = F.silu(x)
	except Exception:
		out = x * torch.sigmoid(x)
	show('SiLU/Swish output', out)


def example_mish():
	"""Mish: x * tanh(softplus(x))"""
	x = torch.tensor([[-2.0, -1.0, 0.0, 1.0, 2.0]])
	try:
		out = F.mish(x)
	except Exception:
		out = x * torch.tanh(F.softplus(x))
	show('Mish output', out)


if __name__ == '__main__':
	print('\n=== ReLU 示例 ===')
	example_relu()

	print('\n=== LeakyReLU 示例 ===')
	example_leaky_relu()

	print('\n=== Sigmoid 示例 ===')
	example_sigmoid()

	print('\n=== Tanh 示例 ===')
	example_tanh()

	print('\n=== Softmax 示例 ===')
	example_softmax()

	print('\n=== GELU 示例 ===')
	example_gelu()

	print('\n=== SiLU/Swish 示例 ===')
	example_silu_swish()

	print('\n=== Mish 示例 ===')
	example_mish()
