"""Pooling layers, especially global pooling."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GlobalMaxPool(nn.Module):
	"""Global Max Pooling
	
	Args:
		dim (int, default -1): Axis to pool along.
	"""

	def __init__(self, dim=-1):
		super().__init__()
		self.dim = dim

	def forward(self, x):
		return x.max(dim=self.dim).values

class GlobalAvgPool(nn.Module):
	"""Global Average Pooling
	
	Args:
		dim (int, default -1): Axis to pool along.
	"""

	def __init__(self, dim=-1):
		super().__init__()
		self.dim = dim

	def forward(self, x):
		return x.mean(dim=self.dim)

class GlobalLPPool(nn.Module):
	"""Global LP Pooling
	
	Args:
		p (int, default 2): Order of the norm.
		dim (int, default -1): Axis to pool along.
		scale (bool, default False): Whether to scale the output by the number
			of elements in the pool (length of the sequence).
	"""

	def __init__(self, p=2, dim=-1, scale=False):
		super().__init__()
		self.p = p
		self.dim = dim
		self.scale = scale

	def forward(self, x):
		pow_avg = F.lp_pool1d(
			x, norm_type=self.p, kernel_size=x.shape[self.dim]
		).squeeze()
		if self.scale:
			return pow_avg / x.shape[self.dim]
		else:
			return pow_avg