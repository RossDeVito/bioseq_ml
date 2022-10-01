"""Utility modules and functions."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Transpose(nn.Module):
	"""Module for transposing dimentions.
	
	Args:
		dim0 (int): First dimention to be transposed.
		dim1 (int): Second dimension to be transposed.
	"""

	def __init__(self, dim0, dim1):
		super().__init__()
		self.dim0 = dim0
		self.dim1 = dim1

	def forward(self, x):
		return x.transpose(self.dim0, self.dim1)


class Concat(nn.Module):
	"""Module for concatenating input tensors along a given dimension.
	
	Args:
		dim (int, default -1): Dimension to concatenate along.
		keys (list, default None): If None, input should be a list of tensors
			they will be concated by calling torch.cat. If not None, input
			should be a dict with keys in keys, and the values will be
			concatenated in the order of keys.
	"""

	def __init__(self, dim=-1, keys=None):
		super().__init__()
		self.dim = dim
		self.keys = keys

	def forward(self, x):
		if self.keys is None:
			return torch.cat(x, dim=self.dim)
		else:
			return torch.cat([x[k] for k in self.keys], dim=self.dim)