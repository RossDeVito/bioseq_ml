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