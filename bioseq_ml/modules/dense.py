"""Dense fully-connected feed-forward neural network modules."""

import torch
import torch.nn as nn


class DenseBlock(nn.Module):
	"""Block of dense feed-forward layers.
	
	Args:
		out_dims (list of ints): List of output dimensions for each layer.
		bias (bool): If layers should use bias. Default True.
		activation (str or None): Activation function to use. Must be a
			name of an activation function class in torch.nn. If None,
			no activation function is used. Default None.
		dropout (float, default 0.0): Dropout probability. If <= 0, no
			dropout is used.
		norm (str or None): Normalization layer to use. Must be None, 'layer',
			or 'batch'. Default None.
	"""

	def __init__(
			self,
			out_dims,
			bias=True,
			activation=None,
			dropout=0.0,
			norm=None
		):
		super().__init__()
		self.out_dims = out_dims
		self.bias = bias
		self.activation = activation
		self.dropout = dropout

		self.layers = nn.ModuleList()
		for out_dim in self.out_dims:
			self.layers.append(nn.LazyLinear(out_dim, bias=self.bias))
			if self.activation is not None:
				self.layers.append(getattr(nn, self.activation)())
			if self.dropout > 0:
				self.layers.append(nn.Dropout(self.dropout))
			if norm == 'layer':
				self.layers.append(nn.LayerNorm(out_dim))
			elif norm == 'batch':
				self.layers.append(nn.BatchNorm1d(out_dim))
			elif norm is not None:
				raise ValueError('Invalid norm: {}'.format(norm))

		self.layers = nn.Sequential(*self.layers)

	def get_output_dim(self):
		"""Returns the output dimension of the module.

		Returns:
			int: Output dimension.
		"""
		return self.out_dims[-1]

	def forward(self, x):
		return self.layers(x)