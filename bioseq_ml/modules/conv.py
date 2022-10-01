"""1D convolutional neural network modules."""

import torch
import torch.nn as nn

from bioseq_ml.modules import Transpose


class ConvLayer(nn.Module):
	"""
	1D Convolutional Layer with optional pooling, normalization, 
	activation, and dropout.

	Args:
		out_channels (int): Number of output channels
		kernel_size: Size of the convolving kernel.
		padding: Zero-padding added to both sides of the input. Default: 'same'.
		norm: Normalization layer. One of 'layer' (default), 'batch', or None.
		norm_kwargs:
		activation (str): Activation function class name in torch.nn or None.
		pooling: Pooling type. None (default), 'max', 'avg', or 'lp'.
		pooling_kwargs: Additional keyword args to pass to pooling.
		dropout: Dropout probability. Default: 0.0.
		**kwargs: Additional keyword arguments passed to nn.Conv1D.
	"""

	def __init__(self, 
		out_channels,
		kernel_size,
		padding='same',
		norm=None,
		norm_kwargs={},
		activation=None,
		pooling=None,
		pooling_kwargs={},
		dropout=0.0,
		**kwargs
	):
		super().__init__()
		self.out_channels = out_channels

		# Create convolution
		self.conv = nn.LazyConv1d(
			out_channels=out_channels,
			kernel_size=kernel_size,
			padding=padding,
			**kwargs
		)

		self.additional_ops = nn.ModuleList()

		# Add normalization
		if norm == 'layer':
			self.additional_ops.append(
				Transpose(1,2)
			)
			self.additional_ops.append(
				nn.LayerNorm(out_channels, **norm_kwargs)
			)
			self.additional_ops.append(
				Transpose(1,2)
			)
		elif norm == 'batch':
			self.additional_ops.append(
				nn.BatchNorm1d(out_channels, **norm_kwargs)
			)
		elif norm is not None:
			raise ValueError

		# Add activation
		if isinstance(activation, str):
			self.additional_ops.append(
				getattr(nn, activation)()
			)
		elif activation is not None:
			raise ValueError

		# Add pooling
		if pooling == 'max':
			self.additional_ops.append(
				nn.MaxPool1d(**pooling_kwargs)
			)
		elif pooling == 'avg':
			self.additional_ops.append(
				nn.AvgPool1d(**pooling_kwargs)
			)
		elif pooling == 'lp':
			self.additional_ops.append(
				nn.LPPool1d(**pooling_kwargs)
			)
		elif pooling is not None:
			raise ValueError

		# Add dropout
		if dropout > 0:
			self.additional_ops.append(
				nn.Dropout1d(dropout)
			)

		# Make additional operations one sequential module
		self.additional_ops = nn.Sequential(*self.additional_ops)

	def forward(self, x):
		return self.additional_ops(self.conv(x))

	def get_output_dim(self):
		return self.out_channels