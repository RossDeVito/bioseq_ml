"""Input embedding modules."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class OneHotEncoding(nn.Module):
	"""One-hot encoding module for int sequence input.
	
	Args:
		num_classes: Number of classes for one-hot encoding.
		output_fmt: Output format for embedding. Either
			'channels_last', 'channels_first', or None (default). None
			is effectively 'channels_first'.
	"""

	def __init__(self, num_classes: int, output_fmt=None):
		super().__init__()
		self.num_classes = num_classes
		self.output_fmt = output_fmt

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		"""Forward pass for one-hot encoding.
		
		Args:
			x: Input tensor of shape (batch_size, seq_len).
		
		Returns:
			One-hot encoded tensor of shape (batch_size, seq_len, num_classes).
		"""
		emb = F.one_hot(x, num_classes=self.num_classes)
		if self.output_fmt == 'channels_last':
			emb = emb.permute(0, 2, 1)
		return emb

	def get_output_dim(self) -> int:
		"""
		Returns:
			Output dimension of embedding.
		"""
		return self.num_classes


class SeqEmbedding(nn.Module):
	"""
	Embedding input module for input consisting of one or more equal length
	int sequence channels to be seperatesly embedded and stacked.

	Args:
		emb_scheme (dict): Dictionary defining what keys are used for
			embedding and how they are embedded. Keys of dict are the
			keys of the input dict of the model to embed. The value for
			each key is a dict with kwargs to nn.Embedding or the key
			'num_classes' for OneHotEncoding. If using OneHotEncoding,
			the 'output_fmt' key should not be used, as it will be handled
			by this module.
		output_fmt (str): Output format, either default 'channels_last' or
			'channels_first'.
	"""

	def __init__(self, emb_scheme, output_fmt='channels_last'):
		super().__init__()
		self.emb_scheme = emb_scheme
		assert output_fmt in ['channels_last', 'channels_first']
		self.output_fmt = output_fmt

		self.embeddings = nn.ModuleDict()
		for key, emb_kwargs in self.emb_scheme.items():
			if 'num_classes' in emb_kwargs:
				if 'output_fmt' in emb_kwargs:
					raise ValueError(
						'output_fmt should not be specified for '
						'OneHotEncoding kwargs when using MultiSeqEmbedding. '
						'Use output_fmt argument of MultiSeqEmbedding instead.'
					)
				self.embeddings[key] = OneHotEncoding(**emb_kwargs)
			else:
				self.embeddings[key] = nn.Embedding(**emb_kwargs)

	def forward(self, x):
		"""
		Args:
			x (dict): Dictionary of input sequences to embed.

		Returns:
			emb (torch.Tensor): Embedded input.
		"""
		emb = []
		for emb_key, emb_module in self.embeddings.items():
			emb.append(emb_module(x[emb_key]))
		emb = torch.cat(emb, dim=-1)

		if self.output_fmt == 'channels_last':
			emb = emb.permute(0, 2, 1)
		
		return emb

	def get_output_dim(self):
		"""
		Returns:
			output_dim (int): Output dimension of embedding.
		"""
		output_dim = 0
		
		for emb_kwargs in self.emb_scheme.values():
			if 'num_classes' in emb_kwargs:
				output_dim += emb_kwargs['num_classes']
			else:
				output_dim += emb_kwargs['embedding_dim']

		return output_dim


class MultiSeqEmbedding(nn.Module):
	"""
	Module for embedding multiple inputs with the same SeqEmbedding type
	embedding.

	Args:
		input_map (dict of dicts): The output of this module will be a dict
			mapping the keys from this dict to the embeddings definded by the
			dicts that are the values. These inner dicts map keys from the
			input dict to input type names, which will be used to map them to
			the propper embedding. The input type names should match the keys
			of emb_scheme.
		emb_scheme (dict): Dictionary defining what how different input type
			names are embedded. Keys of dict are the input type names to embed.
			The value for each key is a dict with kwargs to nn.Embedding or
			the key 'num_classes' for OneHotEncoding. If using OneHotEncoding,
			the 'output_fmt' key should not be used, as it will be handled
			by this module.
		output_fmt (str): Output format, either default 'channels_last' or
			'channels_first'.

	Returns:
		Dict mapping keys of input map to the embedded input.
	"""

	def __init__(self, input_map, emb_scheme, output_fmt='channels_last'):
		super().__init__()
		self.input_map = input_map
		self.output_fmt = output_fmt
		self.embedding = SeqEmbedding(emb_scheme, output_fmt=output_fmt)

	def forward(self, x):
		emb = {}

		for key, emb_map in self.input_map.items():
			input_dict = {v: x[k] for k, v in emb_map.items()}
			emb[key] = self.embedding(input_dict)

		return emb

	def get_output_dim(self):
		return self.embedding.get_output_dim()