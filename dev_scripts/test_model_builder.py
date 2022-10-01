import torch

from bioseq_ml.utils import ModelBuilder
from bioseq_ml.models import BinaryClassifier


class DummyModel(BinaryClassifier):
	def __init__(self, **kwargs):
		super().__init__(**kwargs)
	
	def forward(self, x):
		return self.net(x)


class DummyModule(torch.nn.Module):
	def __init__(self, **kwargs):
		super().__init__()
		self.identity = torch.nn.Identity()
		self.kwargs = kwargs
	
	def forward(self, x):
		return self.identity(x) * 100


if __name__ == '__main__':
	model_spec = {
		'model_type': 'BinaryClassifier',
		'network_spec': [
			{
				'type': 'MultiSeqEmbedding',
				'kwargs': {
					'emb_scheme':{
						'seq': {
							'num_embeddings': 5,
							'embedding_dim': 10,
							'padding_idx': 4,
						},
						'part': {
							'num_classes': 2,
						},
					},
				},
			},
			{
				'type': 'ConvLayer',
				'kwargs': {
					'out_channels': 32,
					'kernel_size': 3,
					'padding': 'same',
					'norm': 'layer',
					'activation': 'GELU',
					'dropout': 0.05,
				},
			},
			{
				'type': 'ConvLayer',
				'kwargs': {
					'out_channels': 32,
					'kernel_size': 7,
					'padding': 'same',
					'norm': 'layer',
					'activation': 'RReLU',
					'pooling': 'max',
					'pooling_kwargs': {'kernel_size': 2},
				},
			},
			{
				'type': 'GlobalMaxPool',
			},
			{
				'type': 'DenseBlock',
				'kwargs': {
					'out_dims': [64, 32],
					'activation': 'ReLU',
					'dropout': 0.1,
				},
			},
			{
				'type': 'DenseBlock',
				'kwargs': {
					'out_dims': [1]
				},
			},
		]
	}

	model_builder = ModelBuilder()
	model = model_builder.build_model(model_spec)

	batch = {
		'seq': torch.randint(0, 4, (8, 24)),	# 8 sequences of length 24
		'part': torch.randint(0, 2, (8, 24)),
	}

	y = model(batch)
	print(y)
	print(y.shape)

	# Using custom models/modules and a module from torch.nn
	custom_model_spec = {
		'model_type': 'DummyModel',
		'network_spec': [
			{
				'type': 'MultiSeqEmbedding',
				'kwargs': {
					'emb_scheme':{
						'seq': {
							'num_embeddings': 5,
							'embedding_dim': 10,
							'padding_idx': 4,
						},
						'part': {
							'num_classes': 2,
						},
					},
				},
			},
			{
				'type': 'Flatten'
			},
			{
				'type': 'DummyModule',
				'kwargs': {
					'something': 1,
				},
			},
			{
				'type': 'DenseBlock',
				'kwargs': {
					'out_dims': [64, 32],
					'activation': 'ReLU',
					'dropout': 0.1,
				},
			},
			{
				'type': 'DenseBlock',
				'kwargs': {
					'out_dims': [1]
				},
			},
		]
	}

	custom_model_builder = ModelBuilder(
		custom_models=[DummyModel],
		custom_modules=[DummyModule]
	)
	custom_model = custom_model_builder.build_model(custom_model_spec)

	y = custom_model(batch)
	print(y)
	print(y.shape)