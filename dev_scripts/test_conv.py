import torch

from bioseq_ml import modules


if __name__ == '__main__':
	conv = modules.ConvLayer(
		out_channels=10,
		kernel_size=3,
		padding='same',
		norm='layer',
		activation='GELU',
		pooling='max',
		pooling_kwargs={'kernel_size': 2},
		dropout=0.5,
	)

	x = torch.randn(8, 16, 24)	# 8 sequences of length 24 with 16 channels

	y = conv(x)

	print(y)
	print(y.shape)
