from pickle import GLOBAL
import torch

from bioseq_ml import modules


if __name__ == '__main__':
	dense = modules.DenseBlock(
		out_dims=[10, 20, 30],
		activation='RReLU',
		norm='batch',
		dropout=0.5,
	)

	x = torch.randn(8, 16)	# 8 sequences of length 16
	y = dense(x)

	print(y)
	print(y.shape)
