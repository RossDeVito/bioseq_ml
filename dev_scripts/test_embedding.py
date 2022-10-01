import torch

from bioseq_ml import modules


if __name__ == '__main__':

	emb_module = modules.MultiSeqEmbedding(
		emb_scheme={
			'seq': {
				'num_embeddings': 5,
				'embedding_dim': 10,
				'padding_idx': 4,
			},
			'part': {
				'num_classes': 2,
			},
		},
		output_fmt='channels_last',
	)

	batch = {
		'seq': torch.randint(0, 4, (8, 24)),	# 8 sequences of length 24
		'part': torch.randint(0, 2, (8, 24)),
	}

	emb = emb_module(batch)

	print(batch)
	print(emb)
	print(emb.shape)

	print(emb_module.get_output_dim())