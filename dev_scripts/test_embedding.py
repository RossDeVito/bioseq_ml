import torch

from bioseq_ml import modules, ModelBuilder


if __name__ == '__main__':

	emb_module = modules.SeqEmbedding(
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

	# Test multi-input wrapper with embeddings for multiple inputs

	ms_emb = modules.MultiSeqEmbedding(
		input_map={
			'pre': {'pre_seq': 'seq', 'pre_part': 'part'},
			'post': {'post_seq': 'seq', 'post_part': 'part'},
		},
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
	common_fn = modules.MultiInputApply(
		[{"type": "Identity"}],
		ModelBuilder()
	)
	concat = modules.Concat(keys=['pre', 'post'])

	batch = {
		'pre_seq': torch.randint(0, 4, (8, 24)),	# 8 sequences of length 24
		'pre_part': torch.randint(0, 2, (8, 24)),
		'post_seq': torch.randint(0, 4, (8, 24)),	# 8 sequences of length 24
		'post_part': torch.randint(0, 2, (8, 24)),
	}

	emb = concat(common_fn(ms_emb(batch)))

	print(batch)
	print(emb)
	print(emb.shape)

