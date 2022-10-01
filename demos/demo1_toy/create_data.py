"""Creates artifical data for the toy example and saves as csv.

The data is created by generating random sequences of length 32, and then
inserting motifs that contribute to the binary way. Motifs will be generated
randomly then assigned some value. If the sum of the values is positive, the
sequence is assigned a 1, otherwise it is assigned a 0. Motif values are
saved as a seperate JSON file.

The data is additionally split into train, val, and test sets by setting values
in the 'split' column. 0 = train, 1 = val, 2 = test.
"""

import json

import numpy as np
import pandas as pd


BASE_TO_INT = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
INT_TO_BASE = {v: k for k, v in BASE_TO_INT.items()}


if __name__ == '__main__':
	# Set random seed for reproducibility
	np.random.seed(36)

	# Set parameters (ranges inclusive)
	num_seqs = 30000
	seq_len = 32
	num_motifs = 48
	max_motifs_per_seq = 3
	motif_len_range = (5, 9)
	scale_val_by_len = True
	motif_val_range = (-10, 10)
	split_prop = (0.8, 0.1, 0.1)

	# Generate motifs and values
	motifs = set()

	while len(motifs) < num_motifs:
		motif_len = np.random.randint(motif_len_range[0], motif_len_range[1] + 1)
		motif = np.random.choice(list(BASE_TO_INT.keys()), motif_len)
		motifs.add(''.join(motif))

	motifs = list(motifs)
	motif_vals = np.random.uniform(motif_val_range[0], motif_val_range[1], num_motifs)
	
	if scale_val_by_len:
		motif_lens = np.array([len(motif) for motif in motifs])
		motif_vals = (motif_vals * motif_lens) / np.max(motif_lens)

	motif_dict = dict(zip(motifs, motif_vals))
	motif_dict = dict(sorted(motif_dict.items(), key=lambda item: item[1]))

	# Generate sequences
	seqs = set()

	while len(seqs) < num_seqs:
		seq = np.random.choice(list(BASE_TO_INT.keys()), seq_len)
		seqs.add(''.join(seq))

	seqs = list(seqs)

	# Insert motifs into sequences and assign values and labels
	seqs_with_motifs = []

	for seq in seqs:
		sample_num_motifs = np.random.randint(1, max_motifs_per_seq + 1)
		motifs = np.random.choice(list(motif_dict.keys()), sample_num_motifs, replace=False)

		sample_dict = dict()
		total_val = 0

		for i in range(max_motifs_per_seq):
			sample_dict[f'motif{i}'] = motifs[i] if i < sample_num_motifs else ''
			sample_dict[f'motif{i}_val'] = motif_dict[motifs[i]] if i < sample_num_motifs else 0
			total_val += sample_dict[f'motif{i}_val']

		for motif in motifs:
			start_idx = np.random.randint(0, seq_len - len(motif) + 1)
			seq = seq[:start_idx] + motif + seq[start_idx + len(motif):]

		sample_dict['seq'] = seq
		sample_dict['total_val'] = total_val
		sample_dict['label'] = 1 if total_val > 0 else 0
		sample_dict['split'] = np.random.choice([0, 1, 2], p=split_prop)

		seqs_with_motifs.append(sample_dict)

	data = pd.DataFrame(seqs_with_motifs)

	# Save data
	data.to_csv(f'data/data_sl{seq_len}_nmotifs{num_motifs}.csv', index=False)
	with open(f'data/motif_dict_sl{seq_len}_nmotifs{num_motifs}.json', 'w') as f:
		json.dump(motif_dict, f, indent=4)

	