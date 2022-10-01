"""Simple dataset for toy example."""

import pandas as pd
import torch
from torch.utils.data import Dataset

from create_data import BASE_TO_INT


class ToyDataset(Dataset):
	"""Dataset for toy example.
	
	Args:
		data (pd.DataFrame): Dataframe containing data.
		seq_col (str): Column name containing sequences.
		label_col (str): Column name containing target values.
		return_data (bool, default False): If True, will return the
			original data row along with the sequence and label.
	"""

	def __init__(
		self,
		data,
		seq_col,
		label_col,
		return_data=False,
	):
		self.data = data
		self.seq_col = seq_col
		self.label_col = label_col
		self.return_data = return_data
		
		self.base_to_int = BASE_TO_INT
	
		# Convert sequences to integers
		self.seqs = self.data[self.seq_col].map(
			lambda x: torch.tensor([self.base_to_int[base] for base in x])
		).to_dict()

	def __len__(self):
		return len(self.data)

	def __getitem__(self, idx):
		seq = self.seqs[idx]
		label = self.data.loc[idx, self.label_col]
		if self.return_data:
			return {
				'seq': seq,
				'label': label,
				'data': self.data.loc[idx],
			}
		else:
			return {
				'seq': seq,
				'label': label,
			}


def toy_collate_fn(batch):
	"""Collate function for toy example.
	
	Args:
		batch (list): List of dicts from ToyDataset.__getitem__.

	Returns:
		batch (dict): With keys:
			'input' (dict): Batched sequences with key 'seq'.
			'label': Batched labels.
			'data': Optional list of DataFrame rows if return_data
				is True in ToyDataset.
	"""
	batch = {key: [item[key] for item in batch] for key in batch[0]}
	
	ret_dict = dict()
	ret_dict['input'] = {
		'seq': torch.nn.utils.rnn.pad_sequence(batch['seq'], batch_first=True).long()
	}
	ret_dict['label'] = torch.tensor(batch['label']).long()
	if 'data' in batch:
		ret_dict['data'] = batch['data']
	return ret_dict


if __name__ == '__main__':
	data_path = 'data/data_sl32_nmotifs24.csv'
	data = pd.read_csv(data_path)
	dataset = ToyDataset(data, 'seq', 'label', return_data=True)

	# Test dataset
	batch_list = [dataset[i] for i in range(100, 164)]
	batch = toy_collate_fn(batch_list)

	print(batch['input']['seq'])
	print(batch['input']['seq'].shape)
