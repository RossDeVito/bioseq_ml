"""Base pytorch-lightning style data module class."""

import pandas as pd
from torch.utils.data import DataLoader
import pytorch_lightning as pl


class CSVDataModule(pl.LightningDataModule):
	"""Base pytorch-lightning style data module class.
	
	Designed to load all data from a CSV (or other file readable with
	pd.read_csv), split it based on a 0 (train), 1 (validation), or 2 (test)
	value in the 'split_col' column, and then create the corresponding
	torch.utils.data.Datasets and Dataloaders. This should be able to work
	with most custom Dataset classes that can be initialized by passing the
	corresponding pandas DataFrame to the __init__ method along with keyword
	arguments.

	Will shuffle training data, but not validation or test.

	Args:
		data_path: Path to CSV data file.
		read_csv_kwargs (dict): Additional keyword arguments to pass to
			pd.read_csv to load data.
		split_col (str): Column name in data to use for splitting data into
			train, validation, and test sets.
		dataset_class (type): Dataset class to use for creating Datasets.
		dataset_kwargs (dict): Keyword arguments to pass to dataset_class
			__init__ method in addition to the DataFrame containing sample
			data (which will be the first positional arg).
		testval_dataset_kwargs (dict, default None): Keyword arguments to
			pass to dataset_class for validation and test Datasets. If None,
			will use dataset_kwargs. Useful for stuff like data augmentation.
		collate_fn (callable, optional): Function to use to collate data
			batches. Defaults to None.
		batch_size (int, default 32): Batch size to use for Dataloaders.
		num_workers (int, default 0): Number of workers to use for Dataloaders.
	"""

	def __init__(
		self,
		data_path,
		read_csv_kwargs={},
		split_col='split',
		dataset_class=None,
		dataset_kwargs={},
		testval_dataset_kwargs=None,
		collate_fn=None,
		batch_size=32,
		num_workers=0,
	):
		if dataset_class is None:
			raise ValueError('Must provide dataset_class')

		super().__init__()
		self.data_path = data_path
		self.read_csv_kwargs = read_csv_kwargs
		self.split_col = split_col
		self.dataset_class = dataset_class
		self.dataset_kwargs = dataset_kwargs
		self.testval_dataset_kwargs = testval_dataset_kwargs
		self.collate_fn = collate_fn
		self.batch_size = batch_size
		self.num_workers = num_workers

		# If testval_dataset_kwargs is None, use dataset_kwargs
		if testval_dataset_kwargs is None:
			self.testval_dataset_kwargs = dataset_kwargs

	def setup(self, stage=None):
		all_data_df = pd.read_csv(self.data_path, **self.read_csv_kwargs)

		self.train_dataset = self.dataset_class(
			all_data_df[all_data_df[self.split_col] == 0].reset_index(),
			**self.dataset_kwargs,
		)
		self.val_dataset = self.dataset_class(
			all_data_df[all_data_df[self.split_col] == 1].reset_index(),
			**self.testval_dataset_kwargs,
		)
		self.test_dataset = self.dataset_class(
			all_data_df[all_data_df[self.split_col] == 2].reset_index(),
			**self.testval_dataset_kwargs,
		)

	def train_dataloader(self):
		return DataLoader(
			self.train_dataset,
			batch_size=self.batch_size,
			shuffle=True,
			num_workers=self.num_workers,
			collate_fn=self.collate_fn,
		)

	def val_dataloader(self):
		return DataLoader(
			self.val_dataset,
			batch_size=self.batch_size,
			shuffle=False,
			num_workers=self.num_workers,
			collate_fn=self.collate_fn,
		)

	def test_dataloader(self):
		return DataLoader(
			self.test_dataset,
			batch_size=self.batch_size,
			shuffle=False,
			num_workers=self.num_workers,
			collate_fn=self.collate_fn,
		)


