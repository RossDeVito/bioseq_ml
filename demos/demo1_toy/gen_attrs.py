"""Generate attributions, dealing with embedding layer."""

import os
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from captum import attr
import logomaker as lm

from bioseq_ml import ModelBuilder
from bioseq_ml.data_modules import CSVDataModule

from dataset import ToyDataset, toy_collate_fn

if __name__ == '__main__':
	__spec__ = None

	# model_save_dir = 'training_output/toy_24/version_3'
	# state_dict_rel_path = 'checkpoints/epoch=52-best_val_loss.ckpt'
	model_save_dir = 'training_output/toy_48/version_1'
	state_dict_rel_path = 'checkpoints/epoch=50-best_val_loss.ckpt'
	
	# Load model state dict (weights)
	state_dict = torch.load(os.path.join(model_save_dir, state_dict_rel_path))['state_dict']

	# Load model spec
	with open(os.path.join(model_save_dir, 'model_spec.json'), 'r') as f:
		model_spec = json.load(f)
	
	# Create model from model_spec
	model_builder = ModelBuilder()

	model = model_builder.build_model(model_spec)
	model.load_state_dict(state_dict)
	model.eval()

	# Seperate embedding layer from model
	embedding_layer = model.net.pop(0)
	
	# Load data
	data_spec_path = os.path.join(model_save_dir, 'data_spec.json')
	with open(data_spec_path, 'r') as f:
		data_spec = json.load(f)
	data_spec['dataset_kwargs']['return_data'] = True
	data_spec['batch_size'] = 2024

	data_module = CSVDataModule(
		dataset_class=ToyDataset,
		collate_fn=toy_collate_fn,
		**data_spec,
	)
	data_module.setup()
	
	# Generate attributions
	test_data = data_module.test_dataloader()
	batch = next(iter(test_data))

	# Generate input embeddings
	input_emb = embedding_layer(batch['input'])

	ig = attr.IntegratedGradients(model)#, multiply_by_inputs=True)
	attrs, approx_error = ig.attribute(
		input_emb,
		# no baseline will use zeros as baseline
		internal_batch_size=256,
		return_convergence_delta=True,
	)
	preds = model(input_emb).detach().numpy().flatten()

	# Get all data into one dataframe
	attr_df = pd.DataFrame(batch['data'])
	attr_df['pred'] = preds
	attr_df['mean_ig'] = attrs.mean(axis=1).tolist()

	attr_df.sort_values('pred', inplace=True)

	# Plot underlying score vs prediction with vertical line at pred (y) = 0.5
	# and horizontal line at score (x) = 0
	sns.scatterplot(data=attr_df, x='total_val', y='pred')
	plt.axvline(0, color='red')
	plt.axhline(0.5, color='red')
	plt.show()

	# Plot attributions
	plot_n = 5

	print("Plotting attributions for lowest {} predicted probability".format(plot_n))
	for example_idx, row in attr_df.head(plot_n).iterrows():
		# Make logomaker style df
		logo_df = pd.DataFrame({
			'A': [0] * len(row.seq),
			'C': [0] * len(row.seq),
			'G': [0] * len(row.seq),
			'T': [0] * len(row.seq),
		})

		for i, base in enumerate(row.seq):
			logo_df.loc[i, base] = row.mean_ig[i]

		# Plot
		logo_plot = lm.Logo(logo_df)
		plt.title("Predicted probability: {:.3f}".format(row.pred))
		print(row)
		plt.show()

	print("Plotting attributions for highest {} predicted probability".format(plot_n))
	for example_idx, row in attr_df.tail(plot_n).iterrows():
		# Make logomaker style df
		logo_df = pd.DataFrame({
			'A': [0] * len(row.seq),
			'C': [0] * len(row.seq),
			'G': [0] * len(row.seq),
			'T': [0] * len(row.seq),
		})

		for i, base in enumerate(row.seq):
			logo_df.loc[i, base] = row.mean_ig[i]

		# Plot
		logo_plot = lm.Logo(logo_df)
		plt.title("Predicted probability: {:.3f}".format(row.pred))
		print(row)
		plt.show()
