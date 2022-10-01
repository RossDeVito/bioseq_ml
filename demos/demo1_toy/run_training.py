"""Demo training script for bioseq_ml.

TODO: Extend bioseq_ml to handle all specs.
"""

import os
import json
import platform

import torch
import pytorch_lightning as pl

from bioseq_ml import ModelBuilder
from bioseq_ml.utils import count_params
from bioseq_ml.data_modules import CSVDataModule

from dataset import ToyDataset, toy_collate_fn


if __name__ == '__main__':
	__spec__ = None

	# Select model, data, and training specs
	model_spec_path = os.path.join('specs', 'model', 'cnn_bincls_1.json')
	data_spec_path = os.path.join('specs', 'data', 'toy_48.json')
	train_spec_path = os.path.join('specs', 'training', 'bin_toy48_1.json')

	# Load specs
	with open(model_spec_path, 'r') as f:
		model_spec = json.load(f)
	with open(data_spec_path, 'r') as f:
		data_spec = json.load(f)
	with open(train_spec_path, 'r') as f:
		train_spec = json.load(f)
	trainer_args = train_spec.copy()	# Will pop keys from this dict. Want
										# to keep train_spec intact to save.

	# Create model from model_spec
	model_builder = ModelBuilder()

	model_class = model_builder.get_model_class(model_spec['model_type'])
	model_args_in_train_spec = [
		k for k in trainer_args.keys() if k in model_class.__init__.__code__.co_varnames
	]
	build_model_kwargs = {k: trainer_args.pop(k) for k in model_args_in_train_spec}

	model = model_builder.build_model(model_spec, **build_model_kwargs)
	print("Model:\n", model)

	# Create and setup data module
	data_module_class = CSVDataModule
	dataset_class = ToyDataset
	collate_fn = toy_collate_fn

	# Batch size (and 'num_workers' since it's more dependent on the training
	# setup than the data) are part of the training spec, but are passed to the
	# data module here
	if 'batch_size' in train_spec:
		data_spec['batch_size'] = trainer_args.pop('batch_size')
	if 'num_workers' in train_spec:
		data_spec['num_workers'] = trainer_args.pop('num_workers')

	data_module = data_module_class(
		dataset_class=dataset_class,
		collate_fn=collate_fn,
		**data_spec,
	)

	# Create trainer
	platform_info = platform.platform()
	if 'mac' in platform_info.lower() and 'arm' in platform_info.lower():
		trainer_args['accelerator'] = 'mps'
		trainer_args['devices'] = 1
		# temp fix: export PYTORCH_ENABLE_MPS_FALLBACK=1

	callbacks = [
		pl.callbacks.ModelCheckpoint(
			monitor="val_loss",
			filename='{epoch}-best_val_loss'
		)
	]
	if 'early_stopping_patience' in trainer_args:
		callbacks.append(
			pl.callbacks.EarlyStopping(
				monitor='val_loss',
				verbose=True,
				patience=trainer_args.pop('early_stopping_patience')
			)
		)

	trainer_args['logger'] = pl.loggers.TensorBoardLogger(
		os.path.join(os.getcwd(), trainer_args.pop('training_output_dir')), 
		trainer_args.pop('experiment_name'),
		default_hp_metric=False
	)
	trainer_args['callbacks'] = callbacks
	trainer_args['log_every_n_steps'] = 1
	trainer = pl.Trainer(**trainer_args)

	# Train model
	trainer.fit(model, data_module)

	# Get performance on test set
	best_val = trainer.test(
		ckpt_path='best',
		dataloaders=data_module.test_dataloader()
	)[0]
	best_val['num_params'] = count_params(model.net)

	# Save performance on test set
	with open(os.path.join(trainer.logger.log_dir, 'test_res.json'), 'w') as f:
		json.dump(best_val, f)
	
	# Save specs
	with open(os.path.join(trainer.logger.log_dir, 'model_spec.json'), 'w') as f:
		json.dump(model_spec, f)
	with open(os.path.join(trainer.logger.log_dir, 'data_spec.json'), 'w') as f:
		json.dump(data_spec, f)
	with open(os.path.join(trainer.logger.log_dir, 'training_spec.json'), 'w') as f:
		json.dump(train_spec, f)
	



