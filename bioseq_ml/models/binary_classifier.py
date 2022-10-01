"""Base binary classification model class."""

import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
import torchvision

from torchmetrics import MetricCollection
from torchmetrics import Precision, Recall, F1Score, ConfusionMatrix


class BinaryClassifier(pl.LightningModule):
	"""Classifier for binary data.

	Args:
		net (nn.Module): Neural network to use.
		optimizer (str, optional): Optimizer to use. Must be string name of
			optimizer class in torch.optim. Defaults to 'Adam'.
		optimizer_kwargs (dict, optional): Keyword arguments to pass to
			optimizer. Defaults to {}. 'lr' will be overwritten by
			learning_rate, do not include it here.
		learning_rate (float, optional): Learning rate. Defaults to 1e-3.
		reduce_lr_on_plateau (bool, optional): Reduce learning rate on plateau.
			Defaults to False.
		reduce_lr_factor (float, optional): Factor to reduce learning rate by
			on plateau if above True. Defaults to 0.1.
		reduce_lr_patience (int, optional): Patience for learning rate scheduler.
			Defaults to 10.
		pos_weight (float, optional): Weight for positive class. Defaults
			to None (no weighting).
	"""

	def __init__(
		self,
		net,
		optimizer='Adam',
		optimizer_kwargs={},
		learning_rate=1e-3,
		reduce_lr_on_plateau=False,
		reduce_lr_factor=0.1,
		reduce_lr_patience=10,
		pos_weight=None,
	):
		super().__init__()
		self.net = net
		self.optimizer = optimizer
		self.optimizer_kwargs = optimizer_kwargs
		self.learning_rate = learning_rate
		self.reduce_lr_on_plateau = reduce_lr_on_plateau
		self.reduce_lr_factor = reduce_lr_factor
		self.reduce_lr_patience = reduce_lr_patience
		self.pos_weight = pos_weight

		self.save_hyperparameters(
			'learning_rate', 'reduce_lr_on_plateau', 
			'reduce_lr_factor', 'reduce_lr_patience', 'pos_weight'
		)

		# Metrics
		metrics = MetricCollection([
			Precision(num_classes=2, average='macro', multiclass=True),
			Recall(num_classes=2, average='macro', multiclass=True),
			F1Score(num_classes=2, average='macro', multiclass=True),
		])
		self.train_metrics = metrics.clone(prefix='train_')
		self.val_metrics = metrics.clone(prefix='val_')
		self.test_metrics = metrics.clone(prefix='test_')

	def forward(self, x):
		return torch.sigmoid(self.net(x))

	def shared_step(self, batch):
		y = batch['label']
		logits = self.net(batch['input'])

		if self.pos_weight is not None:
			weight = torch.tensor([self.pos_weight], device=self.device)
		else:
			weight = None

		loss = F.binary_cross_entropy_with_logits(
			logits, y.unsqueeze(1).float(), weight=weight
		)
		return loss, logits, y

	def training_step(self, batch, batch_idx):
		loss, logits, y = self.shared_step(batch)
		metrics_dict = self.train_metrics(torch.sigmoid(logits), y.long())
		self.log_dict(metrics_dict, on_epoch=True)
		self.log("train_loss", loss, on_step=True, on_epoch=True,
					prog_bar=True)
		return loss

	def validation_step(self, batch, batch_idx):
		loss, logits, y = self.shared_step(batch)
		metrics_dict = self.val_metrics(torch.sigmoid(logits), y.long())
		self.log_dict(metrics_dict, prog_bar=True)
		self.log("val_loss", loss, prog_bar=True)
		return {'logits': logits, 'y_true': y}

	def validation_epoch_end(self, outs):
		"""from https://stackoverflow.com/questions/65498782/how-to-dump-confusion-matrix-using-tensorboard-logger-in-pytorch-lightning/73388839#73388839 """
		tb = self.logger.experiment  # noqa

		outputs = torch.cat([tmp['logits'] for tmp in outs])
		labels = torch.cat([tmp['y_true'] for tmp in outs])

		print("outputs", outputs.get_device(), flush=True)

		if outputs.get_device() == 0 and torch.has_mps:
			confusion = ConfusionMatrix(num_classes=2).to(torch.device("mps"))
		else:
			confusion = ConfusionMatrix(num_classes=2).to(outputs.get_device())
		confusion(torch.sigmoid(outputs), labels)
		computed_confusion = confusion.compute().detach().cpu().numpy().astype(int)

		# confusion matrix
		df_cm = pd.DataFrame(computed_confusion)

		fig, ax = plt.subplots(figsize=(10, 5))
		fig.subplots_adjust(left=0.05, right=.65)
		sns.set(font_scale=1.2)
		sns.heatmap(df_cm, annot=True, annot_kws={"size": 16}, fmt='d', ax=ax)
		ax.set_xlabel('Predicted')
		ax.set_ylabel('True')
		buf = io.BytesIO()

		plt.savefig(buf, format='jpeg', bbox_inches='tight')
		buf.seek(0)
		im = Image.open(buf)
		im = torchvision.transforms.ToTensor()(im)
		tb.add_image("val_confusion_matrix", im, global_step=self.current_epoch)
		plt.close()
		self.log("val_TN", float(computed_confusion[0, 0]), prog_bar=True)
		self.log("val_FP", float(computed_confusion[0, 1]), prog_bar=True)
		self.log("val_FN", float(computed_confusion[1, 0]), prog_bar=True)
		self.log("val_TP", float(computed_confusion[1, 1]), prog_bar=True)

	def test_step(self, batch, batch_idx):
		loss, logits, y = self.shared_step(batch)
		metrics_dict = self.test_metrics(torch.sigmoid(logits), y.long())
		self.log_dict(metrics_dict, prog_bar=True)
		self.log("test_loss", loss, prog_bar=True)

		return {'logits': logits, 'y_true': y}

	def test_epoch_end(self, outs):
		outputs = torch.cat([tmp['logits'] for tmp in outs])
		labels = torch.cat([tmp['y_true'] for tmp in outs])

		# confusion matrix
		if outputs.get_device() == 0 and torch.has_mps:
			confusion = ConfusionMatrix(num_classes=2).to(torch.device("mps"))
		else:
			confusion = ConfusionMatrix(num_classes=2).to(outputs.get_device())
		confusion(torch.sigmoid(outputs), labels)
		computed_confusion = confusion.compute().detach().cpu()

		self.log("test_TN", computed_confusion[0, 0])
		self.log("test_FP", computed_confusion[0, 1])
		self.log("test_FN", computed_confusion[1, 0])
		self.log("test_TP", computed_confusion[1, 1])

	def configure_optimizers(self):
		params = self.parameters()
		optimizer = getattr(torch.optim, self.optimizer)(
			params, 
			lr=self.learning_rate,
			**self.optimizer_kwargs
		)

		if self.reduce_lr_on_plateau:
			scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
				optimizer, 
				factor=self.reduce_lr_factor, 
				patience=self.reduce_lr_patience,
				verbose=True
			)
			return {
				'optimizer': optimizer,
				'lr_scheduler': {
					'scheduler': scheduler,
					'monitor': 'val_loss',
				}
			}
		else:
			return optimizer

