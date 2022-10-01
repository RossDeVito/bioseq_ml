"""
Wrappers for creating networks with multiple sequence inputs that have
some function (modules) applied to them before being concatenated.
"""

from torch import nn


class MultiInputApply(nn.Module):
	"""
	Applies same function to all items in an input dict or list. Output
	will be a dict with the same keys or a list in the same order.

	Args:
		network_spec (list): List of dicts specifying modules in
			model's network that is applied to all inputs.
		model_builder (bioseq_ml.utils.model_utils.ModelBuilder): Model
			builder for building modules in 'network_spec'.
		input_type (str, None): Type of input to save time checking. If
			None, will check and set input type on first forward pass.
	"""

	def __init__(self, network_spec, model_builder, input_type=None):
		super().__init__()
		self.network_spec = network_spec
		self.input_type = input_type

		if self.input_type is not None:
			assert self.input_type in ['dict', 'list'], (
				'input_type must be "dict" or "list" if not None'
			)
			self.dict_input = self.input_type == 'dict'
			self.list_input = self.input_type == 'list'

		self.network = model_builder.build_network(self.network_spec)

	def forward(self, x):
		if self.input_type is None:
			self.dict_input = isinstance(x, dict)
			self.list_input = isinstance(x, list)
			self.input_type = 'dict' if self.dict_input else 'list'

		if self.dict_input:
			return {key: self.network(value) for key, value in x.items()}
		elif self.list_input:
			return [self.network(item) for item in x]
		else:
			raise ValueError('Input must be dict or list')

	@staticmethod
	def requires_model_builder():
		return True