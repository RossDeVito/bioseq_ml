"""Utilities for models."""

from ast import Not
import inspect

from torch import nn

from bioseq_ml import modules
from bioseq_ml import models


def count_params(model, trainable_only=True):
	"""Count number of parameters in a model
	Args:
		model (nn.Module): model to count parameters
		trainable_only (bool): count only trainable parameters
	Returns:
		number of parameters
	"""
	if trainable_only:
		return sum(p.numel() for p in model.parameters() if p.requires_grad)
	else:
		return sum(p.numel() for p in model.parameters())


class ModelBuilder:
	"""Class for building models and modules from specifications.

	Args:
		custom_models (list): List of custom model classes to possibly use.
		custom_modules (list): List of custom module classes to possibly use.
	"""

	def __init__(self, custom_models=None, custom_modules=None):
		self.custom_models = custom_models
		self.custom_modules = custom_modules

		if self.custom_models is not None:
			self.custom_models = {c.__name__: c for c in self.custom_models}
		if self.custom_modules is not None:
			self.custom_modules = {c.__name__: c for c in self.custom_modules}

	def build_module(self, module_spec):
		"""Create module from spec.

		Module classes with the function 'requires_model_builder' where the
		return value is True will be passed the model builder as the
		'model_builder' keyword argument. This allows modules to internally
		use the module builder to build other potentially custom modules.
		
		Args:
			module_spec (dict): With key 'type' giving module class name
				and optional key kwargs providing key work arguments to that
				class.

		Returns:
			module (nn.Module): Module built from specification.
		"""
		module_type = module_spec['type']
		module_kwargs = module_spec.get('kwargs', {})

		# Check custom modules
		if self.custom_modules is not None and module_type in self.custom_modules:
			module_class = self.custom_modules[module_type]
		# Check bioseq_ml.modules
		elif hasattr(modules, module_type):
			module_class = getattr(modules, module_type)
		# Check torch.nn
		elif hasattr(nn, module_type):
			module_class = getattr(nn, module_type)
		else:
			raise ValueError(f'Unknown module type: {module_type}')

		if (
			hasattr(module_class, 'requires_model_builder') 
			and module_class.requires_model_builder()
		):
			module_kwargs['model_builder'] = self

		return module_class(**module_kwargs)

	def build_network(self, network_spec):
		"""Build network from spec.

		Args:
			network_spec (list): List of dicts specifying modules in
				model's network.

		Returns:
			network (nn.Module): Network built from specification.
		"""
		modules = nn.ModuleList()
		for module_spec in network_spec:
			modules.append(self.build_module(module_spec))
		return nn.Sequential(*modules)

	def get_model_class(self, model_type):
		"""Get model class from type string.

		Args:
			model_type (str): Type of model to build. Searches classes in
				custom_models then in bioseq_ml.models.

		Returns:
			model_class (pl.LightningModule): Model class.
		"""
		if self.custom_models is not None and model_type in self.custom_models:
			model_class = self.custom_models[model_type]
		elif hasattr(models, model_type):
			model_class = getattr(models, model_type)
		else:
			raise ValueError(f'Unknown model type: {model_type}')

		return model_class

	def build_model(self, model_spec, **kwargs):
		"""Build a model from a specification dict.

		Args:
			model_spec (dict): Specification dict for model. Includes keys:
				'model_type' (str): Type of model to build. Searches classes in
					custom_models then in bioseq_ml.models.
				'network_spec' (list): List of dicts specifying modules in
					model's network (which will be passed to the model
					constructor as the 'net' keyword argument). Dicts should
					have keys:
						'type' (str): Type of module to create. Can be class
							name from custom_modules, 'bioseq_ml.modules', or
							'torch.nn'. The three potential module sources are
							searched in that order.
						'kwargs' (dict, optional): Keyword arguments for module.
							If not specified, no kwargs are passed.
			kwargs: Additional keyword arguments to pass to model class.
		
		Returns:
			model (pl.LightningModule): Model built from specification.
		"""
		model_class = self.get_model_class(model_spec['model_type'])
		net = self.build_network(model_spec['network_spec'])
		return model_class(net=net, **kwargs)



# def create_module_from_spec(module_spec, custom_modules=None):
# 	"""Create module from specification dict.

# 	Currently supports modules from bioseq_ml.modules and torch.nn.

# 	Args:
# 		module_spec (dict): Specification dict for module. Includes keys:
# 			'type' (str): Type of module to create. Can be class name from
# 				custom_modules, 'bioseq_ml.modules', or 'torch.nn'. The
# 				three potential module sources are searched in that order.
# 			'kwargs' (dict, optional): Keyword arguments for module. If not
# 				specified, no kwargs are passed.
# 		custom_modules (list, optional): List of custom modules to include.
# 	"""
# 	module_type = module_spec['type']
# 	module_kwargs = module_spec.get('kwargs', {})

# 	if custom_modules is not None:
# 		custom_modules = {m.__name__: m for m in custom_modules}
# 		if module_type in custom_modules:
# 			return custom_modules[module_type](**module_kwargs)
	
# 	bioseq_modules = dict(
# 		inspect.getmembers(modules, inspect.isclass)
# 	)
# 	if module_type in bioseq_modules:
# 		return bioseq_modules[module_type](**module_kwargs)
# 	elif module_type in nn.__dict__:
# 		return nn.__dict__[module_type](**module_kwargs)
# 	else:
# 		raise ValueError(f'Unknown module type: {module_type}')
