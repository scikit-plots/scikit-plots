# from __future__ import (
#   absolute_import,
#   division,
#   print_function,
#   unicode_literals
# )
from .classifiers import classifier_factory
from .clustering import clustering_factory
from . import (
	metrics,
	cluster,
	decomposition,
	estimators
)

__version__ = '0.3.7'