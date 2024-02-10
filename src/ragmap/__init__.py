"""
Initializes the ragmap module and exposes the main classes, constants and functions.
"""

from .constants import (
	ModelProvider,
	DimensionReduction,
	IndexCategory,
	plot_settings,
	plot3d_settings,
	wrap_width
)

from .ragmap import (
	RAGmap
)

__version__ = "0.1.0"

__all__ = [
	'ModelProvider',
	'DimensionReduction',
	'IndexCategory',
	'plot_settings',
	'plot3d_settings',
	'wrap_width',
    'RAGmap',
]
