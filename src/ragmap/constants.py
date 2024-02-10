"""
Defines constants used throughout the ragmap module
including plot settings, the supported model providers
and dimensionality reduction techniques as well as the
available categories for indexed documents.
"""

from enum import Enum

from openTSNE import TSNE
from sklearn.decomposition import PCA
from umap import UMAP

class ModelProvider(Enum):
    """
    Supported embedding model providers.
    """
    AMAZON_BEDROCK = "Amazon Bedrock ⛰️"
    HUGGING_FACE = "HuggingFace 🤗"
    OPENAI = "OpenAI"

class DimensionReduction(Enum):
    """
    Available dimensionality reduction techniques.
    """
    UMAP = UMAP
    TSNE = TSNE
    PCA = PCA

class IndexCategory(Enum):
    """
    The different categories for each indexed chunk.
    """
    CHUNK = "Chunk"
    QUERY = "Query"
    RETRIEVED = "Retrieved"
    SUB_QUERY = "Sub-query"
    HYPOTHETICAL_ANSWER = "Hypothetical Answer"

# Default wrap width
wrap_width = 100  # pylint: disable=invalid-name

# Default 2D plot settings
plot_settings = {
    IndexCategory.CHUNK: {
        'color': 'blue',
        'opacity': 0.5,
        'symbol': 'circle',
        'size': 10,
    },
    IndexCategory.QUERY: {
        'color': 'red',
        'opacity': 1,
        'symbol': 'cross',
        'size': 15,
    },
    IndexCategory.RETRIEVED: {
        'color': 'green',
        'opacity': 1,
        'symbol': 'star',
        'size': 15,
    },
    IndexCategory.SUB_QUERY: {
        'color': 'purple',
        'opacity': 1,
        'symbol': 'square',
        'size': 15,
    },
    IndexCategory.HYPOTHETICAL_ANSWER: {
        'color': 'purple',
        'opacity': 1,
        'symbol': 'square',
        'size': 15,
    },
}

# Default 3D plot settings
plot3d_settings = {
    IndexCategory.CHUNK: {
        'color': 'blue',
        'opacity': 0.5,
        'symbol': 'circle',
        'size': 10,
    },
    IndexCategory.QUERY: {
        'color': 'red',
        'opacity': 1,
        'symbol': 'circle',
        'size': 15,
    },
    IndexCategory.RETRIEVED: {
        'color': 'green',
        'opacity': 1,
        'symbol': 'diamond',
        'size': 10,
    },
    IndexCategory.SUB_QUERY: {
        'color': 'purple',
        'opacity': 1,
        'symbol': 'square',
        'size': 10,
    },
    IndexCategory.HYPOTHETICAL_ANSWER: {
        'color': 'purple',
        'opacity': 1,
        'symbol': 'square',
        'size': 10,
    },
}
