r"""
   ___  ___  _____
  / _ \/ _ |/ ___/_ _  ___ ____
 / , _/ __ / (_ /  ' \/ _ `/ _ \
/_/|_/_/ |_\___/_/_/_/\_,_/ .__/
                         /_/
            ___,
       _.-'` __|__
     .'  ,-:` \;',`'-,
    /  .'-;_,;  ':-;_,'.
   /  /;   '/    ,  _`.-\
  |  | '`. (`     /` ` \`|
  |  |:.  `\`-.   \_   / |
  |  |     (   `,  .`\ ;'|
   \  \     | .'     `-'/
    \  `.   ;/        .'
     '._ `'-._____.-'`
        `-.____|
          _____|_____
    jgs  /___________\

A simple Python module to help visualize document chunks and queries in embedding space.
"""

import logging

from textwrap import wrap
from typing import (
    Any,
    Dict,
    List,
    Union
)

import langchain
import numpy as np
import pandas as pd
import plotly.graph_objs as go

from .constants import (
    DimensionReduction,
    IndexCategory,
    wrap_width
)

from .databases import VectorDatabase

from .utils import (
    process_file,
    plot_projections
)

logger = logging.getLogger(__name__)

class RAGmapError(Exception):
    """
    Raised any time RAGmap fails.
    """

class RAGmap():
    """
    RAGmap class for visualizing and exploring embeddings.
    """

    def __init__(self, text_splitter: Any, vectordb: VectorDatabase):
        logger.info("Initializing RAGmap")
        self.text_splitter = text_splitter
        self.vectordb = vectordb

        # Use to track the last indexed document
        self._last_id = 0

    def _split_text(
        self,
        text: str) -> List[str]:
        """
        Splits incoming text into smaller chunks.
        """
        if langchain.text_splitter.TextSplitter in type(self.text_splitter).__mro__:
            return self.text_splitter.split_text(text)
        if callable(self.text_splitter):
            return self.text_splitter(text)
        raise RAGmapError(f"Text splitter of type {type(self.text_splitter)} is not supported!")

    def load_file(
        self,
        input: Any  # pylint: disable=redefined-builtin
    ) -> None:
        """
        Indexes and stores a single file in the vector DB.
        """
        # Process file
        logger.info("Loading >>> %s", input)
        texts, metadata = process_file(input)

        # Split text into chunks
        logger.info("Splitting text into chunks")
        text = "\n\n".join(texts)
        chunks = self._split_text(text)

        # Store everything in the vector database
        logger.info("Storing data in the vector database")
        self.vectordb.add(
            ids=list(map(str, range(self._last_id, self._last_id + len(chunks)))),
            documents=chunks,
            metadatas=[metadata] * len(chunks)
        )
        self._last_id += len(chunks)

    def load_files(self, inputs: List[str]) -> None:
        """
        Indexes and stores multiple files in the vector database.
        """
        for input in inputs:  # pylint: disable=redefined-builtin
            self.load_file(input)

    def _fit_transform(
        self,
        dimension_reduction: DimensionReduction = DimensionReduction.UMAP,
        n_components: int = 2,
        **kwargs):
        """
        Returns a dimensionality reduction transform fitted to the embedding collection.
        """
        logger.info("Retrieving vectors from datastore")
        embeddings = np.array(self.vectordb.get(include=['embeddings'])['embeddings'])
        logger.info("Fitting %iD %s transform to embeddings", \
                    n_components, dimension_reduction.value.__name__)
        return dimension_reduction.value(
            n_components=n_components,
            **kwargs
        ).fit(embeddings)

    def create_projections(
        self,
        dimension_reduction: DimensionReduction = DimensionReduction.UMAP,
        n_components: int = 2,
        **kwargs):
        """
        Uses dimensionality reduction techniques to project an embedding collection.
        """
        embeddings = np.array(
            self.vectordb.get(include=['embeddings'])['embeddings']
        )
        logger.info("Generating %iD %s projections", \
                    n_components, dimension_reduction.value.__name__)
        dim_redux = self._fit_transform(dimension_reduction, n_components, **kwargs)
        return dim_redux.transform(embeddings), dim_redux

    def plot(  # pylint: disable=too-many-locals
        self,
        dimension_reduction: DimensionReduction = DimensionReduction.UMAP,
        n_components: int = 2,
        query: Union[Dict, None] = None,
        dimension_reduction_kwargs: Union[Dict, None] = None) -> go.Figure:
        """
        Creates a 2D or 3D plot of the reduced embedding space.
        """
        # Check projection components
        assert n_components in [2, 3], \
            f"Unsupported number of components (Got: {n_components}, Expected: 2 or 3)"

        # Retrieve data from vector store
        data = self.vectordb.get(
            include=['documents']
        )
        ids = data['ids']
        documents = data['documents']

        # Generate projections
        if dimension_reduction_kwargs is None:
            dimension_reduction_kwargs = {}
        emb_projs, dim_redux = self.create_projections(
            dimension_reduction,
            n_components,
            **dimension_reduction_kwargs
        )

        # Run query and process results
        if query is not None:
            logger.info("Running query\n%s", query)
            retrieved_ids = []
            for result in self.vectordb.query(**query)['ids']:
                retrieved_ids.extend(map(int, result))
        else:
            retrieved_ids = []

        # Prepare embedding projections dataframe for visualization
        logger.info("Preparing embeddings projection dataframe for visualization")
        df = pd.DataFrame({
            'id': [int(id) for id in ids],
            'x': emb_projs[:, 0],
            'y': emb_projs[:, 1],
            'z': emb_projs[:, 2] if n_components == 3 else None,
            'document_cleaned': [
                "<br>".join(wrap(doc, wrap_width))
                    for doc in documents
            ],
            'category': IndexCategory.CHUNK
        })

        # Flag retrieved chunks
        df.loc[df['id'].isin(retrieved_ids), 'category'] = IndexCategory.RETRIEVED

        # Add query projections
        if query is not None:
            query_embs = query.get(
                'query_embeddings', 
                self.vectordb.embed(
                    [query['query_texts']] \
                        if isinstance(query['query_texts'], str) \
                        else query['query_texts']
                )
            )
            query_projs = dim_redux.transform(query_embs)
            df_query = pd.DataFrame({
                'x': query_projs[:, 0],
                'y': query_projs[:, 1],
                'z': query_projs[:, 2] if n_components == 3 else None,
                'document_cleaned': query.get('query_texts', None),
                'category': [IndexCategory.QUERY] + \
                            [IndexCategory.SUB_QUERY] * (len(query_projs) - 1)
            })
            df = pd.concat([df, df_query], axis=0)

        # Create visualization
        return plot_projections(
            df,
            self.vectordb.provider,
            self.vectordb.model,
            dimension_reduction,
            n_components
        )
