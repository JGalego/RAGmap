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
import uuid

from textwrap import wrap
from typing import (
    Any,
    Dict,
    List,
    Optional,
    TypeVar,
    Union
)

import boto3
import chromadb
import langchain
import numpy as np
import pandas as pd
import plotly.graph_objs as go

from pydantic import BaseModel

# Embedding functions
from chromadb.utils.embedding_functions import (
    AmazonBedrockEmbeddingFunction,
    OpenAIEmbeddingFunction,
    SentenceTransformerEmbeddingFunction
)

from .constants import (
	ModelProvider,
    DimensionReduction,
    IndexCategory,
    wrap_width
)

from .utils import (
	process_file,
    plot_projections
)

T = TypeVar("T")
OneOrMany = Union[T, List[T]]

# Log initialization
logger = logging.getLogger(__name__)

class RAGmapError(Exception):
    """
    Raised any time RAGmap fails.
    """

class RAGmap(BaseModel):
    """
    RAGmap class for visualizing and exploring embeddings.
    """
    text_splitter: Any
    embedding_model: str
    provider: ModelProvider
    boto3_sess_args: Union[Dict, None] = None
    embed_func_kwargs: Union[Dict, None] = None
    collection_metadata: Union[Dict, None] = None
    _embedding_function: Optional[chromadb.api.types.EmbeddingFunction] = None
    _vectordb: Optional[chromadb.Collection] = None
    _last_id: int = 0

    def __init__(self, **data):
        logger.info("Initializing RAGmap")
        super().__init__(**data)
        self._init_embed_func()
        self._init_vect_store()

    def _init_embed_func(self) -> None:
        """
        Defines the embedding function based on the selected model provider.
        """
        logger.info("Setting embedding function for provider %s", self.provider.value)
        if self.embed_func_kwargs is None:
            self.embed_func_kwargs = {}
        if self.provider == ModelProvider.AMAZON_BEDROCK:
            if self.boto3_sess_args is None:
                self.boto3_sess_args = {}
            self._embedding_function = AmazonBedrockEmbeddingFunction(
                session=boto3.Session(**self.boto3_sess_args),
                model_name=self.embedding_model,
                **self.embed_func_kwargs
            )
        elif self.provider == ModelProvider.HUGGING_FACE:
            self._embedding_function = SentenceTransformerEmbeddingFunction(
                model_name=self.embedding_model,
                **self.embed_func_kwargs
            )
        elif self.provider == ModelProvider.OPENAI:
            self._embedding_function = OpenAIEmbeddingFunction(
                model_name=self.embedding_model,
                **self.embed_func_kwargs
            )

    def _init_vect_store(self) -> None:
        """
        Initializes the vector data store.
        """
        logger.info("Creating vector store")
        chroma_client = chromadb.Client()
        self._vectordb = chroma_client.create_collection(
            name=uuid.uuid4().hex,
            embedding_function=self._embedding_function,
            metadata=self.collection_metadata
        )

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
        self._vectordb.add(
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

    def embed(
        self,
        input: OneOrMany[str]  # pylint: disable=redefined-builtin
	):
        """Converts incoming texts into embeddings."""
        return self._embedding_function(input=input)

    def query(  # pylint: disable=too-many-arguments
        self,
        query_embeddings: Union[OneOrMany[chromadb.api.types.Embedding], \
                          OneOrMany[np.ndarray], \
                          None] = None,
        query_texts: Union[OneOrMany[str],None] = None,
        n_results: int = 10,
        where: Union[chromadb.api.types.Where, None] = None,
        where_document: Union[chromadb.api.types.WhereDocument, None] = None):
        """
        Returns similar entries in the vector database for provided embeddings or texts.
        """
        return self._vectordb.query(
            query_texts=query_texts,
            query_embeddings=query_embeddings,
            n_results=n_results,
            where=where,
            where_document=where_document,
            include=[
                'documents',
                'embeddings',
                'metadatas'
            ]
        )

    def _fit_transform(
        self,
        dimension_reduction: DimensionReduction = DimensionReduction.UMAP,
        n_components: int = 2,
        **kwargs):
        """
        Returns a dimensionality reduction transform fitted to the embedding collection.
        """
        logger.info("Retrieving vectors from datastore")
        embeddings = np.array(self._vectordb.get(include=['embeddings'])['embeddings'])
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
            self._vectordb.get(include=['embeddings'])['embeddings']
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
        data = self._vectordb.get(
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
            for result in self.query(**query)['ids']:
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
                self.embed(
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
            self.provider,
            self.embedding_model,
            dimension_reduction,
            n_components
        )
