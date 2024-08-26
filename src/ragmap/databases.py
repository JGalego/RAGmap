"""
Vector Databases
"""

import logging
import uuid

from abc import (
    ABC,
    abstractmethod
)

from typing import (
    Any,
    Dict,
    List,
    Union
)

import boto3
import chromadb

from chromadb.utils.embedding_functions import (  # pylint: disable=no-name-in-module
    AmazonBedrockEmbeddingFunction,
    OpenAIEmbeddingFunction,
    SentenceTransformerEmbeddingFunction
)

from .constants import ModelProvider

from .utils import (
    Embedding,
    Embeddings,
    OneOrMany,
)

logger = logging.getLogger(__name__)

class VectorDatabase(ABC):
    """
    Abstract class for vector databases.
    """

    name = uuid.uuid4().hex

    @abstractmethod
    def __init__(
        self,
        model: str,
        provider: ModelProvider,
        embed_func_kwargs: Union[Dict, None] = None,
        metadata: Union[Dict, None] = None
    ) -> None:
        ...

    @abstractmethod
    def _init_embed_func(self) -> None:
        """
        Initializes the embedding function based on the model and model provider.
        """

    @abstractmethod
    def embed(
        self,
        inputs: OneOrMany[str]
    ) -> Embeddings:
        """Converts incoming texts into embeddings."""

    @abstractmethod
    def add(
        self,
        ids: OneOrMany[Any],
        documents: OneOrMany[str],
        metadatas: OneOrMany[Dict]
    ) -> None:
        """
        Adds a document to the vector database.
        """

    @abstractmethod
    def get(
        self,
        include: List[str]
    ) -> Dict:
        """
        Retrieves data from the vector store.
        """

    @abstractmethod
    def query(
        self,
        query_embeddings: Union[OneOrMany[Embedding], None] = None,
        query_texts: Union[OneOrMany[str], None] = None,
        n_results: int = 10):
        """
        Returns similar entries in the vector database for provided embeddings or texts.
        """


class ChromaDb(VectorDatabase):
    """
    ChromaDB database
    """

    def __init__(self, model, provider, embed_func_kwargs = None, metadata = None):
        self.model = model
        self.provider = provider
        self.embed_func_kwargs = embed_func_kwargs
        self._init_embed_func()

        self._client = chromadb.Client()
        self.metadata = metadata
        self._init_vectordb()

    def _init_embed_func(self):
        """
        Defines the embedding function based on the selected model provider.
        """

        if self.embed_func_kwargs is None:
            self.embed_func_kwargs = {}

        logger.info(
            "Setting embedding function (model: %s, provider: %s)",
            self.model, self.provider.value
        )
        if self.provider == ModelProvider.AMAZON_BEDROCK:
            boto3_sess_args = self.embed_func_kwargs.pop('boto3_sess_args', {})
            self._embed_func = AmazonBedrockEmbeddingFunction(
                session=boto3.Session(**boto3_sess_args),
                model_name=self.model,
                **self.embed_func_kwargs
            )
        elif self.provider == ModelProvider.HUGGING_FACE:
            self._embed_func = SentenceTransformerEmbeddingFunction(
                model_name=self.model,
                **self.embed_func_kwargs
            )
        elif self.provider == ModelProvider.OPENAI:
            self._embed_func = OpenAIEmbeddingFunction(
                model_name=self.model,
                **self.embed_func_kwargs
            )

    def _init_vectordb(self):
        """
        Initializes a Chroma collection.
        """
        logger.info(
            "Initializing Chroma collection (name: %s, metadata: %s)",
            self.name, self.metadata
        )
        self._vectordb = self._client.create_collection(
            self.name,
            embedding_function=self._embed_func,
            metadata=self.metadata
        )

    def embed(self, inputs):
        return self._embed_func(input=inputs)

    def add(self, ids, documents, metadatas):
        self._vectordb.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas
        )

    def get(self, include):
        logger.info("Retrieving data from vector store: %s", include)
        return self._vectordb.get(include=include)

    def query(
        self,
        query_embeddings = None,
        query_texts = None,
        n_results = 10):

        assert query_embeddings or query_texts

        return self._vectordb.query(
            query_texts=query_texts,
            query_embeddings=query_embeddings,
            n_results=n_results,
            include=[
                'documents',
                'embeddings',
                'metadatas'
            ]
        )
