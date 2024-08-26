"""
A simple RAGmap demo
"""

import sys
import logging
import warnings

from langchain.text_splitter import RecursiveCharacterTextSplitter

from ragmap.constants import ModelProvider
from ragmap.databases import ChromaDb
from ragmap import RAGmap

# Initialize logs
logging.basicConfig(
	stream=sys.stdout,
	level=logging.INFO,
	format='%(levelname)s: %(message)s'
)

# Disable warnings
warnings.simplefilter(action='ignore')

# Define text splitter
character_splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n", "\n", ". ", " ", ""],
    chunk_size=256,
    chunk_overlap=0
)

# -1. Initialize vector database
vectordb = ChromaDb(
    provider=ModelProvider.HUGGING_FACE,
    model='all-MiniLM-L6-v2'
)

# 0. Initialize RAGmap
rm = RAGmap(
    text_splitter=character_splitter,
	vectordb=vectordb
)

# 1. Load + Index a document
rm.load_file("all_amazon_shareholder_letters.pdf")

# 2. Query document
query_results = rm.vectordb.query(
    query_texts="How many services and features were released by AWS in 2012?",
    n_results=3
)
print(query_results['documents'][0])

# 3. Plot projected embeddings
fig = rm.plot(
    query={
        'query_texts': "How many services and features were released by AWS in 2012?",
        'n_results': 3
    },
    dimension_reduction_kwargs={
        'random_state': 0,
        'transform_seed': 0
    }
).update_layout(
    width=800,
    height=800
)

# 4. Save the results
# Note: requires installing kaleido
fig.write_image("result.png")
