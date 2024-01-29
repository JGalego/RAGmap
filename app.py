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

A simple Streamlit application that helps visualize document chunks and queries in embedding space.

Inspired by DeepLearning.ai's short course on 'Advanced Retrieval for AI with Chroma'
https://www.deeplearning.ai/short-courses/advanced-retrieval-for-ai/
and Gabriel Chua's award-winning RAGxplorer
https://github.com/gabrielchua/RAGxplorer/
"""

import io
import json
import uuid

from textwrap import wrap

import boto3
import chromadb
import umap

import numpy as np
import pandas as pd
import plotly.graph_objs as go
import streamlit as st

from chromadb.utils.embedding_functions import (
    AmazonBedrockEmbeddingFunction
)

from langchain.prompts import ChatPromptTemplate

from langchain.text_splitter import (
    RecursiveCharacterTextSplitter
)

from langchain_community.chat_models import BedrockChat

from langchain_core.output_parsers import StrOutputParser

#############
# Constants #
#############

# 2D
plot_settings = {
    'chunk': {
        'color': 'blue',
        'opacity': 0.5,
        'symbol': 'circle',
        'size': 10,
    },
    'query': {
        'color': 'red',
        'opacity': 1,
        'symbol': 'cross',
        'size': 15,
    },
    'retrieved': {
        'color': 'green',
        'opacity': 1,
        'symbol': 'star',
        'size': 15,
    },
    'sub-query': {
        'color': 'purple',
        'opacity': 1,
        'symbol': 'square',
        'size': 15,
    },
    'hypothetical answer': {
        'color': 'purple',
        'opacity': 1,
        'symbol': 'square',
        'size': 15,
    },
}

# 3D
plot3d_settings = {
    'chunk': {
        'color': 'blue',
        'opacity': 0.5,
        'symbol': 'circle',
        'size': 10,
    },
    'query': {
        'color': 'red',
        'opacity': 1,
        'symbol': 'circle',
        'size': 15,
    },
    'retrieved': {
        'color': 'green',
        'opacity': 1,
        'symbol': 'diamond',
        'size': 10,
    },
    'sub-query': {
        'color': 'purple',
        'opacity': 1,
        'symbol': 'square',
        'size': 10,
    },
    'hypothetical answer': {
        'color': 'purple',
        'opacity': 1,
        'symbol': 'square',
        'size': 10,
    },
}

################
# Helper Utils #
################

st.cache_data()
def list_embedding_models():
    """
    Returns a list of embedding models available in Amazon Bedrock
    """
    return bedrock.list_foundation_models(
        byOutputModality='EMBEDDING')['modelSummaries']


st.cache_data()
def list_text_models():
    """
    Returns a list of text models available in Amazon Bedrock
    """
    return bedrock.list_foundation_models(
        byOutputModality='TEXT')['modelSummaries']


def process_document(filename):
    """
    Loads and extracts text from a document
    """
    # PDF
    if filename.type == "application/pdf":
        from PyPDF2 import PdfReader  # pylint: disable=import-outside-toplevel
        doc = PdfReader(filename)
        texts = [p.extract_text().strip() for p in doc.pages]
    # DOCX
    elif filename.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        from docx import Document  # pylint: disable=import-outside-toplevel
        doc = Document(filename)
        texts = [paragraph.text for paragraph in doc.paragraphs]

    # Filter the empty strings
    texts = [text for text in texts if text]
    return texts


def chunk_texts(texts):
    """
    Turns input texts into chunks
    """
    character_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ". ", " ", ""],
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunks = character_splitter.split_text('\n\n'.join(texts))
    return chunks


def build_collection():
    """
    Creates and populates a chroma collection
    """
    # Load and extract text from PDF document
    with st.spinner("Loading document"):
        texts = process_document(uploaded_file)

    # Split text into chunks
    with st.spinner("Splitting text into chunks"):
        chunks = chunk_texts(texts)

    # Create and populate chroma collection
    with st.spinner("Creating collection"):
        chroma_client = chromadb.Client()
        document_name = uuid.uuid4().hex
        collection = chroma_client.create_collection(
            document_name,
            embedding_function=st.session_state.embedding_function
        )
    with st.spinner("Populating collection"):
        collection.add(
            ids=list(map(str, range(len(chunks)))),
            documents=chunks
        )

    st.session_state.collection = collection


def get_embeddings(text):
    """
    Transforms input text into embeddings
    """
    text_embeddings = st.session_state.embedding_function([text])
    return text_embeddings


def project_embeddings(embeddings, umap_transform):
    """
    Projects text embeddings using a umap transform
    """
    umap_embeddings = np.empty((len(embeddings), umap_transform.n_components))
    umap_embeddings = umap_transform.transform(embeddings)
    return umap_embeddings


def create_projections():
    """
    Transforms document embeddings into umap projections
    """
    # Get documents and embeddings
    with st.spinner("Retrieving document and embeddings"):
        res = st.session_state.collection.get(
            include=['documents', 'embeddings']
        )
        st.session_state.ids = res['ids']
        st.session_state.embeddings = res['embeddings']
        st.session_state.documents = res['documents']

    # Fit umap transformer to embeddings
    with st.spinner("Fitting umap to embeddings"):
        umap_transform = umap.UMAP(
            random_state=0, transform_seed=0, n_components=n_components
        ).fit(st.session_state.embeddings)
        st.session_state.umap_transform = umap_transform

    # Get projections
    with st.spinner("Generating projections"):
        st.session_state.projections = umap_transform.transform(st.session_state.embeddings)


def plot_projections(df_projs):
    """
    Generates a plot of embedding projections
    """
    fig = go.Figure()
    for category in df['category'].unique():
        df_cat = df_projs[df_projs['category'] == category]
        # 2D
        if n_components == 2:
            category_settings = plot_settings[category]
            trace = go.Scatter(
                x=df_cat['x'],
                y=df_cat['y'],
                mode='markers',
                name=category,
                marker={
                    'color': category_settings['color'],
                    'opacity': category_settings['opacity'],
                    'symbol': category_settings['symbol'],
                    'size': category_settings['size'],
                    'line_width': 0
				},
                hoverinfo='text',
                text=df_cat['document_cleaned']
            )
        # 3D
        elif n_components == 3:
            category_settings = plot3d_settings[category]
            trace = go.Scatter3d(
                x=df_cat['x'],
                y=df_cat['y'],
                z=df_cat['z'],
                mode="markers",
                name=category,
                marker={
                    'color': category_settings['color'],
                    'opacity': category_settings['opacity'],
                    'symbol': category_settings['symbol'],
                    'size': category_settings['size'],
                    'line_width': 0
				},
                hoverinfo='text',
                text=df_cat['document_cleaned']
            )

        fig.add_trace(trace)

    fig.update_layout(
        legend={
			'x': 0.5,
            'y': 100,
            'xanchor': "center",
            'yanchor': "top",
            'orientation': "h"
		}
    )

    st.plotly_chart(fig, use_container_width=True)
    return fig


def multiple_queries_expansion(user_query, df_query_original, model_id='anthropic.claude-v2:1'):
    """
    Expands a user query by creating multiple sub-queries
    """
    system_msg = "Given a query, your task is to generate 3 to 5 simple sub-queries related to the original query. These sub-queries must be short. Format your reply in JSON with numbered keys. Skip the preamble."  # pylint: disable=line-too-long
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_msg),
        ("human", "\"{user_query}\"")
    ])
    model = BedrockChat(model_id=model_id)
    output_parser = StrOutputParser()
    chain = prompt | model | output_parser
    output = chain.invoke({"user_query": user_query})
    try:
        st.session_state.query_expansions = list(json.loads(output).values())
        st.session_state.query_expansion_projections = np.array([
            project_embeddings(
                get_embeddings(expansion),
                st.session_state.umap_transform
            ) for expansion in st.session_state.query_expansions
        ])

        df_query_expansions = pd.DataFrame({
            'x': st.session_state.query_expansion_projections[:, 0, 0],
            'y': st.session_state.query_expansion_projections[:, 0, 1],
            'z': st.session_state.query_expansion_projections[:, 0, 2] \
                    if n_components == 3 else None,
            'document_cleaned': st.session_state.query_expansions,
            'category': ["sub-query"] * len(st.session_state.query_expansions),
        })

        df_query_expanded = pd.concat([df_query_original, df_query_expansions])
        return st.session_state.query_expansions, df_query_expanded
    except json.decoder.JSONDecodeError:
        st.warning("Model failed to expand query, falling back to the naive approach!")
        return [user_query], df_query_original


def generated_answer_expansion(user_query, df_query_original, model_id='anthropic.claude-v2:1'):
    """
    Expands a user query by generating an hypothetical answer
    """
    system_msg = "Given a query, your task is to generate a template for the hypothetical answer. Do not include any facts, and instead label them as <PLACEHOLDER>. Skip the preamble."  # pylint: disable=line-too-long
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_msg),
        ("human", "\"{user_query}\"")
    ])
    model = BedrockChat(model_id=model_id)
    output_parser = StrOutputParser()
    chain = prompt | model | output_parser
    output = chain.invoke({"user_query": user_query})
    try:
        st.session_state.query_expansions = [output]
        st.session_state.query_expansion_projections = np.array([
            project_embeddings(
                get_embeddings(expansion),
                st.session_state.umap_transform
            ) for expansion in st.session_state.query_expansions
        ])

        df_query_expansions = pd.DataFrame({
            'x': st.session_state.query_expansion_projections[:, 0, 0],
            'y': st.session_state.query_expansion_projections[:, 0, 1],
            'z': st.session_state.query_expansion_projections[:, 0, 2] \
                    if n_components == 3 else None,
            'document_cleaned': [
                "<br>".join(wrap(doc, 100)) for doc in st.session_state.query_expansions
            ],
            'category': ["hypothetical answer"] * len(st.session_state.query_expansions),
        })

        df_query_expanded = pd.concat([df_query_original, df_query_expansions])
        return st.session_state.query_expansions, df_query_expanded
    except json.decoder.JSONDecodeError:
        st.warning("Model failed to expand query, falling back to the naive approach!")
        return [user_query], df_query_original

########
# Main #
########

st.title("RAGMap 🗺️🔍")
st.text("Visualize document chunks and queries in embedding space")

# Initialize session state
for key in ["collection", "documents", "projections", "query_projections", "retrieved_ids"]:
    if key not in st.session_state:
        st.session_state[key] = None

# Initialize session
session = boto3.Session()

# Initialize bedrock clients
bedrock = session.client('bedrock')
bedrock_runtime = session.client('bedrock-runtime')

st.markdown("### 1. Upload a document 📄")

uploaded_file = st.file_uploader(
    label="Upload a file",
    label_visibility="hidden",
    type=["pdf", "docx"],
)

st.markdown("### 2. Build a vector database 💫")

chunk_size = st.number_input(
    label="Chunk Size",
    min_value=1,
    max_value=2000,
    value=256,
    step=1,
    key="chunk_size",
    help="Number of tokens in each chunk",
)

chunk_overlap = st.number_input(
    label="Chunk Overlap",
    min_value=0,
    max_value=1000,
    value=0,
    step=1,
    key="chunk_overlap",
    help="Number of tokens shared between consecutive chunks to maintain context",
)

embedding_model = st.selectbox(
    label="Embedding Model",
    options=list_embedding_models(),
    index=0,
    format_func=lambda option: f"{option['modelName']} ({option['modelId']})",
    key="embedding_model",
    help="The model used to encapsulate information \
        into dense representations in a multi-dimensional space",
)

# For more information, see
# https://umap-learn.readthedocs.io/en/latest/parameters.html#n-components
n_components = st.radio(
    label="UMAP Components",
    options=[2, 3],
    index=0,
    key="n_components",
    help="The dimensionality of the reduced dimension space",
)

if embedding_model is not None:
    st.session_state.embedding_function = AmazonBedrockEmbeddingFunction(
        session=session,
        model_name=embedding_model['modelId']
    )

if st.button(label="Build"):
    if uploaded_file is None:
        st.error("No file uploaded!")
    else:
        if st.session_state.collection is None:
            build_collection()

        if st.session_state.projections is None:
            create_projections()

        st.success("Vector database created!")

if st.session_state.projections is not None:

    st.markdown("### 3. Explore the embedding space 🖼️")

    if len(st.session_state.projections[0]) != n_components:
        create_projections()

    df = pd.DataFrame({
        'id': [int(id) for id in st.session_state.ids],
        'x': st.session_state.projections[:, 0],
        'y': st.session_state.projections[:, 1],
        'z': st.session_state.projections[:, 2] if n_components == 3 else None,
        'document_cleaned': [
            "<br>".join(wrap(doc, 100))
                for doc in st.session_state.documents
        ],
        'category': "chunk",
    })

    n_results = st.number_input(
        label="Number of results",
        min_value=1,
        max_value=10,
        value=5,
        key='n_results',
    )

    retrieval_strategy = st.radio(
        label="Retrieval Strategy",
        options=[
            "Naive",
            "Multiple Queries",
            "Generated Answer",
        ],
        captions=[
            "Retrieves chunks with high similarity to the user query",
            "Expands the original query by generating additional sub-queries",
            "Expands the original query by generating an hypothetical answer",
        ],
        key='retrieval_strategy',
    )

    query = st.text_input(
        label="Query",
        key='query',
    )

    if st.button(label="Search"):
        # Get query projections
        st.session_state.query_projections = project_embeddings(
            get_embeddings(query),
            st.session_state.umap_transform
        )
        df_query = pd.DataFrame({
            'x': st.session_state.query_projections[:, 0],
            'y': st.session_state.query_projections[:, 1],
            'z': st.session_state.query_projections[:, 2] if n_components == 3 else None,
            'document_cleaned': query,
            'category': "query",
        })

        # Use the query as is or expand it
        if retrieval_strategy == "Naive":
            chroma_query = query
        elif retrieval_strategy == "Multiple Queries":
            chroma_query, df_query = multiple_queries_expansion(query, df_query)
        elif retrieval_strategy == "Generated Answer":
            chroma_query, df_query = generated_answer_expansion(query, df_query)

        # Search query and process results
        results = st.session_state.collection.query(
            query_texts=chroma_query,
            n_results=n_results,
            include=['documents', 'embeddings']
        )

        st.session_state.retrieved_ids = []
        for ids in results['ids']:
            st.session_state.retrieved_ids.extend(ids)
        st.session_state.retrieved_ids = map(int, st.session_state.retrieved_ids)
        df.loc[df['id'].isin(st.session_state.retrieved_ids), 'category'] = "retrieved"

        # Append query projections
        df = pd.concat([df, df_query], axis=0)

    # Display all projections
    st.markdown("#### Projections")
    fig = plot_projections(df)
    buffer = io.StringIO()
    fig.write_html(buffer, include_plotlyjs="cdn")
    html_bytes = buffer.getvalue().encode()

    # Download plot as HTML
    st.download_button(
        label="Download HTML",
        data=html_bytes,
        file_name=f"{uploaded_file.name.split('.')[0]}.html",
        mime="text/html",
    )

    if st.session_state.retrieved_ids is not None:
        # Display query results
        st.markdown("#### Results")
        st.dataframe(
            df[df['category'] == "retrieved"].drop(['category'], axis=1),
            use_container_width=True,
            hide_index=True,
            column_config={
                'x': st.column_config.NumberColumn(
                    width="small"
                ),
                'document_cleaned': st.column_config.TextColumn(
                    "chunk",
                    width="large"
                )
            },
        )

        # Download query results as CSV
        st.download_button(
            label="Download CSV",
            data=df.to_csv(),
            file_name=f"results.csv",
            mime="text/csv",
        )
