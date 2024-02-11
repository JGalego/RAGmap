# RAGmap 🗺️🔍

## Overview

RAGmap is a simple RAG visualization tool for exploring document chunks and queries in embedding space.

> Inspired by DeepLearning.ai's short course on [Advanced Retrieval for AI with Chroma](https://www.deeplearning.ai/short-courses/advanced-retrieval-for-ai/) and Gabriel Chua's award-winning [RAGxplorer](https://github.com/gabrielchua/RAGxplorer).

**Updates:**
* 👨‍💻 [RAGmap](https://pypi.org/project/ragmap) is now available as a standalone Python package!
* 🌐 Try the [live demo](https://ragmap.streamlit.app/) hosted on Streamlit Community Cloud.
* 📢 RAGmap now supports [Hugging Face 🤗](https://huggingface.co) models!

<img src="https://github.com/JGalego/RAGmap/raw/main/images/ragmap_hf_support.gif" width="75%"/>

## What's inside?

RAGmap supports the following features:

* Multiple document formats 📄
	- `PDF`
	- `DOCX`
	- `PPTX`
* Multiple embedding models
	- Hugging Face 🤗
	- Amazon Bedrock ⛰️
		- [Titan Text Embeddings](https://docs.aws.amazon.com/bedrock/latest/userguide/titan-embedding-models.html)
		- [Titan Multimodal Embeddings](https://docs.aws.amazon.com/bedrock/latest/userguide/titan-multiemb-models.html)
		- [Cohere Embed English](https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-embed.html)
		- [Cohere Embed Multilingual](https://aws.amazon.com/blogs/aws/amazon-bedrock-now-provides-access-to-cohere-command-light-and-cohere-embed-english-and-multilingual-models/)
* Dimensionality reduction (2D and 3D)
	- [UMAP](https://umap-learn.readthedocs.io/en/latest/)
	- [t-SNE](https://opentsne.readthedocs.io/en/stable/)
	- [PCA](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html)
* Natural language queries
* Advanced query augmentation
	- Generated Answers (HyDE)
	- Multiple Queries
* ... and more!

☝️⚠️ **Important notice:** As of January 2024, chromadb's `AmazonBedrockEmbeddingFunction` only works with Titan models. Feel free to upvote this [PR](https://github.com/chroma-core/chroma/pull/1675) to add support for [Cohere Embed models](https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-embed.html).

<img src="https://github.com/JGalego/RAGmap/raw/main/images/ragmap.gif" width="75%"/>

## Prerequisites

### Amazon Bedrock

Enable access to the embedding ([Titan Embeddings](https://docs.aws.amazon.com/bedrock/latest/userguide/titan-embedding-models.html), [Cohere Embed](https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-embed.html)) and text ([Anthropic Claude](https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-claude.html)) models via Amazon Bedrock.

> For more information on how to request model access, please refer to the [Amazon Bedrock User Guide](https://docs.aws.amazon.com/bedrock/latest/userguide/model-access.html) (**Set up > Model access**)

<img src="https://github.com/JGalego/RAGmap/raw/main/images/bedrock_model_access.png" width="75%"/>

## How to use

### Option 1 💻

1. Install dependencies

	```
	pip install -r requirements.txt
	```

2. Run the application

	```
	streamlit run app.py
	```

3. Point your browser to http://localhost:8501

### Option 2 🐳

1. Run the following command to start the application

	```
	docker-compose up
	```

2. Once the service is up and running, head over to http://localhost:8501

## Option 3 👨‍💻

1. Install the `ragmap` package

	```
	pip install ragmap
	```

2. Start building your own apps.

	> Check out the [examples](examples) folder to get started!

## Example: [Amazon shareholder letters](https://medium.com/@austenallred/every-amazon-shareholder-letter-as-downloadable-pdf-4eb2ae886018)

<img src="https://github.com/JGalego/RAGmap/raw/main/images/amazon_shareholder_letters.png" width="70%">

## References

* (AWS) [What is Retrieval-Augmented Generation?](https://aws.amazon.com/what-is/retrieval-augmented-generation/)
* (DeepLearning.ai) [Advanced Retrieval for AI with Chroma](https://www.deeplearning.ai/short-courses/advanced-retrieval-for-ai/)