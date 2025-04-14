# RAGmap 🗺️🔍

## Overview

RAGmap is a simple RAG visualization tool for exploring document chunks and queries in embedding space.

> Inspired by DeepLearning.ai's short course on [Advanced Retrieval for AI with Chroma](https://www.deeplearning.ai/short-courses/advanced-retrieval-for-ai/) and Gabriel Chua's award-winning [RAGxplorer](https://github.com/gabrielchua/RAGxplorer).

**Updates:**
* 👨‍💻 [RAGmap](https://pypi.org/project/ragmap) is now available as a standalone Python package!
* 📢 RAGmap now supports [Hugging Face 🤗](https://huggingface.co), [OpenAI](https://platform.openai.com/docs/models) and [Google GenAI](https://cloud.google.com/vertex-ai/generative-ai/docs/models) models!

<img src="https://github.com/JGalego/RAGmap/raw/main/images/ragmap_hf_support.gif" width="75%"/>

## What's inside?

RAGmap supports the following features:

* Multiple document formats 📄
	- `PDF`
	- `DOCX`
	- `PPTX`
* Multiple embedding models
	- Amazon Bedrock ⛰️
	- Hugging Face 🤗
	- OpenAI ֎
	- Google GenAI 🔵🔴🟡🟢
* Dimensionality reduction (2D and 3D)
	- [UMAP](https://umap-learn.readthedocs.io/en/latest/)
	- [t-SNE](https://opentsne.readthedocs.io/en/stable/)
	- [PCA](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html)
* Natural language queries
* Advanced query augmentation
	- Generated Answers (HyDE)
	- Multiple Queries
* ... and more!

<img src="https://github.com/JGalego/RAGmap/raw/main/images/ragmap.gif" width="75%"/>