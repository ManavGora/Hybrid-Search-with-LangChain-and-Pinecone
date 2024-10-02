# Hybrid Search with LangChain and Pinecone

This project demonstrates a hybrid search solution using **LangChain**, **Pinecone**, and **HuggingFace Sentence Transformers**. The approach combines dense vector embeddings with sparse BM25 encoding to achieve more effective search results, incorporating both semantic and keyword-based relevance.

![](https://github.com/ManavGora/Hybrid-Search-with-LangChain-and-Pinecone/blob/main/Pinecone.png)

## Table of Contents
- [Overview](#overview)
- [Technologies Used](#technologies-used)
- [Setup Instructions](#setup-instructions)
- [How It Works](#how-it-works)
- [Usage](#usage)
- [Examples](#examples)


## Overview
In modern search systems, combining semantic search (dense embeddings) with traditional keyword-based search (sparse encoding like BM25) is an effective way to improve retrieval quality. This project implements such a hybrid search using the following components:
- **Dense embeddings** using HuggingFace's pre-trained Sentence Transformers.
- **Sparse encoding** using Pinecone's BM25Encoder.
- **Pinecone** as the vector database for efficient large-scale indexing and searching.

## Technologies Used
- **LangChain**: Framework for building applications with LLMs.
- **Pinecone**: Scalable vector search engine with hybrid search capabilities.
- **HuggingFace**: Pre-trained sentence transformers for dense embeddings.
- **BM25**: Sparse keyword-based retrieval mechanism.
- **Python**: The main programming language used.

## Setup Instructions

### Prerequisites
Make sure you have the following installed:
- Python 3.7+
- A Pinecone account (get your API key from [Pinecone](https://www.pinecone.io/)).
- HuggingFace account for Sentence Transformers.

### Installation
1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/hybrid-search-langchain.git
    cd hybrid-search-langchain
    ```

2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Set up environment variables for Pinecone and HuggingFace:
    - Create a `.env` file in the project root and add your API keys:
      ```bash
      PINECONE_API_KEY=your_pinecone_api_key
      HF_TOKEN=your_huggingface_token
      ```

### Create Pinecone Index
Ensure you have created a Pinecone index for hybrid search:
```python
from pinecone import Pinecone, ServerlessSpec

api_key = "your_pinecone_api_key"
pc = Pinecone(api_key=api_key)
index_name = "hybrid-search-langchain-pinecone"

if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=384,  # match the embedding size
        metric='dotproduct',
        spec=ServerlessSpec(cloud='aws', region='us-east-1')
    )
```

## How It Works

1. **Dense Embedding**: Sentences or documents are converted into dense vector representations using HuggingFace Sentence Transformers.
2. **Sparse Encoding**: The BM25 algorithm is used to create sparse vectors based on word occurrences.
3. **Hybrid Search**: Combines the results of dense and sparse searches, leveraging both the semantic and keyword-based relevance to return more accurate results.
4. **Retrieval**: Results are retrieved and ranked based on combined scores from both dense and sparse searches.

## Usage

### Add Texts to Pinecone Index
```python
retriever.add_texts([
    "In 2023, I visited Pune ashram",
    "In 2024, I will visit Oregon",
    "In 2025, I will build my Empire"
])
```

### Query the Hybrid Search
```python
results = retriever.invoke("What will I do in 2025")
for doc in results:
    print(doc.page_content)
```

### Expected Output:
```
In 2025, I will build my Empire
In 2024, I will visit Oregon
In 2023, I visited Pune ashram
```

## Examples

1. Add texts to the index:
    ```python
    sentences = [
        "In 2023, I visited Pune ashram",
        "In 2024, I will visit Oregon",
        "In 2025, I will build my Empire"
    ]
    retriever.add_texts(sentences)
    ```

2. Invoke the hybrid search:
    ```python
    query = "What will I do in 2025"
    results = retriever.invoke(query)
    for doc in results:
        print(doc.page_content)
    ```

3. Save and load BM25 values:
    ```python
    bm25_encoder.dump("bm25_values.json")
    bm25_encoder = BM25Encoder().load("bm25_values.json")
    ```

