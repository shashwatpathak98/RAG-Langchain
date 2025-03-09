# Retrieval-Augmented Generation (RAG) with LangChain and Google Gemini

A powerful RAG system that enhances AI responses with factual knowledge retrieval, built using LangChain and Google's Gemini models.

## Overview

This project implements a Retrieval-Augmented Generation (RAG) system that combines the strengths of retrieval-based and generation-based AI approaches. RAG enhances large language models by providing them with relevant information retrieved from a knowledge base, resulting in more accurate and factually grounded responses.

## Features

- **Document Processing**: Load and process text documents from a local directory
- **Semantic Chunking**: Split documents into manageable pieces with customizable chunk size and overlap
- **Vector Embeddings**: Convert text chunks into vector representations using Google's embedding models
- **Similarity Search**: Efficiently retrieve relevant information using Chroma vector database
- **Enhanced Generation**: Generate accurate responses using Google Gemini models with retrieved context
- **Interactive Interface**: Simple command-line interface for querying the system
- **Debug Mode**: Examine retrieved documents to understand system behavior

## Installation

1. **Clone the repository:**

   ```bash
   git clone <repository-url>
   cd RAG-Langchain
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate
   // # On Windows:
   venv\Scripts\activate
   ```
