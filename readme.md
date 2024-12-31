# Uplifted RAG Systems with CometML Opik
## Building an Agentic Knowledge Graph for Enhanced Information Retrieval

This repository demonstrates an advanced implementation of a Retrieval-Augmented Generation (RAG) system that combines graph-based knowledge representation with agentic capabilities, monitored and optimized through CometML Opik. The system transforms traditional RAG approaches by enabling context-aware, multi-hop reasoning while maintaining comprehensive observability.

## Table of Contents
- [Overview](#overview)
- [Key Features](#key-features)
- [System Architecture](#system-architecture)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)

## Overview

Traditional RAG systems often struggle with linear document retrieval, context blindness, and limited reasoning capabilities. This implementation addresses these challenges by:
- Implementing a graph-based knowledge structure for complex relationship modeling
- Integrating autonomous agents for dynamic query processing
- Providing comprehensive monitoring through CometML Opik
- Enabling multi-hop reasoning across interconnected documents

## Key Features

### Advanced RAG Capabilities
- Graph-based knowledge representation using Neo4j
- Semantic search with vector embeddings
- Multi-hop reasoning across document relationships
- Context-aware query processing

### Agent Integration
- Dynamic tool selection based on query context
- Adaptive exploration strategies
- Comprehensive conversation memory
- Fallback mechanisms for robust operation

### Monitoring & Optimization
- Real-time performance tracking with CometML Opik
- Detailed metrics collection and visualization
- Hyperparameter optimization
- Model versioning and experiment tracking

### User Interface
- Streamlit-based interactive interface
- Component-based architecture
- Session state management
- Metric visualization

## System Architecture

The system consists of several key components:

1. **Data Pipeline**
   - Raw data preprocessing
   - Neo4j graph database integration
   - Parallel data ingestion system
   - Vector store bridging

2. **Core RAG Components**
   - Question-answering pipeline
   - Paper lookup service
   - Agent-based tool orchestration
   - Coordinator pattern implementation

3. **Monitoring Infrastructure**
   - Real-time metric collection
   - Performance visualization
   - Error tracking
   - Session analytics
  
## Prerequisites
![Copy of Template Diagrame (1200 x 800 px) (1200 x 900 px) (1200 x 700 px)(3)](https://github.com/user-attachments/assets/79ac7fc8-03f1-41f7-b5f7-37e55955ad11)


- Neo4j Database
- CometML Account
- OpenAI API Key
- [arXiv Dataset from Kaggle](https://www.kaggle.com/datasets/Cornell-University/arxiv)

## Installation

1. **Clone the Repository**
   ```bash
   git clone https://github.com/mlvanguards/agentic-graph-rag-evaluation-cometml.git
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Environment Setup**
   Create a `.env` file with the following credentials:
   ```
   NEO4J_URI=your_neo4j_uri
   NEO4J_USER=your_username
   NEO4J_PASSWORD=your_password
   COMET_API_KEY=your_comet_key
   OPENAI_API_KEY=your_openai_key
   ```

## Usage

1. **Data Preprocessing**
   ```python
   python scripts/preprocess_data.py --input arxiv_data.json --output processed_data.json
   ```

2. **Database Ingestion**
   ```python
   cd src/
   python components/database/ingest.py
   ```

3. **Start the Application**
   ```python
   streamlit run main.py
   ```
