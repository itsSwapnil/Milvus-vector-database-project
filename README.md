# Milvus-vector-database-project
This project asynchronously scrapes web content, generates semantic text chunks using sentence embeddings, and stores them in a Milvus vector database for efficient similarity search. Built with Python, Langchain, SentenceTransformers, and Milvus for scalable vector-based retrieval.

# Semantic Web Scraper with Milvus Vector Search

This project scrapes content from multiple websites asynchronously, tokenizes and embeds the content into semantic chunks using Sentence Transformers, and stores them in a Milvus vector database for efficient similarity search and retrieval.

### 📦 Dependencies
Python 3.9
aiohttp
nltk
pandas
sentence-transformers
pymilvus
langchain
scikit-learn
numpy

## 🚀 Features

- Asynchronous web scraping with `aiohttp` and `langchain`
- Semantic chunking using NLTK sentence tokenization
- Embedding with `sentence-transformers/all-MiniLM-L6-v2`
- Vector similarity search using Milvus
- Dockerized setup with Milvus, MinIO, Etcd, and Python environment

## 🐳 Docker Setup

### 1. Clone the Repository

``` git clone https://github.com/yourusername/semantic-web-milvus.git
cd semantic-web-milvus
2. Start Docker Services

docker-compose up --build -d
This will spin up:

Milvus vector database

Etcd (metadata service)

MinIO (object storage)

Python container (milvus-python) with all dependencies pre-installed

3. Access Python Container

docker exec -it milvus-python bash
Run your main script:
python your_script.py
```

### 📂 Project Structure

├── docker-compose.yml

├── Dockerfile.python

├── scripts/

│   └── your_script.py

├── volumes/

│   ├── etcd/

│   ├── milvus/

│   └── minio/

### ✨ Use Cases
Building search engines over scraped web content

Knowledge base construction with semantic search

Content recommendation systems

### 📜 License
This project is licensed under the MIT License.
