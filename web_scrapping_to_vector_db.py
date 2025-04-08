"""from langchain.document_loaders import WebBaseLoader
import re
urls=["https://batteryuniversity.com/article/bu-804a-corrosion-shedding-and-internal-short"]
loader=WebBaseLoader(urls)
docs=loader.load()
texts = " ".join(doc.page_content for doc in docs)
texts= re.sub(r'\s+', ' ', texts).strip()"""

import asyncio
import aiohttp
import re
import logging
from typing import List, Dict
from nltk.tokenize import sent_tokenize
import nltk
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from pymilvus import connections, Collection, CollectionSchema, FieldSchema, DataType, Index
from langchain.document_loaders import WebBaseLoader
nltk.download('punkt')
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


# docker-compose up --build -d
# docker exec -it milvus-python bash

class SemanticTextChunker:
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.embeddings = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    async def async_scrape_web_text(self, urls: List[str]) -> Dict[str, str]:
        """Scrape text from multiple URLs asynchronously."""
        async def fetch(url, session):
            try:
                loader = WebBaseLoader([url])
                docs = loader.load()
                text = " ".join(doc.page_content for doc in docs) if docs else ""
                text = re.sub(r'\s+', ' ', text).strip()
                if not text:
                    logging.warning(f"Empty content from {url}. Skipping.")
                return url, text
            except Exception as e:
                logging.error(f"Error scraping {url}: {e}")
                return url, ""
        async with aiohttp.ClientSession() as session:
            tasks = [fetch(url, session) for url in urls]
            results = await asyncio.gather(*tasks)
        return dict(results)
    def create_semantic_chunks(self, text: str, source_url: str) -> List[Dict]:
        """Create semantic chunks ensuring they fit within the 1024-character limit."""
        if not text.strip():
            logging.warning(f"Empty text received from {source_url}. Skipping chunking.")
            return []
        sentences = sent_tokenize(text)
        chunks = []
        chunk = ""
        for sentence in sentences:
            if len(chunk) + len(sentence) < self.chunk_size:
                chunk += " " + sentence
            else:
                chunk = chunk.strip()
                if len(chunk) > 8192:
                    logging.warning(f"Truncating chunk from {source_url} (Length: {len(chunk)})")
                    chunk = chunk[:8192]  # Ensure max length
                chunks.append(chunk)
                chunk = sentence
        if chunk:
            chunk = chunk.strip()
            if len(chunk) > 8192:
                logging.warning(f"Truncating last chunk from {source_url} (Length: {len(chunk)})")
                chunk = chunk[:8192]
            chunks.append(chunk)
        if not chunks:
            logging.warning(f"No valid chunks created from {source_url}. Skipping embeddings.")
            return []
        # Generate embeddings
        embeddings = self.embeddings.encode(chunks, convert_to_numpy=True)
        # Store chunks
        chunk_data = []
        for i, chunk in enumerate(chunks):
            chunk_data.append({
                'chunk_id': i,
                'content': chunk,
                'source_url': source_url,
                'embedding': embeddings[i],
                'length': len(chunk)
            })
        return chunk_data
    def insert_chunks_to_vector_db(self, collection_name: str, chunks_df: pd.DataFrame):
        """Insert chunked data into Milvus vector database with URL metadata."""
        if chunks_df.empty:
            logging.warning("No chunks to insert into Milvus. Skipping database insertion.")
            return
        # Ensure all 'content' fields are within 1024 characters before inserting
        chunks_df["content"] = chunks_df["content"].apply(lambda x: x[:8192] if len(x) > 8192 else x)
        # Log the max chunk length before insertion
        max_len = max(chunks_df["content"].apply(len))
        logging.info(f"Max chunk length before insertion: {max_len}")
        # Ensure correct columns exist
        expected_columns = {"content", "source_url", "embedding"}
        actual_columns = set(chunks_df.columns)
        missing_columns = expected_columns - actual_columns
        if missing_columns:
            logging.error(f"Missing columns in DataFrame: {missing_columns}")
            return
        # Connect to Milvus
        connections.connect("default", host="milvus-standalone", port="19530")
        # Define Milvus schema
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=8192),
            FieldSchema(name="source_url", dtype=DataType.VARCHAR, max_length=512),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=384),
        ]
        schema = CollectionSchema(fields, description="Semantic sentense Chunker Data")
        collection = Collection(name=collection_name, schema=schema)
        # Prepare data for insertion
        contents = chunks_df["content"].tolist()
        urls = chunks_df["source_url"].tolist()
        vector_data = chunks_df["embedding"].tolist()
        # Insert data
        collection.insert([contents, urls, vector_data])
        # Create index for optimized search
        index_params = {"metric_type": "IP", "index_type": "HNSW", "params": {"M": 16, "efConstruction": 200}}
        Index(collection, "embedding", index_params)
        # Load collection for queries
        collection.load()
        logging.info(f"Inserted {len(contents)} chunks into '{collection_name}' collection.")



async def main():
    collection_name = "web_scraping_semantic_chunks"
    chunker = SemanticTextChunker(chunk_size=500, chunk_overlap=50)
    urls = [
        "https://www.allaboutcircuits.com/technical-articles/category/automotive/",
        "https://rotontek.com/",
        "https://www.fukuta-motor.com.tw/en/news/K06/1",
        "https://www.mightyelectricmotors.com/a/blog/ev-motor-efficiency-a-comprehensive-guide?",
        "https://www.integrasources.com/blog/",
        "https://circuitdigest.com/tech-articles",
        "https://www.danatm4.com/products/systems/",
        "https://batteryuniversity.com/",
        "https://www.infineon.com/cms/en/product/"
    ]
    # Scrape text
    scraped_data = await chunker.async_scrape_web_text(urls)
    # Process chunks
    all_chunks = []
    for url, text in scraped_data.items():
        if text:
            chunks = chunker.create_semantic_chunks(text, url)
            all_chunks.extend(chunks)
    # Convert to DataFrame
    chunks_df = pd.DataFrame(all_chunks)
    # Insert into Milvus
    chunker.insert_chunks_to_vector_db(collection_name, chunks_df)

if __name__ == "__main__":
    asyncio.run(main())