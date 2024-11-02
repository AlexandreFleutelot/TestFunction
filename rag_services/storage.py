# rag_services/storage.py

import os
from typing import List, Dict, Any
from pymongo import MongoClient
from .models import Chunk
import numpy as np

class ChunkStore:
    def __init__(self, connection_string: str, database_name: str, container_name: str):
        self.client = MongoClient(connection_string)
        self.db = self.client[database_name]
        self.collection = self.db[container_name]

    def store_chunks(self, chunks: List[Chunk]):
        """Store a list of chunks in the database."""
        for chunk in chunks:
            self.collection.insert_one(chunk.to_dict())

    def get_chunks(self, query: Dict[str, Any] = {}) -> List[Chunk]:
        """Retrieve chunks from the database based on a query."""
        chunk_dicts = self.collection.find(query)
        chunks = []
        for chunk_dict in chunk_dicts:
            chunks.append(Chunk.from_dict(chunk_dict))
        return chunks

    def search_similar_chunks(self, embedding: np.ndarray, top_k: int = 5) -> List[Chunk]:
        """Search for chunks with similar embeddings."""
        # Assuming embeddings are stored as lists in the database
        all_chunks = self.get_chunks()
        similarities = []
        for chunk in all_chunks:
            if chunk.embedding is not None:
                similarity = np.dot(embedding, chunk.embedding) / (np.linalg.norm(embedding) * np.linalg.norm(chunk.embedding))
                similarities.append((similarity, chunk))
        
        # Sort by similarity and return the top_k chunks
        similarities.sort(key=lambda x: x[0], reverse=True)
        return [chunk for _, chunk in similarities[:top_k]]