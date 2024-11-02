# rag_services/chunking.py

import torch
import numpy as np
from typing import List, Tuple
from .models import Chunk

def get_optimal_boundaries(
    token_embeddings: torch.Tensor,
    min_chunk_size: int = 100,
    max_chunk_size: int = 200
) -> List[Tuple[int, int]]:
    embeddings = token_embeddings.numpy()
    
    similarity_matrix = np.dot(embeddings, embeddings.T)
    mean_similarity = np.mean(similarity_matrix[np.triu_indices(similarity_matrix.shape[0], k=1)])
    similarity_matrix = similarity_matrix - mean_similarity
    np.fill_diagonal(similarity_matrix, 0)

    n = similarity_matrix.shape[0]
    dp = np.zeros(n)
    segmentation = np.zeros(n, dtype=int)

    for i in range(n):
        max_reward = float('-inf')
        best_start = i
        
        for size in range(min_chunk_size, min(max_chunk_size + 1, i + 2)):
            if i - size + 1 >= 0:
                reward = np.sum(similarity_matrix[i - size + 1:i + 1, i - size + 1:i + 1])
                if i - size >= 0:
                    reward += dp[i - size]
                if reward > max_reward:
                    max_reward = reward
                    best_start = i - size + 1
        
        dp[i] = max_reward
        segmentation[i] = best_start

    boundaries = []
    i = n - 1
    while i >= 0:
        boundaries.append((segmentation[i], i))
        i = segmentation[i] - 1

    boundaries.reverse()
    return boundaries

def create_chunks(
    text: str,
    token_embeddings: torch.Tensor,
    offset_mapping: List[Tuple[int, int]],
    min_chunk_size: int = 100,
    max_chunk_size: int = 200
) -> List[Chunk]:
    
    boundaries = get_optimal_boundaries(
            token_embeddings, 
            min_chunk_size=min_chunk_size, 
            max_chunk_size=max_chunk_size
            )
    
    chunks = []
    for start_idx, end_idx in boundaries:
        text_start = offset_mapping[start_idx][0]
        text_end = offset_mapping[end_idx][1]
        chunk_text = text[text_start:text_end]
        
        chunk_embedding = token_embeddings[start_idx:end_idx + 1].mean(dim=0).numpy()
        
        chunk = Chunk(
            content=chunk_text,
            embedding=chunk_embedding,
            metadata={
                "text_start": text_start, 
                "text_end": text_end
                }
        )
        chunks.append(chunk)
    return chunks