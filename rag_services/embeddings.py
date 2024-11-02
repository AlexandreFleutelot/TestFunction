# rag_services/embeddings.py
import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np
from typing import List, Tuple

class EmbeddingService:
    def __init__(self, model_name: str = "jinaai/jina-embeddings-v3"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(self.device)
        self.model.eval()

    def get_position_weights(self, size: int) -> np.ndarray:
        x = np.arange(size)
        center = size / 2
        weights = np.exp(-0.5 * ((x - center)/(size/4))**2)
        return weights

    def get_token_embeddings(
            self,
            text: str,
            window_size: int = 6000,
            overlap_size: int = 3000,
        ) -> Tuple[torch.Tensor, List[Tuple[int, int]]]:
        
        tokens = self.tokenizer(
            text,
            return_offsets_mapping=True,
            return_attention_mask=True,
            add_special_tokens=False,
            return_tensors="pt"
        )
        
        total_tokens = len(tokens.input_ids[0])
        token_embeddings_sum = torch.zeros((total_tokens, self.model.config.hidden_size))
        token_weights_sum = torch.zeros(total_tokens)
        
        for start_idx in range(0, total_tokens, window_size - overlap_size):
            end_idx = min(start_idx + window_size, total_tokens)
            
            input_ids = tokens.input_ids[0][start_idx:end_idx].unsqueeze(0)
            attention_mask = tokens.attention_mask[0][start_idx:end_idx].unsqueeze(0)

            with torch.no_grad():
                model_result = self.model(input_ids, attention_mask=attention_mask, return_dict=True)
                window_embeddings = model_result.last_hidden_state[0]
            
            weights = torch.tensor(
                self.get_position_weights(end_idx - start_idx), 
                dtype=torch.float32
            )
            
            token_embeddings_sum[start_idx:end_idx] += window_embeddings * weights.unsqueeze(1)
            token_weights_sum[start_idx:end_idx] += weights
        
        final_token_embeddings = token_embeddings_sum / token_weights_sum.unsqueeze(1)
        
        return final_token_embeddings, tokens.offset_mapping[0].tolist()