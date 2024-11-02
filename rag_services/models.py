# rag_services/models.py

from dataclasses import dataclass, asdict, field
from typing import Dict, Any, Optional
import numpy as np
import uuid

@dataclass
class Chunk:
    """Represents a document chunk with its content and metadata"""
    content: str
    _id: Optional[str] = field(default_factory=lambda: str(uuid.uuid4()))
    embedding: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert the Chunk instance to a dictionary, excluding None values."""
        data = asdict(self)
        if self.embedding is not None:
            data['embedding'] = self.embedding.tolist()
        # Remove keys with None values
        return {k: v for k, v in data.items() if v is not None}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Chunk':
        """Create a Chunk instance from a dictionary, converting embedding back to numpy array if it exists."""
        chunk_data = data.copy()
        if 'embedding' in chunk_data and chunk_data['embedding'] is not None:
            chunk_data['embedding'] = np.array(chunk_data['embedding'])
        return cls(**chunk_data)