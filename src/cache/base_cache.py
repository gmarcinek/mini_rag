from abc import ABC, abstractmethod
from pathlib import Path
import hashlib
import json
import numpy as np
from typing import List, Tuple, Dict, Optional, Any
from src.chunking import Chunk
from src.embeddings import BertEmbedder

class BaseCache(ABC):
    def __init__(self, cache_dir: str = "cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.embeddings_dir = self.cache_dir / "embeddings"
        self.embeddings_dir.mkdir(exist_ok=True)
        self.chunks_info_path = self.cache_dir / "chunks_info.json"
    
    def get_embedding(self, text: str, embedder: BertEmbedder) -> np.ndarray:
        """Get embedding for text, using cache if available."""
        text_hash = hashlib.md5(text.encode()).hexdigest()
        embedding_path = self.embeddings_dir / f"{text_hash}.npy"

        if embedding_path.exists():
            return np.load(embedding_path)

        print(f"Generating new embedding for text: {text[:50]}...")
        embedding = embedder.get_embedding(text)
        np.save(embedding_path, embedding)
        return embedding

    def load_cache(self) -> Tuple[List[Chunk], List[np.ndarray]]:
        if not self.chunks_info_path.exists():
            return [], []

        try:
            with self.chunks_info_path.open('r', encoding='utf-8') as f:
                chunks_data = json.load(f)

            documents = []
            embeddings = []
            for chunk_data in chunks_data:
                chunk = Chunk(
                    text=chunk_data['text'],
                    doc_id=chunk_data['doc_id'],
                    chunk_id=chunk_data['chunk_id']
                )
                embedding_path = self.embeddings_dir / f"{chunk_data['embedding_hash']}.npy"
                if embedding_path.exists():
                    embedding = np.load(embedding_path)
                    documents.append(chunk)
                    embeddings.append(embedding)
                    print(f"Loaded cached chunk and embedding for doc_id: {chunk.doc_id}, chunk_id: {chunk.chunk_id}")
            return documents, embeddings
        except Exception as e:
            print(f"Error loading cache: {e}")
            self.clear_cache()
            return [], []

    def save_cache(self, documents: List[Chunk], embeddings: List[np.ndarray]) -> None:
        chunks_info = []
        for chunk, embedding in zip(documents, embeddings):
            embedding_hash = self._calculate_hash(embedding)
            
            np_path = self.embeddings_dir / f"{embedding_hash}.npy"
            if not np_path.exists():
                np.save(np_path, embedding)
            
            chunks_info.append({
                "text": chunk.text,
                "doc_id": chunk.doc_id,
                "chunk_id": chunk.chunk_id,
                "embedding_hash": embedding_hash
            })
        
        try:
            with self.chunks_info_path.open('w', encoding='utf-8') as f:
                json.dump(chunks_info, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Error saving cache: {e}")
            self.clear_cache()

    def clear_cache(self) -> None:
        print("Clearing cache...")
        if self.chunks_info_path.exists():
            self.chunks_info_path.unlink()
        for file in self.embeddings_dir.glob("*.npy"):
            file.unlink()

    def _calculate_hash(self, data: np.ndarray) -> str:
        return hashlib.md5(data.tobytes()).hexdigest()