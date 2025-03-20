from abc import ABC, abstractmethod
from pathlib import Path
import hashlib
import json
import numpy as np
from typing import List, Tuple, Dict, Optional, Any
from src.chunking import Chunk
from src.embeddings import PolishLegalEmbedder
from src.chunking.hierarchical_chunker import LegalChunk

class BaseCache(ABC):
    def __init__(self, cache_dir: str = "cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.embeddings_dir = self.cache_dir / "embeddings"
        self.embeddings_dir.mkdir(exist_ok=True)
        self.chunks_info_path = self.cache_dir / "chunks_info.json"
    
    def get_embedding(self, text: str, embedder: PolishLegalEmbedder) -> np.ndarray:
        """Get embedding for text, using cache if available."""
        text_hash = hashlib.md5(text.encode()).hexdigest()
        embedding_path = self.embeddings_dir / f"{text_hash}.npy"

        if embedding_path.exists():
            return np.load(embedding_path)

        print(f"Generating new embedding for text: {text[:50]}...")
        embedding = embedder.get_embedding(text)
        np.save(embedding_path, embedding)
        return embedding

    def save_cache(self, documents: List[LegalChunk], embeddings: List[np.ndarray]) -> None:
        """
        Zapisuje chunki i ich embeddingi w cache z pełnym kontekstem strukturalnym.
        
        Args:
            documents: Lista obiektów LegalChunk
            embeddings: Lista odpowiadających embeddingów numpy
        """
        chunks_info = []
        for chunk, embedding in zip(documents, embeddings):
            embedding_hash = self._calculate_hash(embedding)
            
            # Zapisz embedding jeśli nie istnieje
            np_path = self.embeddings_dir / f"{embedding_hash}.npy"
            if not np_path.exists():
                np.save(np_path, embedding)
            
            # Tworzenie rozszerzonej struktury JSON z pełnym kontekstem
            chunk_info = {
                # Podstawowe informacje o chunku
                "text": chunk.text,
                "doc_id": chunk.doc_id,
                "chunk_id": chunk.chunk_id,
                "embedding_hash": embedding_hash,
                
                # Informacje strukturalne
                "section_type": chunk.section_type,
                "section_id": chunk.section_id,
                "line_start": chunk.line_start,
                "line_end": chunk.line_end,
                
                # Pełny kontekst hierarchiczny
                "context_path": chunk.context_path,
            }
            
            chunks_info.append(chunk_info)
        
        try:
            with self.chunks_info_path.open('w', encoding='utf-8') as f:
                json.dump(chunks_info, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Error saving cache: {e}")
            self.clear_cache()

    def load_cache(self) -> Tuple[List[LegalChunk], List[np.ndarray]]:
        """
        Ładuje chunki i ich embeddingi z cache.
        
        Returns:
            Krotka (lista chunków, lista embeddingów)
        """
        if not self.chunks_info_path.exists():
            return [], []

        try:
            with self.chunks_info_path.open('r', encoding='utf-8') as f:
                chunks_data = json.load(f)

            documents = []
            embeddings = []
            for chunk_data in chunks_data:
                # Tworzenie obiektu LegalChunk z pełnymi danymi
                chunk = LegalChunk(
                    text=chunk_data['text'],
                    doc_id=chunk_data['doc_id'],
                    chunk_id=chunk_data['chunk_id'],
                    section_type=chunk_data.get('section_type', ''),
                    section_id=chunk_data.get('section_id', ''),
                    context_path=chunk_data.get('context_path', []),
                    line_start=chunk_data.get('line_start', 0),
                    line_end=chunk_data.get('line_end', 0)
                )
                
                # Ładowanie embeddingu
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

    def clear_cache(self) -> None:
        print("Clearing cache...")
        if self.chunks_info_path.exists():
            self.chunks_info_path.unlink()
        for file in self.embeddings_dir.glob("*.npy"):
            file.unlink()

    def _calculate_hash(self, data: np.ndarray) -> str:
        return hashlib.md5(data.tobytes()).hexdigest()