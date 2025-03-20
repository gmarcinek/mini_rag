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

    def _text_hash(self, text: str) -> str:
        return hashlib.md5(text.encode()).hexdigest()

    def get_embedding(self, text: str, embedder: PolishLegalEmbedder) -> np.ndarray:
        """Get embedding for text, using cache if available."""
        text_hash = self._text_hash(text)
        embedding_path = self.embeddings_dir / f"{text_hash}.npy"

        if embedding_path.exists():
            try:
                return np.load(embedding_path)
            except Exception as e:
                print(f"Błąd odczytu {embedding_path}: {e}, regeneruję...")

        print(f"Generating new embedding for text: {text[:50]}...")
        embedding = embedder.get_embedding(text)
        np.save(embedding_path, embedding)
        return embedding

    def save_cache(self, documents: List[LegalChunk], embeddings: List[np.ndarray]) -> None:
        """
        Zapisuje chunki i ich embeddingi w cache z pełnym kontekstem strukturalnym.
        """
        existing_chunk_ids = set()
        chunks_info = []

        # Wczytaj istniejące dane (jeśli są)
        if self.chunks_info_path.exists():
            try:
                with self.chunks_info_path.open('r', encoding='utf-8') as f:
                    existing_chunks = json.load(f)
                    for chunk_data in existing_chunks:
                        existing_chunk_ids.add(chunk_data['chunk_id'])
                        chunks_info.append(chunk_data)
            except Exception as e:
                print(f"Error reading existing cache: {e}")
                self.clear_cache()
                chunks_info = []

        # Dodaj nowe
        for chunk, embedding in zip(documents, embeddings):
            if chunk.chunk_id in existing_chunk_ids:
                continue  # Pomijamy duplikaty

            text_hash = self._text_hash(chunk.text)
            np_path = self.embeddings_dir / f"{text_hash}.npy"
            if not np_path.exists():
                np.save(np_path, embedding)

            chunk_info = {
                "text": chunk.text,
                "doc_id": chunk.doc_id,
                "chunk_id": chunk.chunk_id,
                "embedding_hash": text_hash,
                "section_type": chunk.section_type,
                "section_id": chunk.section_id,
                "line_start": chunk.line_start,
                "line_end": chunk.line_end,
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
        """
        if not self.chunks_info_path.exists():
            return [], []

        try:
            with self.chunks_info_path.open('r', encoding='utf-8') as f:
                chunks_data = json.load(f)

            documents = []
            embeddings = []
            for chunk_data in chunks_data:
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

                embedding_path = self.embeddings_dir / f"{chunk_data['embedding_hash']}.npy"
                if embedding_path.exists():
                    try:
                        embedding = np.load(embedding_path)
                        documents.append(chunk)
                        embeddings.append(embedding)
                        print(f"Loaded cached chunk and embedding for doc_id: {chunk.doc_id}, chunk_id: {chunk.chunk_id}")
                    except Exception as e:
                        print(f"Nie można załadować embeddingu {embedding_path}: {e}")
                else:
                    print(f"Brak embeddingu: {embedding_path}, pomijam chunk {chunk.chunk_id}")

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
