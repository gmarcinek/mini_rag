# Standard library
from pathlib import Path
import hashlib
import json
from typing import List, Tuple, Dict, Optional, Any

# Third party
import numpy as np

# Local
from src.chunking import Chunk, SimpleTextSplitter
from src.generation.ollama import OllamaGenerator
from src.cache import BaseCache
from src.embeddings import PolishLegalEmbedder
from src.retrieval.semantic import SemanticRetriever

class MiniRAG:
    def __init__(self, 
            use_gpu: bool = False,
            cache_dir: str = "cache", 
            chunker: SimpleTextSplitter = None, 
            generator_model: str = "llama3.2",
            min_score_threshold: float = 0.8,
            max_top_k: int = 10,
            max_context_length: int = 32000):

        self.embedder = PolishLegalEmbedder(use_gpu=use_gpu)
        self.cache = BaseCache(cache_dir)
        self.retriever = SemanticRetriever(
            embedder=self.embedder,
            min_score_threshold=min_score_threshold,
            max_top_k=max_top_k
        )
        
        self.max_context_length = max_context_length
        self.documents, self.embeddings = self.cache.load_cache()
        self.chunker = chunker if chunker is not None else SimpleTextSplitter()
        
        self.generator = OllamaGenerator(
            model_name=generator_model,
            max_context_length=self.max_context_length
        )

    def clear_documents(self):
        self.documents = []
        self.embeddings = []
        self.cache.clear_cache()

    def add_documents(self, texts: List[str]):
        print(f"Processing {len(texts)} documents...")
        for i, text in enumerate(texts):
            doc_hash = hashlib.md5(text.encode()).hexdigest()
            
            if any(chunk.doc_id.startswith(f"doc_hash_{doc_hash}") for chunk in self.documents):
                print(f"Dokument {i} już istnieje, pomijam...")
                continue

            chunks = self.chunker.split_text(text, doc_id=f"doc_hash_{doc_hash}")
            for chunk in chunks:
                embedding = self.cache.get_embedding(chunk.text, self.embedder)
                self.documents.append(chunk)
                self.embeddings.append(embedding)
        
        self.cache.save_cache(self.documents, self.embeddings)
        print(f"Łączna liczba chunków po dodaniu: {len(self.documents)}")

    def retrieve(self, query: str, top_k: Optional[int] = None, 
                min_score: Optional[float] = None) -> List[Tuple[Chunk, float]]:
        return self.retriever.retrieve(
            query=query,
            documents=self.documents,
            embeddings=self.embeddings,
            top_k=top_k,
            min_score=min_score
        )

    def query(self, question: str, top_k: Optional[int] = None, 
        min_score: Optional[float] = None) -> Dict[str, Any]:
        retrieved_chunks = self.retrieve(question, top_k, min_score)
        
        print(f"\nWybrane chunki dla zapytania: '{question}'")
        for chunk, score in retrieved_chunks:
            print(f"- Score: {score:.3f}, Chunk ID: {chunk.chunk_id}")
            print(f"  Treść: {chunk.text}")
            print("---")  # Separator między chunkami
        
        contexts = [chunk for chunk, score in retrieved_chunks]
        answer = self.generator.generate(question, contexts)
        
        return {
            "answer": answer,
            "retrieved_chunks": retrieved_chunks
        }