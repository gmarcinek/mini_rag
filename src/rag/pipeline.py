# Standard library
from pathlib import Path
import hashlib
import json
import time
from typing import List, Tuple, Dict, Optional, Any

# Third party
import numpy as np

# Local
from src.chunking import Chunk, SimpleTextSplitter
from src.generation.anthropic import AnthropicGenerator
from src.cache import BaseCache
from src.embeddings import PolishLegalEmbedder
from src.retrieval.semantic import SemanticRetriever

class MiniRAG:
    """
    Implementacja prostego systemu Retrieval-Augmented Generation (RAG).
    Łączy wyszukiwanie semantyczne z generacją odpowiedzi.
    """
    
    def __init__(self, 
            use_gpu: bool = False,
            cache_dir: str = "cache", 
            chunker: SimpleTextSplitter = None, 
            generator_model: str = "llama3.2",
            min_score_threshold: float = 0.6,
            max_top_k: int = 10,
            max_context_length: int = 32000,
            debug_mode: bool = False):

        self.debug_mode = debug_mode
        self.max_context_length = max_context_length
        
        # Inicjalizacja komponentów
        self.embedder = PolishLegalEmbedder(use_gpu=use_gpu)
        self.cache = BaseCache(cache_dir)
        self.retriever = SemanticRetriever(
            embedder=self.embedder,
            min_score_threshold=min_score_threshold,
            max_top_k=max_top_k
        )
        
        self.generator = AnthropicGenerator(
            # model_name=generator_model,
            # max_context_length=self.max_context_length
        )
        
        self.chunker = chunker if chunker is not None else SimpleTextSplitter()
        
        # Wczytaj zapisane dokumenty i embeddingi
        self.documents, self.embeddings = self.cache.load_cache()
        
        if self.debug_mode:
            print(f"Zainicjalizowano MiniRAG z {len(self.documents)} dokumentami")

    def clear_documents(self):
        """Wyczyść wszystkie dokumenty i embeddingi z pamięci i cache."""
        self.documents = []
        self.embeddings = []
        self.cache.clear_cache()
        if self.debug_mode:
            print("Wyczyszczono wszystkie dokumenty z pamięci i cache")

    def add_documents(self, texts: List[str], doc_ids: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Dodaj nowe dokumenty do systemu, podziel je na chunki i oblicz embeddingi.
        
        Args:
            texts: Lista tekstów dokumentów do dodania
            doc_ids: Opcjonalna lista identyfikatorów dokumentów. Jeśli nie podano,
                     wygenerowane zostaną identyfikatory bazujące na hash'u tekstu.
                     
        Returns:
            Słownik ze statystykami dodawania dokumentów
        """
        start_time = time.time()
        
        if doc_ids is None:
            doc_ids = [f"doc_hash_{hashlib.md5(text.encode()).hexdigest()}" for text in texts]
        
        if len(texts) != len(doc_ids):
            raise ValueError("Liczba tekstów musi być równa liczbie identyfikatorów dokumentów")
        
        stats = {
            "total_documents": len(texts),
            "added_documents": 0,
            "skipped_documents": 0,
            "total_chunks": 0,
            "processing_time": 0
        }
        
        if self.debug_mode:
            print(f"Przetwarzanie {len(texts)} dokumentów...")
        
        for i, (text, doc_id) in enumerate(zip(texts, doc_ids)):
            # Sprawdź, czy dokument już istnieje
            if any(chunk.doc_id.startswith(doc_id) for chunk in self.documents):
                if self.debug_mode:
                    print(f"Dokument {i} ({doc_id}) już istnieje, pomijam...")
                stats["skipped_documents"] += 1
                continue
            
            # Podziel tekst na chunki
            chunks = self.chunker.split_text(text, doc_id=doc_id)
            
            # Dodaj chunki i embeddingi
            for chunk in chunks:
                embedding = self.cache.get_embedding(chunk.text, self.embedder)
                self.documents.append(chunk)
                self.embeddings.append(embedding)
                stats["total_chunks"] += 1
            
            stats["added_documents"] += 1
        
        # Zapisz do cache
        if stats["added_documents"] > 0:
            self.cache.save_cache(self.documents, self.embeddings)
        
        stats["processing_time"] = time.time() - start_time
        
        if self.debug_mode:
            print(f"Dodano {stats['added_documents']} dokumentów, pominięto {stats['skipped_documents']}")
            print(f"Łączna liczba chunków: {len(self.documents)}")
            print(f"Czas przetwarzania: {stats['processing_time']:.2f}s")
        
        return stats

    def retrieve(self, query: str, top_k: Optional[int] = None, 
                min_score: Optional[float] = None) -> List[Tuple[Chunk, float]]:
        """
        Wyszukaj dokumenty semantycznie podobne do zapytania.
        
        Args:
            query: Zapytanie użytkownika
            top_k: Opcjonalna liczba najlepszych wyników do zwrócenia
            min_score: Opcjonalny minimalny próg podobieństwa
            
        Returns:
            Lista krotek (chunk, score) zawierających znalezione fragmenty i ich podobieństwo
        """
        if not self.documents:
            if self.debug_mode:
                print("Brak dokumentów do przeszukania")
            return []
        
        return self.retriever.retrieve(
            query=query,
            documents=self.documents,
            embeddings=self.embeddings,
            top_k=top_k,
            min_score=min_score
        )

    def query(self, question: str, top_k: Optional[int] = None, 
        min_score: Optional[float] = None) -> Dict[str, Any]:
        """
        Wykonaj zapytanie do systemu RAG - wyszukaj dokumenty i wygeneruj odpowiedź.
        
        Args:
            question: Pytanie użytkownika
            top_k: Opcjonalna liczba najlepszych wyników do użycia
            min_score: Opcjonalny minimalny próg podobieństwa
            fix_text: Czy poprawiać tekst wygenerowanej odpowiedzi
            
        Returns:
            Słownik z odpowiedzią oraz dodatkowymi metadanymi
        """
        start_time = time.time()
        result = {
            "question": question,
            "answer": "",
            "retrieved_chunks": [],
            "retrieval_time": 0,
            "generation_time": 0,
            "total_time": 0,
            "sources": []
        }
        
        # Pobierz dokumenty
        retrieval_start = time.time()
        retrieved_chunks = self.retrieve(question, top_k, min_score)
        result["retrieval_time"] = time.time() - retrieval_start
        result["retrieved_chunks"] = retrieved_chunks
        
        if self.debug_mode:
            print(f"\nWybrane chunki dla zapytania: '{question}'")
            for chunk, score in retrieved_chunks:
                print(f"- Score: {score:.3f}, Doc ID: {chunk.doc_id}, Chunk ID: {chunk.chunk_id}")
                print(f"  Treść: {(chunk.text[:100] + '...') if len(chunk.text) > 100 else chunk.text}")
                print("---")
        
        # Wygeneruj odpowiedź
        if not retrieved_chunks:
            result["answer"] = "Na podstawie dostępnych mi danych, nie mogę odpowiedzieć na to pytanie."
        else:
            generation_start = time.time()
            contexts = [chunk for chunk, _ in retrieved_chunks]
            
            answer = self.generator.generate(question, contexts)
            
            result["answer"] = answer
            result["generation_time"] = time.time() - generation_start
            
            # Przygotuj informacje o źródłach
            for chunk, score in retrieved_chunks:
                result["sources"].append({
                    "doc_id": chunk.doc_id,
                    "chunk_id": chunk.chunk_id,
                    "score": float(score),  # Konwersja na typ, który można serializować do JSON
                    "text": chunk.text[:200] + "..." if len(chunk.text) > 200 else chunk.text
                })
        
        result["total_time"] = time.time() - start_time
        
        return result
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Zwraca podstawowe statystyki systemu RAG.
        
        Returns:
            Słownik ze statystykami
        """
        unique_docs = set()
        doc_stats = {}
        
        for chunk in self.documents:
            doc_id = chunk.doc_id
            base_doc_id = doc_id.split('_chunk_')[0] if '_chunk_' in doc_id else doc_id
            
            unique_docs.add(base_doc_id)
            
            if base_doc_id not in doc_stats:
                doc_stats[base_doc_id] = 0
            doc_stats[base_doc_id] += 1
        
        return {
            "total_chunks": len(self.documents),
            "unique_documents": len(unique_docs),
            "documents_breakdown": doc_stats,
            "cache_size": self.cache.get_cache_size(),
            "embedding_dimensions": self.embeddings[0].shape if self.embeddings else None
        }
    
    def export_to_json(self, filepath: str) -> None:
        """
        Eksportuje metadane dokumentów do pliku JSON.
        
        Args:
            filepath: Ścieżka do pliku wyjściowego
        """
        docs_metadata = []
        
        for i, chunk in enumerate(self.documents):
            docs_metadata.append({
                "index": i,
                "doc_id": chunk.doc_id,
                "chunk_id": chunk.chunk_id,
                "text_length": len(chunk.text)
            })
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(docs_metadata, f, ensure_ascii=False, indent=2)
            
        if self.debug_mode:
            print(f"Wyeksportowano metadane {len(docs_metadata)} chunków do {filepath}")