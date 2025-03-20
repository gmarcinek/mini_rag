from typing import  List, Tuple, Dict, Any, Optional
import time
import json
import hashlib
import os
from src.chunking import Chunk, SimpleTextSplitter
from src.embeddings import PolishLegalEmbedder
from src.cache import BaseCache
from src.retrieval.semantic import SemanticRetriever
from src.generation.ollama import OllamaGenerator
import requests

class LegalRAGPipeline:
    """
    Pełny pipeline RAG zoptymalizowany dla dokumentów prawnych.
    Integruje wszystkie usprawnione komponenty:
    - SemanticRetriever z poprawioną metryką podobieństwa
    - OllamaGenerator z lepszą obsługą długich kontekstów
    """
    
    def __init__(self, 
                 use_gpu: bool = False,
                 cache_dir: str = "cache",
                 chunker: SimpleTextSplitter = None,
                 embedder_model: str = "BAAI/bge-m3",
                 generator_model: str = "llama3.2",
                 ollama_url: str = "http://localhost:11434",
                 min_score_threshold: float = 0.6,
                 max_top_k: int = 5,
                 max_context_length: int = 32000,
                 debug_mode: bool = False):
        
        self.debug_mode = debug_mode
        
        # Inicjalizacja embeddera
        if self.debug_mode:
            print(f"Inicjalizacja embeddera {embedder_model}...")
        self.embedder = PolishLegalEmbedder(model_name=embedder_model, use_gpu=use_gpu)
        
        # Inicjalizacja cache'u
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        self.cache = BaseCache(cache_dir)
        
        # Inicjalizacja retrievera
        if self.debug_mode:
            print("Inicjalizacja retrievera...")
        self.retriever = SemanticRetriever(
            embedder=self.embedder,
            min_score_threshold=min_score_threshold,
            max_top_k=max_top_k
        )
        
        # Inicjalizacja generatora
        if self.debug_mode:
            print(f"Inicjalizacja generatora z modelem {generator_model}...")
        self.generator = OllamaGenerator(
            model_name=generator_model,
            base_url=ollama_url,
            max_context_length=max_context_length
        )
        
        # Inicjalizacja chunkera
        self.chunker = chunker if chunker is not None else SimpleTextSplitter()
        
        # Wczytanie dokumentów i embeddingów z cache'u
        self.documents, self.embeddings = self.cache.load_cache()
        if self.debug_mode:
            print(f"Wczytano {len(self.documents)} dokumentów z cache'u")
    
    def add_document(self, document: str, doc_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Dodaje pojedynczy dokument do systemu.
        
        Args:
            document: Tekst dokumentu
            doc_id: Opcjonalny identyfikator dokumentu
            
        Returns:
            Słownik z wynikami dodawania dokumentu
        """
        return self.add_documents([document], [doc_id] if doc_id else None)
    
    def add_documents(self, documents: List[str], doc_ids: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Dodaje dokumenty do systemu, dzieli je na chunki i oblicza embeddingi.
        
        Args:
            documents: Lista tekstów dokumentów
            doc_ids: Opcjonalna lista identyfikatorów dokumentów
            
        Returns:
            Słownik ze statystykami dodawania dokumentów
        """
        start_time = time.time()
        
        if doc_ids is None:
            doc_ids = [f"doc_{hashlib.md5(doc.encode()).hexdigest()[:10]}" for doc in documents]
        
        if len(documents) != len(doc_ids):
            raise ValueError("Liczba dokumentów musi być równa liczbie identyfikatorów")
        
        stats = {
            "added_documents": 0,
            "skipped_documents": 0,
            "total_chunks": 0,
            "new_chunks": 0,
            "time_chunking": 0,
            "time_embedding": 0,
            "total_time": 0
        }
        
        for doc, doc_id in zip(documents, doc_ids):
            # Sprawdzamy, czy dokument już istnieje
            if any(chunk.doc_id == doc_id for chunk in self.documents):
                if self.debug_mode:
                    print(f"Dokument {doc_id} już istnieje, pomijam...")
                stats["skipped_documents"] += 1
                continue
            
            # Dzielimy dokument na chunki
            chunk_start = time.time()
            chunks = self.chunker.split_text(doc, doc_id=doc_id)
            stats["time_chunking"] += time.time() - chunk_start
            
            # Obliczamy embeddingi i dodajemy do systemu
            embed_start = time.time()
            for chunk in chunks:
                embedding = self.cache.get_embedding(chunk.text, self.embedder)
                self.documents.append(chunk)
                self.embeddings.append(embedding)
                stats["new_chunks"] += 1
            stats["time_embedding"] += time.time() - embed_start
            
            stats["added_documents"] += 1
        
        stats["total_chunks"] = len(self.documents)
        
        # Zapisujemy do cache'u tylko jeśli dodano nowe dokumenty
        if stats["added_documents"] > 0:
            self.cache.save_cache(self.documents, self.embeddings)
        
        stats["total_time"] = time.time() - start_time
        
        if self.debug_mode:
            print(f"Dodano {stats['added_documents']} dokumentów ({stats['new_chunks']} chunków)")
            print(f"Czas przetwarzania: {stats['total_time']:.2f}s")
        
        return stats
    
    def smart_query(self, question: str, top_k: Optional[int] = None, 
              min_score: Optional[float] = None, batch_threshold: int = 3) -> Dict[str, Any]:
        """
        Inteligentnie wybiera między standardowym query a query_large_context
        w zależności od liczby wyników wyszukiwania.
        
        Args:
            question: Pytanie użytkownika
            top_k: Opcjonalna liczba najlepszych dokumentów do użycia
            min_score: Opcjonalny minimalny próg podobieństwa
            batch_threshold: Próg liczby chunków, od którego używane jest przetwarzanie wsadowe
            
        Returns:
            Słownik z odpowiedzią i metadanymi
        """
        # Najpierw wykonaj wyszukiwanie, aby sprawdzić liczbę znalezionych chunków
        start_time = time.time()
        
        # Sprawdź od razu, czy mamy dokumenty
        if not self.documents:
            return {
                "question": question,
                "answer": "Brak dokumentów do przeszukania. Dodaj dokumenty przed zadawaniem pytań.",
                "sources": [],
                "chunks": [],
                "time_retrieval": 0,
                "time_generation": 0,
                "total_time": time.time() - start_time
            }
        
        # Etap 1: Wyszukiwanie semantyczne
        retrieved_chunks = self.retriever.retrieve(
            query=question,
            documents=self.documents,
            embeddings=self.embeddings,
            top_k=top_k,
            min_score=min_score
        )
        
        # W zależności od liczby znalezionych chunków, wybierz odpowiednią metodę przetwarzania
        if len(retrieved_chunks) > batch_threshold:
            print(f"Znaleziono {len(retrieved_chunks)} chunków. Używam query_large_context.")
            # Użyj procesu batchowanego
            return self.query_large_context(
                question=question, 
                retrieved_chunks=retrieved_chunks,  # Przekaż już znalezione chunki
                min_score=min_score
            )
        else:
            print(f"Znaleziono {len(retrieved_chunks)} chunków. Używam standardowego przetwarzania.")
            # Użyj standardowego procesu
            return self.query(
                question=question,
                retrieved_chunks=retrieved_chunks,  # Przekaż już znalezione chunki
                min_score=min_score
            )
            
    def query(self, question: str, top_k: Optional[int] = None, 
          min_score: Optional[float] = None, retrieved_chunks: Optional[List[Tuple[Chunk, float]]] = None) -> Dict[str, Any]:
        """
        Wykonuje zapytanie do systemu RAG.
        
        Args:
            question: Pytanie użytkownika
            top_k: Opcjonalna liczba najlepszych dokumentów do użycia
            min_score: Opcjonalny minimalny próg podobieństwa
            retrieved_chunks: Opcjonalna lista już znalezionych chunków
            
        Returns:
            Słownik z odpowiedzią i metadanymi
        """
        start_time = time.time()
        
        result = {
            "question": question,
            "answer": "",
            "sources": [],
            "chunks": [],
            "time_retrieval": 0,
            "time_generation": 0,
            "total_time": 0
        }
        
        if not self.documents:
            result["answer"] = "Brak dokumentów do przeszukania. Dodaj dokumenty przed zadawaniem pytań."
            result["total_time"] = time.time() - start_time
            return result
        
        # Etap 1: Wyszukiwanie semantyczne (jeśli nie zostało już wykonane)
        retrieval_start = time.time()
        if retrieved_chunks is None:
            retrieved_chunks = self.retriever.retrieve(
                query=question,
                documents=self.documents,
                embeddings=self.embeddings,
                top_k=top_k,
                min_score=min_score
            )
        result["time_retrieval"] = time.time() - retrieval_start
        
        # Zapisujemy informacje o znalezionych chunkach
        for chunk, score in retrieved_chunks:
            result["chunks"].append({
                "doc_id": chunk.doc_id,
                "chunk_id": chunk.chunk_id,
                "score": float(score),
                "text_length": len(chunk.text)
            })
        
        if not retrieved_chunks:
            result["answer"] = "Na podstawie dostępnych danych nie mogę odpowiedzieć na to pytanie."
            result["total_time"] = time.time() - start_time
            return result
        
        # Etap 2: Generacja odpowiedzi
        generation_start = time.time()
        contexts = [chunk for chunk, _ in retrieved_chunks]
        raw_answer = self.generator.generate(question, contexts)
        result["time_generation"] = time.time() - generation_start
        result["answer"] = raw_answer
        
        # Przygotowanie informacji o źródłach
        for chunk, score in retrieved_chunks:
            chunk_preview = chunk.text[:150] + "..." if len(chunk.text) > 150 else chunk.text
            result["sources"].append({
                "doc_id": chunk.doc_id,
                "chunk_id": chunk.chunk_id,
                "score": float(score),
                "preview": chunk_preview
            })
        
        result["total_time"] = time.time() - start_time
        
        return result
    
    def clear(self) -> None:
        """Czyści wszystkie dokumenty i embeddingi z systemu."""
        self.documents = []
        self.embeddings = []
        self.cache.clear_cache()
        if self.debug_mode:
            print("Wyczyszczono wszystkie dokumenty i cache")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Zwraca statystyki systemu.
        
        Returns:
            Słownik ze statystykami
        """
        # Zbieramy unikalne dokumenty
        unique_docs = set()
        for chunk in self.documents:
            unique_docs.add(chunk.doc_id)
        
        # Zbieramy statystyki dla każdego dokumentu
        doc_stats = {}
        for doc_id in unique_docs:
            doc_chunks = [chunk for chunk in self.documents if chunk.doc_id == doc_id]
            doc_stats[doc_id] = {
                "chunks": len(doc_chunks),
                "total_text_length": sum(len(chunk.text) for chunk in doc_chunks)
            }
        
        # Zbieramy statystyki embeddera
        embedder_info = {
            "model": self.embedder.model_name,
            "embedding_dim": self.embeddings[0].shape[0] if self.embeddings else 0,
            "using_gpu": self.embedder.use_gpu
        }
        
        return {
            "documents": {
                "count": len(unique_docs),
                "total_chunks": len(self.documents),
                "per_document": doc_stats
            },
            "embedder": embedder_info,
            "cache": {
                "size_mb": self.cache.get_cache_size(),
                "directory": self.cache.cache_dir
            },
            "retriever": {
                "min_score_threshold": self.retriever.min_score_threshold,
                "max_top_k": self.retriever.max_top_k
            }
        }
    
    def process_in_batches(self, question: str, chunks: List[Tuple[Chunk, float]], 
                        batch_size: int = 4, max_batches: int = 4) -> str:
        """
        Przetwarza duże zestawy chunków w mniejszych wsadach i konsoliduje odpowiedzi.
        
        Args:
            question: Pytanie użytkownika
            chunks: Lista chunków (Chunk, score) zwróconych przez retriever
            batch_size: Rozmiar pojedynczego wsadu (liczba chunków)
            max_batches: Maksymalna liczba wsadów do przetworzenia
            
        Returns:
            Skonsolidowana odpowiedź ze wszystkich wsadów
        """
        if not chunks:
            return "Nie znaleziono odpowiednich fragmentów dla tego pytania."
        
        # Dzielimy chunki na mniejsze wsady
        batches = []
        chunks_sorted = sorted(chunks, key=lambda x: x[1], reverse=True)  # Sortuj według score
        
        # Limity
        total_chunks = min(len(chunks_sorted), batch_size * max_batches)
        
        for i in range(0, total_chunks, batch_size):
            batch = chunks_sorted[i:i + batch_size]
            batches.append(batch)
        
        if self.debug_mode:
            print(f"Podzielono {total_chunks} chunków na {len(batches)} wsady po {batch_size}.")
        
        # Generujemy odpowiedź dla każdego wsadu
        batch_answers = []
        for i, batch in enumerate(batches):
            if self.debug_mode:
                print(f"Przetwarzanie wsadu {i+1}/{len(batches)}...")
            
            # Przygotuj konteksty dla tego wsadu
            contexts = [chunk for chunk, _ in batch]
            
            # Wygeneruj odpowiedź dla tego wsadu chunków
            batch_question = f"{question} (część {i+1}/{len(batches)})"
            batch_answer = self.generator.generate(batch_question, contexts)
            
            batch_answers.append({
                "batch_id": i+1,
                "question": batch_question,
                "answer": batch_answer,
                "chunks": [{"doc_id": c.doc_id, "chunk_id": c.chunk_id, "score": s} for c, s in batch]
            })
        
        # Konsolidujemy odpowiedzi z wszystkich wsadów
        if len(batch_answers) == 1:
            return batch_answers[0]["answer"]
        
        # Przygotuj odpowiedzi do konsolidacji
        answers_to_consolidate = [f"CZĘŚĆ {a['batch_id']}/{len(batches)}:\n{a['answer']}" for a in batch_answers]
        answers_text = "\n\n".join(answers_to_consolidate)
        
        # Konsolidujemy odpowiedzi
        system_prompt = """
        Jesteś ekspertem AI, który potrafi konsolidować informacje z wielu źródeł.
        Twoim zadaniem jest stworzenie jednej spójnej, kompleksowej odpowiedzi na podstawie kilku 
        częściowych odpowiedzi dotyczących tego samego pytania.
        
        Zasady:
        1. Łącz informacje logicznie, usuwając powtórzenia
        2. Zachowaj wszystkie unikalne informacje z każdej części
        3. Uporządkuj odpowiedź w logiczną strukturę
        4. Zachowaj oryginalną terminologię
        5. Usuń wszystkie odwołania do "części" lub "fragmentów" odpowiedzi
        6. Stwórz zwięzłą ale kompletną odpowiedź na oryginalne pytanie
        """
        
        consolidation_prompt = f"""
        Poniżej znajduje się {len(batch_answers)} części odpowiedzi na pytanie:
        
        PYTANIE: {question}
        
        CZĘŚCI ODPOWIEDZI:
        {answers_text}
        
        Stwórz jedną spójną, kompleksową odpowiedź łączącą wszystkie informacje
        z powyższych części. Usuń powtórzenia i połącz informacje logicznie.
        """
        
        consolidated_answer = self._call_ollama(
            prompt=consolidation_prompt,
            system_prompt=system_prompt,
            temperature=0.4,
            max_tokens=4000,
            timeout=60
        )
        
        if not consolidated_answer:
            # Jeśli konsolidacja się nie powiodła, zwróć połączone odpowiedzi
            return "\n\n".join([f"Część {a['batch_id']}/{len(batches)}:\n{a['answer']}" for a in batch_answers])
        
        return consolidated_answer

    def query_large_context(self, question: str, top_k: Optional[int] = None, 
                       min_score: Optional[float] = None, 
                       batch_size: int = 2,
                       max_batches: int = 8,
                       retrieved_chunks: Optional[List[Tuple[Chunk, float]]] = None) -> Dict[str, Any]:
        """
        Wersja metody query obsługująca duże zestawy chunków poprzez przetwarzanie wsadowe.
        
        Args:
            question: Pytanie użytkownika
            top_k: Opcjonalna liczba najlepszych dokumentów do użycia (może być duża)
            min_score: Opcjonalny minimalny próg podobieństwa
            batch_size: Rozmiar pojedynczego wsadu (liczba chunków)
            max_batches: Maksymalna liczba wsadów do przetworzenia
            retrieved_chunks: Opcjonalna lista już znalezionych chunków
            
        Returns:
            Słownik z odpowiedzią i metadanymi
        """
        start_time = time.time()
        
        result = {
            "question": question,
            "answer": "",
            "sources": [],
            "chunks": [],
            "batch_details": [],
            "time_retrieval": 0,
            "time_generation": 0,
            "time_consolidation": 0,
            "total_time": 0
        }
        
        if not self.documents:
            result["answer"] = "Brak dokumentów do przeszukania. Dodaj dokumenty przed zadawaniem pytań."
            result["total_time"] = time.time() - start_time
            return result
        
        # Etap 1: Wyszukiwanie semantyczne (jeśli nie zostało już wykonane)
        retrieval_start = time.time()
        if retrieved_chunks is None:
            effective_top_k = top_k if top_k is not None else batch_size * max_batches
            
            retrieved_chunks = self.retriever.retrieve(
                query=question,
                documents=self.documents,
                embeddings=self.embeddings,
                top_k=effective_top_k,
                min_score=min_score
            )
        result["time_retrieval"] = time.time() - retrieval_start
        
        # Zapisujemy informacje o znalezionych chunkach
        for chunk, score in retrieved_chunks:
            result["chunks"].append({
                "doc_id": chunk.doc_id,
                "chunk_id": chunk.chunk_id,
                "score": float(score),
                "text_length": len(chunk.text)
            })
        
        if not retrieved_chunks:
            result["answer"] = "Na podstawie dostępnych danych nie mogę odpowiedzieć na to pytanie."
            result["total_time"] = time.time() - start_time
            return result
        
        # Etap 2: Generacja odpowiedzi z użyciem batchowania
        generation_start = time.time()
        consolidated_answer = self.process_in_batches(
            question=question,
            chunks=retrieved_chunks,
            batch_size=batch_size,
            max_batches=max_batches
        )
        result["time_generation"] = time.time() - generation_start
        
        result["answer"] = consolidated_answer
        
        # Przygotowanie informacji o źródłach
        for chunk, score in retrieved_chunks[:batch_size * max_batches]:
            chunk_preview = chunk.text[:100] + "..." if len(chunk.text) > 100 else chunk.text
            result["sources"].append({
                "doc_id": chunk.doc_id,
                "chunk_id": chunk.chunk_id,
                "score": float(score),
                "preview": chunk_preview
            })
        
        result["total_time"] = time.time() - start_time
        
        return result
    
    def _call_ollama(self, 
                    prompt: str, 
                    system_prompt: str,
                    temperature: float = 0, 
                    max_tokens: int = 4000, 
                    timeout: int = 30) -> str:
        """
        Pomocnicza metoda do wykonywania zapytań do API Ollama.
        """
        try:
            config = {
                "temperature": temperature,
                "num_predict": max_tokens,
            }
            
            response = requests.post(
                f"{self.generator.base_url}/api/generate",
                json={
                    "model": self.generator.model,
                    "prompt": prompt,
                    "stream": False,
                    "system": system_prompt,
                    "options": config
                },
                timeout=timeout
            )
            
            if response.status_code == 200:
                return response.json().get("response", "")
            else:
                print(f"Błąd API Ollama: {response.status_code} - {response.text}")
                return ""
            
        except Exception as e:
            print(f"Wyjątek podczas zapytania do Ollama: {str(e)}")
            return ""
        def export_to_json(self, filepath: str) -> None:
            """
            Eksportuje metadane dokumentów do pliku JSON.
            
            Args:
                filepath: Ścieżka do pliku wyjściowego
            """
            metadata = []
            
            for i, chunk in enumerate(self.documents):
                metadata.append({
                    "index": i,
                    "doc_id": chunk.doc_id,
                    "chunk_id": chunk.chunk_id,
                    "text_length": len(chunk.text),
                    "text_preview": chunk.text[:100] + "..." if len(chunk.text) > 100 else chunk.text
                })
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)
            
            if self.debug_mode:
                print(f"Metadane {len(metadata)} chunków wyeksportowano do {filepath}")


# Przykład użycia:
"""
# Inicjalizacja systemu
rag = LegalRAGPipeline(
    use_gpu=True,
    debug_mode=True,
)

# Dodanie dokumentów
with open("dokument_prawny.txt", "r", encoding="utf-8") as f:
    document = f.read()
    
rag.add_document(document, doc_id="ustawa_kodeks_cywilny")

# Zadanie pytania
result = rag.query("Jakie są zasady dziedziczenia ustawowego?")
print(f"\nOdpowiedź: {result['answer']}")
print(f"\nCzas wyszukiwania: {result['time_retrieval']:.2f}s")
print(f"Czas generacji: {result['time_generation']:.2f}s")
print(f"Całkowity czas: {result['total_time']:.2f}s")

# Wyświetlenie statystyk
stats = rag.get_stats()
print(f"\nLiczba dokumentów: {stats['documents']['count']}")
print(f"Łączna liczba chunków: {stats['documents']['total_chunks']}")
"""