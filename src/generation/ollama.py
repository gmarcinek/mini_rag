from typing import List, Dict, Any, Optional, Tuple
import requests
import time
from src.chunking import Chunk

class OllamaGenerator:
    def __init__(self, 
                 model_name: str = "llama3.2", 
                 base_url: str = "http://localhost:11434",
                 timeout: int = 30,
                 max_context_length: int = 32000,
                 retry_attempts: int = 3,
                 retry_delay: int = 2):
        self.model = model_name
        self.base_url = base_url.rstrip('/')
        self.base_timeout = timeout
        self.max_context_length = max_context_length
        self.retry_attempts = retry_attempts
        self.retry_delay = retry_delay

    def generate(self, query: str, contexts: List[Chunk], max_tokens: int = 4000) -> Dict[str, Any]:
        """
        Generuje odpowiedź na podstawie zapytania i dostarczonych kontekstów.
        
        Args:
            query: Pytanie użytkownika
            contexts: Lista fragmentów dokumentów znalezionych przez retriever
            max_tokens: Maksymalna liczba tokenów w odpowiedzi
            
        Returns:
            Słownik zawierający wygenerowaną odpowiedź i metadane
        """
        system_prompt = self._format_system_prompt(contexts)
        config = self._get_generation_config(max_tokens)
        
        # Dynamiczne dostosowanie timeoutu w zależności od rozmiaru kontekstu i max_tokens
        dynamic_timeout = self._calculate_dynamic_timeout(contexts, max_tokens)
        
        prioritized_contexts = self._prioritize_contexts(contexts, query)
        
        result = {
            "answer": "",
            "sources": [],
            "error": None,
            "processing_time": 0
        }
        
        start_time = time.time()
        
        try:
            response = self._make_api_request(
                query=query,
                system_prompt=system_prompt,
                config=config,
                timeout=dynamic_timeout
            )
            
            if isinstance(response, dict) and "response" in response:
                answer = response["response"]
                
                # Sprawdź, czy odpowiedź nie jest ucięta
                if self._is_truncated_response(answer):
                    continuation = self._generate_continuation(query, answer, prioritized_contexts)
                    if continuation:
                        answer = f"{answer.rstrip('.')} {continuation}"
                
                result["answer"] = answer
                result["sources"] = self._extract_sources_from_contexts(prioritized_contexts, answer)
            else:
                result["error"] = f"Nieoczekiwana odpowiedź API: {response}"
                
        except Exception as e:
            result["error"] = f"Wystąpił błąd podczas generowania odpowiedzi: {str(e)}"
            
        result["processing_time"] = time.time() - start_time
        
        return result
    
    def _make_api_request(self, query: str, system_prompt: str, config: dict, timeout: int) -> Dict[str, Any]:
        """
        Wykonuje zapytanie do API Ollama z obsługą ponowień w przypadku błędów.
        """
        prompt = f"Odpowiedz na postawione pytanie zwięźle i merytorycznie: {query}"
        
        for attempt in range(self.retry_attempts):
            try:
                response = requests.post(
                    f"{self.base_url}/api/generate",
                    json={
                        "model": self.model,
                        "prompt": prompt,
                        "stream": False,
                        "system": system_prompt,
                        "options": config
                    },
                    timeout=timeout
                )
                
                if response.status_code == 200:
                    return response.json()
                
                if response.status_code == 404:
                    return {"response": f"Model {self.model} nie został znaleziony."}
                    
                if attempt < self.retry_attempts - 1:
                    time.sleep(self.retry_delay)
                    
            except (requests.exceptions.Timeout, requests.exceptions.ConnectionError):
                if attempt < self.retry_attempts - 1:
                    time.sleep(self.retry_delay)
        
        return {"response": "Nie udało się uzyskać odpowiedzi z modelu po kilku próbach."}
    
    def _get_generation_config(self, max_tokens: int) -> dict:
        return {
            "temperature": 0,
            "num_predict": max_tokens,
            "top_p": 0.9,
            "top_k": 40,
        }

    def _format_system_prompt(self, contexts: List[Chunk]) -> str:
        if not isinstance(contexts, list):
            raise TypeError("Contexts must be a list")
                
        contexts = [ctx for ctx in contexts if ctx and hasattr(ctx, 'text') and ctx.text and ctx.text.strip()]
        
        if not contexts:
            return """Jesteś asystentem AI działu prawnego w firmie Nationale Nederlanden. 
            Niestety nie masz uprawnień udzielać informacji na jakiekolwiek pytanie.
            Odpowiedz grzecznie że nie masz danych."""
        
        contexts = self._truncate_contexts(contexts)
            
        context_text = "\n\n".join([
            f"Kontekst {i+1} [Dokument: {getattr(ctx, 'doc_id', 'unknown')}, Chunk: {getattr(ctx, 'chunk_id', i)}]:\n{ctx.text}" 
            for i, ctx in enumerate(contexts)
        ])
        
        return f"""Jesteś asystentem AI w firmie Nationale Nederlanden, który ma prawo udzielać wszelkich informacji zawartych we FRAGMENTACH OWU DO ANALIZY.
     
        TWOJE ZASADY ODPOWIADANIA:
        1. Zawsze udzielaj pełnej, kompletnej odpowiedzi spójnej logicznie.
        2. Cytuj konkretny rozdział, paragraf, artykuł, punkt.
        3. Nie używaj zwrotów typu "Według kontekstu" czy "Z dostępnych informacji" - po prostu odpowiadaj rzeczowo.
        4. Jeśli brakuje informacji w kontekście, napisz wprost: "BRAK DANYCH"
        5. Zachowuj oryginalną terminologię z poniższych fragmentów.
        6. Wskazuj dokładne źródła z których czerpiesz informacje (numer rozdziału).
        
        ---
        KONTEKST:
        {context_text}
        """

    def _truncate_contexts(self, contexts: List[Chunk]) -> List[Chunk]:
        """
        Inteligentne przycinanie kontekstów z uwzględnieniem ich ważności.
        """
        total_chars = sum(len(ctx.text) for ctx in contexts)
        
        if total_chars <= self.max_context_length:
            return contexts
            
        # Sortuj konteksty według score, ale zachowaj różnorodność dokumentów
        # Grupuj konteksty według doc_id
        docs_groups = {}
        for ctx in contexts:
            doc_id = getattr(ctx, 'doc_id', 'unknown')
            if doc_id not in docs_groups:
                docs_groups[doc_id] = []
            docs_groups[doc_id].append(ctx)
        
        # Wybierz po jednym najlepszym chunka z każdego dokumentu, 
        # potem po drugim najlepszym itd.
        selected_contexts = []
        current_length = 0
        
        # Sortuj każdą grupę według score
        for doc_id, ctx_group in docs_groups.items():
            docs_groups[doc_id] = sorted(ctx_group, key=lambda x: getattr(x, 'score', 0), reverse=True)
        
        # Dodawaj po jednym najlepszym chunka z każdego dokumentu
        i = 0
        while current_length < self.max_context_length and any(docs_groups.values()):
            for doc_id in list(docs_groups.keys()):
                if docs_groups[doc_id]:
                    ctx = docs_groups[doc_id].pop(0)
                    # Sprawdź, czy zmieści się cały chunk
                    if current_length + len(ctx.text) <= self.max_context_length:
                        selected_contexts.append(ctx)
                        current_length += len(ctx.text)
                    else:
                        # Przytnij tekst, aby zmieścił się w limicie
                        truncated_text = ctx.text[:self.max_context_length - current_length]
                        if len(truncated_text) > 100:  # Jeśli warto dodać
                            ctx.text = truncated_text
                            selected_contexts.append(ctx)
                            current_length += len(truncated_text)
                        break
                if current_length >= self.max_context_length:
                    break
            i += 1
            # Bezpiecznik
            if i > 100:
                break
                
        return selected_contexts

    def _calculate_dynamic_timeout(self, contexts: List[Chunk], max_tokens: int) -> int:
        """
        Oblicza dynamiczny timeout w zależności od rozmiaru kontekstu i liczby tokenów.
        """
        # Szacuj, że generowanie 1000 tokenów zajmuje około 5 sekund
        context_size = sum(len(ctx.text) for ctx in contexts)
        # Zakładamy, że średnio 4 znaki to 1 token
        estimated_context_tokens = context_size / 4
        estimated_response_time = (estimated_context_tokens / 1000 * 2) + (max_tokens / 1000 * 5)
        
        # Minimum 10 sekund, maksimum 120 sekund
        return max(10, min(120, int(estimated_response_time) + self.base_timeout))

    def _is_truncated_response(self, answer: str) -> bool:
        """
        Sprawdza, czy odpowiedź została ucięta.
        """
        answer = answer.rstrip()
        return (answer.endswith(('...', '.')) and 
                not answer.endswith(('...', '!', '?', '.', ':"', '."')) or
                answer.endswith(('....')))

    def _prioritize_contexts(self, contexts: List[Chunk], query: str) -> List[Chunk]:
        """
        Priorytezuje konteksty na podstawie ich relewancji do zapytania.
        """
        if not contexts:
            return []
            
        # Tu możesz dodać bardziej zaawansowaną logikę priorytyzacji
        # Na razie po prostu sortujemy według score
        return sorted(contexts, key=lambda x: getattr(x, 'score', 0), reverse=True)

    def _extract_sources_from_contexts(self, contexts: List[Chunk], answer: str) -> List[Dict[str, str]]:
        """
        Wyciąga informacje o źródłach, które najprawdopodobniej były użyte w odpowiedzi.
        """
        sources = []
        
        for ctx in contexts:
            doc_id = getattr(ctx, 'doc_id', 'unknown')
            chunk_id = getattr(ctx, 'chunk_id', 'unknown')
            
            # Sprawdź, czy tekst z tego kontekstu jest używany w odpowiedzi
            # To jest proste podejście, które można później ulepszyć
            if len(ctx.text) > 30:  # Sprawdzaj tylko dłuższe fragmenty
                # Szukaj kilku fragmentów tekstu z kontekstu w odpowiedzi
                text_samples = [ctx.text[i:i+20] for i in range(0, len(ctx.text), len(ctx.text)//5) if i+20 < len(ctx.text)]
                for sample in text_samples:
                    if sample in answer:
                        sources.append({
                            "doc_id": doc_id,
                            "chunk_id": chunk_id,
                            "relevance": getattr(ctx, 'score', 0)
                        })
                        break
        
        return sources

    def _generate_continuation(self, query: str, previous_answer: str, contexts: List[Chunk]) -> str:
        """
        Generuje kontynuację odpowiedzi, jeśli została ucięta.
        """
        continuation_prompt = (
            f"Poprzednia odpowiedź została przerwana. Kontynuuj odpowiedź od miejsca: "
            f"{previous_answer[-200:]}"  # Ostatnie 200 znaków jako kontekst
        )
        
        config = self._get_generation_config(2000)  # Duży limit dla kontynuacji
        system_prompt = self._format_system_prompt(contexts)
        
        try:
            response = self._make_api_request(
                query=continuation_prompt,
                system_prompt=system_prompt,
                config=config,
                timeout=self.base_timeout * 2  # Podwójny timeout dla kontynuacji
            )
            
            if isinstance(response, dict) and "response" in response:
                continuation = response["response"]
                return continuation.lstrip()  # Usuń początkowe białe znaki
                
        except Exception:
            pass
            
        return ""  # Jeśli nie udało się wygenerować kontynuacji