from typing import List, Dict, Any, Optional, Tuple
import time
import os
from src.chunking import Chunk
from dotenv import load_dotenv
import anthropic

class AnthropicGenerator:
    def __init__(self, 
                 model_name: str = "claude-3-7-sonnet-20250219", 
                 max_context_length: int = 200000,
                 retry_attempts: int = 3,
                 retry_delay: int = 2,
                 api_key: str = None):
        self.model = model_name
        self.api_key = api_key
        self.base_timeout = 30
        self.max_context_length = max_context_length  # Claude 3.7 Sonnet ma większy kontekst niż Ollama
        self.retry_attempts = retry_attempts
        self.retry_delay = retry_delay
        
        # Załaduj zmienne środowiskowe
        load_dotenv()
        
        # Inicjalizacja klienta tylko w razie potrzeby
        self._client = None

    @property
    def client(self):
        """Leniwa inicjalizacja klienta Anthropic"""
        if self._client is None:
            # Najpierw sprawdź bezpośredni klucz API
            api_key = self.api_key
            
            # Jeśli nie istnieje, użyj zmiennej środowiskowej
            if not api_key:
                api_key = os.getenv('ANTHROPIC_API_KEY')
                
            # Utwórz klienta jeśli mamy klucz API
            if api_key:
                self._client = anthropic.Client(api_key=api_key)
                
        return self._client

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
        # Sprawdź, czy możemy zainicjalizować klienta
        if not self.client:
            return {
                "answer": "Brak klucza API Anthropic. Nie można wygenerować odpowiedzi.",
                "sources": [],
                "error": "Brak klucza API Anthropic",
                "processing_time": 0
            }
            
        system_prompt = self._format_system_prompt(contexts)
        
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
            answer = self._call_anthropic(
                prompt=f"Odpowiedz na postawione pytanie zwięźle i merytorycznie: {query}",
                system_prompt=system_prompt,
                temperature=0,
                max_tokens=max_tokens,
                timeout=dynamic_timeout
            )
            
            if answer:
                # Sprawdź, czy odpowiedź nie jest ucięta
                if self._is_truncated_response(answer):
                    continuation = self._generate_continuation(query, answer, prioritized_contexts)
                    if continuation:
                        answer = f"{answer.rstrip('.')} {continuation}"
                
                result["answer"] = answer
                result["sources"] = self._extract_sources_from_contexts(prioritized_contexts, answer)
            else:
                result["error"] = "Brak odpowiedzi z API Anthropic"
                
        except Exception as e:
            result["error"] = f"Wystąpił błąd podczas generowania odpowiedzi: {str(e)}"
            
        result["processing_time"] = time.time() - start_time
        
        return result
    
    def _call_anthropic(self, 
                    prompt: str, 
                    system_prompt: str,
                    temperature: float = 0, 
                    max_tokens: int = 4000, 
                    timeout: int = 30) -> str:
        """
        Pomocnicza metoda do wykonywania zapytań do API Anthropic z użyciem oficjalnego klienta.
        """
        if not self.client:
            print("Ostrzeżenie: Brak klucza API Anthropic. Generator zwróci pustą odpowiedź.")
            return ""
            
        for attempt in range(self.retry_attempts):
            try:
                # Wywołanie API Anthropic za pomocą oficjalnego klienta
                message = self.client.messages.create(
                    model=self.model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    system=system_prompt,
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                    timeout=timeout
                )
                
                # Pobierz tekst z odpowiedzi
                return "".join([block.text for block in message.content if block.type == "text"])
                
            except anthropic.APITimeoutError:
                if attempt < self.retry_attempts - 1:
                    time.sleep(self.retry_delay)
                else:
                    print(f"Timeout podczas zapytania do Anthropic API")
                    return ""
            except anthropic.APIError as e:
                if attempt < self.retry_attempts - 1:
                    time.sleep(self.retry_delay)
                else:
                    print(f"Błąd API Anthropic: {str(e)}")
                    return ""
            except Exception as e:
                if attempt < self.retry_attempts - 1:
                    time.sleep(self.retry_delay)
                else:
                    print(f"Wyjątek podczas zapytania do Anthropic: {str(e)}")
                    return ""
        
        return ""

    def _format_system_prompt(self, contexts: List[Chunk]) -> str:
        if not isinstance(contexts, list):
            raise TypeError("Contexts must be a list")
                
        contexts = [ctx for ctx in contexts if ctx and hasattr(ctx, 'text') and ctx.text and ctx.text.strip()]
        
        if not contexts:
            return """Jesteś asystentem AI działu prawnego. 
            Niestety nie masz uprawnień udzielać informacji na jakiekolwiek pytanie.
            Odpowiedz grzecznie że nie masz danych."""
        
        contexts = self._truncate_contexts(contexts)
            
        context_text = "\n\n".join([
            f"Kontekst {i+1} [Dokument: {getattr(ctx, 'doc_id', 'unknown')}, Chunk: {getattr(ctx, 'chunk_id', i)}]:\n{ctx.text}" 
            for i, ctx in enumerate(contexts)
        ])
        
        return f"""Jesteś asystentem AI który ma prawo udzielać wszelkich informacji zawartych w kontekstach.
     
        TWOJE ZASADY ODPOWIADANIA:
        1. Zawsze udzielaj pełnej, kompletnej odpowiedzi spójnej logicznie.
        2. Cytuj konkretny rozdział, paragraf, artykuł, punkt.
        3. Usuwaj zwroty typu "Według kontekstu" czy "Z dostępnych informacji" - odpowiadaj rzeczowo.
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
        Claude może być szybszy niż Ollama, ale wciąż potrzebuje odpowiedniego timeoutu
        """
        # Szacuj, że generowanie 1000 tokenów zajmuje około 3 sekundy
        context_size = sum(len(ctx.text) for ctx in contexts)
        # Zakładamy, że średnio 4 znaki to 1 token
        estimated_context_tokens = context_size / 4
        estimated_response_time = (estimated_context_tokens / 1000 * 1) + (max_tokens / 1000 * 3)
        
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
        
        system_prompt = self._format_system_prompt(contexts)
        
        try:
            continuation = self._call_anthropic(
                prompt=continuation_prompt,
                system_prompt=system_prompt,
                temperature=0,
                max_tokens=2000,  # Duży limit dla kontynuacji
                timeout=self.base_timeout * 2  # Podwójny timeout dla kontynuacji
            )
            
            if continuation:
                return continuation.lstrip()  # Usuń początkowe białe znaki
                
        except Exception:
            pass
            
        return ""  # Jeśli nie udało się wygenerować kontynuacji
        
    def export_to_json(self, filepath: str) -> None:
        """
        Eksportuje metadane dokumentów do pliku JSON.
        
        Args:
            filepath: Ścieżka do pliku wyjściowego
        """
        # Ta metoda nie ma sensu w przypadku generatora, ale zostawiam ją dla kompatybilności,
        # gdyby była wywoływana w kodzie kliencie
        print(f"Metoda export_to_json nie jest zaimplementowana dla klasy AnthropicGenerator")