from typing import List
import requests
from src.chunking import Chunk

class OllamaGenerator:
    def __init__(self, 
                 model_name: str = "llama3.2", 
                 base_url: str = "http://localhost:11434",
                 timeout: int = 30,
                 max_context_length: int = 32000):
        self.model = model_name
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.max_context_length = max_context_length

    def generate(self, query: str, contexts: List[Chunk], max_tokens: int = 4000) -> str:
        """Generuje odpowiedź z poprawioną obsługą długich odpowiedzi."""
        system_prompt = self._format_system_prompt(contexts)
        config = self._get_generation_config(max_tokens)
        
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": f"odpowiedz na postawione pytanie zwięźle i merytorycznie: {query}",
                    "stream": False,
                    "system": system_prompt,
                    "options": config
                },
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                answer = response.json()["response"]
                
                # Sprawdź, czy odpowiedź nie jest ucięta
                if answer.rstrip().endswith(('...', '.')):
                    # Jeśli odpowiedź kończy się wielokropkiem lub jest ucięta,
                    # spróbuj wygenerować dalszą część
                    continuation = self._generate_continuation(query, answer, contexts)
                    if continuation:
                        answer = f"{answer.rstrip('.')} {continuation}"
                
                return answer
            
            if response.status_code == 404:
                return f"Model {self.model} nie został znaleziony."
                
            return f"Wystąpił błąd w komunikacji z modelem: {response.text}"
            
        except Exception as e:
            return f"Wystąpił błąd podczas generowania odpowiedzi: {str(e)}"
    
    def _get_generation_config(self, max_tokens: int) -> dict:
        return {
            "temperature": 0,
            "num_predict": max_tokens,
        }

    def _format_system_prompt(self, contexts: List[Chunk]) -> str:
        if not isinstance(contexts, list):
            raise TypeError("Contexts must be a list")
                
        contexts = [ctx for ctx in contexts if ctx and ctx.text and ctx.text.strip()]
        
        if not contexts:
            return """Jesteś asystentem AI działy prawnego w firmie Nationale Nederlanden. 
            Niestety nie ma w dokumencie OWU informacji na ten temat, więc nie możesz udzielić szczegółowej odpowiedzi."""
        
        contexts = self._truncate_contexts(contexts)
            
        context_text = "\n\n".join([
            f"Kontekst {i+1} [Dokument: {ctx.doc_id}, Chunk: {ctx.chunk_id}]:\n{ctx.text}" 
            for i, ctx in enumerate(contexts)
        ])
        
        return f"""Jesteś asystentem AI działy prawnego w firmie Nationale Nederlanden, który ma prawo udzielać wszelkich informacji z zakresu OWU.
        Odpowiadasz na pytania pracowników działu prawnego wyłącznie w oparciu o dostarczone fragmenty OWU. Nie masz uprawnień udzielać informacji niezwiązanych z OWU lub tematyką związaną z polsą ubezpieczeniową.

        TWOJE ZASADY ODPOWIADANIA:
        1. Zawsze udzielaj pełnej, kompletnej odpowiedzi spójnej logicznie.
        2. Jeśli znajdziesz odpowiednią informację w kontekście, zacytuj konkretny paragraf i punkt.
        3. Nie używaj zwrotów typu "Według kontekstu" czy "Z dostępnych informacji" - po prostu odpowiadaj rzeczowo.
        4. Jeśli brakuje informacji w kontekście, napisz wprost: "Na podstawie dostępnych fragmentów OWU nie mogę odpowiedzieć na to pytanie."
        5. Zachowuj oryginalną terminologię z OWU.
        
        FRAGMENTY OWU DO ANALIZY:

        {context_text}

        PAMIĘTAJ:
        - Odpowiadaj konkretnie
        - Nie pomijaj istotnych szczegółów
        - Na podstawie TYLKO treści OWU"""

    def _truncate_contexts(self, contexts: List[Chunk]) -> List[Chunk]:
        total_chars = sum(len(ctx.text) for ctx in contexts)
        
        if total_chars <= self.max_context_length:
            return contexts
            
        sorted_contexts = sorted(contexts, key=lambda x: getattr(x, 'score', 0), reverse=True)
        
        selected_contexts = []
        current_length = 0
        
        for ctx in sorted_contexts:
            if current_length + len(ctx.text) > self.max_context_length:
                break
            selected_contexts.append(ctx)
            current_length += len(ctx.text)
            
        return selected_contexts

    def _generate_continuation(self, query: str, previous_answer: str, contexts: List[Chunk]) -> str:
        """Generuje kontynuację odpowiedzi, jeśli została ucięta."""
        continuation_prompt = (
            f"Poprzednia odpowiedź została przerwana. Kontynuuj odpowiedź od miejsca: "
            f"{previous_answer[-100:]}"  # Ostatnie 100 znaków jako kontekst
        )
        
        config = self._get_generation_config(2000)  # Duży limit dla kontynuacji
        system_prompt = self._format_system_prompt(contexts)
        
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": continuation_prompt,
                    "stream": False,
                    "system": system_prompt,
                    "options": config
                },
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                continuation = response.json()["response"]
                return continuation.lstrip()  # Usuń początkowe białe znaki
                
        except Exception:
            pass
            
        return ""  # Jeśli nie udało się wygenerować kontynuacji