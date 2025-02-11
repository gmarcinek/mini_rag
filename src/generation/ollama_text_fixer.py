from typing import List
import requests
from src.chunking import Chunk

class OllamaTextFixer:
    def __init__(self, 
                 model_name: str = "llama3.2:latest", 
                 base_url: str = "http://localhost:11434",
                 timeout: int = 30,
                 max_context_length: int = 32000):
        self.model = model_name
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.max_context_length = max_context_length  # Używamy instancyjnej zmiennej

    def generate(self, query: str, contexts: List[Chunk], max_tokens: int = 4000) -> str:
        """Generuje odpowiedź z poprawioną obsługą długich odpowiedzi."""
        config = self._get_generation_config(max_tokens)
        
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": query,
                    "stream": False,
                    "system": "You are a Polish text editor. No comments are available. You only format input and return reformated output as is. No describing what you did.",
                    "options": config
                },
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                answer = response.json().get("response", "")
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

