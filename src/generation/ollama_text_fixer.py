from typing import List, Optional
import requests
import time
from src.chunking import Chunk

class OllamaTextFixer:
    """
    Klasa odpowiedzialna za formatowanie i poprawianie tekstu wygenerowanego przez model LLM.
    Wykorzystuje Ollama API do uruchomienia dedykowanego modelu do korekcji tekstu.
    """
    
    def __init__(self, 
                 model_name: str = "llama3.2:latest", 
                 base_url: str = "http://localhost:11434",
                 timeout: int = 30,
                 max_context_length: int = 32000,
                 retry_attempts: int = 2):
        self.model = model_name
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.max_context_length = max_context_length
        self.retry_attempts = retry_attempts

    def fix_text(self, text: str, max_tokens: int = 16000) -> str:
        """
        Poprawia i formatuje podany tekst.
        
        Args:
            text: Tekst do poprawienia
            max_tokens: Maksymalna liczba tokenów w odpowiedzi
            
        Returns:
            Poprawiony tekst
        """
        if not text or len(text.strip()) == 0:
            return text
            
        # Przygotuj instrukcję do poprawy tekstu
        prompt = (
            "Popraw poniższy tekst pod kątem formatowania, interpunkcji i spójności. "
            "Zachowaj oryginalną treść i znaczenie. Nie dodawaj żadnych komentarzy "
            "ani wyjaśnień - zwróć tylko poprawiony tekst:\n\n"
            f"{text}"
        )
        
        # Użyj pustej listy kontekstów, bo nie są potrzebne
        fixed_text = self.generate(prompt, [], max_tokens)
        
        return fixed_text

    def generate(self, query: str, contexts: List[Chunk], max_tokens: int = 16000) -> str:
        """
        Generuje poprawiony tekst używając modelu Ollama.
        
        Args:
            query: Instrukcja z tekstem do poprawienia
            contexts: Lista kontekstów (nieużywana w tym przypadku)
            max_tokens: Maksymalna liczba tokenów w odpowiedzi
            
        Returns:
            Poprawiony tekst
        """
        config = self._get_generation_config(max_tokens)
        start_time = time.time()
        
        for attempt in range(self.retry_attempts):
            try:
                response = requests.post(
                    f"{self.base_url}/api/generate",
                    json={
                        "model": self.model,
                        "prompt": query,
                        "stream": False,
                        "system": "You are a Polish text editor. No comments are allowed. You only format input and return reformatted output as is. Fix grammar, spelling, punctuation and formatting. Do not change the meaning. Do not add any explanations or notes about what you did.",
                        "options": config
                    },
                    timeout=self.timeout
                )
                
                if response.status_code == 200:
                    answer = response.json().get("response", "")
                    processing_time = time.time() - start_time
                    print(f"Tekst poprawiony w {processing_time:.2f}s")
                    return answer
                
                if response.status_code == 404:
                    print(f"Model {self.model} nie został znaleziony.")
                    return query  # Zwracamy oryginalny tekst w przypadku problemu
                
                # Jeśli to nie ostatnia próba, spróbuj ponownie
                if attempt < self.retry_attempts - 1:
                    time.sleep(1)  # Krótki delay przed kolejną próbą
                    continue
                    
                print(f"Wystąpił błąd w komunikacji z modelem: {response.text}")
                return query  # Zwracamy oryginalny tekst w przypadku problemu
                
            except Exception as e:
                print(f"Wystąpił błąd podczas poprawiania tekstu: {str(e)}")
                # Jeśli to nie ostatnia próba, spróbuj ponownie
                if attempt < self.retry_attempts - 1:
                    time.sleep(1)
                    continue
                
                return query  # Zwracamy oryginalny tekst w przypadku problemu
        
        # Jeśli wszystkie próby zakończyły się niepowodzeniem
        return query
    
    def _get_generation_config(self, max_tokens: int) -> dict:
        """
        Zwraca konfigurację dla generacji tekstu.
        
        Args:
            max_tokens: Maksymalna liczba tokenów w odpowiedzi
            
        Returns:
            Słownik z konfiguracją
        """
        return {
            "temperature": 0,  # Najniższa temperatura dla deterministycznych wyników
            "num_predict": max_tokens,
            "top_p": 0.95,
            "top_k": 40
        }

    def fix_formatting(self, text: str) -> str:
        """
        Naprawia tylko formatowanie tekstu, bez ingerencji w jego treść.
        Przydatne dla przypadków, gdzie chcemy tylko poprawić formatowanie
        bez ryzyka zmiany znaczenia.
        
        Args:
            text: Tekst do poprawienia formatowania
            
        Returns:
            Tekst z poprawionym formatowaniem
        """
        if not text or len(text.strip()) == 0:
            return text
            
        # Proste reguły poprawiające formatowanie
        lines = text.split('\n')
        fixed_lines = []
        
        for i, line in enumerate(lines):
            # Usuń nadmiarowe spacje
            line = ' '.join(line.split())
            
            # Popraw spacje przed znakami interpunkcyjnymi
            line = line.replace(' ,', ',').replace(' .', '.').replace(' ;', ';').replace(' :', ':')
            line = line.replace(' )', ')').replace('( ', '(')
            
            # Dodaj spacje po znakach interpunkcyjnych (jeśli nie ma)
            for punct in [',', '.', ';', ':', '!', '?']:
                line = line.replace(f"{punct}a", f"{punct} a")
                line = line.replace(f"{punct}b", f"{punct} b")
                line = line.replace(f"{punct}c", f"{punct} c")
                line = line.replace(f"{punct}d", f"{punct} d")
                line = line.replace(f"{punct}e", f"{punct} e")
                line = line.replace(f"{punct}f", f"{punct} f")
                line = line.replace(f"{punct}g", f"{punct} g")
                line = line.replace(f"{punct}h", f"{punct} h")
                line = line.replace(f"{punct}i", f"{punct} i")
                line = line.replace(f"{punct}j", f"{punct} j")
                line = line.replace(f"{punct}k", f"{punct} k")
                line = line.replace(f"{punct}l", f"{punct} l")
                line = line.replace(f"{punct}m", f"{punct} m")
                line = line.replace(f"{punct}n", f"{punct} n")
                line = line.replace(f"{punct}o", f"{punct} o")
                line = line.replace(f"{punct}p", f"{punct} p")
                line = line.replace(f"{punct}r", f"{punct} r")
                line = line.replace(f"{punct}s", f"{punct} s")
                line = line.replace(f"{punct}t", f"{punct} t")
                line = line.replace(f"{punct}u", f"{punct} u")
                line = line.replace(f"{punct}w", f"{punct} w")
                line = line.replace(f"{punct}v", f"{punct} v")
                line = line.replace(f"{punct}x", f"{punct} x")
                line = line.replace(f"{punct}y", f"{punct} y")
                line = line.replace(f"{punct}z", f"{punct} z")
            
            fixed_lines.append(line)
        
        # Łączenie linii z powrotem w tekst
        fixed_text = '\n'.join(fixed_lines)
        
        # Dodaj pustą linię przed punktami (jeśli nie są już oddzielone)
        fixed_text = fixed_text.replace('\n1.', '\n\n1.')
        fixed_text = fixed_text.replace('\n2.', '\n\n2.')
        fixed_text = fixed_text.replace('\n3.', '\n\n3.')
        fixed_text = fixed_text.replace('\n4.', '\n\n4.')
        fixed_text = fixed_text.replace('\n5.', '\n\n5.')
        
        return fixed_text
    
    def summariseResponse(self, response: str, max_length: int = 200) -> str:
        """
        Tworzy zwięzłe podsumowanie wygenerowanej odpowiedzi.
        
        Args:
            response: Pełna odpowiedź do podsumowania
            max_length: Maksymalna długość podsumowania (w znakach)
            
        Returns:
            Zwięzłe podsumowanie odpowiedzi
        """
        if not response or len(response.strip()) == 0:
            return ""
        
        # Prompt systemowy do podsumowania
        system_prompt = """
        Jesteś ekspertem w tworzeniu zwięzłych podsumowań tekstów prawnych.
        Twoje podsumowania są:
        - Krótkie i treściwe (nie przekraczają wyznaczonego limitu znaków)
        - Zawierają najważniejsze informacje i konkluzje
        - Zachowują oryginalną terminologię prawną
        - Są obiektywne i neutralne
        - Są napisane w czasie teraźniejszym, w trzeciej osobie
        
        NIE zawierają:
        - Wprowadzeń typu "Podsumowanie:" lub "W skrócie:"
        - Własnych opinii lub interpretacji
        - Zbędnych szczegółów lub przykładów
        
        Po prostu przedstaw najważniejsze fakty i konkluzje w zwięzłej formie.
        """
        
        # Prompt użytkownika do podsumowania
        user_prompt = f"""
        Stwórz zwięzłe podsumowanie poniższego tekstu prawnego. 
        Podsumowanie powinno mieć maksymalnie {max_length} znaków.
        
        TEKST DO PODSUMOWANIA:
        {response}
        """
        
        try:
            # Konfiguracja dla generatora
            config = {
                "temperature": 0,
                "num_predict": max_length * 2  # Trochę więcej niż wymagane
            }
            
            result = requests.post(
                f"{self.generator.base_url}/api/generate",
                json={
                    "model": self.generator.model,
                    "prompt": user_prompt,
                    "stream": False,
                    "system": system_prompt,
                    "options": config
                },
                timeout=15  # Krótszy timeout dla podsumowania
            )
            
            if result.status_code == 200:
                summary = result.json().get("response", "")
                
                # Upewnij się, że podsumowanie nie przekracza maksymalnej długości
                if len(summary) > max_length:
                    summary = summary[:max_length].rsplit('. ', 1)[0] + '.'
                    
                return summary
            else:
                if self.debug_mode:
                    print(f"Błąd podczas generowania podsumowania: {result.text}")
                return response[:max_length] + "..." if len(response) > max_length else response
                
        except Exception as e:
            if self.debug_mode:
                print(f"Wyjątek podczas podsumowywania: {str(e)}")
            return response[:max_length] + "..." if len(response) > max_length else response

    def generalizeResponse(self, response: str) -> str:
        """
        Generalizuje odpowiedź, przekształcając ją w bardziej ogólne zasady lub wytyczne.
        
        Args:
            response: Szczegółowa odpowiedź do uogólnienia
            
        Returns:
            Uogólniona wersja odpowiedzi
        """
        if not response or len(response.strip()) == 0:
            return ""
        
        # Prompt systemowy do generalizacji
        system_prompt = """
        Jesteś ekspertem prawnym, który potrafi przekształcać szczegółowe odpowiedzi na tematy prawne 
        w ogólne zasady i wytyczne. Twoje generalizacje:
        
        - Wyodrębniają uniwersalne zasady i reguły z konkretnych przypadków
        - Formułują ogólne wytyczne zamiast szczegółowych procedur
        - Używają obiektywnego, neutralnego języka
        - Zachowują poprawność merytoryczną
        - Są napisane w formie bezosobowej lub trzeciej osobie
        - Stosują język zrozumiały dla niespecjalistów, zachowując kluczową terminologię prawną
        
        NIE wprowadzaj nowych informacji, których nie ma w oryginalnej odpowiedzi.
        NIE rozpoczynaj od fraz typu "Oto ogólne zasady:" lub "W ogólności:"
        
        Po prostu przeformułuj szczegółową odpowiedź w formę ogólnych zasad i wytycznych.
        """
        
        # Prompt użytkownika do generalizacji
        user_prompt = f"""
        Przekształć poniższą szczegółową odpowiedź prawną w zestaw ogólnych zasad i wytycznych.
        Zachowaj wszystkie kluczowe informacje, ale sformułuj je w sposób bardziej uniwersalny.
        
        ODPOWIEDŹ DO GENERALIZACJI:
        {response}
        """
        
        try:
            # Konfiguracja dla generatora
            config = {
                "temperature": 0.1,  # Lekko zwiększona temperatura dla bardziej kreatywnej generalizacji
                "num_predict": len(response) * 2  # Zachowaj podobną długość
            }
            
            result = requests.post(
                f"{self.generator.base_url}/api/generate",
                json={
                    "model": self.generator.model,
                    "prompt": user_prompt,
                    "stream": False,
                    "system": system_prompt,
                    "options": config
                },
                timeout=30
            )
            
            if result.status_code == 200:
                generalized = result.json().get("response", "")
                return generalized
            else:
                if self.debug_mode:
                    print(f"Błąd podczas generalizacji odpowiedzi: {result.text}")
                return response
                
        except Exception as e:
            if self.debug_mode:
                print(f"Wyjątek podczas generalizacji: {str(e)}")
            return response

    def reasonifyResponse(self, response: str, question: str) -> str:
        """
        Wzbogaca odpowiedź o wyjaśnienie rozumowania, uzasadnienie i kontekst.
        
        Args:
            response: Podstawowa odpowiedź do uzasadnienia
            question: Oryginalne pytanie, na które udzielono odpowiedzi
            
        Returns:
            Wzbogacona odpowiedź z wyjaśnieniem rozumowania
        """
        if not response or len(response.strip()) == 0:
            return ""
        
        # Prompt systemowy do uzasadnienia
        system_prompt = """
        Jesteś ekspertem prawnym, który potrafi wzbogacać odpowiedzi o dogłębne wyjaśnienia i uzasadnienia.
        Twoje rozszerzone odpowiedzi:
        
        - Wyjaśniają rozumowanie stojące za konkluzjami
        - Pokazują związki przyczynowo-skutkowe
        - Omawiają kontekst i znaczenie przepisów prawnych
        - Przedstawiają logikę i argumentację prawną
        - Wyjaśniają, dlaczego dane przepisy mają zastosowanie w tej sytuacji
        - W razie potrzeby wyjaśniają niejasne terminy prawne
        - Podają przykłady praktycznego zastosowania, jeśli to pomoże w zrozumieniu
        
        NIE wprowadzaj nowych faktów, których nie ma w oryginalnej odpowiedzi.
        NIE spekuluj na temat informacji, których nie posiadasz.
        NIE używaj fraz typu "uzasadnienie jest następujące:" - po prostu płynnie wpleć uzasadnienie w tekst.
        
        Zachowaj wszystkie informacje z oryginalnej odpowiedzi, ale wzbogać je o wyjaśnienia
        przyczyn, konsekwencji i kontekstu, tak aby odpowiedź była bardziej zrozumiała i przekonująca.
        """
        
        # Prompt użytkownika do uzasadnienia
        user_prompt = f"""
        Rozszerz poniższą odpowiedź prawną o dogłębne wyjaśnienia, uzasadnienia i kontekst.
        Zachowaj wszystkie kluczowe informacje, ale wzbogać je o wyjaśnienie rozumowania
        i uzasadnienie konkluzji.
        
        ORYGINALNE PYTANIE:
        {question}
        
        ODPOWIEDŹ DO UZASADNIENIA:
        {response}
        """
        
        try:
            # Konfiguracja dla generatora
            config = {
                "temperature": 0.2,  # Lekko zwiększona temperatura dla bardziej naturalnych wyjaśnień
                "num_predict": len(response) * 2  # Uzasadniona odpowiedź może być dłuższa
            }
            
            result = requests.post(
                f"{self.generator.base_url}/api/generate",
                json={
                    "model": self.generator.model,
                    "prompt": user_prompt,
                    "stream": False,
                    "system": system_prompt,
                    "options": config
                },
                timeout=45  # Dłuższy timeout dla bardziej złożonego zadania
            )
            
            if result.status_code == 200:
                reasonified = result.json().get("response", "")
                return reasonified
            else:
                if self.debug_mode:
                    print(f"Błąd podczas uzasadniania odpowiedzi: {result.text}")
                return response
                
        except Exception as e:
            if self.debug_mode:
                print(f"Wyjątek podczas uzasadniania: {str(e)}")
            return response
