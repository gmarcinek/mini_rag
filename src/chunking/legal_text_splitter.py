from dataclasses import dataclass
from typing import List, Dict, Optional
import re
from pathlib import Path
from src.chunking.config import SECTION_PATTERNS
from src.generation.ollama_text_fixer import OllamaTextFixer

@dataclass
class LegalChunk:
    text: str
    section_type: str
    section_number: str
    doc_id: str = ""
    chunk_id: int = 0
    parent_section: Optional[str] = None

class ChunkTransformer:
    """Bazowa klasa dla transformacji chunków."""
    def transform(self, text: str) -> str:
        return text

class OllamaChunkTransformer(ChunkTransformer):
    """Transformuje tekst chunka używając modelu Ollama."""
    def __init__(self, model_name: str = "llama3.2"):
        self.generator = OllamaTextFixer(model_name=model_name)

    def transform(self, text: str) -> str:
        prompt = f"Oczyść poniższy tekst OWU z błędów edytorskich i szumu:\n nie komentuj nie halucynuj, nie dodawaj nic od siebie, utrzymaj strukture i znaczenie zdań \n\n{text}"

        fixed_text = self.generator.generate(prompt, [])
        return fixed_text


class LegalTextSplitter:
    def __init__(self, transformer: Optional[ChunkTransformer] = None):
        self.transformer = transformer or ChunkTransformer()
        self.patterns = SECTION_PATTERNS
        
        # Tworzymy katalog chunks jeśli nie istnieje
        self.chunks_dir = Path("chunks")
        self.chunks_dir.mkdir(exist_ok=True)

    def find_section_matches(self, text: str) -> List[tuple]:
        matches = []
        for section_type, pattern in self.patterns.items():
            for match in re.finditer(pattern, text, re.MULTILINE | re.IGNORECASE):
                start = match.start()
                section_text = match.group()
                number = re.search(r'(?:[IVXLCDM]+|\d+)', section_text)
                number = number.group() if number else ''
                matches.append((start, section_type, number, section_text))
        return sorted(matches, key=lambda x: x[0])
    
    def split_text(self, text: str, doc_id: str = "") -> List[LegalChunk]:
        matches = self.find_section_matches(text)
        chunks = []
        
        for i in range(len(matches)):
            start = matches[i][0]
            end = matches[i+1][0] if i < len(matches)-1 else len(text)
            
            next_match_in_chunk = None
            for j in range(i+1, len(matches)):
                if matches[j][0] < end:
                    next_match_in_chunk = matches[j][0]
                    break
            
            if next_match_in_chunk:
                end = next_match_in_chunk
                
            section_text = text[start:end].strip()
            transformed_text = self.transformer.transform(section_text)
            
            chunk = LegalChunk(
                text=transformed_text,
                section_type=matches[i][1],
                section_number=matches[i][2],
                doc_id=doc_id,
                chunk_id=len(chunks)
            )
            chunks.append(chunk)
            
            # Używamy self.chunks_dir do tworzenia ścieżki
            chunk_path = self.chunks_dir / f"{doc_id}_{chunk.chunk_id}.txt"
            chunk_path.write_text(transformed_text, encoding='utf-8')

        return chunks