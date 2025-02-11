import hashlib
from typing import List, Tuple, Set
from src.chunking import Chunk

class DocumentSimilarity:
    def __init__(self):
        self.seen_content: Set[str] = set()
    
    def group_similar_chunks(self, chunks_with_scores: List[Tuple[Chunk, float]]) -> List[Tuple[Chunk, float]]:
        """Grupuje podobne chunki, eliminując duplikaty na podstawie ich zawartości."""
        grouped_chunks = []
        self.seen_content.clear()
        
        for chunk, score in chunks_with_scores:
            content_hash = self.get_content_hash(chunk.text)
            if content_hash not in self.seen_content:
                self.seen_content.add(content_hash)
                grouped_chunks.append((chunk, score))
        
        return grouped_chunks

    @staticmethod
    def get_content_hash(text: str) -> str:
        """Generuje hash dla znormalizowanej zawartości tekstu."""
        normalized_text = ' '.join(text.lower().split())
        return hashlib.md5(normalized_text[:100].encode()).hexdigest()