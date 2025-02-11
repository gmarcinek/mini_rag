from dataclasses import dataclass
from typing import List

@dataclass
class Chunk:
    text: str
    doc_id: str = ""
    chunk_id: int = 0

class SimpleTextSplitter:
    def __init__(self, 
                 chunk_size: int = 1500,  # zwiększone z 512
                 chunk_overlap: int = 200,  # zwiększone dla lepszej ciągłości
                 separator: str = "\n"):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separator = separator

    def split_text(self, text: str, doc_id: str = "") -> List[Chunk]:
        # Najpierw dzielimy tekst na większe sekcje używając separatora
        sections = [s.strip() for s in text.split(self.separator) if s.strip()]
        chunks = []
        current_chunk = []
        current_length = 0
        chunk_id = 0
        
        for section in sections:
            # Jeśli sekcja jest większa niż chunk_size, dzielimy ją na zdania
            if len(section) > self.chunk_size:
                # Proste dzielenie na zdania (można ulepszyć)
                sentences = [s.strip() + '.' for s in section.split('.') if s.strip()]
                for sentence in sentences:
                    if current_length + len(sentence) > self.chunk_size:
                        # Zapisz obecny chunk
                        if current_chunk:
                            chunk_text = ' '.join(current_chunk)
                            chunks.append(Chunk(text=chunk_text, doc_id=doc_id, chunk_id=chunk_id))
                            chunk_id += 1
                            
                            # Zachowaj overlap
                            overlap_size = 0
                            overlap_chunks = []
                            for c in reversed(current_chunk):
                                if overlap_size + len(c) <= self.chunk_overlap:
                                    overlap_chunks.insert(0, c)
                                    overlap_size += len(c)
                                else:
                                    break
                            current_chunk = overlap_chunks
                            current_length = overlap_size
                    
                    current_chunk.append(sentence)
                    current_length += len(sentence)
            else:
                if current_length + len(section) > self.chunk_size:
                    # Zapisz obecny chunk
                    if current_chunk:
                        chunk_text = ' '.join(current_chunk)
                        chunks.append(Chunk(text=chunk_text, doc_id=doc_id, chunk_id=chunk_id))
                        chunk_id += 1
                        
                        # Zachowaj overlap
                        overlap_size = 0
                        overlap_chunks = []
                        for c in reversed(current_chunk):
                            if overlap_size + len(c) <= self.chunk_overlap:
                                overlap_chunks.insert(0, c)
                                overlap_size += len(c)
                            else:
                                break
                        current_chunk = overlap_chunks
                        current_length = overlap_size
                
                current_chunk.append(section)
                current_length += len(section)
        
        # Nie zapomnij o ostatnim chunku
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            chunks.append(Chunk(text=chunk_text, doc_id=doc_id, chunk_id=chunk_id))
        
        print(f"Split text into {len(chunks)} chunks")
        return chunks