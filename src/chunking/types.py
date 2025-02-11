from dataclasses import dataclass
from .text_splitter import Chunk  # importujemy Chunk z tego samego modułu

@dataclass
class ChunkInfo:
    chunk: Chunk
    embedding_hash: str