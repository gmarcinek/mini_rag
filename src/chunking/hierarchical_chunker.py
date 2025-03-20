from dataclasses import dataclass
from typing import List, Dict, Optional, Set, Tuple
import re
from pathlib import Path
from collections import defaultdict

from src.analyzers.legal_text_structure_analyzer import LegalTextStructureAnalyzer

@dataclass
class LegalChunk:
    """Reprezentuje chunk tekstu prawnego z pełnym kontekstem strukturalnym."""
    text: str
    section_type: str
    section_id: str
    doc_id: str
    chunk_id: int
    context_path: List[Dict[str, str]]  # Ścieżka kontekstu: lista [{"type": typ, "id": id, "name": nazwa}, ...]
    line_start: int
    line_end: int
    subtype: str = ""
    original_marker: Optional[Dict] = None
    
    def get_full_context_str(self):
        """
        Zwraca pełny kontekst jako string do wyświetlenia w nagłówku chunka.
        Na przykład: "Rozdział II, Artykuł 4"
        """
        if not self.context_path:
            return ""
        
        context_parts = []
        for ctx in self.context_path:
            # Obsługa zarówno starego (type/name) jak i nowego (category/number) formatu
            if 'category' in ctx:
                section_type = ctx['category']
                section_name = ctx['number']
            else:
                # Fallback do starego formatu dla kompatybilności
                section_type = ctx.get('type', '')
                section_name = ctx.get('name', '')
            
            # Tłumaczenie typów na bardziej czytelne nazwy (dla starego formatu)
            type_mapping = {
                'rozdzial': 'Rozdział',
                'art': 'Artykuł',
                'paragraf': 'Paragraf',
                'ustep': 'Ustęp',
                'zalacznik': 'Załącznik',
                'sekcja': 'Sekcja',
                'owu': 'OWU',
                'definicje': 'Definicje',
                'postanowienia': 'Postanowienia',
                'punkt': 'Punkt'
            }
            
            if section_type in type_mapping:
                section_type = type_mapping[section_type]
            
            # Dodajemy tytuł sekcji, jeśli jest dostępny (tylko dla nowego formatu)
            if 'title' in ctx and ctx['title']:
                context_parts.append(f"{section_type} {section_name} ({ctx['title']})")
            else:
                context_parts.append(f"{section_type} {section_name}")
        
        return ", ".join(context_parts)


class HierarchicalLegalChunker:
    """
    Ulepszony chunker tekstu prawnego oparty na hierarchii dokumentu.
    Zapewnia inteligentny podział na chunki z zachowaniem pełnego kontekstu strukturalnego.
    """
    
    def __init__(self, text_analyzer=None):
        """
        Inicjalizuje chunker.
        
        Args:
            text_analyzer: Ignorowany parametr dla kompatybilności ze starym API
        """
        self.chunks_dir = Path("chunks")
        self.chunks_dir.mkdir(exist_ok=True)
    
   

    def split_text(self, text: str, doc_id: str = "") -> List[LegalChunk]:
        """
        Dzieli tekst prawny na chunki z uwzględnieniem struktury dokumentu.
        Zachowuje naturalną strukturę dokumentu bez ograniczeń rozmiaru.
        
        Args:
            text: Tekst dokumentu do chunkowania
            doc_id: Identyfikator dokumentu
            
        Returns:
            Lista obiektów LegalChunk z pełnym kontekstem
        """
        # Analizujemy strukturę dokumentu
        analyzer = LegalTextStructureAnalyzer(text)
        lines = text.splitlines()
        
        # Przygotowujemy wynikową listę chunków
        chunks = []
        chunk_id = 0
        
        
        # 1. Tworzymy chunki dla artykułów (jako całość)
        articles = analyzer.get_sections_by_type('art')
        for article in articles:
            art_id = article['id']
            art_start, art_end = article['line'], article['content_end']
            article_text = "\n".join(lines[art_start:art_end])
            
            # Budujemy pełną ścieżkę kontekstu
            context_path = analyzer.build_context_path(art_id)
            
            # Tworzymy chunk dla całego artykułu
            chunk = LegalChunk(
                text=article_text,
                section_type='art',
                section_id=art_id,
                doc_id=doc_id,
                chunk_id=chunk_id,
                context_path=context_path,
                line_start=art_start,
                line_end=art_end,
                original_marker=article
            )
            chunks.append(chunk)
            chunk_id += 1
        
        # 2. Uwzględniamy pozostałe typy sekcji
        for section_type in ['paragraf', 'ustep', 'sekcja']:
            sections = analyzer.get_sections_by_type(section_type)
            for section in sections:
                section_id = section['id']
                section_start, section_end = section['line'], section['content_end']
                section_text = "\n".join(lines[section_start:section_end])
                    
                context_path = analyzer.build_context_path(section_id)
                
                chunk = LegalChunk(
                    text=section_text,
                    section_type=section_type,
                    section_id=section_id,
                    doc_id=doc_id,
                    chunk_id=chunk_id,
                    context_path=context_path,
                    line_start=section_start,
                    line_end=section_end,
                    original_marker=section
                )
                chunks.append(chunk)
                chunk_id += 1
        
        # Opcjonalne: Zapisujemy chunki do plików 
        self._save_chunks_to_files(chunks)
        
        return chunks
    
    def _save_chunks_to_files(self, chunks: List[LegalChunk]) -> None:
        """
        Zapisuje chunki do plików tekstowych.
        
        Args:
            chunks: Lista chunków do zapisania
        """
        for chunk in chunks:
            # Tworzymy nazwę pliku zawierającą identyfikator dokumentu i chunka
            if chunk.section_type == 'punkt' and chunk.subtype:
                filename = f"{chunk.doc_id}_{chunk.chunk_id}_{chunk.section_type}_{chunk.subtype}.txt"
            else:
                filename = f"{chunk.doc_id}_{chunk.chunk_id}_{chunk.section_type}.txt"
                
            filepath = self.chunks_dir / filename
            
            # Dodajemy nagłówek z kontekstem
            context_header = f"[{chunk.get_full_context_str()}]\n"
            
            # Formatujemy nazwę sekcji odpowiednio
            if chunk.section_type == 'punkt' and chunk.subtype:
                section_name = chunk.section_id.split('_')[1]  # Pobierz nazwę punktu (np. 'a', '1', itd.)
                if chunk.subtype == 'literowy_punkt':
                    section_display = f"{section_name})"
                elif chunk.subtype == 'punkt':
                    section_display = f"{section_name})"
                elif chunk.subtype == 'podpunkt':
                    section_display = f"{section_name})"
                elif chunk.subtype == 'wyliczenie':
                    section_display = f"- {section_name}"
                elif chunk.subtype == 'stopniowy_punkt':
                    section_display = f"{section_name}°"
                else:
                    section_display = f"{section_name})"
            else:
                # Dla innych typów sekcji, pobierz nazwę z ID (np. 'art_1' -> '1')
                section_display = chunk.section_id.split('_', 1)[1] if '_' in chunk.section_id else chunk.section_id
            
            # Zapisujemy tekst z nagłówkiem kontekstu
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(context_header + chunk.text)
