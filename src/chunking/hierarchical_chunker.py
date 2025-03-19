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
    
    def get_full_context_str(self) -> str:
        """Zwraca sformatowany string z pełną ścieżką kontekstu."""
        context_parts = []
        for ctx in self.context_path:
            section_type = ctx['type']
            name = ctx['name']
            subtype = ctx.get('subtype', '')
            
            # Formatowanie w zależności od typu sekcji
            if section_type == 'rozdzial':
                formatted = f"Rozdział {name}"
            elif section_type == 'art':
                formatted = f"Artykuł {name}"
            elif section_type == 'paragraf':
                formatted = f"§ {name}"
            elif section_type == 'ustep':
                formatted = f"Ustęp {name}"
            elif section_type == 'punkt':
                # Formatowanie w zależności od podtypu punktu
                if subtype == 'literowy punkt':
                    formatted = f"{name})"
                elif subtype == 'punkt':
                    formatted = f"{name})"
                elif subtype == 'podpunkt':
                    formatted = f"{name})"
                elif subtype == 'wyliczenie':
                    formatted = f"- {name}"
                elif subtype == 'stopniowy punkt':
                    formatted = f"{name}°"
                else:
                    formatted = f"{name})"
            else:
                formatted = f"{section_type.capitalize()} {name}"
            
            context_parts.append(formatted)
        
        return " > ".join(context_parts)


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
        
        
        # 3. Tworzymy chunki dla rozdziałów (jako całość)
        chapters = analyzer.get_sections_by_type('rozdzial')
        for chapter in chapters:
            chapter_id = chapter['id']
            chapter_start, chapter_end = chapter['line'], chapter['content_end']
            chapter_text = "\n".join(lines[chapter_start:chapter_end])
                
            context_path = analyzer.build_context_path(chapter_id)
            
            chunk = LegalChunk(
                text=chapter_text,
                section_type='rozdzial',
                section_id=chapter_id,
                doc_id=doc_id,
                chunk_id=chunk_id,
                context_path=context_path,
                line_start=chapter_start,
                line_end=chapter_end,
                original_marker=chapter
            )
            chunks.append(chunk)
            chunk_id += 1
        
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
            
            # 2. Tworzymy chunki dla punktów w artykule
            # Znajdujemy wszystkie punkty, które są dziećmi tego artykułu
            point_children = [
                marker for marker in analyzer.section_markers 
                if marker['type'] == 'punkt' and 
                analyzer.create_child_to_parent_map().get(marker['id']) == art_id
            ]
            
            for point in point_children:
                point_id = point['id']
                point_start, point_end = point['line'], point['content_end']
                
                # Budujemy ścieżkę kontekstu dla punktu
                point_context = analyzer.build_context_path(point_id)
                
                # Treść punktu
                point_text = "\n".join(lines[point_start:point_end])
                
                chunk = LegalChunk(
                    text=point_text,
                    section_type='punkt',
                    section_id=point_id,
                    doc_id=doc_id,
                    chunk_id=chunk_id,
                    context_path=point_context,
                    line_start=point_start,
                    line_end=point_end,
                    subtype=point.get('subtype', ''),
                    original_marker=point
                )
                chunks.append(chunk)
                chunk_id += 1
        
        
        
        # 4. Uwzględniamy pozostałe typy sekcji
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
            context_header = f"KONTEKST: {chunk.get_full_context_str()}\n"
            
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
            
            context_header += f"SEKCJA: {chunk.section_type} {section_display}\n"
            context_header += "-" * 80 + "\n\n"
            
            # Zapisujemy tekst z nagłówkiem kontekstu
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(context_header + chunk.text)