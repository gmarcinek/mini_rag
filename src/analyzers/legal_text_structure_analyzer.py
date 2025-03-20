import re
from typing import List, Dict, Tuple, Optional, Set
from pathlib import Path
from collections import defaultdict

class LegalTextStructureAnalyzer:
    """
    Narzędzie do analizy struktury tekstu prawnego - zoptymalizowane pod kątem chunkowania.
    Koncentruje się na wykrywaniu i hierarchizacji elementów strukturalnych tekstu.
    """
    
    # Wzorce rozpoznawania sekcji strukturalnych
    SECTION_PATTERNS = {
        'rozdzial': r'^(?:Rozdział|Rozdz\.|R\.)\s*(\d+|[IVXLCDM]+)\.?',  # Dodane \.? aby obsługiwać opcjonalną kropkę
        'art': r'^(?:Art\.|Artykuł)\s*(\d+)\.?',  # Również dodane \.?
        'paragraf': r'^(?:§|Par\.)\s*(\d+)\.?',
        'ustep': r'^(?:Ust\.|Ustęp|U\.)\s*(\d+)\.?',
        'zalacznik': r'^(?:Zał\.|Załącznik|Z\.)\s*(\d+)\.?',
        'sekcja': r'^(?:Sekcja|Sek\.)\s*(\d+)\.?',
        'owu': r'^(?:OWU|Ogólne\s*Warunki\s*Umowy)\s*(\d+)\.?',
        'definicje': r'^(?:Definicje|Def\.)\s*(\d+)\.?',
        'postanowienia': r'^(?:Postanowienia)\s*(?:Ogólne|Szczegółowe|Końcowe)\s*(\d+)\.?',
    }
    
    # Wzorce dla punktów i podpunktów
    POINT_PATTERNS = [
        (r'^(\d+)\)\s', 'punkt'),                 # np. "1) Treść punktu"
        (r'^(\d+\.\d+)\)\s', 'podpunkt'),         # np. "1.1) Treść podpunktu"
        (r'^([a-z])\)\s', 'literowy_punkt'),      # np. "a) Treść punktu"
        (r'^(-|\*)\s', 'wyliczenie'),             # np. "- Treść punktu" lub "* Treść punktu"
        (r'^\s*(\d+\°)\s', 'stopniowy_punkt')     # np. "1° Treść punktu"
    ]

    HIERARCHY = {
        'rozdział': 0,
        'sekcja': 0,
        'zalacznik': 0,
        'owu': 0,
        'definicje': 1,
        'postanowienia': 1,
        'artykuł': 1,
        'art': 1,
        'paragraf': 1,
        'punkt': 2,
        'ustep': 2,
        'stopniowy_punkt': 2,
        'podpunkt': 3,
        'wyliczenie': 3,
        'literowy_punkt': 3,
    }
    
    def __init__(self, text: str):
        """
        Inicjalizacja analizatora z tekstem.
        
        Args:
            text: Tekst dokumentu do analizy
        """
        self.text = text
        self.lines = text.splitlines()
        self.section_markers = self._identify_all_section_markers()
        self.hierarchy = self._build_document_hierarchy()
    
    def _identify_all_section_markers(self) -> List[Dict]:
        markers = []
        
        for line_idx, line in enumerate(self.lines):
            line = line.strip()
            if not line:
                continue
            
            # Najpierw sprawdzamy sekcje strukturalne
            section_found = False
            for section_type, pattern in self.SECTION_PATTERNS.items():
                if match := re.match(pattern, line):
                    section_name = match.group(1)
                    section_id = f"{section_type}_{section_name}"
                    
                    # Upewniamy się, że używamy spójnego klucza do HIERARCHY
                    # Jeśli section_type to 'rozdzial', konwertujemy na 'rozdział' dla zgodności z HIERARCHY
                    hierarchy_key = 'rozdział' if section_type == 'rozdzial' else section_type
                    
                    markers.append({
                        'type': section_type,
                        'hierarchy_key': hierarchy_key,  # Dodajemy klucz do hierarchii
                        'name': section_name,
                        'id': section_id,
                        'line': line_idx,
                        'content_start': line_idx,
                        'text': line
                    })
                    section_found = True
                    break
            
            # Jeśli nie znaleziono sekcji strukturalnej, sprawdzamy czy to punkt
            if not section_found:
                # Dodajemy obsługę punktów numerycznych jako domyślny typ
                punkt_match = re.match(r'^(\d+)\.\s*(.+)', line)
                if punkt_match:
                    punkt_name = punkt_match.group(1)
                    punkt_id = f"punkt_numeryczny_{punkt_name}_{line_idx}"
                    
                    markers.append({
                        'type': 'punkt',
                        'hierarchy_key': 'punkt',  # Dodajemy klucz do hierarchii
                        'subtype': 'punkt_numeryczny',
                        'name': punkt_name,
                        'id': punkt_id,
                        'line': line_idx,
                        'content_start': line_idx,
                        'text': line
                    })
                
                # Zachowujemy istniejące wzorce punktów
                elif not punkt_match:
                    for pattern, punkt_type in self.POINT_PATTERNS:
                        if punkt_match := re.match(pattern, line):
                            punkt_name = punkt_match.group(1)
                            punkt_id = f"{punkt_type}_{punkt_name}_{line_idx}"
                            
                            markers.append({
                                'type': 'punkt',
                                'hierarchy_key': punkt_type,  # Dodajemy klucz do hierarchii
                                'subtype': punkt_type,
                                'name': punkt_name,
                                'id': punkt_id,
                                'line': line_idx,
                                'content_start': line_idx,
                                'text': line
                            })
                            break
        
        # Sortujemy markery według pozycji w tekście
        markers = sorted(markers, key=lambda x: x['line'])
        
        # Przypisujemy hierarchię do każdego markera
        for marker in markers:
            # Używamy hierarchy_key do uzyskania poziomu hierarchii
            hierarchy_key = marker.get('hierarchy_key', marker['type'])
            marker['hierarchy_level'] = self.HIERARCHY.get(hierarchy_key, 3)
        
        # Dodajemy informacje o końcu zawartości każdej sekcji
        for i, marker in enumerate(markers):
            current_type = marker['type']
            current_level = marker['hierarchy_level']
            
            # Znajdź koniec sekcji
            for j in range(i + 1, len(markers)):
                next_marker = markers[j]
                next_type = next_marker['type']
                next_level = next_marker['hierarchy_level']
                
                # Sekcja kończy się gdy:
                # 1. Napotkamy sekcję tego samego typu (np. artykuł -> artykuł)
                # 2. LUB napotkamy sekcję wyższego poziomu (np. artykuł -> rozdział)
                # Mniejsza liczba w HIERARCHY oznacza wyższy poziom w hierarchii
                if next_type == current_type or next_level <= current_level:
                    marker['content_end'] = next_marker['line'] - 1
                    break
            else:
                # Jeśli to ostatnia sekcja w dokumencie
                marker['content_end'] = len(self.lines)
        
        return markers
    
    def _build_document_hierarchy(self) -> Dict[str, List[str]]:
        """
        Buduje hierarchię dokumentu (relacje rodzic-dziecko między sekcjami).
        
        Returns:
            Słownik {parent_id: [child_id1, child_id2, ...]}
        """
        hierarchy = defaultdict(list)
        
        # Hierarchia typów sekcji (od najwyższego do najniższego poziomu)
        hierarchy_order = [
            'rozdzial', 'sekcja', 'postanowienia', 'definicje', 'owu', 'zalacznik',
            'art', 'paragraf', 'ustep', 'punkt'
        ]
        
        # Aktualny kontener dla każdego poziomu hierarchii
        current_containers = {}
        
        for marker in self.section_markers:
            section_type = marker['type']
            section_id = marker['id']
            
            # Jeśli to typ strukturalny (nie punkt), resetujemy wszystkie niższe poziomy
            if section_type != 'punkt':
                section_index = hierarchy_order.index(section_type)
                
                # Usuwamy wszystkie kontenery niższych poziomów
                for lower_type in hierarchy_order[section_index+1:]:
                    current_containers.pop(lower_type, None)
                
                # Ustawiamy ten jako kontener dla jego poziomu
                current_containers[section_type] = section_id
                
                # Dodajemy jako dziecko najniższego wyższego kontenera
                parent_found = False
                for higher_type in reversed(hierarchy_order[:section_index]):
                    if higher_type in current_containers:
                        parent_id = current_containers[higher_type]
                        hierarchy[parent_id].append(section_id)
                        parent_found = True
                        break
                
                # Jeśli nie znaleziono rodzica, dodajemy do głównego dokumentu
                if not parent_found:
                    hierarchy['dokument'].append(section_id)
            else:
                # Dla punktów szukamy najbliższego kontenera
                parent_found = False
                for container_type in ['art', 'paragraf', 'ustep']:
                    if container_type in current_containers:
                        parent_id = current_containers[container_type]
                        hierarchy[parent_id].append(section_id)
                        parent_found = True
                        break
                
                # Jeśli nie znaleziono bezpośredniego kontenera, używamy najniższego dostępnego
                if not parent_found:
                    for container_type in reversed(hierarchy_order):
                        if container_type in current_containers:
                            parent_id = current_containers[container_type]
                            hierarchy[parent_id].append(section_id)
                            break
                
                # Ustawiamy ten punkt jako kontener dla poziomu punkt
                current_containers['punkt'] = section_id
        
        return dict(hierarchy)
    
    def get_section_content(self, section_id: str) -> str:
        for marker in self.section_markers:
            if marker['id'] == section_id:
                start = marker['content_start']
                end = marker['content_end']
                return "\n".join(self.lines[start:end])
        
        return ""
    
    def create_child_to_parent_map(self) -> Dict[str, str]:
        child_to_parent = {}
        
        for parent, children in self.hierarchy.items():
            for child in children:
                child_to_parent[child] = parent
        
        return child_to_parent
    
    def build_context_path(self, section_id: str) -> List[Dict[str, str]]:
        path = []
        current_id = section_id
        
        # Znajdź marker dla danej sekcji
        marker = next((m for m in self.section_markers if m['id'] == section_id), None)
        if marker:
            path.append({
                'type': marker['type'],
                'id': marker['id'],
                'name': marker['name'],
                'subtype': marker.get('subtype', '')
            })
        
        # Budujemy ścieżkę w górę hierarchii
        child_to_parent = self.create_child_to_parent_map()
        while current_id in child_to_parent:
            parent_id = child_to_parent[current_id]
            if parent_id == 'dokument':
                break
            
            parent_marker = next((m for m in self.section_markers if m['id'] == parent_id), None)
            if parent_marker:
                path.append({
                    'type': parent_marker['type'],
                    'id': parent_marker['id'],
                    'name': parent_marker['name'],
                    'subtype': parent_marker.get('subtype', '')
                })
            
            current_id = parent_id
        
        # Odwracamy, żeby ścieżka była od korzenia do liścia
        return list(reversed(path))
    
    def get_section_bounds(self, section_id: str) -> Optional[Tuple[int, int]]:
        for marker in self.section_markers:
            if marker['id'] == section_id:
                return marker['line'], marker['content_end']
        
        return None
    
    def get_children_for_section(self, section_id: str) -> List[Dict]:
        if section_id not in self.hierarchy:
            return []
        
        child_ids = self.hierarchy[section_id]
        return [marker for marker in self.section_markers if marker['id'] in child_ids]
    
    def get_sections_by_type(self, section_type: str) -> List[Dict]:
        return [marker for marker in self.section_markers if marker['type'] == section_type]
    
    def format_context_path(self, path: List[Dict[str, str]]) -> str:
        context_parts = []
        
        for ctx in path:
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