import re
from pathlib import Path
from typing import Dict, List, Set, Optional
from collections import defaultdict
from .metrics import (
    TextQualityMetrics,
    StructureMetrics,
    ReferenceMetrics,
    DocumentMetrics
)

class DocumentAnalyzer:
    """Analizator jakości dokumentów wejściowych"""
    SECTION_PATTERNS = {
        'rozdzial': r'^(?:Rozdział|Rozdz\.|R\.)\s*(\d+|[IVXLCDM]+)',
        'art': r'^(?:Art\.|Artykuł)\s*(\d+)',
        'paragraf': r'^(?:§|Par\.)\s*(\d+)',
        'ustep': r'^(?:Ust\.|Ustęp|U\.)\s*(\d+)',
        'zalacznik': r'^(?:Zał\.|Załącznik|Z\.)\s*(\d+)',
        'sekcja': r'^(?:Sekcja|Sek\.)\s*(\d+)',
        'owu': r'^(?:OWU|Ogólne\s*Warunki\s*Umowy)\s*(\d+)',
        'definicje': r'^(?:Definicje|Def\.)\s*(\d+)',
        'postanowienia': r'^(?:Postanowienia)\s*(?:Ogólne|Szczegółowe|Końcowe)\s*(\d+)',
    }

    # Wzorce referencji
    REFERENCE_PATTERNS = {
        'internal': r'(?:zgodnie z|patrz|w art|w punk|w rozdziale|w rozdz|por\.|zob\.)?\s*(?:§|art\.|rozdz\.|pkt\.|ust\.)?\s*\d+',
        'external': r'(?:Dz\.U\.|M\.P\.)\s*(?:z\s*\d{4}\s*r\.)?\s*(?:Nr\s*\d+)?',
    }

    def __init__(self, file_path: Path):
        self.file_path = file_path
        self.content = self._load_document()
        self.lines = self.content.splitlines()
        self.words = re.findall(r'\w+', self.content.lower())

    def _load_document(self) -> str:
        """Wczytuje zawartość dokumentu"""
        return self.file_path.read_text(encoding='utf-8')

    def analyze_text_quality(self) -> TextQualityMetrics:
        """Analizuje podstawowe metryki jakości tekstu"""
        # Czyścimy linie z whitespace'ów przed analizą
        content_lines = [line.strip() for line in self.lines if line.strip()]
        empty_lines = len([line for line in self.lines if not line.strip()])
        total_lines = len(self.lines)
        
        # Zliczamy znaki specjalne z większą precyzją
        special_chars = len(re.findall(r'[^a-zA-Z0-9\s\.\,\-]', self.content))
        words = re.findall(r'\b\w+\b', self.content.lower())
        
        return TextQualityMetrics(
            total_chars=len(self.content),
            total_lines=total_lines,
            avg_chars_per_line=len(self.content) / total_lines if total_lines else 0,
            empty_lines_ratio=empty_lines / total_lines if total_lines else 0,
            special_chars_ratio=special_chars / len(self.content) if self.content else 0,
            avg_word_length=sum(len(w) for w in words) / len(words) if words else 0,
            unique_words_ratio=len(set(words)) / len(words) if words else 0
        )

    def analyze_structure(self) -> StructureMetrics:
        """Analizuje strukturę dokumentu z obsługą wielu typów sekcji"""
        sections = defaultdict(list)
        hierarchy = defaultdict(list)
        
        # Słownik do śledzenia bieżących kontenerów na różnych poziomach
        current_containers = {}
        
        # Hierarchia typów sekcji (od najwyższego do najniższego poziomu)
        hierarchy_order = [
            'rozdzial',
            'sekcja',
            'postanowienia',
            'definicje',
            'owu',
            'zalacznik',
            'art',
            'paragraf',
            'ustep',
            'punkt'
        ]
        
        # Liczniki wystąpień nazw sekcji (do rozróżniania powtórzeń)
        section_counters = defaultdict(int)
        
        # Słownik mapujący nazwę sekcji na jej unikalny identyfikator
        section_ids = {}
        
        for line_idx, line in enumerate(self.lines):
            matched = False
            
            # Próba dopasowania znanych typów sekcji z SECTION_PATTERNS
            for section_type, pattern in self.SECTION_PATTERNS.items():
                if match := re.match(pattern, line):
                    matched = True
                    section_name = match.group(1)
                    base_section_id = f"{section_type}_{section_name}"
                    
                    # Inkrementuj licznik dla tej nazwy sekcji
                    section_counters[base_section_id] += 1
                    counter = section_counters[base_section_id]
                    
                    # Utwórz unikalny identyfikator dla tej sekcji
                    unique_section_id = f"{base_section_id}" if counter == 1 else f"{base_section_id} ({counter})"
                    
                    # Zapisz informację o tej sekcji
                    sections[section_type].append(section_name)
                    section_ids[(section_type, section_name, line_idx)] = unique_section_id
                    
                    # Wyczyść wszystkie kontenery na niższych poziomach
                    section_index = hierarchy_order.index(section_type)
                    for lower_type in hierarchy_order[section_index+1:]:
                        current_containers.pop(lower_type, None)
                    
                    # Ustaw tę sekcję jako kontener dla jej poziomu
                    current_containers[section_type] = unique_section_id
                    
                    # Logika hierarchii - dodaj tę sekcję jako dziecko najniższego kontenera wyższego poziomu
                    parent_found = False
                    for higher_type in reversed(hierarchy_order[:section_index]):
                        if higher_type in current_containers:
                            parent_id = current_containers[higher_type]
                            hierarchy[parent_id].append(unique_section_id)
                            parent_found = True
                            break
                    
                    # Jeśli nie znaleziono rodzica, dodaj do głównego dokumentu
                    if not parent_found:
                        hierarchy['dokument'].append(unique_section_id)
                    
                    break
            
            # Jeśli nie znaleziono standardowej sekcji, sprawdź czy to punkt artykułu
            if not matched:
                # Wzorce dla punktów i podpunktów
                punkt_patterns = [
                    r'^(\d+)\)\s',             # np. "1) Treść punktu"
                    r'^(\d+\.\d+)\)\s',        # np. "1.1) Treść podpunktu"
                    r'^([a-z])\)\s',           # np. "a) Treść punktu"
                    r'^(-|\*)\s',              # np. "- Treść punktu" lub "* Treść punktu"
                    r'^\s*(\d+\°)\s'           # np. "1° Treść punktu"
                ]
                
                for punkt_pattern in punkt_patterns:
                    if punkt_match := re.match(punkt_pattern, line):
                        punkt_id = f"punkt_{punkt_match.group(1)}"
                        
                        # Zapisz punkt jako element sekcji
                        sections['punkt'].append(punkt_match.group(1))
                        
                        # Znajdź najbliższy kontener dla punktu
                        parent_found = False
                        for container_type in ['art', 'paragraf', 'ustep']:
                            if container_type in current_containers:
                                parent_id = current_containers[container_type]
                                hierarchy[parent_id].append(punkt_id)
                                parent_found = True
                                break
                        
                        # Jeśli nie znaleziono kontenera, spróbuj dodać do najniższego kontenera
                        if not parent_found:
                            for container_type in reversed(hierarchy_order):
                                if container_type in current_containers:
                                    parent_id = current_containers[container_type]
                                    hierarchy[parent_id].append(punkt_id)
                                    break
                        
                        # Ustaw punkt jako aktualny kontener dla "punkt"
                        current_containers['punkt'] = punkt_id
                        break
        
        return StructureMetrics(
            total_sections=sum(len(s) for s in sections.values()),
            section_types={k: len(v) for k, v in sections.items()},
            max_depth=max(len(path) for path in hierarchy.values()) if hierarchy else 0,
            section_hierarchy=dict(hierarchy),
            incomplete_sections=self._find_incomplete_sections(sections),
        )

    def analyze_references(self) -> ReferenceMetrics:
        """Analizuje referencje w dokumencie"""
        internal_refs = re.findall(self.REFERENCE_PATTERNS['internal'], self.content)
        external_refs = re.findall(self.REFERENCE_PATTERNS['external'], self.content)
        
        return ReferenceMetrics(
            total_references=len(internal_refs) + len(external_refs),
            internal_references=len(internal_refs),
            external_references=len(external_refs),
            broken_references=self._find_broken_references(internal_refs),
            reference_targets=self._map_reference_targets(internal_refs),
            circular_references=self._find_circular_references()
        )

    def get_full_metrics(self) -> DocumentMetrics:
        """Oblicza pełne metryki dokumentu"""
        text_quality = self.analyze_text_quality()
        structure = self.analyze_structure()
        references = self.analyze_references()
        
        # Obliczanie metryk całościowych
        overall_quality = self._calculate_overall_quality(text_quality)
        structure_completeness = self._calculate_structure_completeness(structure)
        reference_validity = self._calculate_reference_validity(references)
        noise_level = self._calculate_noise_level(text_quality)
        
        return DocumentMetrics(
            file_path=self.file_path,
            text_quality=text_quality,
            structure=structure,
            references=references,
            overall_quality_score=overall_quality,
            structure_completeness=structure_completeness,
            reference_validity=reference_validity,
            noise_level=noise_level
        )


    def _find_incomplete_sections(self, sections: Dict[str, List[str]]) -> List[str]:
        """Znajduje niekompletne sekcje (bez treści lub z niepełną treścią)."""
        incomplete = []
        
        section_content = defaultdict(list)
        current_section = None
        
        # Przetwarzamy linie
        for line in self.lines:
            line = line.strip()
            if not line:
                continue
                
            # Sprawdzamy czy to nowa sekcja
            is_new_section = False
            for section_type, pattern in self.SECTION_PATTERNS.items():
                if match := re.match(pattern, line):
                    if current_section and len(section_content[current_section]) < 1:
                        incomplete.append(f"Niekompletna sekcja: {current_section}")
                    
                    # Dodaj sprawdzenie istnienia grupy
                    if match.groups():  # Sprawdź, czy są jakieś grupy
                        current_section = f"{section_type}_{match.group(1)}"
                    else:
                        current_section = f"{section_type}_unknown"
                    
                    is_new_section = True
                    break
                    
            if not is_new_section and current_section:
                section_content[current_section].append(line)
        
        # Sprawdzamy ostatnią sekcję
        if current_section and len(section_content[current_section]) < 1:
            incomplete.append(f"Niekompletna sekcja: {current_section}")
        
        return incomplete

    def _calculate_noise_level(self, text_quality: TextQualityMetrics) -> float:
        """Oblicza poziom szumu w tekście (0-1, gdzie 1 to maksymalny szum)."""
        noise_indicators = [
            text_quality.special_chars_ratio * 0.3,  # Wysoki współczynnik znaków specjalnych
            text_quality.empty_lines_ratio * 0.2,    # Dużo pustych linii
            (1 - text_quality.unique_words_ratio) * 0.2,  # Mała różnorodność słów
            
            # Dodatkowe wskaźniki szumu
            len([l for l in self.lines if len(l.strip()) < 3]) / len(self.lines) * 0.15,  # Krótkie linie
            len([l for l in self.lines if "  " in l]) / len(self.lines) * 0.15,  # Wielokrotne spacje
        ]
        
        return min(1.0, sum(noise_indicators))

    def _calculate_structure_completeness(self, structure: StructureMetrics) -> float:
        """Oblicza kompletność struktury dokumentu (0-1)."""
        weights = {
            'missing_sections': 0.4,  # Waga dla brakujących sekcji
            'incomplete_sections': 0.3,  # Waga dla niekompletnych sekcji
            'hierarchy': 0.3,  # Waga dla hierarchii
        }
        
        # Punkty za brakujące sekcje
        missing_score = max(0.0, missing_score)
        
        # Punkty za niekompletne sekcje
        incomplete_score = 1.0 - (len(structure.incomplete_sections) * 0.1)
        incomplete_score = max(0.0, incomplete_score)
        
        # Punkty za hierarchię
        expected_depth = 3  # Oczekiwana głębokość hierarchii
        hierarchy_score = min(1.0, structure.max_depth / expected_depth)
        
        return (
            weights['missing_sections'] * missing_score +
            weights['incomplete_sections'] * incomplete_score +
            weights['hierarchy'] * hierarchy_score
        )

    def _calculate_overall_quality(self, text_quality: TextQualityMetrics) -> float:
        """Oblicza ogólną jakość dokumentu (0-1)."""
        weights = {
            'chars_per_line': 0.2,    # Waga dla średniej długości linii
            'empty_lines': 0.1,       # Waga dla pustych linii
            'special_chars': 0.2,     # Waga dla znaków specjalnych
            'word_length': 0.2,       # Waga dla długości słów
            'word_diversity': 0.3,    # Waga dla różnorodności słów
        }
        
        # Optymalne wartości
        optimal = {
            'chars_per_line': 80,     # Optymalna liczba znaków w linii
            'empty_lines_ratio': 0.1, # Optymalny stosunek pustych linii
            'special_chars_ratio': 0.15, # Optymalny stosunek znaków specjalnych
            'avg_word_length': 6,     # Optymalna średnia długość słowa
            'unique_words_ratio': 0.7, # Optymalny stosunek unikalnych słów
        }
        
        # Obliczanie składowych jakości
        chars_per_line_score = max(0, 1 - abs(text_quality.avg_chars_per_line - optimal['chars_per_line']) / optimal['chars_per_line'])
        empty_lines_score = max(0, 1 - abs(text_quality.empty_lines_ratio - optimal['empty_lines_ratio']) / 0.2)
        special_chars_score = max(0, 1 - text_quality.special_chars_ratio / optimal['special_chars_ratio'])
        word_length_score = max(0, 1 - abs(text_quality.avg_word_length - optimal['avg_word_length']) / 5)
        word_diversity_score = min(1, text_quality.unique_words_ratio / optimal['unique_words_ratio'])
        
        return (
            weights['chars_per_line'] * chars_per_line_score +
            weights['empty_lines'] * empty_lines_score +
            weights['special_chars'] * special_chars_score +
            weights['word_length'] * word_length_score +
            weights['word_diversity'] * word_diversity_score
        )

    def _find_broken_references(self, references: List[str]) -> List[str]:
        """
        Znajduje niepoprawne referencje (odnoszące się do nieistniejących sekcji).
        """
        broken_refs = []
        
        # Mapowanie istniejących sekcji
        existing_sections = {
            'par': set(),   # paragrafy
            'art': set(),   # artykuły
            'rozdz': set(), # rozdziały
            'ust': set(),   # ustępy
            'pkt': set()    # punkty
        }

        # Wzorce do rozpoznawania sekcji (rozszerzone)
        section_patterns = [
            (r'^§\s*(\d+)', 'par'),
            (r'^Art\.\s*(\d+)', 'art'),
            (r'^Rozdział\s*(\d+)', 'rozdz'),
            (r'^(\d+)[.)]', 'ust'),
            (r'^[a-z][.)]', 'pkt')
        ]
        
        # Zbieranie istniejących sekcji z dokumentu
        for line in self.lines:
            line = line.strip()
            for pattern, section_type in section_patterns:
                match = re.match(pattern, line, re.IGNORECASE)
                if match and match.groups():
                    try:
                        existing_sections[section_type].add(match.group(1))
                    except IndexError:
                        # Log or handle the error if needed
                        print(f"Nieprawidłowe dopasowanie dla linii: {line}")
        
        # Wzorce do rozpoznawania referencji (rozszerzone)
        ref_patterns = [
            (r'(?:zgodnie z\s+)?(?:§|par\.|paragraf)\s*(\d+)', 'par'),
            (r'(?:zgodnie z\s+)?(?:art\.|artykuł)\s*(\d+)', 'art'),
            (r'(?:zgodnie z\s+)?(?:rozdz\.|rozdział)\s*(\d+)', 'rozdz'),
            (r'(?:zgodnie z\s+)?(?:ust\.|ustęp)\s*(\d+)', 'ust'),
            (r'(?:zgodnie z\s+)?(?:pkt\.|punkt)\s*([a-z])', 'pkt')
        ]
        
        # Sprawdzanie referencji
        for ref in references:
            for pattern, section_type in ref_patterns:
                match = re.search(pattern, ref, re.IGNORECASE)
                if match and match.groups():
                    try:
                        section_num = match.group(1)
                        if section_num not in existing_sections[section_type]:
                            type_map = {
                                'par': '§',
                                'art': 'Art.',
                                'rozdz': 'Rozdziału',
                                'ust': 'ustępu',
                                'pkt': 'punktu'
                            }
                            broken_refs.append(f"Brak {type_map[section_type]} {section_num}")
                    except IndexError:
                        # Log or handle the error if needed
                        print(f"Nieprawidłowe dopasowanie dla referencji: {ref}")
        
        return list(set(broken_refs))  # Usuń powtórzenia

    def _map_reference_targets(self, references: List[str]) -> Dict[str, List[Dict]]:
        """
        Mapuje referencje na ich cele, tworząc graf powiązań.
        """
        reference_map = defaultdict(list)
        current_section = None
        
        # Najpierw identyfikujemy wszystkie sekcje i ich referencje
        for line_num, line in enumerate(self.lines):
            # Sprawdzamy czy to początek nowej sekcji
            for section_type, pattern in self.SECTION_PATTERNS.items():
                match = re.match(pattern, line)
                if match:
                    # Dodaj sprawdzenie, czy match ma grupy
                    try:
                        if match.groups():
                            current_section = f"{section_type}_{match.group(1)}"
                            break
                    except IndexError:
                        # Jeśli nie ma grupy, pomiń tę sekcję
                        print(f"Nie można dopasować sekcji w linii: {line}")
                        continue

            # Sprawdzanie referencji
            for ref in references:
                # Usuń białe znaki i nowe linie
                ref = ref.strip()
                
                # Pomiń puste referencje
                if not ref:
                    continue
                
                # Sprawdź, czy ref pasuje do wzorca referencji
                for ref_type, ref_pattern in self.REFERENCE_PATTERNS.items():
                    if re.search(ref_pattern, ref):
                        if current_section:
                            reference_map[current_section].append({
                                'reference': ref,
                                'line_number': line_num + 1,
                                'context': line.strip()
                            })
                        break
        
        return dict(reference_map)

    def _find_circular_references(self) -> List[tuple]:
        def build_reference_graph():
            graph = defaultdict(set)
            current_section = None
            
            for line in self.lines:
                # Sprawdzamy czy to nowa sekcja
                for section_type, pattern in self.SECTION_PATTERNS.items():
                    match = re.match(pattern, line)
                    if match:
                        # Sprawdź, czy match ma grupy
                        try:
                            current_section = f"{section_type}_{match.group(1)}"
                            break
                        except IndexError:
                            # Jeśli nie ma grupy, pomiń tę sekcję
                            continue
                
                if current_section:
                    # Szukamy referencji w linii
                    for ref_pattern in self.REFERENCE_PATTERNS.values():
                        for match in re.finditer(ref_pattern, line):
                            ref_text = match.group(0)
                            # Próbujemy zidentyfikować sekcję, do której się odnosi
                            ref_match = re.search(r'(?:§|art\.|rozdz\.)\s*(\d+)', ref_text)
                            if ref_match:
                                try:
                                    target_section = f"section_{ref_match.group(1)}"
                                    graph[current_section].add(target_section)
                                except IndexError:
                                    # Pomiń, jeśli nie można dopasować grupy
                                    continue
            
            return graph

        def find_cycles(graph):
            def dfs(node, visited, path):
                if node in path:
                    cycle_start = path.index(node)
                    return path[cycle_start:]
                
                if node in visited:
                    return None
                    
                visited.add(node)
                path.append(node)
                
                # Konwertujemy na listę by uniknąć modyfikacji podczas iteracji
                neighbors = list(graph[node])
                for neighbor in neighbors:
                    cycle = dfs(neighbor, visited, path.copy())  # Używamy kopii ścieżki
                    if cycle:
                        return cycle
                
                path.pop()
                return None
            
            cycles = []
            visited = set()
            
            # Konwertujemy na listę by uniknąć modyfikacji podczas iteracji
            nodes = list(graph.keys())
            for node in nodes:
                if node not in visited:
                    cycle = dfs(node, visited, [])
                    if cycle:
                        cycles.append(tuple(cycle))
            
            return cycles
        
        # Obsługa możliwego braku wzorców
        if not hasattr(self, 'SECTION_PATTERNS') or not hasattr(self, 'REFERENCE_PATTERNS'):
            return []
        
        try:
            # Budujemy graf referencji i szukamy cykli
            reference_graph = build_reference_graph()
            return find_cycles(reference_graph)
        except Exception:
            # W razie jakiegokolwiek błędu zwracamy pustą listę
            return []
    

    def _calculate_reference_validity(self, references: ReferenceMetrics) -> float:
        """
        Oblicza ogólną poprawność referencji (0-1).
        """
        if references.total_references == 0:
            return 0.0
            
        weights = {
            'broken_refs': 0.4,    # Waga dla zepsutych referencji
            'circular_refs': 0.3,  # Waga dla referencji cyklicznych
            'coverage': 0.3        # Waga dla pokrycia referencjami
        }
        
        # Punkty za poprawne referencje
        broken_score = 1.0 - (len(references.broken_references) / references.total_references)
        
        # Punkty za brak cykli
        circular_score = 1.0 - (len(references.circular_references) / (references.total_references + 1))
        
        # Punkty za pokrycie (zakładamy, że każda sekcja powinna mieć min. 1 referencję)
        expected_refs = len([s for s in self.lines if any(re.match(p, s) for p in self.SECTION_PATTERNS.values())])
        coverage_score = min(1.0, references.total_references / (expected_refs + 1))
        
        return (
            weights['broken_refs'] * broken_score +
            weights['circular_refs'] * circular_score +
            weights['coverage'] * coverage_score
        )