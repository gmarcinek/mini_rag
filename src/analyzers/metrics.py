from dataclasses import dataclass
from typing import Dict, List, Optional
from pathlib import Path

@dataclass
class TextQualityMetrics:
    """Podstawowe metryki jakości tekstu"""
    total_chars: int
    total_lines: int
    avg_chars_per_line: float
    empty_lines_ratio: float
    special_chars_ratio: float
    avg_word_length: float
    unique_words_ratio: float

@dataclass
class StructureMetrics:
    """Metryki struktury dokumentu"""
    total_sections: int
    section_types: Dict[str, int]  # typ sekcji -> liczba wystąpień
    max_depth: int
    section_hierarchy: Dict[str, List[str]]  # sekcja -> lista podsekcji
    missing_required_sections: List[str]
    incomplete_sections: List[str]

@dataclass
class ReferenceMetrics:
    """Metryki referencji w dokumencie"""
    total_references: int
    internal_references: int
    external_references: int
    broken_references: List[str]
    reference_targets: Dict[str, List[str]]  # referencja -> lista celów
    circular_references: List[tuple]

@dataclass
class DocumentMetrics:
    """Pełne metryki dokumentu"""
    file_path: Path
    text_quality: TextQualityMetrics
    structure: StructureMetrics
    references: ReferenceMetrics
    
    # Dodatkowe metryki całościowe
    overall_quality_score: float  # 0-1
    structure_completeness: float  # 0-1
    reference_validity: float  # 0-1
    noise_level: float  # 0-1 (gdzie 0 to brak szumu)

    @property
    def has_critical_issues(self) -> bool:
        """Sprawdza czy dokument ma krytyczne problemy"""
        return (
            self.overall_quality_score < 0.5 or
            self.structure_completeness < 0.7 or
            self.reference_validity < 0.8 or
            self.noise_level > 0.3
        )

    def get_summary(self) -> Dict[str, any]:
        """Zwraca podsumowanie metryk w formie słownika"""
        return {
            'file_path': str(self.file_path),
            'quality_score': self.overall_quality_score,
            'structure_completeness': self.structure_completeness,
            'reference_validity': self.reference_validity,
            'noise_level': self.noise_level,
            'has_critical_issues': self.has_critical_issues,
            'total_sections': self.structure.total_sections,
            'missing_sections': len(self.structure.missing_required_sections),
            'broken_references': len(self.references.broken_references)
        }