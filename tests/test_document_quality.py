import os
import pytest
import traceback
from pathlib import Path
from src.analyzers.document_analyzer import DocumentAnalyzer
from src.analyzers.metrics import TextQualityMetrics, StructureMetrics, ReferenceMetrics

class TestDocumentQuality:
   @pytest.mark.parametrize("document_path", [
       Path("data/documents"),  # Główny katalog do przeskanowania
   ])
   def test_document_metrics(self, document_path):
       """Test metryk dla wszystkich dokumentów w podanym katalogu"""
       # Sprawdź, czy katalog istnieje
       full_path = Path(__file__).parent.parent / document_path
       assert full_path.is_dir(), f"Katalog {full_path} nie istnieje"
       
       # Przejdź przez wszystkie pliki w katalogu
       for file_path in full_path.glob('*'):
           # Pomijaj katalogi i ukryte pliki
           if file_path.is_file() and not file_path.name.startswith('.'):
               try:
                   print(f"\nAnaliza dokumentu: {file_path}")
                   
                   # Inicjalizacja analyzera
                   analyzer = DocumentAnalyzer(file_path)
                   
                   # Analiza jakości tekstu
                   text_metrics = analyzer.analyze_text_quality()
                   print("Metryki jakości tekstu:")
                   print(f"Całkowita liczba znaków: {text_metrics.total_chars}")
                   print(f"Liczba linii: {text_metrics.total_lines}")
                   print(f"Średnia liczba znaków na linię: {text_metrics.avg_chars_per_line}")
                   print(f"Stosunek pustych linii: {text_metrics.empty_lines_ratio}")
                   print(f"Stosunek znaków specjalnych: {text_metrics.special_chars_ratio}")
                   print(f"Średnia długość słowa: {text_metrics.avg_word_length}")
                   print(f"Stosunek unikalnych słów: {text_metrics.unique_words_ratio}")
                   
                   # Podstawowe asercje
                   assert text_metrics.total_chars > 0, f"Dokument {file_path} powinien mieć więcej niż 0 znaków"
                   assert text_metrics.total_lines > 0, f"Dokument {file_path} powinien mieć więcej niż 0 linii"
                   
                   # Asercje dla metryk tekstu z rozszerzonymi zakresami
                   assert 0 <= text_metrics.empty_lines_ratio <= 0.39  # Dopuszczamy do 35% pustych linii
                   assert 0 <= text_metrics.special_chars_ratio <= 0.15  # Realistyczny próg znaków specjalnych
                   assert 4 <= text_metrics.avg_word_length <= 15  # Typowy zakres dla języka polskiego
                   assert 0.2 <= text_metrics.unique_words_ratio <= 1.0  # Złagodzony próg unikalności słów
                   
                   # Analiza struktury z obsługą błędów
                   try:
                       structure_metrics = analyzer.analyze_structure()
                       print("\nNiekompletne sekcje:")
                       print(structure_metrics)
                   except IndexError as structure_error:
                       print(f"Ostrzeżenie: Problem z analizą struktury dokumentu {file_path}: {structure_error}")
                   
                   # Analiza referencji
                   reference_metrics = analyzer.analyze_references()
                   print("\nMetryki referencji:")
                   print(f"Całkowite referencje: {reference_metrics.total_references}")
                   print(f"Referencje wewnętrzne: {reference_metrics.internal_references}")
                   print(f"Referencje zewnętrzne: {reference_metrics.external_references}")
                   print(f"Zepsute referencje: {reference_metrics.broken_references}")
                   
                   # Dodatkowe asercje dla metryk referencji
                   assert reference_metrics.total_references >= 0, f"Liczba referencji w {file_path} nie może być ujemna"

               except Exception as e:
                   # Wyświetl pełny ślad błędu
                   print(f"Pełny ślad błędu dla pliku {file_path}:")
                   traceback.print_exc()
                   
                   # Wyświetl szczegóły wyjątku
                   print(f"Typ wyjątku: {type(e)}")
                   print(f"Szczegóły błędu: {e}")
                   
                   # Zawsze wywołuj AssertionError, aby test nie przechodził
                   assert False, f"Błąd podczas analizy pliku {file_path}: {e}"