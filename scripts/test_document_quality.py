import os
import pytest
import traceback
from pathlib import Path
import sys
from src.analyzers.document_analyzer import DocumentAnalyzer
from src.analyzers.metrics import TextQualityMetrics, StructureMetrics, ReferenceMetrics
from colorama import init, Fore

init(autoreset=True)

class TestDocumentQuality:
   @pytest.mark.parametrize("document_path", [
       Path("data/documents"),
   ])
   def test_document_metrics(self, document_path):
       """Test metryk dla wszystkich dokumentów w podanym katalogu"""
       full_path = Path(__file__).parent.parent / document_path
       assert full_path.is_dir(), f"Katalog {full_path} nie istnieje"
       
       for file_path in full_path.glob('*'):
           if file_path.is_file() and not file_path.name.startswith('.'):
               try:
                   print(f"{Fore.YELLOW}\nAnaliza dokumentu: {file_path}{Fore.RESET}")
                   analyzer = DocumentAnalyzer(file_path)
                   
                   text_metrics = analyzer.analyze_text_quality()
                   print("Metryki jakości tekstu:")
                   print(f"Całkowita liczba znaków: {text_metrics.total_chars}")
                   print(f"Liczba linii: {text_metrics.total_lines}")
                   print(f"Średnia liczba znaków na linię: {text_metrics.avg_chars_per_line}")
                   print(f"Stosunek pustych linii: {text_metrics.empty_lines_ratio}")
                   print(f"Stosunek znaków specjalnych: {text_metrics.special_chars_ratio}")
                   print(f"Średnia długość słowa: {text_metrics.avg_word_length}")
                   print(f"Stosunek unikalnych słów: {text_metrics.unique_words_ratio}")

                   self.print_ascii_chart(text_metrics)

                   assert text_metrics.total_chars > 0
                   assert text_metrics.total_lines > 0
                   assert 0 <= text_metrics.empty_lines_ratio <= 0.39
                   assert 0 <= text_metrics.special_chars_ratio <= 0.15
                   assert 4 <= text_metrics.avg_word_length <= 15
                   assert 0.2 <= text_metrics.unique_words_ratio <= 1.0

                   reference_metrics = analyzer.analyze_references()
                   print("\nMetryki referencji:")
                   print(f"Całkowite referencje: {reference_metrics.total_references}")
                   print(f"Referencje wewnętrzne: {reference_metrics.internal_references}")
                   print(f"Referencje zewnętrzne: {reference_metrics.external_references}")
                   print(f"Zepsute referencje: {reference_metrics.broken_references}")
                   
                   assert reference_metrics.total_references >= 0

                   structure_metrics = analyzer.analyze_structure()
                   print(f"\ntotal_sections: {structure_metrics.total_sections}")
                   print(f"\nsection_types: {structure_metrics.section_types}")
                   print(f"\nmax_depth: {structure_metrics.max_depth}")
                   print(f"\nsection_hierarchy: {structure_metrics.section_hierarchy}")
                   print(f"\nmissing_required_sections: {structure_metrics.missing_required_sections}")
                   print(f"\nincomplete_sections: {structure_metrics.incomplete_sections}")

               except Exception as e:
                   print(f"Błąd podczas analizy pliku {file_path}: {e}")
                   traceback.print_exc()
                   assert False

   def print_ascii_chart(self, text_metrics):
       """Drukuje kolorowe ASCII wykresy w terminalu"""
       metrics = {
           'Śr. znaków/linię': (text_metrics.avg_chars_per_line, 115.46),
           'Puste linie %': (text_metrics.empty_lines_ratio, 0.19),
           'Znaki specjalne %': (text_metrics.special_chars_ratio, 0.05),
           'Śr. dł. słowa': (text_metrics.avg_word_length, 6),
           'Unikalne słowa %': (text_metrics.unique_words_ratio, 0.28)
       }
       
       max_bar_length = 30  # Maksymalna szerokość wykresu

       print("\n📊 Wykres ASCII:")
       for label, (value, norm) in metrics.items():
           value_scaled = min(int((value / norm) * max_bar_length), max_bar_length)
           bar_value = self.get_colored_bar(value_scaled, max_bar_length, value, norm)

           print(f"{label:20} | {bar_value} | {value:.2f} (norma: {norm})")

   def get_colored_bar(self, value_scaled, max_bar_length, value, norm):
       """Tworzy kolorowy pasek ASCII o ustalonej szerokości"""
       deviation = abs(value - norm) / norm  # Odchylenie procentowe

       if deviation < 0.1:
           color = Fore.GREEN  # W normie
       elif deviation < 0.3:
           color = Fore.YELLOW  # Lekkie odchylenie
       elif deviation < 0.5:
           color = Fore.MAGENTA  # Średnie
       elif deviation < 0.8:
           color = Fore.RED  # Duże odchylenie
       else:
           color = Fore.MAGENTA + Fore.LIGHTBLACK_EX  # Ekstremalne

       return color + "█" * value_scaled + " " * (max_bar_length - value_scaled) + Fore.RESET

def main():
    sys.exit(pytest.main(["scripts/test_document_quality.py", "--tb=long", "-s"]))
