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
                   print_hierarchy_tree(structure_metrics.section_hierarchy)
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


def print_hierarchy_tree(hierarchy, root=None, indent=0, is_last=False, prefix="", show_counts=True, max_depth=None, current_depth=0):
    """
    Wyświetla hierarchię w estetycznej formie drzewa w terminalu z obsługą wielu typów sekcji.
    
    Args:
        hierarchy (dict): Słownik hierarchii sekcji
        root (str): Bieżący element do wyświetlenia (None dla początku)
        indent (int): Bieżący poziom wcięcia
        is_last (bool): Czy bieżący element jest ostatnim na swoim poziomie
        prefix (str): Prefix linii do wyświetlenia (akumuluje się z głębokością)
        show_counts (bool): Czy pokazywać liczbę dzieci dla każdej sekcji
        max_depth (int, optional): Maksymalna głębokość do wyświetlenia (None = bez limitu)
        current_depth (int): Aktualna głębokość w hierarchii
    """
    if max_depth is not None and current_depth > max_depth:
        return
        
    if root is None:
        # Znajdź początkowe elementy (te, które nie są podrzędne żadnemu innemu)
        all_sections = set(hierarchy.keys())
        all_children = set()
        for children in hierarchy.values():
            all_children.update(children)
        
        # Elementy które nie są dziećmi innych elementów
        root_sections = all_sections - all_children
        
        # Jeśli nie znaleziono elementów początkowych, weź pierwszy dostępny
        if not root_sections and hierarchy:
            # Szukaj elementu 'dokument' lub pierwszego klucza
            if 'dokument' in hierarchy:
                root_sections = ['dokument']
            else:
                root_sections = [next(iter(hierarchy.keys()))]
        
        for i, section in enumerate(sorted(root_sections)):
            is_last_root = i == len(root_sections) - 1
            children = hierarchy.get(section, [])
            
            # Formatuj nazwę sekcji
            if section == 'dokument':
                formatted_name = "Dokument"
            else:
                formatted_name = format_section_name(section)
            
            if show_counts and children:
                print(f"{formatted_name} [{len(children)}]")
            else:
                print(f"{formatted_name}")
            
            for j, child in enumerate(sorted(children)):
                is_last_child = j == len(children) - 1
                print_hierarchy_tree(hierarchy, child, 1, is_last_child, "", show_counts, max_depth, current_depth + 1)
    else:
        # Generuj wcięcie
        if is_last:
            connection = "└── "
            new_prefix = prefix + "    "
        else:
            connection = "├── "
            new_prefix = prefix + "│   "
        
        # Formatuj nazwę sekcji dla wyświetlenia
        formatted_name = format_section_name(root)
        
        # Pobierz dzieci
        children = hierarchy.get(root, [])
        
        # Wyświetl bieżący element
        if show_counts and children:
            print(f"{prefix}{connection}{formatted_name} [{len(children)}]")
        else:
            print(f"{prefix}{connection}{formatted_name}")
        
        # Rekurencyjnie wyświetl dzieci
        for i, child in enumerate(sorted(children)):
            is_last_child = i == len(children) - 1
            print_hierarchy_tree(hierarchy, child, indent + 1, is_last_child, new_prefix, show_counts, max_depth, current_depth + 1)

def format_section_name(section_id):
    """Formatuje identyfikator sekcji na czytelną nazwę dla wyświetlenia"""
    # Obsługa punktów
    if section_id.startswith('punkt_'):
        punkt_num = section_id.replace('punkt_', '')
        return f"Punkt {punkt_num}"
    
    # Obsługa standardowych sekcji
    parts = section_id.split('_')
    if len(parts) >= 2:
        section_type = parts[0]
        section_num = parts[1]
        
        # Sprawdź czy jest to powtórzenie z adnotacją
        if ' (' in section_id:
            # Wyświetl informację o duplikacie
            base_part = section_id.split(' (')[0]
            instance_part = section_id.split(' (')[1]
            
            type_name = get_section_type_name(section_type)
            return f"{type_name} {section_num} ({instance_part}"
        else:
            # Standardowa sekcja bez powtórzeń
            type_name = get_section_type_name(section_type)
            return f"{type_name} {section_num}"
    
    # Zwróć oryginalną nazwę jeśli nie pasuje do żadnego wzorca
    return section_id

def get_section_type_name(section_type):
    """Zwraca pełną nazwę typu sekcji"""
    type_names = {
        'rozdzial': "Rozdział",
        'art': "Artykuł",
        'paragraf': "Paragraf",
        'ustep': "Ustęp",
        'zalacznik': "Załącznik",
        'sekcja': "Sekcja",
        'owu': "OWU",
        'definicje': "Definicje",
        'postanowienia': "Postanowienia"
    }
    return type_names.get(section_type, section_type.capitalize())

def display_document_structure(structure_metrics, show_counts=True, max_depth=None, include_stats=True):
    """
    Wyświetla strukturę dokumentu na podstawie wyników analizy.
    
    Args:
        structure_metrics: Wynik metody analyze_structure
        show_counts (bool): Czy pokazywać liczbę elementów podrzędnych
        max_depth (int, optional): Maksymalna głębokość hierarchii do wyświetlenia
        include_stats (bool): Czy wyświetlać statystyki dokumentu
    """
    print("\nStruktura dokumentu:")
    print_hierarchy_tree(
        structure_metrics.section_hierarchy, 
        show_counts=show_counts, 
        max_depth=max_depth
    )
    
    if include_stats:
        # Wyświetl dodatkowe statystyki
        print("\nStatystyki dokumentu:")
        print(f"- Całkowita liczba sekcji: {structure_metrics.total_sections}")
        
        # Formatuj typy sekcji w czytelny sposób
        if structure_metrics.section_types:
            print("- Typy sekcji:")
            for section_type, count in sorted(structure_metrics.section_types.items(), 
                                             key=lambda x: (x[1], x[0]), reverse=True):
                type_name = get_section_type_name(section_type)
                print(f"  * {type_name}: {count}")
        
        print(f"- Maksymalna głębokość: {structure_metrics.max_depth}")
        
        if hasattr(structure_metrics, 'incomplete_sections') and structure_metrics.incomplete_sections:
            print("\nNiekompletne sekcje:")
            for section_type, sections in structure_metrics.incomplete_sections.items():
                type_name = get_section_type_name(section_type)
                print(f"- {type_name}: {', '.join(sections)}")
                

def main():
    sys.exit(pytest.main(["scripts/test_document_quality.py", "--tb=long", "-s"]))
    