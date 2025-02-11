import re

# Wzorce dla sekcji dokumentów
# SECTION_PATTERNS = {
#     'main_sections': r'^(?:Rozdział|Art\.?|Artykuł)\s*(?:[IVXLCDM]+|\d+)\.?',
# }

# Wzorce dla sekcji dokumentów prawniczych i OWU
SECTION_PATTERNS = {
    'rozdzial': r'^(?:Rozdział|Rozdz\.|Rozdz|R\.|R|Dział|Dz\.|Dz)\s*(?:[IVXLCDM]+|\d+)\.?',
    'art': r'^(?:Art\.|Art\s*\.|Artykuł|Art|Artyku[łl]|A\.|A)\s*(?:[IVXLCDM]+|\d+)\.?',
    'paragraf': r'^(?:§|Par\.|Par\s*\.|Paragraf|P\.|P)\s*(?:[IVXLCDM]+|\d+)\.?',
    'ustep': r'^(?:Ust\.|Ust\s*\.|Ustęp|U\.|U|Punkt|Pkt\.|Pkt|Pkt\s*\.)\s*(?:[IVXLCDM]+|\d+)\.?',
    'zalacznik': r'^(?:Zał\.|Zał\s*\.|Załącznik|Za[łl]|Z\.|Z)\s*(?:[IVXLCDM]+|\d+)\.?',
    'sekcja': r'^(?:Sekcja|Sek\.|Sek|S\.|S)\s*(?:[IVXLCDM]+|\d+)\.?',
    'postanowienia': r'^(?:Postanowienia\s*(?:Ogólne|Szczegółowe|Końcowe)|Post\.|Post)\s*(?:[IVXLCDM]+|\d+)\.?'
}

# Dodatkowe opcje specjalne dla dokumentów prawniczych
SPECIAL_SECTION_PATTERNS = {
    'rozdzial_cyfry': r'^Rozdział\s*[0-9]{1,3}\.?',
    'art_cyfry': r'^Art\.\s*[0-9]{1,3}\.?',
    'paragraf_cyfry': r'^§\s*[0-9]{1,3}\.?',
    'ustep_cyfry': r'^Ust\.\s*[0-9]{1,3}\.?',
}

# Dodatkowe flagi i opcje
SECTION_MATCHING_OPTIONS = {
    'case_sensitive': False,  # Ignorowanie wielkości liter
    'ignore_leading_whitespace': True,  # Ignorowanie białych znaków na początku
}

# Konfiguracja chunkowania
CHUNKING_CONFIG = {
    'context_size': 7000,
    'overlap_size': 400
}

# Konfiguracja katalogów
DIRECTORIES = {
    'uploads_dir': 'uploads',
    'chunks_dir': 'chunks'
}

# Konfiguracja tokenizera
TOKENIZER_CONFIG = {
    'encoding_name': 'cl100k_base'
}

# Konfiguracja transformacji tekstu
TEXT_TRANSFORMERS = [
    {'pattern': r'\s+', 'replacement': ' '},  # Usuwanie nadmiarowych spacji
    {'pattern': r' ,', 'replacement': ','},   # Usuwanie spacji przed przecinkami
]