from src.rag.pipeline import MiniRAG
from src.rag.LegalRAGPipeline import LegalRAGPipeline
from src.chunking.text_splitter import SimpleTextSplitter
from src.chunking.legal_text_splitter import LegalTextSplitter, OllamaChunkTransformer
from src.chunking.hierarchical_chunker import HierarchicalLegalChunker
from pathlib import Path
from src.analyzers.legal_text_structure_analyzer import LegalTextStructureAnalyzer

def load_document(file_path: str) -> str:
   path = Path(file_path)
   if not path.exists():
       raise FileNotFoundError(f"File not found: {file_path}")
   return path.read_text(encoding='utf-8')

def main():
    try:
        document = load_document("data/documents/pdf_content.txt")
        print("Dokument wczytany pomyślnie.")
    except FileNotFoundError as e:
        print(f"Błąd: {e}")
        return
    except Exception as e:
        print(f"Nieoczekiwany błąd: {e}")
        return

    # Tworzymy transformer z Mistralem
    transformer = OllamaChunkTransformer(model_name="llama3.2")

    # Przekazujemy transformer do LegalTextSplitter
    # chunker = LegalTextSplitter(
    chunker = HierarchicalLegalChunker()

    rag = LegalRAGPipeline(
        use_gpu=True,
        chunker=chunker,
        debug_mode=True,
        max_top_k=5
    )
   
    rag.add_documents([document])

    print("\n--- Interaktywny RAG ---")
    print("Wpisz 'exit' lub 'koniec', aby zakończyć.")

    while True:
        query = input("\nTwoje pytanie: ").strip()
        
        if query.lower() in ['exit', 'koniec', 'wyjdź', 'quit']:
            print("Do zobaczenia!")
            break
        
        if not query:
            continue

        try:
            result = rag.smart_query(query)
            print("\n===================================")
            print("\nOdpowiedź:")
            print(result['answer'])
        except Exception as e:
            print(f"Błąd podczas przetwarzania zapytania: {e}")

if __name__ == "__main__":
    main()
