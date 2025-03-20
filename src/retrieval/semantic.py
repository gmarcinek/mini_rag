from typing import List, Tuple, Optional, Set
import numpy as np
from src.chunking import Chunk
from src.embeddings import PolishLegalEmbedder
from src.documents.similarity import DocumentSimilarity

class SemanticRetriever:
    def __init__(self,
                embedder: PolishLegalEmbedder,
                min_score_threshold: float = 0.80,
                max_top_k: int = 10):
        self.embedder = embedder
        self.min_score_threshold = min_score_threshold
        self.max_top_k = max_top_k
        self.doc_similarity = DocumentSimilarity()

        self.broad_query_keywords = {
            'rozdział', 'rozdziały', 'dział', 'działy', 'sekcja', 'sekcje',
            'art', 'artykuł', 'artykuły', 'paragraf', 'paragrafy', 'ustęp', 'ustępy',
            'punkt', 'punkty', 'załącznik', 'załączniki',
            'rozdz', 'art', 'par', 'ust', 'pkt', 'zał',
            'owu', 'ogólne warunki', 'ogólne warunki umowy', 'ogólne warunki ubezpieczenia',
            'wyłączenia', 'warunki', 'wszystkie', 'lista', 'wymień',
            'obowiązki', 'prawa', 'odpowiedzialność', 'zakres', 'termin',
            'postanowienia', 'postanowienia ogólne', 'postanowienia szczegółowe', 'postanowienia końcowe',
        }

        self.specific_query_keywords = {
            'definicja', 'definicje', 'def',
            'co to jest', 'co oznacza', 'jak zdefiniowano',
            'znaczenie', 'interpretacja', 'wykładnia',
            'wyjaśnienie', 'objaśnienie', 'określenie'
        }

    def retrieve(self, 
                query: str, 
                documents: List[Chunk],
                embeddings: List[np.ndarray],
                top_k: Optional[int] = None,
                min_score: Optional[float] = None) -> List[Tuple[Chunk, float]]:
        print(f"\nWyszukiwanie dla zapytania: {query}")
        
        complexity, is_broad_query = self._calculate_query_complexity(query)
        
        if is_broad_query:
            base_min_score = 0.45
            effective_top_k = 10
            print("Wykryto zapytanie wymagające szerokiego kontekstu")
        else:
            base_min_score = min_score if min_score is not None else self.min_score_threshold
            effective_top_k = top_k if top_k is not None else self.max_top_k
        
        adjusted_min_score = self._adjust_min_score(query, base_min_score, is_broad_query)
        print(f"Dostosowany próg podobieństwa: {adjusted_min_score:.3f}")
        
        query_embedding = self.embedder.get_embedding(query)
        similarities = []
        
        for i, doc_embedding in enumerate(embeddings):
            similarity = np.dot(query_embedding.flatten(), doc_embedding.flatten()) / \
                        (np.linalg.norm(query_embedding) * np.linalg.norm(doc_embedding))
            if similarity >= adjusted_min_score:
                similarities.append((i, similarity))
        
        sorted_similarities = sorted(similarities, key=lambda x: x[1], reverse=True)
        
        if is_broad_query:
            results = [(documents[i], score) 
                        for i, score in sorted_similarities[:effective_top_k]]
        else:
            optimal_k = self._get_optimal_top_k(sorted_similarities)
            results = [(documents[i], score) 
                        for i, score in sorted_similarities[:optimal_k]]
        
        print(f"Znalezione fragmenty: {len(results)}")
        print("Scores:", [f"{score:.3f}" for _, score in results])
        
        results = self._check_legal_relations(results)
        results = self.doc_similarity.group_similar_chunks(results)
        return results

    def _calculate_query_complexity(self, query: str) -> Tuple[float, bool]:
        words = query.lower().split()
        
        is_broad_query = any(word in self.broad_query_keywords for word in words)
        
        length_score = min(len(words) / 10, 1.0)
        keywords = {'definicja', 'warunki', 'paragraf', 'artykuł', 'ustęp'}
        keyword_count = sum(1 for word in words if word in keywords)
        keyword_score = min(keyword_count / 3, 1.0)
        
        complexity = 0.7 * length_score + 0.3 * keyword_score
        return complexity, is_broad_query

    def _adjust_min_score(self, query: str, base_score: float, is_broad_query: bool) -> float:
        if is_broad_query:
            return max(base_score - 0.2, 0.4)
        
        complexity = self._calculate_query_complexity(query)[0]
        adjustment = 0.1 * (1 - complexity)
        return min(max(base_score + adjustment, 0.5), 0.8)

    def _get_optimal_top_k(self, similarities: List[Tuple[int, float]], min_results: int = 3) -> int:
        if not similarities:
            return min_results
            
        scores = [score for _, score in similarities]
        
        if len(scores) < min_results:
            return len(scores)
        
        score_diffs = [scores[i] - scores[i+1] for i in range(len(scores)-1)]
        if not score_diffs:
            return min_results
            
        significant_drops = [(i, diff) for i, diff in enumerate(score_diffs) if diff > 0.1]
        if significant_drops:
            first_significant_drop = min(significant_drops, key=lambda x: x[1])[0]
            return max(min_results, first_significant_drop + 1)
        
        return min(min_results, len(scores))

    def _check_legal_relations(self, chunks: List[Tuple[Chunk, float]]) -> List[Tuple[Chunk, float]]:
        if not chunks:
            return chunks
            
        results = []
        seen_signatures = set()
        
        for chunk, score in chunks:
            if hasattr(chunk, 'current_signature') and chunk.current_signature:
                if chunk.current_signature not in seen_signatures:
                    seen_signatures.add(chunk.current_signature)
                    results.append((chunk, score))
        
        return results or chunks