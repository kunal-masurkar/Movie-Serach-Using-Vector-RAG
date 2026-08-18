import json
from typing import List, Dict, Any, Optional
from pathlib import Path
import sys

# Ensure root is in path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from config.settings import DEFAULT_TOP_K, MOVIES_DATA_PATH
from embedding.embedder import MovieEmbedder
from database.vector_store import VectorStoreManager

class MovieSearchEngine:
    """
    High-level search service combining embedder, vector database,
    metadata filtering, and comparative search algorithms.
    """

    def __init__(self):
        self.embedder = MovieEmbedder()
        self.vector_store = VectorStoreManager()

    def search(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
        genre_filter: Optional[str] = None,
        min_rating: Optional[float] = None,
        min_year: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Executes semantic vector similarity search.
        """
        if not query or not query.strip():
            return []

        # Step 1: Query embedding generation
        query_vector = self.embedder.generate_embedding(query)

        # Step 2: Vector DB Similarity Search with Metadata Filters
        results = self.vector_store.search_similar(
            query_embedding=query_vector,
            top_k=top_k,
            genre_filter=genre_filter,
            min_rating=min_rating,
            min_year=min_year
        )

        return results

    def compare_vector_vs_keyword(self, query: str, top_k: int = DEFAULT_TOP_K) -> Dict[str, Any]:
        """
        Runs both Vector Semantic Search and traditional Exact Keyword Search,
        highlighting the power of semantic embeddings over simple string matching.
        """
        # Vector Semantic Search
        vector_results = self.search(query, top_k=top_k)

        # Traditional Keyword Search
        query_words = set(query.lower().split())
        keyword_results = []

        try:
            with open(MOVIES_DATA_PATH, "r", encoding="utf-8") as f:
                all_movies = json.load(f)

            scored_movies = []
            for movie in all_movies:
                text = f"{movie.get('title', '')} {movie.get('description', '')} {' '.join(movie.get('genre', []))}".lower()
                # Count matching query words
                matches = sum(1 for word in query_words if len(word) > 2 and word in text)
                if matches > 0:
                    scored_movies.append((matches, movie))

            scored_movies.sort(key=lambda x: x[0], reverse=True)
            for matches, movie in scored_movies[:top_k]:
                keyword_results.append({
                    "id": movie["id"],
                    "title": movie["title"],
                    "description": movie["description"],
                    "genre": movie["genre"],
                    "year": movie["year"],
                    "rating": movie["rating"],
                    "keyword_matches": matches
                })
        except Exception as e:
            print(f"[SearchEngine] Error running keyword comparison: {e}")

        return {
            "query": query,
            "vector_search_results": vector_results,
            "keyword_search_results": keyword_results,
            "explanation": (
                "Vector search converts text into 384D semantic embeddings to match meanings "
                "(e.g. 'astronaut stuck on planet' matches 'The Martian' even without exact words), "
                "whereas keyword search depends strictly on literal word overlaps."
            )
        }
