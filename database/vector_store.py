import os
import chromadb
from chromadb.config import Settings as ChromaSettings
from typing import List, Dict, Any, Optional
from config.settings import CHROMA_PERSIST_DIRECTORY, COLLECTION_NAME, COSINE_SIMILARITY_THRESHOLD

class VectorStoreManager:
    """
    Manages ChromaDB vector database storage, collection creation,
    embedding indexing, and similarity queries with metadata filtering.
    """

    def __init__(self, persist_dir: str = CHROMA_PERSIST_DIRECTORY, collection_name: str = COLLECTION_NAME):
        self.persist_dir = persist_dir
        self.collection_name = collection_name
        
        os.makedirs(self.persist_dir, exist_ok=True)
        self.client = chromadb.PersistentClient(path=self.persist_dir)
        self._collection = None

    @property
    def collection(self):
        """Lazy access or creation of Chroma collection configured with cosine space."""
        if self._collection is None:
            self._collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
        return self._collection

    def count(self) -> int:
        """Returns total number of vectors in collection."""
        return self.collection.count()

    def add_movies(
        self,
        ids: List[str],
        embeddings: List[List[float]],
        documents: List[str],
        metadatas: List[Dict[str, Any]]
    ):
        """
        Ingests vectors, documents, and metadatas into the ChromaDB collection.
        Chroma requires metadatas values to be primitive types (int, float, str, bool).
        """
        sanitized_metadatas = []
        for meta in metadatas:
            clean_meta = {}
            for k, v in meta.items():
                if isinstance(v, list):
                    clean_meta[k] = ", ".join(map(str, v))
                elif isinstance(v, (str, int, float, bool)):
                    clean_meta[k] = v
                else:
                    clean_meta[k] = str(v)
            sanitized_metadatas.append(clean_meta)

        self.collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=sanitized_metadatas
        )

    def search_similar(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        genre_filter: Optional[str] = None,
        min_rating: Optional[float] = None,
        min_year: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Executes a vector similarity search with optional metadata filters.
        Returns top_k matches with distance/similarity scores and metadata.
        """
        if self.count() == 0:
            return []

        where_clauses = []

        if min_rating is not None and min_rating > 0:
            where_clauses.append({"rating": {"$gte": float(min_rating)}})

        if min_year is not None and min_year > 1900:
            where_clauses.append({"year": {"$gte": int(min_year)}})

        where_filter = None
        if len(where_clauses) == 1:
            where_filter = where_clauses[0]
        elif len(where_clauses) > 1:
            where_filter = {"$and": where_clauses}

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k * 3 if genre_filter else top_k, self.count()),
            where=where_filter,
            include=["documents", "metadatas", "distances"]
        )

        formatted_results = []
        if not results or not results["ids"] or not results["ids"][0]:
            return formatted_results

        ids = results["ids"][0]
        distances = results["distances"][0]
        metadatas = results["metadatas"][0]
        documents = results["documents"][0]

        for m_id, dist, meta, doc in zip(ids, distances, metadatas, documents):
            # Chroma returns cosine distance in [0, 2].
            # Cosine Similarity = 1 - Cosine Distance.
            similarity_score = max(0.0, 1.0 - dist)

            genres_list = [g.strip() for g in meta.get("genre", "").split(",") if g.strip()]

            # Apply Python post-filter for genre if genre_filter is provided (since genre is comma-separated string)
            if genre_filter and genre_filter.lower() != "all":
                if not any(genre_filter.lower() in g.lower() for g in genres_list):
                    continue

            formatted_results.append({
                "id": m_id,
                "title": meta.get("title", "Unknown"),
                "description": meta.get("description", doc),
                "genre": genres_list,
                "year": meta.get("year"),
                "rating": meta.get("rating"),
                "director": meta.get("director", ""),
                "similarity_score": round(similarity_score, 4),
                "distance": round(dist, 4),
                "match_percentage": round(similarity_score * 100, 1)
            })

            if len(formatted_results) >= top_k:
                break

        return formatted_results

    def reset(self):
        """Clears the collection safely."""
        try:
            self.client.delete_collection(self.collection_name)
        except Exception:
            pass
        self._collection = None

