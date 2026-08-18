import json
import os
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from config.settings import MOVIES_DATA_PATH
from embedding.embedder import MovieEmbedder
from database.vector_store import VectorStoreManager

def ingest_movies_dataset(json_path: str = MOVIES_DATA_PATH, force_reset: bool = False):
    """
    Reads movies JSON file, generates embeddings for searchable texts,
    and stores vectors & metadata into ChromaDB vector database.
    """
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Movie dataset not found at {json_path}")

    print(f"[Ingestion] Reading movie data from '{json_path}'...")
    with open(json_path, "r", encoding="utf-8") as f:
        movies = json.load(f)

    print(f"[Ingestion] Loaded {len(movies)} movies.")

    embedder = MovieEmbedder()
    store = VectorStoreManager()

    if force_reset:
        print("[Ingestion] Force reset requested. Clearing existing vector store...")
        store.reset()

    ids = []
    documents = []
    metadatas = []
    searchable_texts = []

    for movie in movies:
        movie_id = movie["id"]
        searchable_text = embedder.format_movie_searchable_text(movie)
        
        ids.append(movie_id)
        searchable_texts.append(searchable_text)
        documents.append(searchable_text)
        metadatas.append({
            "title": movie.get("title", ""),
            "description": movie.get("description", ""),
            "genre": movie.get("genre", []),
            "year": int(movie.get("year", 0)),
            "rating": float(movie.get("rating", 0.0)),
            "director": movie.get("director", "")
        })

    print(f"[Ingestion] Generating embeddings for {len(searchable_texts)} movies using all-MiniLM-L6-v2...")
    embeddings = embedder.generate_embeddings_batch(searchable_texts)

    print("[Ingestion] Upserting records into ChromaDB vector database...")
    store.add_movies(
        ids=ids,
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas
    )

    total_count = store.count()
    print(f"[Ingestion] Success! Ingestion complete. Total vectors in store: {total_count}")
    return total_count

if __name__ == "__main__":
    ingest_movies_dataset(force_reset=True)
