import json
import os
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from config.settings import EMBEDDING_MODEL_NAME, MOVIES_DATA_PATH
from embedding.embedder import MovieEmbedder
from database.vector_store import VectorStoreManager


def ingest_movies_dataset(
    json_path: str = MOVIES_DATA_PATH,
    force_reset: bool = False
):
    """
    Ingest movies into ChromaDB.

    Behavior:
    - force_reset=True:
        Clears the entire vector store and rebuilds it.
    - force_reset=False:
        Adds new movies and updates existing movies.
        Existing unchanged movies are skipped.
    """

    if not os.path.exists(json_path):
        raise FileNotFoundError(
            f"Movie dataset not found at {json_path}"
        )

    print(f"[Ingestion] Reading movie data from '{json_path}'...")

    with open(json_path, "r", encoding="utf-8") as f:
        movies = json.load(f)

    print(f"[Ingestion] Loaded {len(movies)} movies.")

    store = VectorStoreManager()

    # ---------------------------------------------------------
    # RESET MODE
    # ---------------------------------------------------------

    if force_reset:
        print(
            "[Ingestion] Force reset requested. "
            "Clearing existing vector store..."
        )

        store.reset()

    # ---------------------------------------------------------
    # CHECK EXISTING MOVIES
    # ---------------------------------------------------------

    existing_ids = set()

    if not force_reset and store.count() > 0:
        print("[Ingestion] Checking existing movies in ChromaDB...")

        existing_data = store.collection.get(
            include=["documents", "metadatas"]
        )

        existing_ids = set(existing_data.get("ids", []))

        print(
            f"[Ingestion] Found {len(existing_ids)} "
            "existing movies in ChromaDB."
        )

    # ---------------------------------------------------------
    # PREPARE MOVIES
    # ---------------------------------------------------------

    embedder = MovieEmbedder()

    ids = []
    documents = []
    metadatas = []
    searchable_texts = []

    skipped_count = 0

    for movie in movies:

        movie_id = str(movie["id"])

        searchable_text = embedder.format_movie_searchable_text(
            movie
        )

        # -----------------------------------------------------
        # EXISTING MOVIE
        # -----------------------------------------------------

        if not force_reset and movie_id in existing_ids:

            skipped_count += 1
            continue

        # -----------------------------------------------------
        # NEW MOVIE
        # -----------------------------------------------------

        ids.append(movie_id)

        searchable_texts.append(searchable_text)

        documents.append(searchable_text)

        metadatas.append(
            {
                "title": movie.get("title", ""),
                "description": movie.get("description", ""),
                "genre": movie.get("genre", []),
                "year": int(movie.get("year", 0)),
                "rating": float(movie.get("rating", 0.0)),
                "director": movie.get("director", ""),
            }
        )

    print(
        f"[Ingestion] Existing unchanged movies skipped: "
        f"{skipped_count}"
    )

    print(
        f"[Ingestion] Movies requiring ingestion: "
        f"{len(searchable_texts)}"
    )

    # ---------------------------------------------------------
    # NOTHING NEW
    # ---------------------------------------------------------

    if not searchable_texts:

        total_count = store.count()

        print(
            "[Ingestion] No new movies to ingest."
        )

        print(
            f"[Ingestion] Total vectors in store: "
            f"{total_count}"
        )

        return total_count

    # ---------------------------------------------------------
    # GENERATE EMBEDDINGS
    # ---------------------------------------------------------

    print(
        f"[Ingestion] Generating embeddings for "
        f"{len(searchable_texts)} movies using "
        f"{EMBEDDING_MODEL_NAME}..."
    )

    embeddings = embedder.generate_embeddings_batch(
        searchable_texts
    )

    # ---------------------------------------------------------
    # STORE IN CHROMADB
    # ---------------------------------------------------------

    print(
        "[Ingestion] Upserting records into "
        "ChromaDB vector database..."
    )

    store.add_movies(
        ids=ids,
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas,
    )

    total_count = store.count()

    print(
        f"[Ingestion] Success! "
        f"Total vectors in store: {total_count}"
    )

    return total_count


if __name__ == "__main__":
    ingest_movies_dataset(force_reset=True)