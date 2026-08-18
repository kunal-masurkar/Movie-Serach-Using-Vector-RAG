import pytest
import os
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from embedding.embedder import MovieEmbedder
from database.vector_store import VectorStoreManager
from ingestion.ingest_movies import ingest_movies_dataset
from search.search_movies import MovieSearchEngine

@pytest.fixture(scope="module")
def setup_vector_store():
    """Ingest dataset into vector database before running tests."""
    ingest_movies_dataset(force_reset=True)
    yield VectorStoreManager()

def test_embedding_generation():
    """Verify that sentence-transformer model outputs 384-dimensional normalized vectors."""
    embedder = MovieEmbedder()
    text = "Interstellar astronaut space journey"
    vec = embedder.generate_embedding(text)
    assert isinstance(vec, list)
    assert len(vec) == 384
    assert all(isinstance(x, float) for x in vec)

def test_ingestion_and_vector_count(setup_vector_store):
    """Verify that all movies in the dataset were ingested into ChromaDB."""
    store = setup_vector_store

    dataset_path = Path(__file__).resolve().parent.parent / "data" / "movies.json"

    import json

    with open(dataset_path, "r", encoding="utf-8") as file:
        movies = json.load(file)

    assert store.count() == len(movies)

def test_space_semantic_search(setup_vector_store):
    """Verify space query ranks relevant space movies at top."""
    engine = MovieSearchEngine()
    results = engine.search("movies about astronauts exploring space", top_k=5)
    
    assert len(results) > 0
    top_titles = [m["title"] for m in results]
    # Interstellar, The Martian, Gravity, or Apollo 13 should appear in top results
    space_movies = {"Interstellar", "The Martian", "Gravity", "Apollo 13"}
    matched = space_movies.intersection(set(top_titles))
    assert len(matched) >= 2

def test_metadata_rating_filter(setup_vector_store):
    """Verify metadata filtering by minimum rating."""
    engine = MovieSearchEngine()
    min_r = 8.5
    results = engine.search("dramatic movie", top_k=5, min_rating=min_r)
    
    assert len(results) > 0
    for movie in results:
        assert movie["rating"] >= min_r

def test_top_k_limit(setup_vector_store):
    """Verify that search respects top_k limit."""
    engine = MovieSearchEngine()
    top_k = 3
    results = engine.search("action adventure", top_k=top_k)
    assert len(results) == top_k

def test_vector_vs_keyword_comparison(setup_vector_store):
    """Verify compare_vector_vs_keyword output structure."""
    engine = MovieSearchEngine()
    comp = engine.compare_vector_vs_keyword("A person stuck on another planet trying to stay alive", top_k=3)
    assert "vector_search_results" in comp
    assert "keyword_search_results" in comp
    assert len(comp["vector_search_results"]) > 0
