import os
import sys
import json
from pathlib import Path
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, HTTPException, Query, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Ensure project root is in sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

from config.settings import EMBEDDING_MODEL_NAME, EMBEDDING_DIMENSION, DEFAULT_TOP_K, MOVIES_DATA_PATH
from search.search_movies import MovieSearchEngine
from database.vector_store import VectorStoreManager
from ingestion.ingest_movies import ingest_movies_dataset

app = FastAPI(
    title="Movie Semantic Search API",
    description="Vector Database semantic search API powered by ChromaDB & SentenceTransformers",
    version="1.0.0"
)

# Enable CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global Search Engine Instance
search_engine: Optional[MovieSearchEngine] = None

def get_search_engine() -> MovieSearchEngine:
    global search_engine
    if search_engine is None:
        search_engine = MovieSearchEngine()
    return search_engine

# Request / Response Schemas
class SearchRequest(BaseModel):
    query: str = Field(..., description="Natural language search query", example="movies about astronauts exploring space")
    top_k: int = Field(DEFAULT_TOP_K, ge=1, le=20, description="Number of results to return")
    genre: Optional[str] = Field(None, description="Filter by genre, e.g., 'Science Fiction'")
    min_rating: Optional[float] = Field(None, ge=0.0, le=10.0, description="Minimum movie rating (0-10)")
    min_year: Optional[int] = Field(None, ge=1900, le=2030, description="Minimum release year")

class SearchResponseItem(BaseModel):
    id: str
    title: str
    description: str
    genre: List[str]
    year: Optional[int]
    rating: Optional[float]
    director: Optional[str]
    similarity_score: float
    distance: float
    match_percentage: float

class CompareRequest(BaseModel):
    query: str = Field(..., description="Query to compare vector vs keyword search")
    top_k: int = Field(DEFAULT_TOP_K, ge=1, le=20)

@app.on_event("startup")
def startup_event():
    """Ensure vector DB has records on server startup."""
    store = VectorStoreManager()
    if store.count() == 0:
        print("[API Startup] Vector store is empty. Triggering automatic initial ingestion...")
        ingest_movies_dataset(force_reset=False)
    else:
        print(f"[API Startup] Vector store initialized with {store.count()} records.")

@app.post("/api/search", response_model=List[SearchResponseItem])
def api_search(req: SearchRequest):
    """
    Semantic similarity search using vector embeddings and optional metadata filters.
    """
    engine = get_search_engine()
    results = engine.search(
        query=req.query,
        top_k=req.top_k,
        genre_filter=req.genre,
        min_rating=req.min_rating,
        min_year=req.min_year
    )
    return results

@app.post("/api/compare")
def api_compare(req: CompareRequest):
    """
    Compare Vector Semantic Search vs Traditional Exact Keyword Search side-by-side.
    """
    engine = get_search_engine()
    comparison = engine.compare_vector_vs_keyword(query=req.query, top_k=req.top_k)
    return comparison

@app.get("/api/movies")
def api_get_movies():
    """
    Returns full movie catalog metadata and list of available genres.
    """
    if not os.path.exists(MOVIES_DATA_PATH):
        raise HTTPException(status_code=404, detail="Movies data file not found.")

    with open(MOVIES_DATA_PATH, "r", encoding="utf-8") as f:
        movies = json.load(f)

    all_genres = set()
    for m in movies:
        for g in m.get("genre", []):
            all_genres.add(g)

    return {
        "total_movies": len(movies),
        "genres": sorted(list(all_genres)),
        "movies": movies
    }

@app.get("/api/stats")
def api_stats():
    """
    Returns vector database metadata and stats.
    """
    store = VectorStoreManager()
    return {
        "total_vectors": store.count(),
        "embedding_model": EMBEDDING_MODEL_NAME,
        "vector_dimension": EMBEDDING_DIMENSION,
        "collection_name": store.collection_name,
        "persist_directory": store.persist_dir
    }

@app.post("/api/ingest")
def api_ingest(background_tasks: BackgroundTasks):
    """
    Re-triggers vector database ingestion.
    """
    def run_ingest():
        ingest_movies_dataset(force_reset=True)

    background_tasks.add_task(run_ingest)
    return {"message": "Movie dataset re-ingestion started in background."}

# Mount static directory for frontend UI
STATIC_DIR = BASE_DIR / "static"
if os.path.exists(STATIC_DIR):
    app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="127.0.0.1", port=8000, reload=True)
