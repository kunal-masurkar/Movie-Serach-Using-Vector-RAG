import os
from pathlib import Path

# Base Directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Database Configuration
CHROMA_PERSIST_DIRECTORY = str(BASE_DIR / "chroma_db")
COLLECTION_NAME = "movies_semantic_search"

# Embedding Model Configuration
# all-MiniLM-L6-v2 produces 384-dimensional dense vectors
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384

# Default Search Parameters
DEFAULT_TOP_K = 5
COSINE_SIMILARITY_THRESHOLD = 0.0

# Dataset Path
MOVIES_DATA_PATH = str(BASE_DIR / "data" / "movies.json")
