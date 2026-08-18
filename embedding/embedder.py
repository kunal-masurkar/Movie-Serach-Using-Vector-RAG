from typing import List, Union
from sentence_transformers import SentenceTransformer
from config.settings import EMBEDDING_MODEL_NAME, EMBEDDING_DIMENSION

class MovieEmbedder:
    """
    Handles text vectorization using SentenceTransformer.
    Uses 'all-MiniLM-L6-v2' producing 384-dimensional dense vectors.
    """

    def __init__(self, model_name: str = EMBEDDING_MODEL_NAME):
        self.model_name = model_name
        self._model = None

    @property
    def model(self) -> SentenceTransformer:
        """Lazy loading of model to optimize startup time."""
        if self._model is None:
            print(f"[MovieEmbedder] Loading SentenceTransformer model '{self.model_name}'...")
            self._model = SentenceTransformer(self.model_name)
            print(f"[MovieEmbedder] Model loaded successfully. Embedding dimension: {EMBEDDING_DIMENSION}")
        return self._model

    def generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding vector for a single string.
        """
        if not text or not text.strip():
            raise ValueError("Input text cannot be empty.")
        
        vector = self.model.encode(text, convert_to_numpy=True, normalize_embeddings=True)
        return vector.tolist()

    def generate_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embedding vectors for a batch of strings.
        """
        if not texts:
            return []
        
        vectors = self.model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        return vectors.tolist()

    @staticmethod
    def format_movie_searchable_text(movie: dict) -> str:
        """
        Formats movie record into a rich descriptive text for embedding.
        Includes title, description, genres, and director.
        """
        title = movie.get("title", "")
        description = movie.get("description", "")
        genres = ", ".join(movie.get("genre", [])) if isinstance(movie.get("genre"), list) else str(movie.get("genre", ""))
        director = movie.get("director", "")
        
        parts = [f"Title: {title}"]
        if description:
            parts.append(f"Description: {description}")
        if genres:
            parts.append(f"Genres: {genres}")
        if director:
            parts.append(f"Director: {director}")
            
        return ". ".join(parts)
