# 🎬 Movie Semantic Search using Vector Database (ChromaDB)

A **semantic movie search engine** built with Python, **ChromaDB**, **Sentence Transformers (`all-MiniLM-L6-v2`)**, **FastAPI**, and a modern dark-mode Web UI.

Instead of searching strictly by exact keyword matches, this system converts movie descriptions into **384-dimensional dense vector embeddings** and stores them in a vector database. Users can search using natural language (e.g., *"movies about astronauts exploring space"* or *"AI questioning reality"*) to find movies with similar **meanings**, even if the exact search words never appear in the movie description.

---

## 💡 How It Works

```
 ┌─────────────────┐
 │   Movie Data    │  (50 movies in JSON format)
 └────────┬────────┘
          │
          ▼
 ┌─────────────────┐
 │ Searchable Text │  "Title: Interstellar. Overview: A team of explorers... Genres: Sci-Fi..."
 └────────┬────────┘
          │
          ▼
 ┌─────────────────┐
 │ Embedding Model │  all-MiniLM-L6-v2 (SentenceTransformer)
 └────────┬────────┘
          │
          ▼
 ┌─────────────────┐
 │ 384D Vector     │  [0.12, -0.43, 0.71, ..., 0.08]
 └────────┬────────┘
          │
          ▼
 ┌─────────────────┐
 │ Vector DB Store │  ChromaDB (Cosine distance space + Metadata indexing)
 └────────┬────────┘
          │
          ├───────────────────────────────┐
          │                               │
          ▼                               ▼
 ┌─────────────────┐             ┌──────────────────┐
 │ Natural Query   │             │ Metadata Filters │  (Genre, Min Rating, Min Year)
 └────────┬────────┘             └────────┬─────────┘
          │                               │
          ▼                               │
 ┌─────────────────┐                      │
 │ Query Embedding │                      │
 └────────┬────────┘                      │
          │                               │
          └───────────────┬───────────────┘
                          │
                          ▼
                 ┌─────────────────┐
                 │ Similarity      │  Cosine Similarity = 1 - Cosine Distance
                 │ Search (Top-K)  │
                 └────────┬────────┘
                          │
                          ▼
                 ┌─────────────────┐
                 │ Relevant Movies │  Ranked with Match % Badges & Metadata
                 └─────────────────┘
```

### 1. What is an Embedding?
An embedding is a numerical vector representation of text that captures its underlying **semantic meaning**. 
- Using `all-MiniLM-L6-v2`, any sentence is transformed into a dense array of **384 numbers**.
- Texts with similar meanings produce vectors that are mathematically close to each other in 384-dimensional space.

### 2. Searchable Text Construction
For each movie in `data/movies.json`, an informative text representation is generated:
```text
Title: Interstellar. Description: A team of explorers travel through a wormhole in space in an attempt to ensure humanity's survival. Genres: Science Fiction, Adventure, Drama. Director: Christopher Nolan.
```

### 3. Vector Storage in ChromaDB
The vectors, raw documents, and metadata (`title`, `genre`, `year`, `rating`, `director`) are indexed in a local **ChromaDB** vector database collection configured with Cosine distance metric.

### 4. Cosine Similarity & Match Scoring
When a user searches (e.g. *"person stranded on Mars trying to survive"*):
1. The search query is vectorized into a 384D vector.
2. ChromaDB finds the nearest vectors using Cosine distance ($d$).
3. Similarity Score is computed as:
   $$\text{Similarity Score} = 1.0 - d$$
4. Normalized Match Percentage is displayed (e.g. `86.5% Match`).

### 5. Vector Search vs Keyword Search
| Metric / Feature | Traditional Keyword Search | Vector Semantic Search (ChromaDB) |
| :--- | :--- | :--- |
| **Matching Logic** | Exact literal string matching | Semantic meaning & intent matching |
| **"Astronaut stranded on planet"** | ❌ Fails if words "astronaut" or "planet" missing | ✅ Finds *The Martian* & *Cast Away* |
| **Synonyms & Context** | Requires explicit dictionary mapping | Understood automatically via 384D vector space |
| **Filtering** | Basic SQL `WHERE` | Hybrid Vector Similarity + Metadata Filters |

---

## 📁 Project Structure

```text
Movie-Search-Using-Vector/
│
├── config/
│   └── settings.py          # Vector DB paths, embedding model & default settings
│
├── data/
│   └── movies.json          # Curated dataset of 50 movies with detailed metadata
│
├── database/
│   └── vector_store.py      # ChromaDB manager (collection creation, metadata filtering)
│
├── embedding/
│   └── embedder.py          # SentenceTransformers (all-MiniLM-L6-v2) wrapper
│
├── ingestion/
│   └── ingest_movies.py     # Batch ingestion pipeline script
│
├── search/
│   └── search_movies.py     # Search engine & vector vs keyword comparison
│
├── api/
│   └── main.py              # FastAPI REST server & endpoint handlers
│
├── static/
│   ├── index.html           # Dark glassmorphic Web UI HTML
│   ├── style.css            # Custom CSS design system
│   └── app.js               # Frontend interactive logic
│
├── tests/
│   └── test_vector_search.py # Pytest test suite for vector operations
│
├── main.py                  # CLI entry point (search, ingest, serve)
├── requirements.txt         # Project dependencies
└── README.md                # Documentation
```

---

## 🚀 Quick Start & Usage

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Ingest Dataset into Vector DB
Ingest all 50 movies from `data/movies.json` into ChromaDB:
```bash
python main.py --ingest
```

### 3. Launch the Web Interface
Start the FastAPI server and open the web app:
```bash
python main.py --serve
```
Open **`http://127.0.0.1:8000`** in your browser to experience the real-time semantic search UI.

### 4. Interactive CLI Search
Run a command-line search directly in your terminal:
```bash
python main.py
```

### 5. Run Automated Tests
```bash
pytest tests/
```

---

## 🎯 Sample Natural Language Prompts

Try entering these prompts into the search bar or CLI:

- **🚀 Space Exploration**: *"A team of astronauts traveling through deep space searching for a new home"* $\rightarrow$ **Interstellar**, **Apollo 13**, **The Martian**
- **🤖 Artificial Intelligence**: *"Synthetic humanoid AI questioning human emotions and reality"* $\rightarrow$ **Ex Machina**, **The Matrix**, **Her**
- **🏝️ Survival**: *"A person stranded alone trying to stay alive against nature"* $\rightarrow$ **Cast Away**, **The Martian**, **Gravity**
- **🕵️ Dark Mystery**: *"Detectives hunting a serial killer in a dark rainy city using clues"* $\rightarrow$ **Se7en**, **Knives Out**, **Blade Runner 2049**
- **🎭 Multiverse Adventure**: *"Exploring alternate universes and parallel realities"* $\rightarrow$ **Everything Everywhere All at Once**, **Spider-Man: Across the Spider-Verse**

---

## 🛠️ Technology Stack
- **Language**: Python 3.14+
- **Vector Database**: ChromaDB
- **Embedding Model**: `sentence-transformers/all-MiniLM-L6-v2` (384-dimensional dense vectors)
- **Backend API**: FastAPI + Uvicorn
- **Frontend UI**: Vanilla HTML5, CSS3 (Glassmorphism, Dark Mode), JavaScript (ES6+)
- **Testing**: Pytest
