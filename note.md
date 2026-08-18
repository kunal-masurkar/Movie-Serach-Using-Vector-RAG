# Movie Semantic Search — Vector Database Project

## 1. Project Overview

This project is a **semantic movie search system** using a **vector database**.

Instead of searching movies only by exact keywords, the system converts movie descriptions into **embeddings (vectors)** and stores them in a vector database.

Users can then search using natural language such as:

> "I want a movie about astronauts exploring space."

The system finds movies with **similar meaning**, even if the exact words don't appear in the movie description.

### Core idea

```text
Movie Data
    ↓
Create searchable text
    ↓
Embedding Model
    ↓
Vector
    ↓
Vector Database
    ↓
User Query
    ↓
Query Embedding
    ↓
Similarity Search
    ↓
Relevant Movies
```

---

# 2. Project Goals

The main goal is to understand how a vector database works in a real application.

The project should demonstrate:

* Creating embeddings
* Storing embeddings in a vector database
* Semantic similarity search
* Metadata storage
* Metadata filtering
* Top-K results
* Similarity scores
* Connecting a backend to a vector database
* Separating normal application data from vector data
* Eventually integrating an LLM

---

# 3. Example

### Movie

```json
{
  "id": 1,
  "title": "Interstellar",
  "description": "A group of astronauts travel through a wormhole in search of a new home for humanity.",
  "genre": [
    "Science Fiction",
    "Adventure",
    "Drama"
  ],
  "year": 2014,
  "rating": 8.7
}
```

The system creates searchable text:

```text
Interstellar.
A group of astronauts travel through a wormhole
in search of a new home for humanity.
Science Fiction, Adventure, Drama.
```

The embedding model converts this into a vector:

```text
[0.12, -0.43, 0.71, 0.08, ...]
```

The vector is stored in the vector database.

---

# 4. What is an Embedding?

An embedding is a numerical representation of information.

For example:

```text
"Movie about space exploration"
```

becomes:

```text
[0.21, -0.43, 0.72, 0.15, ...]
```

The numbers themselves aren't meaningful to us individually.

The important thing is that **similar meanings tend to produce vectors that are close to one another**.

For example:

```text
"Movie about space exploration"
              ↓
           Vector A

"Film about astronauts traveling through space"
              ↓
           Vector B
```

A and B should be relatively similar.

---

# 5. What is the Vector Database?

The vector database stores the embeddings and allows us to perform similarity searches.

A conceptual record could look like:

```text
Movie ID:
movie_001

Vector:
[0.12, -0.43, 0.71, ...]

Text:
"Interstellar. A group of astronauts..."

Metadata:
{
    title: "Interstellar",
    genre: ["Science Fiction", "Adventure"],
    year: 2014,
    rating: 8.7
}
```

Therefore, a vector database record consists conceptually of:

```text
┌───────────────────────────────┐
│ Vector Database Record        │
├───────────────────────────────┤
│ ID                            │
│ Vector / Embedding            │
│ Searchable Content            │
│ Metadata                      │
└───────────────────────────────┘
```

---

# 6. Why Use a Vector Database?

A traditional keyword search might search for:

```text
"space exploration"
```

and look for those exact words.

A vector database searches for **semantic similarity**.

For example:

```text
User:
"I want movies about astronauts exploring space."

Possible results:

Interstellar
The Martian
Gravity
Apollo 13
```

The movie descriptions don't need to contain exactly:

```text
"astronauts exploring space"
```

The embeddings allow the system to compare the **meaning**.

---

# 7. Project Architecture

The initial project can be structured as:

```text
                    MOVIE DATA
                        │
                        ↓
               Searchable Text
                        │
                        ↓
                Embedding Model
                        │
                        ↓
                     Vector
                        │
                        ↓
                ┌──────────────┐
                │  Vector DB   │
                │              │
                │ Embeddings   │
                │ Text         │
                │ Metadata     │
                └──────┬───────┘
                       ↑
                       │
                User Search
                       │
                       ↓
                Query Embedding
                       │
                       ↓
                Similarity Search
                       │
                       ↓
                Top-K Movies
```

---

# 8. Recommended Technology Stack

For the learning version:

### Language

```text
Python
```

### Embedding model

Use a local sentence-transformer model such as:

```text
all-MiniLM-L6-v2
```

This is useful for learning because you can generate embeddings locally.

### Vector database

Use either:

```text
Chroma
```

or:

```text
Qdrant
```

For the first implementation, **Chroma is a simple choice for learning**.

### Optional backend

Later:

```text
FastAPI
```

### Optional frontend

Later:

```text
React
```

### Optional LLM

Add only after the basic vector search works.

---

# 9. Project Structure

Start with:

```text
movie-vector-search/
│
├── data/
│   └── movies.json
│
├── database/
│   └── vector_store.py
│
├── embedding/
│   └── embedder.py
│
├── ingestion/
│   └── ingest_movies.py
│
├── search/
│   └── search_movies.py
│
├── config/
│   └── settings.py
│
├── main.py
│
├── requirements.txt
│
└── README.md
```

You can simplify this while learning. The separation is mainly to help you understand the responsibilities of each component.

---

# 10. Movie Dataset

Create:

```text
data/movies.json
```

Start with around **20–50 movies**.

Example:

```json
[
  {
    "id": "movie_001",
    "title": "Interstellar",
    "description": "A group of astronauts travel through a wormhole in search of a new home for humanity.",
    "genre": ["Science Fiction", "Adventure", "Drama"],
    "year": 2014,
    "rating": 8.7
  },
  {
    "id": "movie_002",
    "title": "The Martian",
    "description": "An astronaut becomes stranded on Mars and must use science and engineering to survive.",
    "genre": ["Science Fiction", "Adventure", "Drama"],
    "year": 2015,
    "rating": 8.0
  }
]
```

Add more movies from different categories:

```text
Science Fiction
Adventure
Comedy
Drama
Thriller
Horror
Romance
Action
Animation
Mystery
```

This makes semantic search easier to test.

---

# 11. Creating the Searchable Movie Text

You need to decide what information should be converted into an embedding.

A good initial format is:

```text
Title + Description + Genre
```

For example:

```text
Interstellar.

A group of astronauts travel through a wormhole
in search of a new home for humanity.

Science Fiction, Adventure, Drama.
```

Then send this text to the embedding model.

---

# 12. Embedding Generation

The process is:

```text
Movie
 ↓
Searchable text
 ↓
Embedding model
 ↓
Vector
```

For example:

```text
"Interstellar. Astronauts travel through a wormhole..."
```

might produce:

```text
[
    0.123,
   -0.421,
    0.782,
    0.091,
    ...
]
```

The exact values aren't important.

What matters is that each movie receives a vector.

---

# 13. Vector Dimensions

Different embedding models generate vectors with different dimensions.

For example:

```text
Model
 ↓
384-dimensional vector
```

or another model might produce:

```text
768-dimensional vector
```

The vector database collection must be configured for the dimension produced by your embedding model.

For this project, if you use:

```text
all-MiniLM-L6-v2
```

expect a 384-dimensional embedding.

---

# 14. Ingestion Pipeline

Create an ingestion process responsible for putting movies into the vector database.

```text
movies.json
    ↓
Load movie
    ↓
Create searchable text
    ↓
Generate embedding
    ↓
Store embedding
    ↓
Store movie content
    ↓
Store metadata
```

After ingestion:

```text
Vector DB
│
├── movie_001
│   ├── vector
│   ├── text
│   └── metadata
│
├── movie_002
│   ├── vector
│   ├── text
│   └── metadata
│
└── ...
```

---

# 15. What Metadata Should Be Stored?

At minimum:

```text
movie_id
title
genre
year
rating
```

You can later add:

```text
director
cast
language
country
duration
age_rating
```

Example:

```json
{
  "movie_id": "movie_001",
  "title": "Interstellar",
  "genre": "Science Fiction",
  "year": 2014,
  "rating": 8.7
}
```

---

# 16. Searching

When the user enters:

```text
Movies about space exploration
```

the system performs:

```text
User Query
     ↓
Embedding Model
     ↓
Query Vector
     ↓
Vector Database
     ↓
Similarity Search
     ↓
Top-K Results
```

For example:

```text
Query:
Movies about space exploration

Results:

Interstellar       0.91
The Martian        0.86
Gravity            0.79
Apollo 13          0.75
Dune               0.51
```

The scores are similarity scores.

Higher generally means more similar under the configured metric.

Don't interpret a score like `0.91` as "91% correct." It is a similarity measure, not a probability.

---

# 17. Top-K Search

You usually don't want every movie.

You request the top results.

For example:

```text
top_k = 5
```

The vector database returns the five most similar results.

Conceptually:

```text
Query
 ↓
Vector search
 ↓
Top 5
 ↓
Results
```

---

# 18. Metadata Filtering

Now introduce filters.

Suppose the user wants:

> "Science-fiction movies about space exploration rated above 8."

Your system has:

```text
Semantic query:
space exploration

Filter:
genre = Science Fiction
rating >= 8
```

Now you're combining:

```text
Semantic similarity
        +
Metadata filtering
```

This is an important capability to learn.

---

# 19. Example Search Cases

### Search 1

```text
Movies about space exploration
```

Expected candidates:

```text
Interstellar
The Martian
Gravity
Apollo 13
```

---

### Search 2

```text
Movies involving artificial intelligence
```

Potential results:

```text
Ex Machina
Her
The Matrix
Blade Runner 2049
```

---

### Search 3

```text
A person trapped somewhere trying to survive
```

Potential results:

```text
The Martian
Cast Away
Gravity
```

---

### Search 4

```text
Funny movie about friends
```

Potential results:

```text
Comedy/friendship movies
```

Notice that these searches don't necessarily require exact keywords.

---

# 20. Keyword Search vs Vector Search

You should explicitly test this.

Suppose the database contains:

```text
"An astronaut becomes stranded on Mars
and must use science and engineering to survive."
```

Search:

```text
"A person stuck on another planet trying to stay alive."
```

Keyword search may perform poorly because the wording is different.

Vector search should understand the relationship between:

```text
astronaut
```

and:

```text
person
```

and:

```text
stranded
```

and:

```text
stuck
```

and:

```text
survive
```

This experiment is one of the main reasons for building the project.

---

# 21. Similarity Metrics

Vector databases use mathematical distance/similarity measures.

Common choices include:

```text
Cosine similarity
Euclidean distance
Dot product
```

For this project, you can start with **cosine similarity**.

Conceptually:

```text
Vector A
   ↓
Compare
   ↓
Vector B
```

If their direction is similar, their semantic representations are considered similar.

---

# 22. PostgreSQL — Optional

For your first version, you don't need PostgreSQL.

However, a more production-like system could use:

```text
PostgreSQL
     +
Vector Database
```

PostgreSQL could store:

```text
movies
users
favorites
ratings
watch history
```

The vector database stores:

```text
movie embeddings
```

Both can share:

```text
movie_id
```

Example:

```text
PostgreSQL

movie_id = movie_001
title = Interstellar
rating = 8.7
```

Vector DB:

```text
movie_id = movie_001
vector = [...]
```

---

# 23. Why Use Two Databases?

Because they solve different problems.

### PostgreSQL

Good for:

```text
Structured data
Relationships
Transactions
Filtering
User accounts
Business logic
```

### Vector DB

Good for:

```text
Semantic search
Similarity search
Embeddings
Nearest-neighbor retrieval
```

Don't think of a vector database as a replacement for PostgreSQL.

Think of it as a **specialized search database**.

---

# 24. Backend API — Version 2

Once the standalone Python program works, add an API.

For example:

```text
POST /api/search
```

Request:

```json
{
  "query": "movies about space exploration",
  "top_k": 5
}
```

Response:

```json
{
  "results": [
    {
      "movie_id": "movie_001",
      "title": "Interstellar",
      "score": 0.91
    },
    {
      "movie_id": "movie_002",
      "title": "The Martian",
      "score": 0.86
    }
  ]
}
```

Then your architecture becomes:

```text
Frontend
   ↓
REST API
   ↓
Search Service
   ↓
Embedding Model
   ↓
Vector DB
```

---

# 25. Frontend — Version 3

You can eventually create:

```text
┌───────────────────────────────────────┐
│         MOVIE SEMANTIC SEARCH         │
├───────────────────────────────────────┤
│                                       │
│ Search                                │
│ ┌───────────────────────────────────┐ │
│ │ movies about space exploration    │ │
│ └───────────────────────────────────┘ │
│                                       │
│ Genre: [All ▼]                        │
│ Minimum Rating: [7.5]                 │
│ Year: [2010]                          │
│                                       │
│             [ Search ]                │
│                                       │
├───────────────────────────────────────┤
│ Results                               │
│                                       │
│ Interstellar                   0.91   │
│ Science Fiction • 2014                │
│                                       │
│ The Martian                    0.86   │
│ Science Fiction • 2015                │
│                                       │
│ Gravity                       0.79    │
│ Science Fiction • 2013                │
└───────────────────────────────────────┘
```

The UI isn't the important part. The search pipeline is.

---

# 26. Adding an LLM — Version 4

Once your vector search works, you can add an LLM.

Instead of:

```text
User
 ↓
Vector DB
 ↓
Results
```

you can have:

```text
User
 ↓
LLM / Query Understanding
 ↓
Embedding
 ↓
Vector DB
 ↓
Relevant movies
 ↓
LLM
 ↓
Natural-language response
```

For example:

> "I want something similar to Interstellar, but less serious and with more adventure."

The system could retrieve relevant movies and then have the LLM explain why each one matches.

---

# 27. Don't hide the Vector DB behind the LLM

For this learning project, this is important.

Don't start with:

```text
User
 ↓
LLM framework
 ↓
Everything magically happens
```

You'll learn very little about vector databases that way.

Instead, first make this work:

```text
Text
 ↓
Embedding
 ↓
Vector DB
 ↓
Similarity
 ↓
Results
```

Then add the LLM.

---

# 28. Testing Plan

You should create a small test set.

### Test semantic similarity

```text
Query:
Movies about space exploration
```

Check whether:

```text
Interstellar
The Martian
Gravity
```

rank highly.

### Test unrelated search

```text
Query:
Romantic comedy about relationships
```

Check that space movies aren't ranked highly.

### Test metadata

```text
Query:
space exploration

Filter:
rating >= 8
```

Verify that lower-rated movies are excluded.

### Test top-K

```text
top_k = 3
```

Verify only three results are returned.

---

# 29. Evaluation

Don't just look at the results manually.

Create expected results.

Example:

```text
Query:
"Movies about astronauts"

Expected relevant:
Interstellar
The Martian
Gravity
Apollo 13
```

Then compare your vector search results against that expected set.

You can eventually calculate metrics such as:

```text
Precision
Recall
Precision@K
Recall@K
```

This becomes important when you move from a toy project to a real search system.

---

# 30. Development Roadmap

## Phase 1 — Embeddings

Learn:

```text
Movie text
 ↓
Embedding model
 ↓
Vector
```

---

## Phase 2 — Vector DB

Learn:

```text
Vector
 ↓
Store
 ↓
Retrieve
```

---

## Phase 3 — Similarity Search

Implement:

```text
Query
 ↓
Query embedding
 ↓
Vector similarity
 ↓
Top-K movies
```

---

## Phase 4 — Metadata

Add:

```text
Genre
Year
Rating
```

and implement filtering.

---

## Phase 5 — API

Build:

```text
POST /search
```

with FastAPI.

---

## Phase 6 — Frontend

Build a small React search interface.

---

## Phase 7 — LLM

Add natural-language explanations and more complex query understanding.

---

# 31. Final Architecture

The completed learning project could look like:

```text
                         USER
                           │
                           ↓
                      React UI
                           │
                           ↓
                      FastAPI
                           │
                  ┌────────┴────────┐
                  ↓                 ↓
             Query Parser      Filters
                  │                 │
                  ↓                 ↓
             Embedding Model        │
                  │                 │
                  └────────┬────────┘
                           ↓
                    Vector Database
                           │
                    Similarity Search
                           │
                           ↓
                       Top-K Movies
                           │
                           ↓
                    Optional Reranking
                           │
                           ↓
                         Results
```

And the ingestion side:

```text
                    movies.json
                         │
                         ↓
                   Data Processing
                         │
                         ↓
                  Searchable Text
                         │
                         ↓
                  Embedding Model
                         │
                         ↓
                       Vector
                         │
                         ↓
                  ┌──────────────┐
                  │  Vector DB   │
                  │              │
                  │ Vector       │
                  │ Text         │
                  │ Metadata     │
                  └──────────────┘
```

---

# 32. What you should understand after completing it

If you finish this project properly, you should be able to explain:

**What is an embedding?**

> A numerical representation of data that captures semantic characteristics.

**What is a vector database?**

> A database optimized for storing embeddings and finding similar vectors.

**Why not just use SQL?**

> SQL is excellent for structured queries, while vector databases are specialized for similarity/semantic search.

**What is similarity search?**

> Finding stored vectors that are mathematically closest to a query vector.

**What is metadata?**

> Structured information attached to a vector, such as genre, year, or rating, which can be used for filtering.

**What is Top-K?**

> Returning the K most relevant results from a similarity search.

**Where does the embedding model fit?**

> It converts movie text and user queries into vectors so they can be compared.

**Where does the LLM fit?**

> It is optional; it can understand complex queries and generate explanations from retrieved results.

That gives you a complete, manageable project that teaches the **actual mechanics of vector databases**, rather than just using one as a black box.