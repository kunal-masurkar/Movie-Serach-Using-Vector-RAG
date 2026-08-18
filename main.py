import sys
import argparse
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent))

from ingestion.ingest_movies import ingest_movies_dataset
from search.search_movies import MovieSearchEngine

def run_cli_search():
    """Interactive command-line interface for semantic movie search."""
    print("=" * 70)
    print(" 🎬 CineVector — Semantic Movie Search (Vector Database CLI) ")
    print("=" * 70)

    engine = MovieSearchEngine()

    while True:
        print("\n" + "-" * 70)
        query = input("Enter search query (or 'q' to quit): ").strip()
        if not query or query.lower() == 'q':
            print("Exiting CineVector CLI. Goodbye!")
            break

        print(f"\n[CLI] Querying vector database for: '{query}'...")
        results = engine.search(query=query, top_k=5)

        if not results:
            print("[CLI] No matching movies found.")
            continue

        print(f"\nTop {len(results)} Vector Search Results:")
        for idx, res in enumerate(results, 1):
            print(f"\n{idx}. {res['title']} ({res['year']}) — ★ {res['rating']}")
            print(f"   Similarity Match: {res['match_percentage']}% | Cosine Distance: {res['distance']}")
            print(f"   Genres: {', '.join(res['genre'])} | Director: {res.get('director', 'N/A')}")
            print(f"   Overview: {res['description']}")

def main():
    parser = argparse.ArgumentParser(description="Movie Semantic Search Vector DB System")
    parser.add_argument("--ingest", action="store_true", help="Force re-ingest movies.json dataset into ChromaDB")
    parser.add_argument("--query", type=str, help="Run a single vector search query from command line")
    parser.add_argument("--serve", action="store_true", help="Launch FastAPI web server on http://127.0.0.1:8000")

    args = parser.parse_args()

    if args.ingest:
        print("[Main] Running dataset vector ingestion pipeline...")
        ingest_movies_dataset(force_reset=True)
        return

    if args.query:
        engine = MovieSearchEngine()
        results = engine.search(query=args.query, top_k=5)
        print(f"\nQuery: '{args.query}'\nResults:")
        for r in results:
            print(f"- {r['title']} ({r['match_percentage']}% match)")
        return

    if args.serve:
        import uvicorn
        print("[Main] Starting FastAPI web server at http://127.0.0.1:8000 ...")
        uvicorn.run("api.main:app", host="127.0.0.1", port=8000, reload=True)
        return

    # Default: Run interactive CLI
    run_cli_search()

if __name__ == "__main__":
    main()
