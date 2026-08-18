document.addEventListener('DOMContentLoaded', () => {
    // DOM Elements
    const searchInput = document.getElementById('search-input');
    const searchBtn = document.getElementById('search-btn');
    const genreFilter = document.getElementById('genre-filter');
    const ratingFilter = document.getElementById('rating-filter');
    const ratingVal = document.getElementById('rating-val');
    const yearFilter = document.getElementById('year-filter');
    const topkFilter = document.getElementById('topk-filter');
    const moviesGrid = document.getElementById('movies-grid');
    const resultsLoader = document.getElementById('results-loader');
    const resultsCountBadge = document.getElementById('results-count-badge');
    const vectorCountBadge = document.getElementById('vector-count-badge');
    const chips = document.querySelectorAll('.chip');
    const tabBtns = document.querySelectorAll('.tab-btn');
    const tabContents = document.querySelectorAll('.tab-content');

    // Compare Mode Elements
    const compareInput = document.getElementById('compare-input');
    const compareBtn = document.getElementById('compare-btn');
    const compareLoader = document.getElementById('compare-loader');
    const vectorResultsList = document.getElementById('vector-results-list');
    const keywordResultsList = document.getElementById('keyword-results-list');

    // Catalog Elements
    const catalogGrid = document.getElementById('catalog-grid');

    // Load initial metadata and genres
    fetchStats();
    fetchMoviesCatalog();

    // Event Listeners
    searchBtn.addEventListener('click', performSemanticSearch);
    searchInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') performSemanticSearch();
    });

    ratingFilter.addEventListener('input', (e) => {
        ratingVal.textContent = parseFloat(e.target.value).toFixed(1);
    });

    chips.forEach(chip => {
        chip.addEventListener('click', () => {
            searchInput.value = chip.dataset.query;
            performSemanticSearch();
        });
    });

    // Tab Switching
    tabBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            tabBtns.forEach(b => b.classList.remove('active'));
            tabContents.forEach(c => c.classList.remove('active'));
            
            btn.classList.add('active');
            const tabId = btn.dataset.tab;
            document.getElementById(tabId).classList.add('active');

            if (tabId === 'compare-mode' && vectorResultsList.children.length === 0) {
                performCompareSearch();
            }
        });
    });

    compareBtn.addEventListener('click', performCompareSearch);
    compareInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') performCompareSearch();
    });

    // API Functions
    async function fetchStats() {
        try {
            const res = await fetch('/api/stats');
            if (res.ok) {
                const data = await res.json();
                vectorCountBadge.textContent = `${data.total_vectors} Vectors Indexed (${data.embedding_model})`;
                document.getElementById('stat-total-movies').textContent = data.total_vectors;
                document.getElementById('stat-embedding-model').textContent = data.embedding_model;
                document.getElementById('stat-vector-dim').textContent = `${data.vector_dimension}D`;
            }
        } catch (err) {
            console.error('Error fetching stats:', err);
            vectorCountBadge.textContent = 'ChromaDB Local Engine';
        }
    }

    async function fetchMoviesCatalog() {
        try {
            const res = await fetch('/api/movies');
            if (res.ok) {
                const data = await res.json();
                
                // Populate Genre Dropdown
                genreFilter.innerHTML = '<option value="All">All Genres</option>';
                data.genres.forEach(g => {
                    const opt = document.createElement('option');
                    opt.value = g;
                    opt.textContent = g;
                    genreFilter.appendChild(opt);
                });

                // Populate Catalog Grid
                renderCatalog(data.movies);
            }
        } catch (err) {
            console.error('Error fetching catalog:', err);
        }
    }

    async function performSemanticSearch() {
        const query = searchInput.value.trim();
        if (!query) return;

        moviesGrid.innerHTML = '';
        resultsLoader.classList.remove('hidden');
        resultsCountBadge.textContent = 'Searching...';

        const payload = {
            query: query,
            top_k: parseInt(topkFilter.value),
            genre: genreFilter.value === 'All' ? null : genreFilter.value,
            min_rating: parseFloat(ratingFilter.value) > 0 ? parseFloat(ratingFilter.value) : null,
            min_year: yearFilter.value ? parseInt(yearFilter.value) : null
        };

        try {
            const res = await fetch('/api/search', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });

            resultsLoader.classList.add('hidden');

            if (res.ok) {
                const movies = await res.json();
                resultsCountBadge.textContent = `${movies.length} matches returned`;

                if (movies.length === 0) {
                    moviesGrid.innerHTML = `
                        <div class="empty-state">
                            <div class="empty-icon">🔍</div>
                            <h3>No Matching Movies Found</h3>
                            <p>Try broadening your query or lowering the rating/year filters.</p>
                        </div>
                    `;
                    return;
                }

                renderMovieCards(movies, moviesGrid);
            } else {
                resultsCountBadge.textContent = 'Error';
                moviesGrid.innerHTML = `<div class="empty-state"><p>Search request failed.</p></div>`;
            }
        } catch (err) {
            console.error('Search error:', err);
            resultsLoader.classList.add('hidden');
            moviesGrid.innerHTML = `<div class="empty-state"><p>Error connecting to search engine backend.</p></div>`;
        }
    }

    async function performCompareSearch() {
        const query = compareInput.value.trim();
        if (!query) return;

        vectorResultsList.innerHTML = '';
        keywordResultsList.innerHTML = '';
        compareLoader.classList.remove('hidden');

        try {
            const res = await fetch('/api/compare', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ query: query, top_k: 5 })
            });

            compareLoader.classList.add('hidden');

            if (res.ok) {
                const data = await res.json();

                // Render Vector Results
                if (data.vector_search_results.length === 0) {
                    vectorResultsList.innerHTML = '<p class="text-muted">No semantic matches.</p>';
                } else {
                    data.vector_search_results.forEach(m => {
                        const card = document.createElement('div');
                        card.className = 'mini-card';
                        card.innerHTML = `
                            <div class="mini-card-top">
                                <span class="mini-card-title">${m.title}</span>
                                <span class="score-badge">${m.match_percentage}% Match</span>
                            </div>
                            <p class="movie-description">${m.description}</p>
                            <div class="movie-meta">${m.genre.join(', ')} • ${m.year}</div>
                        `;
                        vectorResultsList.appendChild(card);
                    });
                }

                // Render Keyword Results
                if (data.keyword_search_results.length === 0) {
                    keywordResultsList.innerHTML = `
                        <div class="mini-card">
                            <p style="color: #f87171; font-weight: 600;">❌ No Literal Keyword Matches Found</p>
                            <p class="col-sub" style="margin-top: 0.4rem;">Traditional keyword search missed relevant movies because words like '${query}' didn't appear verbatim in the movie descriptions.</p>
                        </div>
                    `;
                } else {
                    data.keyword_search_results.forEach(m => {
                        const card = document.createElement('div');
                        card.className = 'mini-card';
                        card.innerHTML = `
                            <div class="mini-card-top">
                                <span class="mini-card-title">${m.title}</span>
                                <span class="genre-tag">${m.keyword_matches} Word Match(es)</span>
                            </div>
                            <p class="movie-description">${m.description}</p>
                            <div class="movie-meta">${m.genre.join(', ')} • ${m.year}</div>
                        `;
                        keywordResultsList.appendChild(card);
                    });
                }
            }
        } catch (err) {
            console.error('Compare error:', err);
            compareLoader.classList.add('hidden');
        }
    }

    function renderMovieCards(movies, container) {
        container.innerHTML = '';
        movies.forEach(m => {
            const card = document.createElement('div');
            card.className = 'movie-card';

            const genresHtml = m.genre.map(g => `<span class="genre-tag">${g}</span>`).join('');

            card.innerHTML = `
                <div>
                    <div class="card-top">
                        <h3 class="movie-title">${m.title}</h3>
                        <span class="score-badge">${m.match_percentage}% Match</span>
                    </div>
                    <div class="movie-meta">
                        <span>★ ${m.rating}</span>
                        <span>•</span>
                        <span>${m.year}</span>
                        ${m.director ? `<span>•</span><span>${m.director}</span>` : ''}
                    </div>
                    <p class="movie-description">${m.description}</p>
                </div>
                <div class="tags-row">
                    ${genresHtml}
                </div>
            `;
            container.appendChild(card);
        });
    }

    function renderCatalog(movies) {
        catalogGrid.innerHTML = '';
        movies.forEach(m => {
            const card = document.createElement('div');
            card.className = 'movie-card';
            const genresHtml = m.genre.map(g => `<span class="genre-tag">${g}</span>`).join('');
            card.innerHTML = `
                <div>
                    <div class="card-top">
                        <h3 class="movie-title">${m.title}</h3>
                        <span class="genre-tag" style="background: rgba(99, 102, 241, 0.2); color: #818cf8;">ID: ${m.id}</span>
                    </div>
                    <div class="movie-meta">
                        <span>★ ${m.rating}</span>
                        <span>•</span>
                        <span>${m.year}</span>
                        <span>•</span>
                        <span>${m.director}</span>
                    </div>
                    <p class="movie-description">${m.description}</p>
                </div>
                <div class="tags-row">
                    ${genresHtml}
                </div>
            `;
            catalogGrid.appendChild(card);
        });
    }
});
