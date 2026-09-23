// Initialize the application when the window loads
window.onload = async function() {
    try {
        // Display loading message
        const resultElement = document.getElementById('result');
        resultElement.textContent = "Loading movie data...";
        resultElement.className = 'loading';
        
        // Load data
        await loadData();
        
        // Populate dropdown and update status
        populateMoviesDropdown();
        resultElement.textContent = "Data loaded. Please select a movie.";
        resultElement.className = 'success';
    } catch (error) {
        console.error('Initialization error:', error);
        // Error message already set in data.js
    }
};

// Populate the movies dropdown with sorted movie titles
function populateMoviesDropdown() {
    const selectElement = document.getElementById('movie-select');
    
    // Clear existing options except the first placeholder
    while (selectElement.options.length > 1) {
        selectElement.remove(1);
    }
    
    // Sort movies alphabetically by title
    const sortedMovies = [...movies].sort((a, b) => a.title.localeCompare(b.title));
    
    // Add movies to dropdown
    sortedMovies.forEach(movie => {
        const option = document.createElement('option');
        option.value = movie.id;
        option.textContent = movie.title;
        selectElement.appendChild(option);
    });
}

// Build a binary genre vector (length 18) for a movie
function buildGenreVector(genres) {
    return genreNames.map(genre => genres.includes(genre) ? 1 : 0);
}

// Cosine similarity between two binary genre vectors
function cosineSimilarity(vectorA, vectorB) {
    let dot = 0;
    let normA = 0;
    let normB = 0;
    for (let i = 0; i < vectorA.length; i++) {
        dot += vectorA[i] * vectorB[i];
        normA += vectorA[i] * vectorA[i];
        normB += vectorB[i] * vectorB[i];
    }
    if (normA === 0 || normB === 0) return 0;
    return dot / (Math.sqrt(normA) * Math.sqrt(normB));
}

// Watched movies used to build the user profile (user's own picks)
const SEED_MOVIE_IDS = [33, 202, 222];

// Build a user profile vector (length 18): average of the watched movies' genre vectors
function buildUserProfile(movieIds) {
    const profile = new Array(genreNames.length).fill(0);
    let count = 0;
    movieIds.forEach(id => {
        const movie = movies.find(m => m.id === id);
        if (movie) {
            const vector = buildGenreVector(movie.genres);
            for (let i = 0; i < vector.length; i++) {
                profile[i] += vector[i];
            }
            count++;
        }
    });
    return profile.map(value => count > 0 ? value / count : 0);
}

// Profile-based recommendations: compare the user profile vector with all movies
function getProfileRecommendations(watchedIds) {
    const profile = buildUserProfile(watchedIds);
    const candidates = movies.filter(movie => !watchedIds.includes(movie.id));
    const scored = candidates.map(movie => ({
        ...movie,
        score: cosineSimilarity(profile, buildGenreVector(movie.genres))
    }));
    scored.sort((a, b) => b.score - a.score);
    return scored.slice(0, 5);
}

// Main recommendation function
function getRecommendations() {
    const resultElement = document.getElementById('result');
    
    try {
        // Step 1: Get user input
        const selectElement = document.getElementById('movie-select');
        const selectedMovieId = parseInt(selectElement.value);
        
        if (isNaN(selectedMovieId)) {
            resultElement.textContent = "Please select a movie first.";
            resultElement.className = 'error';
            return;
        }
        
        // Step 2: Find the liked movie
        const likedMovie = movies.find(movie => movie.id === selectedMovieId);
        if (!likedMovie) {
            resultElement.textContent = "Error: Selected movie not found in database.";
            resultElement.className = 'error';
            return;
        }
        
        // Show loading message while processing
        resultElement.textContent = "Calculating recommendations...";
        resultElement.className = 'loading';
        
        // Use setTimeout to allow the UI to update before heavy computation
        setTimeout(() => {
            try {
                // Step 3: Prepare for similarity calculation
                const likedVector = buildGenreVector(likedMovie.genres);
                const candidateMovies = movies.filter(movie => movie.id !== likedMovie.id);
                
                // Step 4: Calculate cosine similarity scores
                const scoredMovies = candidateMovies.map(candidate => ({
                    ...candidate,
                    score: cosineSimilarity(likedVector, buildGenreVector(candidate.genres))
                }));
                
                // Step 5: Sort by score in descending order
                scoredMovies.sort((a, b) => b.score - a.score);
                
                // Step 6: Select top recommendations
                const topRecommendations = scoredMovies.slice(0, 5);
                
                // Step 7: Display results (Item-to-Item Top-5 + Profile-Based Top-5)
                const profileRecommendations = getProfileRecommendations(SEED_MOVIE_IDS);
                const recommendationTitles = topRecommendations.map(movie => movie.title);
                const profileTitles = profileRecommendations.map(movie => movie.title);
                if (topRecommendations.length > 0) {
                    resultElement.textContent =
                        `Because you liked "${likedMovie.title}", we recommend (Item-to-Item Top-5): ${recommendationTitles.join(', ')} | ` +
                        `(Profile-Based Top-5): ${profileTitles.join(', ')}`;
                    resultElement.className = 'success';
                } else {
                    resultElement.textContent = `No recommendations found for "${likedMovie.title}".`;
                    resultElement.className = 'error';
                }
            } catch (error) {
                console.error('Error in recommendation calculation:', error);
                resultElement.textContent = "An error occurred while calculating recommendations.";
                resultElement.className = 'error';
            }
        }, 100);
    } catch (error) {
        console.error('Error in getRecommendations:', error);
        resultElement.textContent = "An unexpected error occurred.";
        resultElement.className = 'error';
    }
}
