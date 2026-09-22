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
                
                // Step 7: Display results
                if (topRecommendations.length > 0) {
                    const recommendationTitles = topRecommendations.map(movie => movie.title);
                    resultElement.textContent = `Because you liked "${likedMovie.title}", we recommend: ${recommendationTitles.join(', ')}`;
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
