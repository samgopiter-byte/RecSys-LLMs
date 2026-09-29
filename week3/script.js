// ===========================================================================
// HW3 - memory-based Collaborative Filtering over MovieLens 100K
// ===========================================================================
//
// MISSING-VALUE STRATEGY  (HW3 requires one explicit, documented strategy)
// ---------------------------------------------------------------------------
// The rating matrix is 943 users x 1682 movies = 1 586 126 cells, of which
// only 100 000 are observed -> density 6.30 %, sparsity 93.70 %.
//
// The single strategy used everywhere below is:
//
//     CO-RATED ONLY
//
// Missing cells are NEVER imputed, never filled with 0, and never replaced by
// a global / user / item average. Every similarity and every prediction is
// computed strictly on the INTERSECTION of two profiles, i.e. only on movies
// that both parties actually rated. A missing rating is treated as a structural
// absence that carries no information and is excluded from the computation, so
// it can never leak into a sum or a mean as an implicit 0.
//
// That last point is not a style preference. If a dense matrix were padded with
// zeros, ~94 % of every pair would be (0,0) points, which would inflate the
// co-rated count, drag both means towards 0 and make Pearson meaningless. The
// co-rated-only strategy is what makes the similarity well defined on a 6.3 %
// dense matrix.
//
// Because a prediction can only be assembled from cells that really exist,
// every estimate must justify itself with its own evidence. That is what the
// guards below enforce: when the co-rated support is too thin, the code REFUSES
// to output a number instead of silently falling back to a prior, and the UI
// reports which guard failed.
//
// RELIABILITY WEIGHTING
// ---------------------------------------------------------------------------
// Raw Pearson over a small co-rated set is very noisy and biased upwards, so
// every similarity is shrunk by the amount of evidence behind it:
//
//     weightedSimilarity = rawPearson * min(commonCount / 50, 1)
//
// where commonCount = number of co-rated movies. The factor is exactly 1.0 once
// 50+ movies are co-rated and decays linearly below that, so a pair seen on
// only 5 movies can contribute at most 10 % of its correlation.
//
// WHICH MEAN IS USED WHERE
// ---------------------------------------------------------------------------
//   - Pearson normalises by the means of the two profiles RESTRICTED to the
//     co-rated subset (the standard definition, see pearsonOnCoRated).
//   - The prediction step uses each neighbour's mean over ALL of its ratings,
//     because the target movie is by construction NOT in the co-rated set: either
//     the pair was never rated, or the target cell was held out first (below).
//
// ITEM-BASED CF - SAME STRATEGY, ONE DIFFERENT BASELINE
// ---------------------------------------------------------------------------
// Item-based CF reuses the user-based strategy almost unchanged:
//   - similarity is again Pearson, again on the co-rated intersection only, but
//     the "co-rated" axis is now the set of USERS who rated both movies;
//   - the same reliability weighting, where commonCount is now the number of
//     co-rated USERS rather than co-rated movies;
//   - the same MIN_OVERLAP gate, the same cold-start guards, the same refusal to
//     output a number when the evidence is too thin.
// Prediction form (as in the lecture - the baseline is the TARGET MOVIE, not the
// target user, so this is deliberately NOT a mirror of the user-based one):
//     r_hat_ui = mean(i) + SUM_j w(i,j) * (r_uj - mean(j)) / SUM_j |w(i,j)|
// over the top-K items j that the target user has already rated, where mean(i) is
// the mean rating of the target movie i over all of its raters and mean(j) is the
// mean rating of neighbour movie j over all of its raters.
//
// LEAKAGE-SAFE PREDICTION (Iteration 4)
// ---------------------------------------------------------------------------
// Those "NOT in the co-rated set" statements are only true for a pair that was
// never rated. When the selected pair ALREADY has an observed rating in u.data,
// that rating would otherwise feed the very quantities being estimated:
//   * the target user's mean, which is the User-Based baseline;
//   * the target movie's mean, which is the Item-Based baseline;
//   * every co-rated intersection - the target cell is part of it whenever the
//     target user and a User-Based neighbour both rated the target movie, and
//     whenever the target user rated both the target movie and an Item-Based
//     neighbour item.
//
// predictLeakageSafe() therefore runs BOTH methods inside
// withTargetCellHeldOut() (see data.js), which removes the cell from both
// profiles - ratings map, id list, count, sum and mean - for the duration of the
// call and restores all of it afterwards, including on error. The flat `ratings`
// array and the two data files are never modified, so nothing is lost
// permanently. When the pair was never rated the hold-out is a no-op and the
// calculation is an ordinary missing-value prediction.
// ===========================================================================


// --- CF tuning constants (all explicit, nothing hardcoded inline) ----------
const MIN_OVERLAP = 3;                    // min co-rated movies for a pair to count as a neighbour
const RELIABILITY_SATURATION = 50;        // co-rated count at which the reliability weight reaches 1.0
const TOP_K_NEIGHBORS = 50;               // how many best-scoring neighbours are allowed to vote
const MIN_USER_HISTORY = 5;               // user-side cold-start guard (ratings in the profile)
const MIN_ITEM_HISTORY = 5;               // item-side cold-start guard (ratings on the movie)
const MIN_EVIDENCE = 0.5;                 // min sum of weighted similarities to accept a prediction
const REQUIRE_POSITIVE_SIMILARITY = true; // only positively correlated neighbours get to vote
const MIN_PREDICTION = 1;
const MAX_PREDICTION = 5;

// Demo cases. Every pair below was verified to be a GENUINELY MISSING cell:
// ratingsByUser.get(userId).ratings.has(movieId) === false, so each demo is a
// real recommendation, not a replay of a rating the dataset already contains.
// `search` is what gets typed into the search box so the filter visibly narrows
// down to that movie during a demo.
const QUICK_EXAMPLES = [
    // A: standard case - both methods answer and essentially agree.
    { key: 'A', userId: 308, movieId: 302, search: 'L.A. Confidential' },
    // B: large disagreement - both answer, 1.50 points apart.
    { key: 'B', userId: 206, movieId: 1136, search: 'Ghosts of Mississippi' },
    // C: sparse - User-Based evidence too weak, Item-Based still answers.
    { key: 'C', userId: 196, movieId: 848, search: 'Murder, My Sweet' }
];

let dataReady = false;
let movieSearchIndex = [];   // [{ id, label, titleLower }] - pre-lowercased, built once
let lastRenderedMovieId = null;  // what the visible prediction cards currently describe


// Initialize application when window loads
window.onload = async function() {
    try {
        updateStatus('Loading MovieLens data...');

        await loadData();

        // Build the memory-based CF lookup structures (by user and by item)
        buildRatingIndexes();

        // Pre-lowercase every title once, so filtering is a plain substring scan
        buildMovieSearchIndex();

        // Populate dropdowns
        populateUserDropdown();
        populateMovieDropdown();

        dataReady = true;
        document.getElementById('predict-btn').disabled = false;

        updateStatus(
            `Ready: ${numUsers} users, ${numMovies} movies, ${ratings.length} ratings. ` +
            `User-Based CF and Item-Based CF active (co-rated only, Pearson + reliability weighting).`
        );

        console.info(
            '[HW3] User-Based CF and Item-Based CF are both active on the selected user + movie.'
        );
        console.info(
            `[HW3] CF constants: MIN_OVERLAP=${MIN_OVERLAP}, ` +
            `RELIABILITY_SATURATION=${RELIABILITY_SATURATION}, ` +
            `TOP_K_NEIGHBORS=${TOP_K_NEIGHBORS}, ` +
            `MIN_USER_HISTORY=${MIN_USER_HISTORY}, ` +
            `MIN_ITEM_HISTORY=${MIN_ITEM_HISTORY}, ` +
            `MIN_EVIDENCE=${MIN_EVIDENCE}`
        );
    } catch (error) {
        console.error('Initialization error:', error);
        updateStatus('Error initializing application: ' + error.message, true);
    }
};

// ===========================================================================
// Movie search + dropdown population
// ===========================================================================

function formatMovieLabel(movie) {
    return movie.year ? `${movie.title} (${movie.year})` : movie.title;
}

/**
 * Pre-build the searchable catalogue once at startup: the dropdown label plus an
 * already-lowercased title. Filtering is then a single substring scan over 1682
 * strings per keystroke, so there is no need for debouncing and no page reload.
 */
function buildMovieSearchIndex() {
    movieSearchIndex = movies.map(movie => ({
        id: movie.id,
        label: formatMovieLabel(movie),
        titleLower: movie.title.toLowerCase()
    }));
}

function findMovieById(movieId) {
    const id = parseInt(movieId, 10);
    const entry = movieSearchIndex.find(e => e.id === id);
    if (entry) return entry.label;
    const movie = movies.find(m => m.id === id);
    if (!movie) return `Movie ${movieId}`;
    return formatMovieLabel(movie);
}

function getMovieTitle(movieId) {
    return findMovieById(movieId);
}

/**
 * Repopulate the movie dropdown from a search query and keep the selection sane.
 *
 * Selection rules (HW3):
 *   - the previously selected movie is preserved if it is still among the matches;
 *   - otherwise the first available match becomes the selection;
 *   - if nothing matches the select is emptied and the caller shows a clear state.
 *
 * Returns the number of matches that are currently rendered.
 */
function renderMovieOptions(query) {
    const movieSelect = document.getElementById('movie-select');
    const previousId = movieSelect.value;
    const needle = (query || '').trim().toLowerCase();

    const matches = needle === ''
        ? movieSearchIndex
        : movieSearchIndex.filter(entry => entry.titleLower.includes(needle));

    movieSelect.innerHTML = '';
    matches.forEach(entry => {
        const option = document.createElement('option');
        option.value = entry.id;
        option.textContent = entry.label;
        movieSelect.appendChild(option);
    });

    const stillVisible = matches.some(entry => String(entry.id) === previousId);
    if (stillVisible) {
        movieSelect.value = previousId;
    } else if (matches.length > 0) {
        // The browser already selected the first option, but be explicit.
        movieSelect.value = String(matches[0].id);
    } else {
        movieSelect.value = '';
    }

    updateSearchCount(matches.length, needle);
    return matches.length;
}

function updateSearchCount(count, needle) {
    const el = document.getElementById('search-count');
    if (count === 0) {
        el.textContent = needle ? `No movies found for "${needle}"` : 'No movies found';
        el.className = 'search-count empty';
    } else {
        const word = count === 1 ? 'movie' : 'movies';
        el.textContent = `${count} ${word} found`;
        el.className = 'search-count';
    }
    document.getElementById('search-clear').hidden = needle === '';
}

/** Fired on every keystroke of the search field. */
function onMovieSearchInput() {
    const input = document.getElementById('movie-search');
    const previousMovieId = document.getElementById('movie-select').value;

    renderMovieOptions(input.value);

    const newMovieId = document.getElementById('movie-select').value;
    handleMovieSelectionChanged(previousMovieId, newMovieId);
}

function clearMovieSearch() {
    const input = document.getElementById('movie-search');
    input.value = '';
    input.focus();
    onMovieSearchInput();
}

/**
 * If the effective selection moved because of a UI action, re-run the prediction
 * so the result cards never describe a different movie than the one shown in the
 * dropdown. If nothing changed, do nothing (typing must not spam the console).
 */
function handleMovieSelectionChanged(previousMovieId, newMovieId) {
    if (newMovieId === '' ) {
        lastRenderedMovieId = null;
        setResultCard('result-user', 'User-Based CF',
            '<div class="result-body">No movie selected - adjust the search filter.</div>', 'pending');
        setResultCard('result-item', 'Item-Based CF',
            '<div class="result-body">No movie selected - adjust the search filter.</div>', 'pending');
        setResultCard('result-compare', 'Comparison',
            '<div class="result-body">No movie selected.</div>', 'pending');
        return;
    }
    if (previousMovieId === newMovieId) return;
    if (newMovieId === lastRenderedMovieId) return;

    lastRenderedMovieId = newMovieId;
    predictRating();
}

/**
 * Quick example buttons: pick the user, type the movie name into the search box,
 * apply the filter and select the target movie. The prediction is refreshed so the
 * cards are never stale, but pressing Predict Rating again is still harmless.
 */
function applyQuickExample(index) {
    const example = QUICK_EXAMPLES[index];
    if (!example) return;

    document.getElementById('user-select').value = String(example.userId);
    document.getElementById('movie-search').value = example.search;

    const shown = renderMovieOptions(example.search);

    const movieSelect = document.getElementById('movie-select');
    const found = movieSearchIndex.some(
        entry => entry.id === example.movieId && entry.titleLower.includes(example.search.toLowerCase())
    );
    if (found) {
        movieSelect.value = String(example.movieId);
    } else {
        console.warn(
            `[HW3] Quick example ${example.key}: movie ${example.movieId} is not among the ` +
            `${shown} search result(s) for "${example.search}". Falling back to the first result.`
        );
    }

    const userRec = ratingsByUser.get(example.userId);
    const alreadyRated = !!userRec && userRec.ratings.has(example.movieId);

    console.info(
        `[HW3] Quick example ${example.key}: User ${example.userId} + "${findMovieById(movieSelect.value)}" ` +
        `(movie ${movieSelect.value}), ${shown} search result(s), ` +
        `target cell ${alreadyRated ? 'ALREADY RATED' : 'genuinely missing'}.`
    );
    if (alreadyRated) {
        console.warn(
            `[HW3] Quick example ${example.key} is expected to be a genuinely missing cell, ` +
            `but User ${example.userId} has already rated movie ${example.movieId}. ` +
            `The prediction will still be leakage-safe, but it is no longer a missing-value demo.`
        );
    }

    lastRenderedMovieId = movieSelect.value;
    predictRating();
}

function setDiagnosticsVisible(isVisible) {
    const panel = document.getElementById('diagnostics');
    const button = document.getElementById('diagnostics-toggle');
    panel.classList.toggle('visible', isVisible);
    button.textContent = isVisible ? 'Hide diagnostics' : 'Show diagnostics';
    button.setAttribute('aria-expanded', isVisible ? 'true' : 'false');
}

function toggleDiagnostics() {
    const panel = document.getElementById('diagnostics');
    setDiagnosticsVisible(!panel.classList.contains('visible'));
}

function populateUserDropdown() {
    const userSelect = document.getElementById('user-select');
    userSelect.innerHTML = '';

    // Add users (assuming user IDs are sequential from 1 to numUsers)
    for (let i = 1; i <= numUsers; i++) {
        const option = document.createElement('option');
        option.value = i;
        option.textContent = `User ${i}`;
        userSelect.appendChild(option);
    }
}

function populateMovieDropdown() {
    renderMovieOptions('');
}


// ===========================================================================
// Similarity
// ===========================================================================

/**
 * Pearson correlation computed ONLY on the co-rated movies.
 * Both profile means are the means of the co-rated subset, which is the
 * standard definition of Pearson between two rating profiles.
 *
 * Returns { commonCount, pearson } with pearson === null when the correlation is
 * undefined for this pair (fewer than 2 co-rated movies, or one of the two
 * profiles is constant over the co-rated subset -> zero variance).
 */
function pearsonOnCoRated(userRec, otherRec) {
    const aIds = userRec.itemIds;
    const bIds = otherRec.itemIds;

    // Two-pointer merge over the ascending id arrays = linear co-rated intersection.
    const xa = [];
    const xb = [];
    let i = 0;
    let j = 0;
    while (i < aIds.length && j < bIds.length) {
        const aId = aIds[i];
        const bId = bIds[j];
        if (aId === bId) {
            xa.push(userRec.ratings.get(aId));
            xb.push(otherRec.ratings.get(bId));
            i++;
            j++;
        } else if (aId < bId) {
            i++;
        } else {
            j++;
        }
    }

    const commonCount = xa.length;
    if (commonCount < 2) return { commonCount, pearson: null };

    let meanA = 0;
    let meanB = 0;
    for (let k = 0; k < commonCount; k++) {
        meanA += xa[k];
        meanB += xb[k];
    }
    meanA /= commonCount;
    meanB /= commonCount;

    let cov = 0;
    let varA = 0;
    let varB = 0;
    for (let k = 0; k < commonCount; k++) {
        const da = xa[k] - meanA;
        const db = xb[k] - meanB;
        cov += da * db;
        varA += da * da;
        varB += db * db;
    }

    if (varA === 0 || varB === 0) return { commonCount, pearson: null };

    return { commonCount, pearson: cov / Math.sqrt(varA * varB) };
}

/**
 * Reliability weighting: shrink a raw correlation by how much co-rated
 * evidence supports it. 1.0 at RELIABILITY_SATURATION co-rated movies and above,
 * linear decay below that.
 */
function computeWeightedSimilarity(rawPearson, commonCount) {
    return rawPearson * Math.min(commonCount / RELIABILITY_SATURATION, 1);
}


// ===========================================================================
// User-Based CF
// ===========================================================================

/**
 * All users who rated the target movie, scored against the target user.
 * Candidates are restricted to users with at least MIN_OVERLAP co-rated movies.
 * Returned sorted by weighted similarity, descending.
 */
function findUserNeighbors(targetRec, itemRec, targetUserId) {
    const candidates = [];
    let scanned = 0;
    let rejectedOverlap = 0;
    let rejectedUndefined = 0;

    for (const otherId of itemRec.userIds) {
        if (otherId === targetUserId) continue;
        scanned++;

        const otherRec = ratingsByUser.get(otherId);
        const { commonCount, pearson } = pearsonOnCoRated(targetRec, otherRec);

        if (pearson === null) {
            rejectedUndefined++;
            continue;
        }
        if (commonCount < MIN_OVERLAP) {
            rejectedOverlap++;
            continue;
        }

        candidates.push({
            userId: otherId,
            commonCount,
            rawPearson: pearson,
            weighted: computeWeightedSimilarity(pearson, commonCount),
            rating: otherRec.ratings.get(itemRec.id)
        });
    }

    candidates.sort((a, b) => b.weighted - a.weighted);
    return { candidates, scanned, rejectedOverlap, rejectedUndefined };
}

function clampRating(value) {
    if (!Number.isFinite(value)) return null;
    return Math.min(MAX_PREDICTION, Math.max(MIN_PREDICTION, value));
}

/**
 * Copy the leakage bookkeeping from the currently held-out cell into a result
 * object. A pair that was never rated is a genuine missing-value prediction and
 * needs no correction, so all three fields stay at their false / null defaults.
 */
function recordTargetCell(result, userId, movieId) {
    if (!isHeldOut(userId, movieId)) return result;
    result.targetWasRated = true;
    result.actualRating = getHeldOutTarget().rating;
    result.leaveOneOutApplied = true;
    return result;
}

/**
 * The single entry point every prediction goes through.
 *
 * Both methods are computed INSIDE withTargetCellHeldOut(), so when the selected
 * pair already has a real rating in u.data that rating is removed from both
 * profiles first: it cannot reach the User-Based baseline, the Item-Based
 * baseline, or any co-rated Pearson intersection. The indexes are restored
 * afterwards, including on error, and the flat `ratings` array is never touched.
 *
 * When the pair was never rated this is an ordinary missing-value prediction and
 * the hold-out is a no-op.
 */
function predictLeakageSafe(userId, movieId) {
    return withTargetCellHeldOut(userId, movieId, () => {
        const userResult = predictUserBased(userId, movieId);
        const itemResult = predictItemBased(userId, movieId);
        return { userBased: userResult, itemBased: itemResult };
    });
}

/**
 * User-Based CF prediction with the co-rated only + reliability weighting
 * strategy. Always returns a result object; it never throws and never invents
 * a number when the evidence is too thin.
 */
function predictUserBased(userId, movieId) {
    const result = {
        ok: false,
        status: 'ok',
        message: '',
        prediction: null,
        userId,
        movieId,
        movieTitle: getMovieTitle(movieId),
        userProfile: null,
        itemProfile: null,
        scanned: 0,
        rejectedOverlap: 0,
        rejectedUndefined: 0,
        candidateCount: 0,
        voters: [],
        topNeighbor: null,
        evidence: 0,
        // leakage bookkeeping - filled in from the held-out cell, if any
        targetWasRated: false,
        actualRating: null,
        leaveOneOutApplied: false
    };

    const userRec = ratingsByUser.get(userId);
    const itemRec = ratingsByItem.get(movieId);

    recordTargetCell(result, userId, movieId);

    // --- cold start guards ------------------------------------------------
    if (!userRec) {
        result.status = 'cold-user';
        result.message = `User ${userId} has no rating history at all (new user). ` +
            `User-based CF cannot produce a prediction without past ratings.`;
        return result;
    }
    if (!itemRec) {
        result.status = 'cold-item';
        result.message = `"${result.movieTitle}" has no ratings from anyone (unseen movie). ` +
            `User-based CF needs at least a few users who rated it.`;
        return result;
    }

    result.userProfile = snapshotProfile(userRec);
    result.itemProfile = snapshotProfile(itemRec);

    if (userRec.count < MIN_USER_HISTORY) {
        result.status = 'cold-user';
        result.message = `User ${userId} rated only ${userRec.count} movie(s) ` +
            `(minimum required: ${MIN_USER_HISTORY}). Too little history - no prediction.`;
        return result;
    }
    if (itemRec.count < MIN_ITEM_HISTORY) {
        result.status = 'cold-item';
        result.message = `"${result.movieTitle}" was rated by only ${itemRec.count} user(s) ` +
            `(minimum required: ${MIN_ITEM_HISTORY}). Not enough evidence - no prediction.`;
        return result;
    }

    // --- neighbour search -------------------------------------------------
    const search = findUserNeighbors(userRec, itemRec, userId);
    result.scanned = search.scanned;
    result.rejectedOverlap = search.rejectedOverlap;
    result.rejectedUndefined = search.rejectedUndefined;
    result.candidateCount = search.candidates.length;

    if (search.candidates.length === 0) {
        result.status = 'no-neighbors';
        result.message = `No user rated "${result.movieTitle}" with at least ` +
            `${MIN_OVERLAP} movie(s) in common with User ${userId} ` +
            `(${search.scanned} user(s) checked, ` +
            `${search.rejectedOverlap} rejected for insufficient overlap).`;
        return result;
    }

    // --- weighted voting --------------------------------------------------
    const voters = search.candidates
        .filter(n => (REQUIRE_POSITIVE_SIMILARITY ? n.weighted > 0 : true))
        .slice(0, TOP_K_NEIGHBORS);

    if (voters.length === 0) {
        result.status = 'no-neighbors';
        result.message = `Found ${search.candidates.length} candidate neighbour(s) for ` +
            `User ${userId} and "${result.movieTitle}", but none has a positive weighted ` +
            `similarity, so there is no usable evidence.`;
        return result;
    }

    let numerator = 0;
    let evidence = 0;
    for (const neighbor of voters) {
        numerator += neighbor.weighted * (neighbor.rating - ratingsByUser.get(neighbor.userId).mean);
        evidence += Math.abs(neighbor.weighted);
    }
    result.voters = voters;
    result.evidence = evidence;
    result.topNeighbor = voters[0];

    if (evidence < MIN_EVIDENCE) {
        result.status = 'weak-evidence';
        result.message = `Evidence too weak: ${voters.length} neighbour(s) survived, the most ` +
            `similar one sharing only ${voters[0].commonCount} co-rated movie(s) ` +
            `(weighted similarity ${voters[0].weighted.toFixed(3)}), total evidence only ` +
            `${evidence.toFixed(3)} (minimum: ${MIN_EVIDENCE}). No prediction.`;
        return result;
    }

    // Prediction = target user's own mean, corrected by the neighbours' deviation.
    const raw = userRec.mean + numerator / evidence;
    const prediction = clampRating(raw);
    if (prediction === null) {
        result.status = 'weak-evidence';
        result.message = 'Computation produced a non-finite value. No prediction.';
        return result;
    }

    result.ok = true;
    result.prediction = prediction;
    result.clamped = raw !== prediction;
    result.message = `${voters.length} neighbour(s) voted.`;
    return result;
}


// ===========================================================================
// Item-Based CF
// ===========================================================================

/**
 * Pearson correlation between two MOVIES, computed only over the users who
 * rated both of them. Structurally identical to pearsonOnCoRated, but the
 * intersection axis is the user list instead of the item list.
 *
 * Returns { commonCount, pearson } with pearson === null when the correlation is
 * undefined (fewer than 2 co-rated users, or one movie is rated with a constant
 * score by the co-rated users -> zero variance).
 */
function pearsonOnCoRatedItems(itemRec, otherItemRec) {
    const aIds = itemRec.userIds;
    const bIds = otherItemRec.userIds;

    const xa = [];
    const xb = [];
    let i = 0;
    let j = 0;
    while (i < aIds.length && j < bIds.length) {
        const aId = aIds[i];
        const bId = bIds[j];
        if (aId === bId) {
            // aId / bId is a user who rated BOTH movies
            xa.push(itemRec.ratings.get(aId));
            xb.push(otherItemRec.ratings.get(bId));
            i++;
            j++;
        } else if (aId < bId) {
            i++;
        } else {
            j++;
        }
    }

    const commonCount = xa.length;
    if (commonCount < 2) return { commonCount, pearson: null };

    let meanA = 0;
    let meanB = 0;
    for (let k = 0; k < commonCount; k++) {
        meanA += xa[k];
        meanB += xb[k];
    }
    meanA /= commonCount;
    meanB /= commonCount;

    let cov = 0;
    let varA = 0;
    let varB = 0;
    for (let k = 0; k < commonCount; k++) {
        const da = xa[k] - meanA;
        const db = xb[k] - meanB;
        cov += da * db;
        varA += da * da;
        varB += db * db;
    }

    if (varA === 0 || varB === 0) return { commonCount, pearson: null };

    return { commonCount, pearson: cov / Math.sqrt(varA * varB) };
}

/**
 * Movies the target user has already rated, scored against the target movie.
 * Only the movies with at least MIN_OVERLAP co-rated users survive.
 * Returned sorted by weighted similarity, descending.
 */
function findItemNeighbors(targetItemRec, userRec, targetMovieId) {
    const candidates = [];
    let scanned = 0;
    let rejectedOverlap = 0;
    let rejectedUndefined = 0;

    for (const otherMovieId of userRec.itemIds) {
        if (otherMovieId === targetMovieId) continue;
        scanned++;

        const otherRec = ratingsByItem.get(otherMovieId);
        if (!otherRec) continue;

        const { commonCount, pearson } = pearsonOnCoRatedItems(targetItemRec, otherRec);

        if (pearson === null) {
            rejectedUndefined++;
            continue;
        }
        if (commonCount < MIN_OVERLAP) {
            rejectedOverlap++;
            continue;
        }

        candidates.push({
            movieId: otherMovieId,
            commonCount,
            rawPearson: pearson,
            weighted: computeWeightedSimilarity(pearson, commonCount),
            // the rating the TARGET USER gave this neighbour movie
            userRating: otherRec.ratings.get(userRec.id)
        });
    }

    candidates.sort((a, b) => b.weighted - a.weighted);
    return { candidates, scanned, rejectedOverlap, rejectedUndefined };
}

/**
 * Item-Based CF prediction.
 *
 * Formula (as taught in the lecture):
 *
 *   prediction = mean(target movie)
 *              + SUM  weightedSimilarity(target, neighbour)
 *                       * (userRating(neighbour) - mean(neighbour))
 *                / SUM |weightedSimilarity(target, neighbour)|
 *
 * The baseline is the target movie's mean rating, NOT the target user's mean.
 * Similarity is Pearson over co-rated users only, shrunk towards 0 by the
 * reliability weight min(commonUsers / 50, 1). Never throws; returns
 * prediction === null with an explanatory status instead of falling back to
 * any prior.
 */
function predictItemBased(userId, movieId) {
    const result = {
        ok: false,
        status: 'ok',
        message: '',
        prediction: null,
        userId,
        movieId,
        movieTitle: getMovieTitle(movieId),
        userProfile: null,
        itemProfile: null,
        scanned: 0,
        rejectedOverlap: 0,
        rejectedUndefined: 0,
        candidateCount: 0,
        voters: [],
        topNeighbor: null,
        evidence: 0,
        // lecture formula: prediction = itemRec.mean + correction
        baseline: null,
        correction: 0,
        // leakage bookkeeping - filled in from the held-out cell, if any
        targetWasRated: false,
        actualRating: null,
        leaveOneOutApplied: false
    };

    const userRec = ratingsByUser.get(userId);
    const itemRec = ratingsByItem.get(movieId);

    recordTargetCell(result, userId, movieId);

    // --- cold start guards (same set as the user-based side) ---------------
    if (!userRec) {
        result.status = 'cold-user';
        result.message = `User ${userId} has no rating history at all (new user). ` +
            `Item-based CF cannot find a single rated movie to compare with.`;
        return result;
    }
    if (!itemRec) {
        result.status = 'cold-item';
        result.message = `"${result.movieTitle}" has no ratings from anyone (unseen movie). ` +
            `Item-based CF cannot compute a similarity for it.`;
        return result;
    }

    result.userProfile = snapshotProfile(userRec);
    result.itemProfile = snapshotProfile(itemRec);

    if (userRec.count < MIN_USER_HISTORY) {
        result.status = 'cold-user';
        result.message = `User ${userId} rated only ${userRec.count} movie(s) ` +
            `(minimum required: ${MIN_USER_HISTORY}). Too little history - no prediction.`;
        return result;
    }
    if (itemRec.count < MIN_ITEM_HISTORY) {
        result.status = 'cold-item';
        result.message = `"${result.movieTitle}" was rated by only ${itemRec.count} user(s) ` +
            `(minimum required: ${MIN_ITEM_HISTORY}). Not enough evidence - no prediction.`;
        return result;
    }

    // --- neighbour search over the movies this user has already rated ------
    const search = findItemNeighbors(itemRec, userRec, movieId);
    result.scanned = search.scanned;
    result.rejectedOverlap = search.rejectedOverlap;
    result.rejectedUndefined = search.rejectedUndefined;
    result.candidateCount = search.candidates.length;

    if (search.candidates.length === 0) {
        result.status = 'no-neighbors';
        result.message = `None of the ${search.scanned} movie(s) User ${userId} has rated ` +
            `shares at least ${MIN_OVERLAP} co-rated user(s) with "${result.movieTitle}" ` +
            `(${search.rejectedOverlap} rejected for insufficient overlap).`;
        return result;
    }

    // --- weighted voting --------------------------------------------------
    const voters = search.candidates
        .filter(n => (REQUIRE_POSITIVE_SIMILARITY ? n.weighted > 0 : true))
        .slice(0, TOP_K_NEIGHBORS);

    if (voters.length === 0) {
        result.status = 'no-neighbors';
        result.message = `Found ${search.candidates.length} similar movie(s) for ` +
            `"${result.movieTitle}" based on User ${userId}'s history, but none has a positive ` +
            `weighted similarity, so there is no usable evidence.`;
        return result;
    }

    let numerator = 0;
    let evidence = 0;
    for (const neighbor of voters) {
        const neighborRec = ratingsByItem.get(neighbor.movieId);
        numerator += neighbor.weighted * (neighbor.userRating - neighborRec.mean);
        evidence += Math.abs(neighbor.weighted);
    }
    result.voters = voters;
    result.evidence = evidence;
    result.topNeighbor = voters[0];

    if (evidence < MIN_EVIDENCE) {
        result.status = 'weak-evidence';
        result.message = `Evidence too weak: ${voters.length} similar movie(s) survived, the most ` +
            `similar one sharing only ${voters[0].commonCount} co-rated user(s) ` +
            `(weighted similarity ${voters[0].weighted.toFixed(3)}), total evidence only ` +
            `${evidence.toFixed(3)} (minimum: ${MIN_EVIDENCE}). No prediction.`;
        return result;
    }

    // Lecture formula (item-based collaborative filtering):
    //
    //   prediction = mean(target movie)
    //              + SUM  weightedSimilarity(target, neighbour)
    //                       * (userRating(neighbour) - mean(neighbour))
    //                / SUM |weightedSimilarity(target, neighbour)|
    //
    // The baseline is the TARGET MOVIE's mean rating (itemRec.mean), not the
    // target user's mean. Each neighbour contributes its own deviation from its
    // own item mean, and the whole correction is normalised by the total
    // absolute similarity (evidence). numerator/evidence above is that
    // correction term, so only the baseline is added here.
    result.baseline = itemRec.mean;
    result.correction = numerator / evidence;
    const raw = itemRec.mean + result.correction;
    const prediction = clampRating(raw);
    if (prediction === null) {
        result.status = 'weak-evidence';
        result.message = 'Computation produced a non-finite value. No prediction.';
        return result;
    }

    result.ok = true;
    result.prediction = prediction;
    result.clamped = raw !== prediction;
    result.message = `${voters.length} similar movie(s) voted.`;
    return result;
}


// ===========================================================================
// Prediction entry point (bound to the button via onclick in index.html)
// ===========================================================================

function predictRating() {
    if (!dataReady) {
        setResultCard('result-user', 'User-Based CF',
            '<div class="result-body">Data is still loading. Please wait...</div>', 'warning');
        setResultCard('result-item', 'Item-Based CF',
            '<div class="result-body">Data is still loading. Please wait...</div>', 'warning');
        setResultCard('result-compare', 'Comparison',
            '<div class="result-body">Data is still loading. Please wait...</div>', 'pending');
        return;
    }

    const userId = parseInt(document.getElementById('user-select').value, 10);
    const movieId = parseInt(document.getElementById('movie-select').value, 10);

    if (!Number.isInteger(userId) || !Number.isInteger(movieId)) {
        lastRenderedMovieId = null;
        setResultCard('result-user', 'User-Based CF',
            '<div class="result-body">Please select both a user and a movie.</div>', 'warning');
        setResultCard('result-item', 'Item-Based CF',
            '<div class="result-body">Please select both a user and a movie.</div>', 'warning');
        setResultCard('result-compare', 'Comparison',
            '<div class="result-body">Nothing to compare yet.</div>', 'pending');
        return;
    }

    lastRenderedMovieId = String(movieId);

    // Both methods are computed INSIDE the same target-cell hold-out, so neither
    // one can see the real rating of the pair being predicted. The hold-out is a
    // no-op when the pair was never rated.
    let bundle;
    try {
        bundle = predictLeakageSafe(userId, movieId);
    } catch (error) {
        console.error('Prediction error:', error);
        setResultCard('result-user', 'User-Based CF',
            `<div class="result-body">Error: ${escapeHtml(error.message)}</div>`, 'warning');
        setResultCard('result-item', 'Item-Based CF',
            `<div class="result-body">Error: ${escapeHtml(error.message)}</div>`, 'warning');
        setResultCard('result-compare', 'Comparison',
            '<div class="result-body">Prediction failed.</div>', 'warning');
        return;
    }

    const userResult = bundle.userBased;
    const itemResult = bundle.itemBased;

    // The two methods stay independent: one may fail while the other still has
    // enough evidence, and the surviving result must still be shown.
    try {
        logUserBasedDiagnostics(userResult);
        renderUserBasedResult(userResult);
    } catch (error) {
        console.error('User-Based rendering error:', error);
        setResultCard('result-user', 'User-Based CF',
            `<div class="result-body">Error: ${escapeHtml(error.message)}</div>`, 'warning');
    }

    try {
        logItemBasedDiagnostics(itemResult);
        renderItemBasedResult(itemResult);
    } catch (error) {
        console.error('Item-Based rendering error:', error);
        setResultCard('result-item', 'Item-Based CF',
            `<div class="result-body">Error: ${escapeHtml(error.message)}</div>`, 'warning');
    }

    renderLeakageNote(userResult, itemResult);
    renderComparison(userResult, itemResult);
    renderDiagnostics(userResult, itemResult);
}


// ===========================================================================
// Console diagnostics
// ===========================================================================

function logUserBasedDiagnostics(result) {
    const group = `[User-Based CF] User ${result.userId} -> "${result.movieTitle}" (movie ${result.movieId})`;

    console.group(group);

    console.log('Selected user            :', result.userId, result.userProfile
        ? `(${result.userProfile.count} ratings, mean ${result.userProfile.mean.toFixed(3)})`
        : '(profile not found)');
    console.log('Selected movie           :', result.movieId, `"${result.movieTitle}"`, result.itemProfile
        ? `(${result.itemProfile.count} ratings, mean ${result.itemProfile.mean.toFixed(3)})`
        : '(no ratings)');

    if (result.userProfile && result.itemProfile) {
        console.log('Number of candidate neighbors:', result.scanned, '(users who rated the target movie)');
        console.log(`  rejected, overlap < MIN_OVERLAP (${MIN_OVERLAP}) :`, result.rejectedOverlap);
        console.log('  rejected, undefined Pearson (zero variance)   :', result.rejectedUndefined);
        console.log('  passed the overlap gate                      :', result.candidateCount);
    }

    if (result.topNeighbor) {
        const top = result.topNeighbor;
        console.log('Top neighbour raw Pearson  :', top.rawPearson.toFixed(4));
        console.log('Top neighbour co-rated count:', top.commonCount);
        console.log(
            'Top neighbour weighted sim :', top.weighted.toFixed(4),
            `= ${top.rawPearson.toFixed(4)} * min(${top.commonCount}/${RELIABILITY_SATURATION}, 1)`
        );
        console.log('Total evidence (sum |weighted similarity|):', result.evidence.toFixed(4),
            `over ${result.voters.length} voter(s)`);

        const rows = result.voters.slice(0, 20).map(n => ({
            user: n.userId,
            coRated: n.commonCount,
            rawPearson: Number(n.rawPearson.toFixed(4)),
            weightedSim: Number(n.weighted.toFixed(4)),
            theirRating: n.rating,
            theirMean: Number(ratingsByUser.get(n.userId).mean.toFixed(3)),
            deviation: Number((n.rating - ratingsByUser.get(n.userId).mean).toFixed(3))
        }));
        console.table(rows);
        if (result.voters.length > 20) {
            console.log(`... ${result.voters.length - 20} more voters not shown`);
        }
    }

    if (result.ok) {
        console.log('Final prediction          :', result.prediction.toFixed(2), '/ 5');
        if (result.clamped) {
            console.log('  (clamped into the valid 1..5 range)');
        }
    } else {
        console.warn('Final prediction          : NONE -', result.status, '|', result.message);
    }

    logTargetCell(result);
    console.groupEnd();
}


/**
 * Makes the leakage handling visible in the console, so a reader can tell
 * whether the predicted pair was already known in u.data and whether the
 * target cell was withheld before anything was computed.
 */
function logTargetCell(result) {
    if (result.targetWasRated) {
        console.log('Target was previously rated:', 'yes', `(actual rating ${result.actualRating})`);
        console.log('Leave-one-out applied     :', 'yes',
            '| the target cell was removed from both profiles before this prediction');
    } else {
        console.log('Target was previously rated:', 'no (missing cell)');
        console.log('Leave-one-out applied     :', 'no (nothing to withhold)');
    }
}


function logItemBasedDiagnostics(result) {
    const group = `[Item-Based CF] User ${result.userId} -> "${result.movieTitle}" (movie ${result.movieId})`;

    console.group(group);

    console.log('Selected user            :', result.userId, result.userProfile
        ? `(${result.userProfile.count} rated movies, mean ${result.userProfile.mean.toFixed(3)})`
        : '(profile not found)');
    console.log('Selected movie           :', result.movieId, `"${result.movieTitle}"`, result.itemProfile
        ? `(${result.itemProfile.count} ratings, mean ${result.itemProfile.mean.toFixed(3)})`
        : '(no ratings)');

    if (result.userProfile && result.itemProfile) {
        console.log('Rated items scanned      :', result.scanned,
            '(movies of this user, excluding the target movie)');
        console.log(`  rejected, overlap < MIN_OVERLAP (${MIN_OVERLAP}) :`, result.rejectedOverlap);
        console.log('  rejected, undefined Pearson (zero variance)   :', result.rejectedUndefined);
        console.log('  candidate similar items                      :', result.candidateCount);
    }

    if (result.topNeighbor) {
        const top = result.topNeighbor;
        console.log('Top item raw Pearson     :', top.rawPearson.toFixed(4));
        console.log('Top item co-rated users  :', top.commonCount);
        console.log(
            'Top item weighted sim    :', top.weighted.toFixed(4),
            `= ${top.rawPearson.toFixed(4)} * min(${top.commonCount}/${RELIABILITY_SATURATION}, 1)`
        );
        console.log('User rating on top item  :', top.userRating, `("${getMovieTitle(top.movieId)}")`);
        console.log('Total evidence (sum |weighted similarity|):', result.evidence.toFixed(4),
            `over ${result.voters.length} similar item(s)`);

        const rows = result.voters.slice(0, 20).map(n => {
            const rec = ratingsByItem.get(n.movieId);
            return {
                movie: n.movieId,
                title: getMovieTitle(n.movieId),
                coRatedUsers: n.commonCount,
                rawPearson: Number(n.rawPearson.toFixed(4)),
                weightedSim: Number(n.weighted.toFixed(4)),
                userRating: n.userRating,
                movieMean: Number(rec.mean.toFixed(3)),
                deviation: Number((n.userRating - rec.mean).toFixed(3))
            };
        });
        console.table(rows);
        if (result.voters.length > 20) {
            console.log(`... ${result.voters.length - 20} more similar items not shown`);
        }
    }

    if (result.ok) {
        console.log('Final prediction          :', result.prediction.toFixed(2), '/ 5');
        if (result.clamped) {
            console.log('  (clamped into the valid 1..5 range)');
        }
    } else {
        console.warn('Final prediction          : NONE -', result.status, '|', result.message);
    }

    logTargetCell(result);
    console.groupEnd();
}


// ===========================================================================
// UI rendering
// ===========================================================================

function escapeHtml(value) {
    return String(value)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;');
}

function ratingClassFor(value) {
    if (value >= 4) return 'high';
    if (value <= 2) return 'low';
    return 'medium';
}

function setResultCard(elementId, label, html, className = '') {
    const el = document.getElementById(elementId);
    el.innerHTML = `<div class="result-label">${label}</div>` + html;
    el.className = 'result-card ' + className;
}

function renderUserBasedResult(result) {
    if (!result) return;

    if (!result.ok) {
        setResultCard('result-user', 'User-Based CF',
            '<div class="result-body">No prediction</div>' +
            `<div class="result-body">${escapeHtml(result.message)}</div>`,
            'warning'
        );
        return;
    }

    setResultCard('result-user', 'User-Based CF',
        `<div class="result-value">${result.prediction.toFixed(2)}<span class="result-of">/5</span></div>` +
        `<div class="result-body">User ${result.userId} on "${escapeHtml(result.movieTitle)}": ` +
        `${result.voters.length} user neighbour(s) voted, ` +
        `total weighted similarity ${result.evidence.toFixed(2)}.</div>`,
        ratingClassFor(result.prediction)
    );
}

/**
 * Small neutral line under the comparison card telling the reader whether the
 * predicted pair was already known in the data, and what was done about it.
 * Factual only - it never claims one method is better than the other.
 */
function renderLeakageNote(userResult, itemResult) {
    const el = document.getElementById('leakage-note');
    if (!el) return;

    const result = itemResult || userResult;
    if (!result) {
        el.textContent = '';
        el.className = 'leakage-note';
        return;
    }

    if (result.targetWasRated) {
        el.textContent =
            `Known rating: ${result.actualRating}. ` +
            'Prediction is computed with leave-one-out to avoid target leakage.';
        el.className = 'leakage-note known';
    } else {
        el.textContent =
            'Target rating is missing; this is a genuine recommendation prediction.';
        el.className = 'leakage-note missing';
    }
}

function renderItemBasedResult(result) {
    if (!result) return;

    if (!result.ok) {
        setResultCard('result-item', 'Item-Based CF',
            '<div class="result-body">No prediction</div>' +
            `<div class="result-body">${escapeHtml(result.message)}</div>`,
            'warning'
        );
        return;
    }

    setResultCard('result-item', 'Item-Based CF',
        `<div class="result-value">${result.prediction.toFixed(2)}<span class="result-of">/5</span></div>` +
        `<div class="result-body">User ${result.userId} on "${escapeHtml(result.movieTitle)}": ` +
        `${result.voters.length} similar movie(s) voted, ` +
        `total weighted similarity ${result.evidence.toFixed(2)}.</div>`,
        ratingClassFor(result.prediction)
    );
}

/**
 * Side-by-side verdict. Both predictions are reported exactly as computed and the
 * gap is reported as a plain absolute difference. Neither method is presented as
 * more accurate than the other here - deciding that requires an offline
 * evaluation (RMSE/MAE on a held-out split), not a single pair.
 */
function renderComparison(userResult, itemResult) {
    const NEUTRAL_NOTE =
        'The methods use different neighbourhoods, so predictions may differ.';

    if (!userResult || !itemResult) {
        setResultCard('result-compare', 'Comparison',
            `<div class="result-body">Nothing to compare yet.</div>` +
            `<div class="result-note">${NEUTRAL_NOTE}</div>`, 'pending');
        return;
    }

    const userText = userResult.ok
        ? `<strong>${userResult.prediction.toFixed(2)}</strong>/5`
        : `<span class="result-none">no prediction (${escapeHtml(userResult.status)})</span>`;
    const itemText = itemResult.ok
        ? `<strong>${itemResult.prediction.toFixed(2)}</strong>/5`
        : `<span class="result-none">no prediction (${escapeHtml(itemResult.status)})</span>`;

    let body =
        `<div class="result-body">User-Based CF: ${userText}</div>` +
        `<div class="result-body">Item-Based CF: ${itemText}</div>`;

    if (userResult.ok && itemResult.ok) {
        const absDiff = Math.abs(itemResult.prediction - userResult.prediction);
        body += `<div class="result-body">Absolute difference: ` +
            `<strong>${absDiff.toFixed(2)}</strong></div>`;
    } else {
        body += `<div class="result-body">Absolute difference: n/a ` +
            `(at least one method returned no prediction)</div>`;
    }

    body += `<div class="result-note">${NEUTRAL_NOTE}</div>`;

    const cardClass = (userResult.ok && itemResult.ok) ? 'neutral' : 'warning';
    setResultCard('result-compare', 'Comparison', body, cardClass);
}

function renderDiagnostics(userResult, itemResult) {
    const el = document.getElementById('diagnostics');
    const rows = [];
    const anyResult = itemResult || userResult;

    if (anyResult) {
        rows.push(['-- Target cell --', '']);
        rows.push(['Target was previously rated', anyResult.targetWasRated ? 'yes' : 'no']);
        rows.push(['Leave-one-out applied', anyResult.leaveOneOutApplied ? 'yes' : 'no']);
        if (anyResult.targetWasRated) {
            rows.push(['Actual rating (withheld)', anyResult.actualRating]);
        }
    }

    if (userResult && userResult.userProfile && userResult.itemProfile) {
        rows.push(['-- User-Based CF --', '']);
        rows.push(['Candidates checked', `${userResult.scanned} user(s) rated this movie`]);
        rows.push([`Rejected, overlap < ${MIN_OVERLAP}`, userResult.rejectedOverlap]);
        rows.push(['Rejected, undefined Pearson', userResult.rejectedUndefined]);
        rows.push(['Passed overlap gate', userResult.candidateCount]);
        rows.push([`Voters used (top ${TOP_K_NEIGHBORS})`, userResult.voters.length]);
        rows.push(['Baseline, target user mean', userResult.userProfile.mean.toFixed(3)]);
        rows.push(['Total evidence', userResult.evidence.toFixed(3)]);
    }

    if (itemResult && itemResult.userProfile && itemResult.itemProfile) {
        rows.push(['-- Item-Based CF --', '']);
        if (itemResult.baseline !== null) {
            rows.push(['Baseline, target item mean', itemResult.baseline.toFixed(3)]);
            rows.push(['Weighted correction term', itemResult.correction.toFixed(3)]);
        }
        rows.push(['Rated items scanned', itemResult.scanned]);
        rows.push([`Rejected, overlap < ${MIN_OVERLAP}`, itemResult.rejectedOverlap]);
        rows.push(['Rejected, undefined Pearson', itemResult.rejectedUndefined]);
        rows.push(['Candidate similar items', itemResult.candidateCount]);
        rows.push([`Voters used (top ${TOP_K_NEIGHBORS})`, itemResult.voters.length]);
        rows.push(['Total evidence', itemResult.evidence.toFixed(3)]);
    }

    if (rows.length === 0) {
        el.innerHTML = '<div class="diagnostics-title">Diagnostics</div>' +
            '<div class="diagnostics-row"><span>No prediction has been run yet</span><span>-</span></div>';
        setDiagnosticsVisible(true);
        return;
    }

    el.innerHTML = '<div class="diagnostics-title">Diagnostics</div>' + rows.map(
        ([k, v]) => `<div class="diagnostics-row"><span>${escapeHtml(k)}</span><span>${escapeHtml(v)}</span></div>`
    ).join('');
    setDiagnosticsVisible(true);
}

function updateStatus(message, isError = false) {
    const statusElement = document.getElementById('status');
    statusElement.textContent = message;
    statusElement.style.borderLeftColor = isError ? '#e74c3c' : '#3498db';
    statusElement.style.background = isError ? '#fdedec' : '#f8f9fa';
}
