// Global variables to store parsed data
let movies = [];
let ratings = [];
let numUsers = 0;
let numMovies = 0;

// Movie data structure: { id: number, title: string, year: number|null, genres: string[] }
// Rating data structure: { userId: number, movieId: number, rating: number }

// ---------------------------------------------------------------------------
// MovieLens 100K `u.item` genre layout
// ---------------------------------------------------------------------------
// `u.item` is pipe-separated with 24 fields:
//
//   [0] movie id
//   [1] movie title (e.g. "Toy Story (1995)")
//   [2] release date
//   [3] video release date
//   [4] IMDb URL
//   [5] reserved "unknown" attribute  <-- NOT a genre
//   [6] ... [23]  the 18 real genre flags, in the fixed order below
//
// IMPORTANT: the genre block starts at field index 6, not 5. Field 5 is a
// reserved flag that is always 0; including it shifts every genre label by one
// position (Action <- unknown, Animation <- Action, ... Western <- War), which
// produces a completely wrong genre set. Week 2 shipped exactly that off-by-one
// bug and it was fixed there (commit 0a8f6c0); the same slice is used here so
// the two weeks cannot disagree.
//
// The flag order below is the canonical MovieLens order and is positional - it
// must match the column order in u.item exactly.
// ---------------------------------------------------------------------------
const MOVIELENS_GENRES = [
    'Action',
    'Adventure',
    'Animation',
    "Children's",
    'Comedy',
    'Crime',
    'Documentary',
    'Drama',
    'Fantasy',
    'Film-Noir',
    'Horror',
    'Musical',
    'Mystery',
    'Romance',
    'Sci-Fi',
    'Thriller',
    'War',
    'Western'
];

const ITEM_FIELD_COUNT = 24;
const GENRE_FIELD_START = 6;   // field 5 is the reserved "unknown" attribute
const GENRE_FIELD_END = 24;

/**
 * Read the 18 genre flags out of one already-split u.item record.
 * Returns the genre names whose flag is 1, in canonical order.
 */
function parseGenresFromFields(fields) {
    const genres = [];
    for (let i = 0; i < MOVIELENS_GENRES.length; i++) {
        if (parseInt(fields[GENRE_FIELD_START + i]) === 1) {
            genres.push(MOVIELENS_GENRES[i]);
        }
    }
    return genres;
}

async function loadData() {
    try {
        // Load movie data
        const movieResponse = await fetch('u.item');
        const movieText = await movieResponse.text();
        movies = parseItemData(movieText);
        numMovies = movies.length;

        // Load rating data
        const ratingResponse = await fetch('u.data');
        const ratingText = await ratingResponse.text();
        ratings = parseRatingData(ratingText);
        
        // Calculate number of unique users
        const uniqueUsers = new Set(ratings.map(r => r.userId));
        numUsers = uniqueUsers.size;

        console.log(`Loaded ${movies.length} movies and ${ratings.length} ratings from ${numUsers} users`);
        
        return { movies, ratings, numUsers, numMovies };
    } catch (error) {
        console.error('Error loading data:', error);
        throw error;
    }
}

function parseItemData(text) {
    const lines = text.split('\n');
    const movieData = [];
    
    for (const line of lines) {
        if (line.trim() === '') continue;
        
        const parts = line.split('|');
        if (parts.length >= 2) {
            const id = parseInt(parts[0]);
            // Extract title and year from the title field (format: "Title (Year)")
            const titleMatch = parts[1].match(/(.+)\s+\((\d{4})\)$/);
            let title = parts[1];
            let year = null;
            
            if (titleMatch) {
                title = titleMatch[1].trim();
                year = parseInt(titleMatch[2]);
            }
            
            // Genres come from fields 6..23 only. See MOVIELENS_GENRES above for
            // why field 5 must be excluded.
            let genres = [];
            if (parts.length >= ITEM_FIELD_COUNT) {
                genres = parseGenresFromFields(parts);
            } else {
                console.warn(
                    `[data] movie ${id} has only ${parts.length} fields, expected ` +
                    `${ITEM_FIELD_COUNT}; genres left empty`
                );
            }

            movieData.push({
                id: id,
                title: title,
                year: year,
                genres: genres
            });
        }
    }
    
    return movieData;
}

function parseRatingData(text) {
    const lines = text.split('\n');
    const ratingData = [];
    
    for (const line of lines) {
        if (line.trim() === '') continue;
        
        const parts = line.split('\t');
        if (parts.length >= 3) {
            ratingData.push({
                userId: parseInt(parts[0]),
                movieId: parseInt(parts[1]),
                rating: parseFloat(parts[2])
            });
        }
    }
    
    return ratingData;
}

// ---------------------------------------------------------------------------
// HW3 - memory-based CF lookup structures.
//
// The flat `ratings` array above is a list of OBSERVED cells only. Neighbourhood
// CF needs fast access in both directions (a user's whole profile, and the full
// list of users who rated one movie), plus per-profile statistics, so we build
// two indexes once at startup:
//
//   ratingsByUser : Map<userId, UserProfile>
//   ratingsByItem : Map<movieId, ItemProfile>
//
//   UserProfile = { id, count, sum, mean, ratings: Map<movieId, rating>, itemIds: number[] }
//   ItemProfile = { id, count, sum, mean, ratings: Map<userId, rating>, userIds: number[] }
//
// `itemIds` / `userIds` are kept as ascending-sorted arrays so that the
// co-rated intersection can be computed with a linear two-pointer merge instead
// of a hash lookup per cell. `mean` is each profile's average rating and is used
// both for Pearson normalisation and for the deviation term of the prediction.
//
// IMPORTANT: `count` is the number of OBSERVED cells in that profile. There is
// no dense matrix and no zero padding anywhere, which is exactly what makes the
// "co-rated only" missing-value strategy possible - see the strategy note at the
// top of script.js.
// ---------------------------------------------------------------------------

let ratingsByUser = new Map();
let ratingsByItem = new Map();

function buildRatingIndexes() {
    ratingsByUser = new Map();
    ratingsByItem = new Map();

    let duplicates = 0;

    for (const r of ratings) {
        let user = ratingsByUser.get(r.userId);
        if (!user) {
            user = { id: r.userId, count: 0, sum: 0, mean: 0, ratings: new Map(), itemIds: [] };
            ratingsByUser.set(r.userId, user);
        }
        if (user.ratings.has(r.movieId)) { duplicates++; continue; }
        user.ratings.set(r.movieId, r.rating);
        user.itemIds.push(r.movieId);
        user.count++;
        user.sum += r.rating;

        let item = ratingsByItem.get(r.movieId);
        if (!item) {
            item = { id: r.movieId, count: 0, sum: 0, mean: 0, ratings: new Map(), userIds: [] };
            ratingsByItem.set(r.movieId, item);
        }
        if (item.ratings.has(r.userId)) { duplicates++; continue; }
        item.ratings.set(r.userId, r.rating);
        item.userIds.push(r.userId);
        item.count++;
        item.sum += r.rating;
    }

    for (const user of ratingsByUser.values()) {
        user.itemIds.sort((a, b) => a - b);
        user.mean = user.count > 0 ? user.sum / user.count : 0;
    }
    for (const item of ratingsByItem.values()) {
        item.userIds.sort((a, b) => a - b);
        item.mean = item.count > 0 ? item.sum / item.count : 0;
    }

    if (duplicates > 0) {
        console.warn(`[data] skipped ${duplicates} duplicate (user, movie) cells`);
    }

    const observed = ratingsByUser.size * ratingsByItem.size;
    const density = observed > 0 ? (100 * ratings.length) / observed : 0;

    console.log(
        `[data] rating index built: ${ratingsByUser.size} users, ` +
        `${ratingsByItem.size} items, ${ratings.length} observed cells ` +
        `(density ${density.toFixed(2)}%, sparsity ${(100 - density).toFixed(2)}%)`
    );

    return { ratingsByUser, ratingsByItem };
}


// ---------------------------------------------------------------------------
// Target-cell hold-out (leave-one-out) support
// ---------------------------------------------------------------------------
// Predicting a rating for a pair (user, movie) that ALREADY has an observed
// rating in u.data is a leakage problem: that rating would otherwise feed the
// very quantities we are estimating from -
//   * the target user's mean (User-Based baseline),
//   * the target movie's mean (Item-Based baseline),
//   * the co-rated intersection of every Pearson similarity (the target cell
//     is part of it whenever target user and neighbour both rated the movie,
//     or whenever the target movie and a neighbour item were both rated by
//     the target user).
//
// withTargetCellHeldOut() removes the cell from BOTH profiles for the duration
// of one callback and restores every touched field afterwards, including when
// the callback throws. The flat `ratings` array is never modified, so nothing
// is lost permanently and no data file is touched.
//
// It is a no-op when the pair was never rated - that case is already a genuine
// missing-value prediction and needs no correction.
//
// `heldOutTarget` is what the prediction functions read to report
// targetWasRated / actualRating / leaveOneOutApplied in their result objects.
// ---------------------------------------------------------------------------

let heldOutTarget = null;

/** The rating currently withheld, or null when nothing is withheld. */
function getHeldOutTarget() {
    return heldOutTarget;
}

/** True when this exact (user, movie) cell is the one currently withheld. */
function isHeldOut(userId, movieId) {
    return !!heldOutTarget
        && heldOutTarget.userId === userId
        && heldOutTarget.movieId === movieId;
}

/**
 * Copy a profile so the count/sum/mean recorded in a result object are the ones
 * the prediction actually used.
 *
 * Without this, a result would keep a live reference to the profile: the
 * hold-out restores the profile as soon as the prediction returns, so any later
 * read of `result.userProfile.mean` would report the FULL-data mean instead of
 * the leave-one-out mean the prediction was based on. The ratings map and id
 * list are still shared by reference, which stays consistent because the
 * hold-out restores them exactly.
 */
function snapshotProfile(profile) {
    if (!profile) return null;
    return {
        id: profile.id,
        count: profile.count,
        sum: profile.sum,
        mean: profile.mean,
        ratings: profile.ratings,
        itemIds: profile.itemIds,
        userIds: profile.userIds
    };
}

function withTargetCellHeldOut(userId, movieId, fn) {
    const userRec = ratingsByUser.get(userId);
    const itemRec = ratingsByItem.get(movieId);

    // Nothing to hold out: unknown user, unknown movie, or a genuinely missing cell.
    if (!userRec || !itemRec || !userRec.ratings.has(movieId) || !itemRec.ratings.has(userId)) {
        return fn();
    }

    // Snapshot every field we are about to touch so the restore is exact.
    const userSnap = {
        rating: userRec.ratings.get(movieId),
        index: userRec.itemIds.indexOf(movieId),
        count: userRec.count,
        sum: userRec.sum,
        mean: userRec.mean
    };
    const itemSnap = {
        rating: itemRec.ratings.get(userId),
        index: itemRec.userIds.indexOf(userId),
        count: itemRec.count,
        sum: itemRec.sum,
        mean: itemRec.mean
    };

    const previousHeldOut = heldOutTarget;
    heldOutTarget = { userId, movieId, rating: userSnap.rating };

    try {
        userRec.ratings.delete(movieId);
        userRec.itemIds.splice(userSnap.index, 1);
        userRec.count -= 1;
        userRec.sum -= userSnap.rating;
        userRec.mean = userRec.count > 0 ? userRec.sum / userRec.count : 0;

        itemRec.ratings.delete(userId);
        itemRec.userIds.splice(itemSnap.index, 1);
        itemRec.count -= 1;
        itemRec.sum -= itemSnap.rating;
        itemRec.mean = itemRec.count > 0 ? itemRec.sum / itemRec.count : 0;

        return fn();
    } finally {
        userRec.ratings.set(movieId, userSnap.rating);
        userRec.itemIds.splice(userSnap.index, 0, movieId);
        userRec.count = userSnap.count;
        userRec.sum = userSnap.sum;
        userRec.mean = userSnap.mean;

        itemRec.ratings.set(userId, itemSnap.rating);
        itemRec.userIds.splice(itemSnap.index, 0, userId);
        itemRec.count = itemSnap.count;
        itemRec.sum = itemSnap.sum;
        itemRec.mean = itemSnap.mean;

        heldOutTarget = previousHeldOut;
    }
}
