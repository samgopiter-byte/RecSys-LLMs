# HW3 Implementation Note

## What this project is

A memory-based collaborative filtering recommender for MovieLens 100K, running
entirely in the browser with plain JavaScript. No TensorFlow.js, no Matrix
Factorization, no server.

Two independent neighbourhood methods are implemented and always shown side by
side:

- **User-Based CF** — find users similar to the active user, then predict from
  the ratings those users gave the target movie.
- **Item-Based CF** — find movies similar to the target movie (measured over
  co-raters), then predict from how the active user rated those movies.

The original Week 3 starter used Matrix Factorization with TensorFlow.js. HW3
requires memory-based neighbourhood CF, so the starter was adapted rather than
extended; no TensorFlow code is loaded or referenced.

## Files

| File | Contents |
| --- | --- |
| `index.html` | Two panels: the primary Top-5 view and the secondary Prediction Inspector. |
| `style.css` | Layout, including the two equal-width Top-5 columns. |
| `data.js` | `loadData()`, `parseItemData()`, `parseRatingData()`, the rating indexes, the leave-one-out helpers, and the MovieLens genre parser. |
| `script.js` | Pearson similarity, both predictors, the Top-5 recommenders, and the UI. |
| `u.data` / `u.item` | Original MovieLens 100K files, **unmodified**. |

## The main view: Top-5 recommendations

1. Pick an **Active User**.
2. Press **Generate Top-5 Recommendations** (the list length is also switchable
   to 10).
3. Both methods render their own ranked list, and the summary line reports how
   many candidates were considered, how many produced a defensible prediction,
   how long each method took, and how many titles the two lists share.

Only movies the user has **not** rated are eligible. Movies with fewer than
`MIN_ITEM_HISTORY` raters are skipped up front, which is the same condition the
cold-start guard would reject anyway, so nothing is lost.

If fewer than five movies survive the evidence guards, the list is shown
shorter and the shortfall is stated. **No fallback rating is ever substituted**
for a prediction the evidence does not support.

## The Prediction Inspector (secondary view)

Below the Top-5 view, the inspector explains a single user + movie pair:
neighbour lists, weighted similarities, co-rated counts, evidence, and which
guard rejected the prediction when one does. Movie search, the three quick
examples, and the collapsible diagnostics panel all live here.

## Missing-value strategy

There is exactly **one** missing-value strategy:

> **CO-RATED ONLY** — missing ratings are never imputed, never replaced by `0`,
> never replaced by a mean. Pearson is computed only over movies (or users) that
> both parties actually rated.

Reliability weighting

```
weightedSimilarity = rawPearson * min(commonCount / 50, 1)
```

is an **additional confidence control** applied to an already-computed
similarity, not a second missing-value strategy.

## Target leakage

When the Inspector is asked about a pair that *is* rated, the target cell is
temporarily removed from both profiles (`withTargetCellHeldOut`) before any mean,
overlap, or Pearson value is computed, and restored in a `finally` block. The
original `ratings` array is never mutated. Genuine missing pairs are untouched
by this and are ordinary recommendation predictions.

## CF constants

```
MIN_OVERLAP            = 3     min co-rated observations for a pair to count
RELIABILITY_SATURATION = 50    co-rated count at which the weight reaches 1.0
TOP_K_NEIGHBORS        = 50    how many best-scoring neighbours may vote
MIN_USER_HISTORY       = 5     user-side cold-start guard
MIN_ITEM_HISTORY       = 5     item-side cold-start guard
MIN_EVIDENCE           = 0.5   min sum of |weighted similarity| to accept
```

## Performance

The Top-5 pass does not call the full predictor once per unseen movie.

- Candidates are pre-filtered to movies with enough raters.
- For **User-Based**, `pearsonOnCoRated(activeUser, otherUser)` does not depend
  on the candidate movie, so it is computed once per call and reused for every
  candidate (~335 ms → ~30 ms for User 308, bit-identical results).
- For **Item-Based**, the neighbour pool is always the user's already-rated
  movies, so it is resolved once per call. The remaining per-pair Pearson
  (candidate movie × history movie) genuinely depends on both movies and is
  therefore the residual cost; item-based is the slower of the two and scales
  with profile size.

## Genre handling

`u.item` carries 19 pipe-separated attribute fields. Field 5 is the reserved
"unknown" attribute, so the 18 genre flags are read from fields
`6 ... 23` (`GENRE_FIELD_START = 6`, `GENRE_FIELD_END = 24`) against
`MOVIELENS_GENRES` in canonical MovieLens order. Two movies in this dataset
(`267` and `1373`) have all genre flags unset and are reported as such rather
than being given a fabricated genre.
