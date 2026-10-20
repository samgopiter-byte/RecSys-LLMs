/**
 * week6/script.js — HW6 application layer.
 *
 * Graph-based recommendations with global PageRank, personalized PageRank
 * (PPR) and a matrix-factorization (MF) reference, compared under one shared
 * retrieval protocol: validation-selected alpha, one shared evaluation cohort,
 * Recall@10 and explicit exclusions.
 *
 * This is a plain (classic) script, NOT an ES module. `index.html` loads, in
 * order: the TensorFlow.js CDN runtime, `vendor/d3.min.js`, `data.js`,
 * `graph.js` and finally this file. There are exactly five script tags and no
 * inline script, so the provided MF reference lives inside this file.
 *
 * Globals provided by `data.js` (read, never redefined here):
 *   `movies`   `[{ id, title, genres: string[] }]`
 *   `ratings`  `[{ userId, movieId, rating, timestamp }]`
 *   `numUsers` `numMovies` `ratingMatrix` `genreNames` `SPLIT_EXCLUSIONS`
 *   `loadData()`  `splitByTimestamp()` -> `{ train, val, test }`
 *
 * Globals provided by `graph.js` (read, never redefined here):
 *   `REFERENCE_GRAPH`, `buildGraph`, `rowNormalize`, `pageRank`,
 *   `personalizedPageRank`, `GRAPH`, `setGraph`, `getGraph`,
 *   `renderGraphView`, `fixtureStep`, `fixtureReset`, `GRAPH_VIEW_STATE`
 *
 * WHAT THE STUDENT MUST IMPLEMENT (`TODO(hw6)`, each stub throws until written):
 *   1. `movieScores(p, graph)`                   — movie-node scores, `graph.movieIds` order.
 *   2. `recommendTopK(method, userId, K, split)` — candidate pool, exclusions, tie-breaks.
 *   3. `recallAtK(method, K, split)`             — shared-cohort Recall@K plus exclusions.
 *
 * PROVIDED (instructor scaffolding — not an exercise to rewrite):
 *   - `hw6InitTf`, the formatting/CSV/exporter helpers, the training-split indices,
 *     `popularityScores`, `mfScores` and the deterministic MF reference model,
 *     every panel renderer, the four-method / alpha-sweep / convergence /
 *     two-user-demo runners, the self-check harness and the UI wiring.
 *   - The MF reference is instructor-provided: students must not have to
 *     implement it. Report its selected checkpoint, not a hand-built model.
 *
 * Design notes:
 *   - The graph itself (transition matrix, PageRank, PPR, dangling-mass policy)
 *     lives in `graph.js`. This file never re-implements propagation for the
 *     product path; it calls `pageRank` / `personalizedPageRank` through the
 *     adapter below and treats an unrecognised call shape as an integration gap
 *     (reported PENDING), never as a failed check.
 *   - PPR scores are long-run graph-walk probabilities. They are NOT purchase
 *     probabilities, NOT predicted ratings and NOT comparable in scale to MF
 *     scores. No rating RMSE is ever computed for PPR.
 *   - `alpha` always means the link-following probability; `1 - alpha` restarts.
 *   - Alpha is selected on validation only. Test outcomes never choose settings.
 *
 * @module week6/script
 */

// ---------------------------------------------------------------------------
// Typedefs
// ---------------------------------------------------------------------------

/**
 * @typedef {Object} Movie
 * @property {number} id
 * @property {string} title
 * @property {string[]} genres
 */

/**
 * @typedef {Object} Rating
 * @property {number} userId
 * @property {number} movieId
 * @property {number} rating
 * @property {number} timestamp
 */

/**
 * @typedef {Object} RecommendedMovie
 * @property {number} movieId      Raw MovieLens movie id.
 * @property {string} title        Display title.
 * @property {string[]} genres     Genre names.
 * @property {number} score        Method score (PPR scores are walk probabilities).
 * @property {number} popularity   Distinct users with a positive training rating.
 */

/**
 * @typedef {Object} PropagationResult
 * @property {ArrayLike<number>} scores  Node-probability vector, length = node count.
 * @property {number|null} iterations    Iterations performed, when reported.
 * @property {number|null} residual      Final L1 difference, when reported.
 * @property {boolean|null} converged    `true` only when the tolerance was met.
 * @property {string} source             `"graph.js"` or `"reference"` (self-check path).
 */

/**
 * @typedef {Object} RecallReport
 * @property {number} recall              Macro-average Recall@K over the shared cohort.
 * @property {number[]} cohort            Sorted user ids in the shared cohort.
 * @property {number} excludedUnseen      Relevant movies absent from the training pool.
 * @property {number} excludedNoEligible  Users with no eligible relevant movie.
 * @property {Array<{userId:number, relevant:number, hit:number}>} perUser
 */

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/** Alpha grid swept on validation for PPR (readme section 6.1). */
const ALPHA_GRID = [0.50, 0.70, 0.85, 0.95];

/** Rank cut-off used throughout the assignment. */
const HW6_K = 10;

/**
 * Marker carried by every unimplemented stub error message.
 *
 * Named with the `HW6_` prefix (rather than the bare week5 `TODO_MARKER`) so a
 * same-named `const` in `graph.js` cannot cause a global-lexical redeclaration
 * `SyntaxError` that would take the whole page down. The convention value and
 * behaviour are unchanged: any error whose message carries this marker is
 * reported PENDING, never FAIL.
 */
const HW6_TODO_MARKER = "TODO(hw6)";

/** Marker used for integration gaps; the harness maps it to PENDING. */
const HW6_INTERFACE_MARK = "HW6_INTERFACE";

/** Iteration cap and tolerance reported by the convergence panel (readme 3.4). */
const HW6_MAX_ITERATIONS = 500;
const HW6_TOLERANCE = 1e-6;

/**
 * Cache bounds. PPR holds one node-probability vector per (user, alpha), so the
 * limit bounds memory during an interactive session; PageRank holds one vector
 * per alpha; recommendation lists are small arrays but numerous.
 */
const HW6_CACHE_LIMIT = { pageRank: 64, ppr: 2048, rec: 20000 };

/**
 * Users propagated per batched `tf.matMul` call. Bounds the working set of the
 * async engine (memory and the latency of one batch) without changing results,
 * which are computed per column.
 */
const HW6_BATCH_SIZE = 64;

/**
 * Wall-time budget (ms) between browser yields inside the batched kernel. Small
 * enough that the page paints and Cancel stays responsive, large enough that
 * yielding does not dominate the run.
 */
const HW6_YIELD_GAP_MS = 30;

/**
 * Minimum wall time (ms) between intra-chunk progress repaints. The batched
 * kernel calls `onIteration` every iteration; this throttle keeps a fast
 * convergence from churning the DOM while still letting the panel move during a
 * slow chunk.
 */
const HW6_PROGRESS_MIN_REPAINT_MS = 80;

/**
 * Controls that must not be used while an evaluation is in flight. Run
 * self-checks (`run-tests`) are included because they temporarily replace the
 * active graph, which disposes tensors the live async kernel still holds.
 * Exports are included because they would capture a partial run.
 */
const HW6_RUN_GUARD_IDS = [
  "recall-btn", "alpha-sweep-btn", "two-user-demo-btn", "recommend-btn",
  "run-pagerank", "run-ppr", "build-graph", "run-tests", "step-btn",
  "reset-steps", "export-verification", "export-results",
  "export-recommendations",
];

/** Select/range inputs disabled while an evaluation is in flight. */
const HW6_RUN_GUARD_INPUTS = ["method", "alpha", "user-select", "seed-user-select"];

/** Dimensionality of the provided MF reference model. */
const MF_FACTORS = 16;

/** Fixed seed for the deterministic MF reference (reproducible initialisation). */
const MF_SEED = 20261013;

/**
 * Training settings, keeping the previous assignment's convention. `reg` is the
 * L2 coefficient applied as `sum(factors^2) * reg / actualBatchSize` (the same
 * per-batch normalisation as the submitted HW5 implementation).
 */
const MF_CONFIG = { epochs: 30, lr: 0.01, reg: 0.05, batchSize: 512 };

/**
 * Public description of the provided MF reference model, filled in when the
 * model is trained. The report must state the model and checkpoint actually
 * used, so these fields are the record.
 *
 * `trainedAt` is the lazy-training marker: it stays `null` until the reference
 * has actually been trained, so "not trained yet" is a truthful, checkable flag
 * rather than an inference from a zero score. `checkpoint` stays `null` while
 * untrained. `l2Scaling` records the per-batch L2 normalisation actually used
 * (`reg / actualBatchSize`, matching the HW5 convention).
 *
 * @type {{factors:number, epochs:number, learningRate:number,
 *         regularization:number, batchSize:number, seed:number,
 *         l2Scaling:string, backend:(string|null), checkpoint:(number|null),
 *         validationRMSE:(number|null), trainedAt:(string|null),
 *         history:Array<Object>}}
 */
const MF_INFO = {
  factors: MF_FACTORS,
  epochs: MF_CONFIG.epochs,
  learningRate: MF_CONFIG.lr,
  regularization: MF_CONFIG.reg,
  batchSize: MF_CONFIG.batchSize,
  seed: MF_SEED,
  l2Scaling: "reg / actualBatchSize",
  backend: null,
  checkpoint: null,
  validationRMSE: null,
  trainedAt: null,
  history: [],
};

/**
 * @typedef {Object} MfModel
 * @property {tf.Variable} userFactors Dense `[numUsers + 1, MF_FACTORS]`, row index = raw `userId`.
 * @property {tf.Variable} itemFactors Dense `[maxMovieId + 1, MF_FACTORS]`, row index = raw `movieId`.
 * @property {number} globalMean        Mean training rating.
 */

/**
 * Provided MF reference model, or `null` until the lazy `hw6EnsureMf()` has
 * actually trained it. Nothing MF-related runs on page load.
 *
 * @type {MfModel|null}
 */
let HW6_MF_MODEL = null;

/**
 * In-flight MF training promise. A second need while training is running
 * awaits the same promise instead of starting a second training pass.
 *
 * @type {Promise<MfModel|null>|null}
 */
let HW6_MF_TRAINING = null;

/** Highest raw movie id in the dataset; cached for factor allocation. */
let HW6_MAX_MOVIE_ID = 0;

/** Chronological 80/10/10 split, set once by the bootstrap. */
const HW6_SPLIT = { train: [], val: [], test: [] };

/** Mutable application state (all names prefixed so `graph.js` stays untouched). */
const HW6_STATE = {
  backend: null,
  loaded: false,
  splitReady: false,

  /** Sorted distinct movie ids observed in the training split (readme 5 candidate pool). */
  candidateIds: [],
  /** `Set` view of `candidateIds`. */
  candidateSet: new Set(),
  /** Sorted distinct user ids observed in the training split. */
  trainUserIds: [],
  /** `movieId -> count of distinct users with a positive (>= 4.0) training rating`. */
  popularity: new Map(),
  /** `userId -> Set(movieId)` per split; drives the previously-rated exclusion rules. */
  pairsBySplit: { train: new Map(), val: new Map(), test: new Map() },
  /** `userId -> Map(movieId -> rating)` per split; drives relevance. */
  gradesBySplit: { train: new Map(), val: new Map(), test: new Map() },
  /** Users with at least one positive training edge (readme 7.1). */
  usersWithPositiveTrain: new Set(),
  /** Count of distinct positive training (user, movie) pairs. */
  positivePairs: 0,

  /**
   * Multi-entry propagation caches, keyed by graph epoch + alpha (+ seed user).
   * Keyed by alpha so a slider move never has to discard a still-valid result;
   * a graph rebuild bumps `graphEpoch` (via `hw6InvalidateCaches`) so stale
   * entries can never be served. Bounded by `HW6_CACHE_LIMIT`.
   */
  pageRankCache: new Map(),
  pprCache: new Map(),
  /** Recommendation-list memo: keyed by method|user|K|split|alpha. */
  recCache: new Map(),
  /** Monotonic counter giving cache entries a least-recently-used order. */
  cacheClock: 0,
  /** Incremented whenever the active graph (and therefore every cache) changes. */
  graphEpoch: 0,
  /** Programmatic alpha override; `null` means "read the slider". */
  alphaContext: null,
  /** Alpha selected by the last completed validation sweep, or `null`. */
  selectedAlpha: null,
  /** Completed validation sweep snapshot `{ epoch, rows, best, note }`. */
  sweep: null,
  /** Settings of the last completed primary comparison, for evidence. */
  comparisonConfig: null,
  /** The in-flight async evaluation token, or `null`. */
  activeRun: null,
  /** Convergence records for the panel: `{ pagerank, ppr }`. */
  convergence: { pagerank: null, ppr: null },

  /** Last runner outputs (also the exporters' sources). */
  comparisonRows: null,
  alphaSweepRows: null,
  comparisonNote: null,
  alphaSweepNote: null,
  demo: null,
  recallReport: null,
  recommendationRows: [],
  /** `Set` of recommendation row keys already recorded, to avoid duplicate CSV rows. */
  recommendationKeys: new Set(),

  /** Last self-check run, reused by `serializeVerification()`. */
  lastVerification: null,

  /** Throttle clock for intra-chunk progress repaints. */
  lastProgressPaint: 0,
  /** Summary of the last finished run, kept visible after the panel hides. */
  lastRunSummary: null,

  /**
   * Lifecycle of the most recent evaluation: `"idle"`, `"active"`,
   * `"cancelled"`, `"failed"` or `"completed"`. Set by `hw6RunEvaluationAsync`
   * and read by the exporters so a partial run can never masquerade as final.
   */
  runState: "idle",
  /**
   * Immutable snapshot of the last fully successful primary evaluation:
   * `{ identity, config, comparisonRows, alphaSweepRows, selectedAlpha,
   * convergence, recallReport, graphEpoch, completedAt }`. Exports and the
   * two-user demo read from here when the current run is not a clean completed
   * one, so a cancelled or failed run cannot mix with earlier evidence, and a
   * later slider change cannot mutate it.
   */
  completedSnapshot: null,
  /** Monotonic counter giving each completed run a human-readable identity. */
  runCounter: 0,
  /** Prior disabled-state of guarded controls, captured while a run is active. */
  guardPriorDisabled: null,
};

// ---------------------------------------------------------------------------
// Small helpers (provided)
// ---------------------------------------------------------------------------

/**
 * Format a number to fixed decimals; non-finite and missing values render as an
 * em dash so tables never show `NaN` or `undefined`.
 *
 * @param {number|null|undefined} x
 * @param {number} [d=4] Decimal places.
 * @returns {string}
 */
function hw6Fmt(x, d) {
  const places = d === undefined ? 4 : d;
  if (x === null || x === undefined) return "\u2014";
  const n = Number(x);
  if (!Number.isFinite(n)) return "\u2014";
  return n.toFixed(places);
}

/**
 * Escape text before inserting it into HTML.
 *
 * @param {*} text
 * @returns {string}
 */
function hw6EscapeHtml(text) {
  return String(text)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

/**
 * Look up a movie record by id.
 *
 * @param {number|string} id
 * @returns {Movie|undefined}
 */
function hw6MovieById(id) {
  const numericId = Number(id);
  for (let i = 0; i < movies.length; i++) {
    if (movies[i].id === numericId) return movies[i];
  }
  return undefined;
}

/**
 * Look up a movie title with a readable fallback for unknown ids.
 *
 * @param {number|string} id
 * @returns {string}
 */
function hw6Title(id) {
  const movie = hw6MovieById(id);
  return movie ? movie.title : "Movie #" + id;
}

/**
 * Genre list for a movie id (empty array when unknown).
 *
 * @param {number|string} id
 * @returns {string[]}
 */
function hw6Genres(id) {
  const movie = hw6MovieById(id);
  return movie && Array.isArray(movie.genres) ? movie.genres : [];
}

/**
 * Build the error used for an unrecognised `graph.js` interface shape. The
 * harness maps this to PENDING: it is an integration gap, not a wrong number.
 *
 * @param {string} message
 * @returns {Error}
 */
function hw6InterfaceError(message) {
  const error = new Error(HW6_INTERFACE_MARK + ": " + message);
  error.hw6Interface = true;
  return error;
}

/**
 * Read a numeric DOM value with a fallback.
 *
 * @param {string} id
 * @param {number} fallback
 * @returns {number}
 */
function hw6ReadNumber(id, fallback) {
  const el = document.getElementById(id);
  if (!el) return fallback;
  const raw = el.value;
  // An empty select / number input must fall back rather than read as 0.
  if (raw === undefined || raw === null || String(raw).trim() === "") return fallback;
  const value = Number(raw);
  return Number.isFinite(value) ? value : fallback;
}

/**
 * Read the selected method, falling back to `popularity` for unknown values.
 *
 * @returns {"popularity"|"pagerank"|"ppr"|"mf"}
 */
function hw6ReadMethod() {
  const el = document.getElementById("method");
  const value = el ? String(el.value) : "popularity";
  return value === "pagerank" || value === "ppr" || value === "mf" ? value : "popularity";
}

/**
 * Read the alpha slider as the link-following probability.
 *
 * @returns {number} Alpha clamped into `[0.50, 0.95]`.
 */
function hw6ReadAlpha() {
  const value = hw6ReadNumber("alpha", 0.85);
  return Math.min(0.95, Math.max(0.50, value));
}

/**
 * Set the alpha slider, refresh the live readout, and invalidate cached results.
 * The slider must never dispatch an event here (that would recurse).
 *
 * @param {number} value
 * @returns {void}
 */
function hw6SetAlpha(value) {
  const slider = document.getElementById("alpha");
  const readout = document.getElementById("alpha-value");
  const numeric = Number(value);
  const clamped = Math.min(0.95, Math.max(0.50, Number.isFinite(numeric) ? numeric : 0.85));
  if (slider) slider.value = String(clamped);
  if (readout) readout.textContent = clamped.toFixed(2);
  // Pure setter: propagation caches are keyed by alpha, so changing the value
  // must NOT drop still-valid results for other alphas (that no-op invalidation
  // was what made a 1 -> 2 -> 1 sequence recompute every walk).
}

/**
 * The alpha used for programmatic evaluation. An explicit context (`alphaContext`)
 * wins, so the validation sweep and the primary comparison can run at their own
 * alpha without moving the user's demo slider; otherwise the slider is read.
 *
 * @returns {number}
 */
function hw6EvalAlpha() {
  const context = HW6_STATE.alphaContext;
  return Number.isFinite(context) ? Number(context) : hw6ReadAlpha();
}

/**
 * Stable string key for one alpha at four decimals (the slider's own precision).
 *
 * @param {number} alpha
 * @returns {string}
 */
function hw6AlphaKey(alpha) {
  return Number(alpha).toFixed(4);
}

/**
 * Read a cache entry, refreshing its recency.
 *
 * @param {Map} map
 * @param {string} key
 * @returns {*} The cached value, or `undefined`.
 */
function hw6CacheGet(map, key) {
  const entry = map.get(key);
  if (entry === undefined) return undefined;
  entry.clock = (HW6_STATE.cacheClock += 1);
  return entry.value;
}

/**
 * Write a cache entry, evicting the least-recently-used one when over `limit`.
 *
 * @param {Map} map
 * @param {string} key
 * @param {*} value
 * @param {number} limit
 * @returns {void}
 */
function hw6CacheSet(map, key, value, limit) {
  map.set(key, { value: value, clock: (HW6_STATE.cacheClock += 1) });
  if (limit > 0 && map.size > limit) {
    let oldestKey = null;
    let oldestClock = Infinity;
    map.forEach(function (entry, entryKey) {
      if (entry.clock < oldestClock) {
        oldestClock = entry.clock;
        oldestKey = entryKey;
      }
    });
    if (oldestKey !== null) map.delete(oldestKey);
  }
}

/**
 * Drop cached PageRank / PPR / recommendation results because the graph (or the
 * underlying data) changed. `graph.js` owns the graph and any cache of its own;
 * if it exposes a `stale` flag we set it, but no field is invented here.
 *
 * A graph change is the only event that invalidates these caches: alpha and the
 * target user are part of every key, so they need no invalidation.
 *
 * @returns {void}
 */
function hw6InvalidateCaches() {
  HW6_STATE.pageRankCache.clear();
  HW6_STATE.pprCache.clear();
  HW6_STATE.recCache.clear();
  HW6_STATE.graphEpoch += 1;
  HW6_STATE.sweep = null;
  HW6_STATE.selectedAlpha = null;
  HW6_STATE.comparisonConfig = null;
  const viewState = typeof GRAPH_VIEW_STATE !== "undefined" ? GRAPH_VIEW_STATE : null;
  if (viewState && typeof viewState === "object" && "stale" in viewState) {
    viewState.stale = true;
  }
}

/**
 * Write the status line without ever throwing.
 *
 * @param {string} message
 * @returns {void}
 */
function hw6SetStatus(message) {
  const status = document.getElementById("status");
  if (status) status.textContent = message;
}

/**
 * Render a readable "not available yet" state that distinguishes an
 * unimplemented stub from a genuine failure from an interface gap.
 *
 * @param {string} what Human label, e.g. "Recommendations".
 * @param {*} error     Thrown value (consumed defensively).
 * @returns {string} HTML for a panel body.
 */
function hw6UnavailableHtml(what, error) {
  const message = String(error && error.message ? error.message : error);
  const head = '<p class="empty-state">' + hw6EscapeHtml(what);
  const tail = '<p class="hint"><code>' + hw6EscapeHtml(message) + "</code></p>";
  if (message.indexOf(HW6_TODO_MARKER) !== -1) {
    return head + " not available yet \u2014 implement the <code>TODO(hw6)</code> function, then run again.</p>" + tail;
  }
  if (message.indexOf(HW6_INTERFACE_MARK) !== -1) {
    return head + " unavailable \u2014 the <code>graph.js</code> interface did not match this page's adapter.</p>" + tail;
  }
  return head + " failed \u2014 the page stays usable; details below.</p>" + tail;
}

/**
 * Convert a value into a valid CSV field (RFC 4180 quoting).
 *
 * @param {*} value
 * @returns {string}
 */
function hw6CsvField(value) {
  if (value === null || value === undefined) return "";
  const text = String(value);
  if (/[",\r\n]/.test(text)) return '"' + text.replace(/"/g, '""') + '"';
  return text;
}

/**
 * Join one CSV row.
 *
 * @param {Array<*>} values
 * @returns {string}
 */
function hw6CsvRow(values) {
  return values.map(hw6CsvField).join(",");
}

/**
 * Build a CSV document from a header and rows.
 *
 * @param {string[]} header
 * @param {Array<Array<*>>} rows
 * @returns {string}
 */
function hw6CsvDocument(header, rows) {
  const lines = [hw6CsvRow(header)];
  for (let i = 0; i < rows.length; i++) lines.push(hw6CsvRow(rows[i]));
  return lines.join("\r\n") + "\r\n";
}

/**
 * Trigger a client-side download. Uses a `Blob` plus `URL.createObjectURL` and
 * an `<a download>` element, which works when the page is opened from `file://`
 * as well as over HTTP.
 *
 * @param {string} filename Suggested file name.
 * @param {string} text     File body.
 * @param {string} [mime]   MIME type; a `;charset=utf-8` suffix is appended.
 * @returns {boolean} `true` when the download was triggered.
 */
function download(filename, text, mime) {
  const type = (mime || "text/plain") + ";charset=utf-8";
  try {
    const blob = new Blob([text], { type: type });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = filename;
    anchor.style.display = "none";
    document.body.appendChild(anchor);
    anchor.click();
    window.setTimeout(function () {
      if (anchor.parentNode) anchor.parentNode.removeChild(anchor);
      URL.revokeObjectURL(url);
    }, 1500);
    return true;
  } catch (error) {
    console.error("Download failed:", error);
    return false;
  }
}

// ---------------------------------------------------------------------------
// TensorFlow.js backend bootstrap (provided)
// ---------------------------------------------------------------------------

/**
 * Select the available TensorFlow.js backend, report it, and warn on CPU.
 *
 * Writes `TensorFlow.js backend: <name>` into `#tf-status` and an explanatory
 * line into `#backend-note`. When the CDN script failed, `tf` is undefined and
 * this renders a readable message and returns `null` instead of throwing.
 *
 * @returns {Promise<string|null>} Selected backend name, or `null` when TF.js is missing.
 */
async function hw6InitTf() {
  const statusEl = document.getElementById("tf-status");
  const noteEl = document.getElementById("backend-note");

  if (typeof tf === "undefined") {
    const message =
      "TensorFlow.js failed to load. Check the CDN script and your network connection, then reload the page.";
    if (statusEl) {
      statusEl.textContent = message;
      statusEl.className = "error";
    }
    if (noteEl) noteEl.textContent = "No backend available; the dataset summary stays readable.";
    return null;
  }

  try {
    await tf.setBackend("webgl");
  } catch (error) {
    try {
      await tf.setBackend("cpu");
    } catch (fallbackError) {
      // Leave the default in place; `tf.getBackend()` below reports what it is.
    }
  }
  await tf.ready();

  const backend = tf.getBackend();
  HW6_STATE.backend = backend;
  MF_INFO.backend = backend;
  if (statusEl) statusEl.textContent = "TensorFlow.js backend: " + backend;
  if (noteEl) {
    noteEl.textContent =
      backend === "cpu"
        ? "CPU backend selected (WebGL unavailable). Runs are slower; backend and runtime are recorded, not compared across devices."
        : "Backend and runtime are recorded, not compared across devices.";
  }
  return backend;
}

// ---------------------------------------------------------------------------
// Training-split indices and provided scoring helpers
// ---------------------------------------------------------------------------

/**
 * Build every training-split lookup used by the provided runners and by
 * `recommendTopK` / `recallAtK`.
 *
 * Indexing assumptions (readme section 3.1 and 5):
 *   - The candidate pool is every movie observed in the training split, sorted
 *     ascending by raw movie id. A zero in `R` means "no positive training
 *     edge", never a negative preference.
 *   - Popularity counts distinct users whose training rating for the movie is
 *     `>= 4.0`; a repeated (user, movie) pair is counted once.
 *   - `pairsBySplit` stores the raw rated-movie sets so the exclusion rules can
 *     ignore the rating value ("rated in training or validation regardless of
 *     the rating value").
 *
 * @param {{train: Rating[], val: Rating[], test: Rating[]}} split
 * @returns {void}
 */
function hw6BuildSplitIndices(split) {
  const names = ["train", "val", "test"];
  const movieIds = new Set();
  const userIds = new Set();
  const positiveUsersPerMovie = new Map();
  const positiveUsers = new Set();
  let positivePairs = 0;

  for (let s = 0; s < names.length; s++) {
    const name = names[s];
    const list = split[name] || [];
    for (let i = 0; i < list.length; i++) {
      const row = list[i];
      const pairs = HW6_STATE.pairsBySplit[name];
      const grades = HW6_STATE.gradesBySplit[name];
      let pairSet = pairs.get(row.userId);
      if (!pairSet) {
        pairSet = new Set();
        pairs.set(row.userId, pairSet);
      }
      pairSet.add(row.movieId);
      let gradeMap = grades.get(row.userId);
      if (!gradeMap) {
        gradeMap = new Map();
        grades.set(row.userId, gradeMap);
      }
      gradeMap.set(row.movieId, row.rating);
    }
  }

  const train = split.train || [];
  for (let i = 0; i < train.length; i++) {
    const row = train[i];
    movieIds.add(row.movieId);
    userIds.add(row.userId);
    if (row.rating >= 4.0) {
      let users = positiveUsersPerMovie.get(row.movieId);
      if (!users) {
        users = new Set();
        positiveUsersPerMovie.set(row.movieId, users);
      }
      if (!users.has(row.userId)) {
        users.add(row.userId);
        positivePairs++;
      }
      positiveUsers.add(row.userId);
    }
  }

  HW6_STATE.candidateIds = Array.from(movieIds).sort(function (a, b) {
    return a - b;
  });
  HW6_STATE.candidateSet = movieIds;
  HW6_STATE.trainUserIds = Array.from(userIds).sort(function (a, b) {
    return a - b;
  });
  HW6_STATE.popularity = new Map();
  positiveUsersPerMovie.forEach(function (users, movieId) {
    HW6_STATE.popularity.set(movieId, users.size);
  });
  HW6_STATE.usersWithPositiveTrain = positiveUsers;
  HW6_STATE.positivePairs = positivePairs;
  HW6_STATE.splitReady = true;
}

/**
 * Training popularity of one movie (readme section 7.3).
 *
 * @param {number} movieId
 * @returns {number} Distinct users with a positive training rating; `0` when unobserved.
 */
function hw6PopularityOf(movieId) {
  const value = HW6_STATE.popularity.get(movieId);
  return value === undefined ? 0 : value;
}

/**
 * Provided popularity scorer: distinct users with a training rating `>= 4.0`.
 *
 * @returns {Object<number, number>} Map-like object keyed by raw movie id.
 */
function popularityScores() {
  const out = {};
  HW6_STATE.popularity.forEach(function (count, movieId) {
    out[movieId] = count;
  });
  return out;
}

/**
 * Movies the user rated in the splits that must be excluded for an evaluation
 * split (readme section 5).
 *
 *   - `"test"` excludes movies rated in training OR validation, regardless of
 *     the rating value.
 *   - `"val"`  excludes movies rated in training only.
 *
 * @param {string} splitName `"val"` or `"test"`.
 * @param {number} userId
 * @returns {Set<number>}
 */
function hw6ExclusionSet(splitName, userId) {
  const out = new Set();
  const addFrom = function (name) {
    const set = HW6_STATE.pairsBySplit[name].get(userId);
    if (set) set.forEach(function (movieId) { out.add(movieId); });
  };
  addFrom("train");
  if (splitName === "test") addFrom("val");
  return out;
}

/**
 * The shared evaluation cohort for a split (readme section 7.1).
 *
 * A user is evaluable when the user appears in training with at least one
 * positive training edge and at least one relevant movie remains eligible in
 * the split. Relevant means: rating `>= 4.0` in the split, the movie belongs to
 * the training candidate pool, and it passes that split's previously-rated
 * exclusion rule. The MF-scoring condition is in principle satisfiable for
 * every training-movie candidate, so it does not remove users here.
 *
 * All four methods must use exactly this cohort for the primary comparison.
 *
 * @param {string} splitName `"val"` or `"test"` (anything else is treated as `"test"`).
 * @returns {number[]} Ascending user ids.
 */
function hw6SharedCohort(splitName) {
  const name = splitName === "val" ? "val" : "test";
  const grades = HW6_STATE.gradesBySplit[name];
  const cohort = [];
  for (let i = 0; i < HW6_STATE.trainUserIds.length; i++) {
    const userId = HW6_STATE.trainUserIds[i];
    if (!HW6_STATE.usersWithPositiveTrain.has(userId)) continue;
    const gradeMap = grades.get(userId);
    if (!gradeMap) continue;
    const excluded = hw6ExclusionSet(name, userId);
    let relevant = 0;
    gradeMap.forEach(function (rating, movieId) {
      if (rating >= 4.0 && HW6_STATE.candidateSet.has(movieId) && !excluded.has(movieId)) relevant++;
    });
    if (relevant > 0) cohort.push(userId);
  }
  return cohort;
}

/**
 * Count users and relevant movies that the cohort rule set aside, so the recall
 * number is interpretable (readme section 7.2).
 *
 * @param {string} splitName
 * @returns {{candidateUsers:number, cohortSize:number, excludedUnseen:number,
 *            excludedNoEligible:number, excludedNoPositiveEdge:number,
 *            unseenIds:number[]}}
 */
function hw6CohortExclusions(splitName) {
  const name = splitName === "val" ? "val" : "test";
  const grades = HW6_STATE.gradesBySplit[name];
  let candidateUsers = 0;
  let excludedNoPositiveEdge = 0;
  let excludedNoEligible = 0;
  let excludedUnseen = 0;
  const unseen = new Set();

  HW6_STATE.trainUserIds.forEach(function (userId) {
    const gradeMap = grades.get(userId);
    if (!gradeMap) return;
    candidateUsers++;
    if (!HW6_STATE.usersWithPositiveTrain.has(userId)) excludedNoPositiveEdge++;
    const excluded = hw6ExclusionSet(name, userId);
    let relevant = 0;
    gradeMap.forEach(function (rating, movieId) {
      if (rating < 4.0) return;
      if (!HW6_STATE.candidateSet.has(movieId)) {
        excludedUnseen++;
        unseen.add(movieId);
        return;
      }
      if (excluded.has(movieId)) return;
      relevant++;
    });
    if (relevant === 0) excludedNoEligible++;
  });

  return {
    candidateUsers: candidateUsers,
    cohortSize: hw6SharedCohort(name).length,
    excludedUnseen: excludedUnseen,
    excludedNoEligible: excludedNoEligible,
    excludedNoPositiveEdge: excludedNoPositiveEdge,
    unseenIds: Array.from(unseen).sort(function (a, b) { return a - b; }),
  };
}

// ---------------------------------------------------------------------------
// Provided matrix-factorization reference (instructor scaffolding)
//
// A 16-dimensional dot-product MF with a global mean, trained on the same
// chronological split, using `tf.train.adam`, checkpointed by validation RMSE
// (ties resolved toward the earliest epoch). It is deterministic: factors are
// initialised from a seeded PRNG and the mini-batch order is drawn from the same
// generator, so two runs on the same data produce the same model.
//
// Loss (per mini-batch of actual size `size`):
//   L = mean((rating - (u . i + globalMean))^2)
//     + (sum(uFactors^2) + sum(iFactors^2)) * reg / size
// The L2 term is divided by the ACTUAL batch size, so the last partial batch
// (train.length % batchSize) is regularised by its own size; this matches the
// previous assignment's normalisation convention.
//
// Differences from the submitted HW5 model, disclosed rather than hidden: HW6
// uses a fixed batchSize of 512 (HW5 default 64), 30 epochs (HW5 default 20),
// and a seeded Box-Muller initialisation (HW5 used unseeded tf.randomNormal).
// No previously saved HW5 weights are loaded: nothing here reads stored weights
// from disk or storage, so this is the same model FAMILY and objective as HW5,
// not the identical trained parameters.
//
// This block is NOT part of the homework stubs. Students report the checkpoint
// recorded in `MF_INFO`; they do not rebuild the model.
// ---------------------------------------------------------------------------

/**
 * Deterministic 32-bit PRNG (mulberry32). Returns a function producing values
 * in `[0, 1)`.
 *
 * @param {number} seed
 * @returns {() => number}
 */
function hw6Mulberry32(seed) {
  let a = seed >>> 0;
  return function () {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/**
 * Dispose the provided MF variables so repeated training does not leak memory.
 *
 * @returns {void}
 */
function hw6DisposeMf() {
  if (!HW6_MF_MODEL) return;
  try {
    if (HW6_MF_MODEL.userFactors) HW6_MF_MODEL.userFactors.dispose();
    if (HW6_MF_MODEL.itemFactors) HW6_MF_MODEL.itemFactors.dispose();
  } catch (error) {
    // Disposal is best-effort; the reference is replaced wholesale below.
  }
  HW6_MF_MODEL = null;
}

/**
 * Batched training RMSE for a split, using the provided MF variables.
 *
 * @param {Rating[]} split
 * @returns {number} RMSE, or `NaN` when the model is unavailable.
 */
function hw6MfRmse(split) {
  if (!HW6_MF_MODEL || !split || split.length === 0) return Number.NaN;
  const model = HW6_MF_MODEL;
  return tf.tidy(function () {
    const n = split.length;
    const uRows = model.userFactors.shape[0];
    const iRows = model.itemFactors.shape[0];
    const userIds = new Int32Array(n);
    const movieIds = new Int32Array(n);
    const actual = new Float32Array(n);
    for (let k = 0; k < n; k++) {
      const row = split[k];
      userIds[k] = Math.min(Math.max(0, row.userId), uRows - 1);
      movieIds[k] = Math.min(Math.max(0, row.movieId), iRows - 1);
      actual[k] = row.rating;
    }
    const u = tf.gather(model.userFactors, tf.tensor1d(userIds, "int32"));
    const i = tf.gather(model.itemFactors, tf.tensor1d(movieIds, "int32"));
    const pred = tf.sum(tf.mul(u, i), 1).add(model.globalMean);
    const diff = pred.sub(tf.tensor1d(actual, "float32"));
    return Math.sqrt(tf.mean(diff.square()).dataSync()[0]);
  });
}

/**
 * Train the provided MF reference at most once (idempotent, lazy). Nothing here
 * runs on page load: the first caller that actually needs MF scores triggers it,
 * and every later caller reuses the cached model.
 *
 * The training pass is asynchronous and `await`s the next animation frame after
 * every epoch, so the page keeps painting and the `#status` line can report
 * progress instead of freezing for minutes. Concurrent callers share one
 * in-flight promise, so a second need never starts a second training run.
 *
 * Deterministic; leaves the best-validation variables in place. `MF_INFO.trainedAt`
 * is set only on success, and `checkpoint` stays `null` while untrained.
 *
 * @returns {Promise<MfModel|null>} The trained model, or `null` when the data or
 *   the TensorFlow.js runtime is not ready, or when training throws.
 */
function hw6EnsureMf(run) {
  if (HW6_MF_TRAINING) return HW6_MF_TRAINING;
  if (HW6_MF_MODEL && MF_INFO.trainedAt !== null) return Promise.resolve(HW6_MF_MODEL);
  if (typeof tf === "undefined" || !HW6_STATE.splitReady) return Promise.resolve(null);
  const train = HW6_SPLIT.train;
  if (train.length === 0) return Promise.resolve(null);

  HW6_MF_TRAINING = hw6TrainMf(train, run).then(
    function (model) {
      HW6_MF_TRAINING = null;
      // MF scores changed, so any memoised MF recommendation list is stale.
      HW6_STATE.recCache.clear();
      return model;
    },
    function (error) {
      HW6_MF_TRAINING = null;
      // A cancellation is not a training failure: leave no partial model and
      // propagate the cancellation so the surrounding run is marked cancelled.
      if (hw6IsCancelledError(error)) {
        hw6DisposeMf();
        MF_INFO.checkpoint = null;
        MF_INFO.validationRMSE = null;
        MF_INFO.trainedAt = null;
        return Promise.reject(error);
      }
      console.error("MF reference training failed:", error);
      // Cover failures raised before the inner try (variable / tensor setup).
      hw6DisposeMf();
      MF_INFO.checkpoint = null;
      MF_INFO.validationRMSE = null;
      MF_INFO.trainedAt = null;
      return null;
    },
  );
  return HW6_MF_TRAINING;
}

/**
 * The actual MF training loop used by `hw6EnsureMf`. Kept separate so the
 * in-flight promise is installed before the first `await`, which is what makes
 * concurrent first-needs share a single run.
 *
 * Uses `tf.tidy` per batch, disposes the scalar loss returned by
 * `optimizer.minimize`, and never puts an async callback inside `tf.tidy`.
 *
 * @param {Rating[]} train Training rows.
 * @param {{cancelled:boolean, startedAt:number}} [run] Optional active-run token:
 *   when supplied, MF epoch progress is painted into the evaluation panel and a
 *   cooperative Cancel is honoured at epoch boundaries (never mid-batch, so the
 *   random order, objective, hyperparameters and checkpoint rule are unchanged).
 * @returns {Promise<MfModel|null>}
 */
async function hw6TrainMf(train, run) {
  hw6DisposeMf();
  const F = MF_FACTORS;
  const userRows = Math.max(1, numUsers + 1);
  const itemRows = Math.max(1, HW6_MAX_MOVIE_ID + 1);
  const rand = hw6Mulberry32(MF_SEED);
  const normal = function () {
    const u = Math.max(rand(), 1e-12);
    const v = rand();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  };

  const uData = new Float32Array(userRows * F);
  for (let k = 0; k < uData.length; k++) uData[k] = normal() * 0.05;
  const iData = new Float32Array(itemRows * F);
  for (let k = 0; k < iData.length; k++) iData[k] = normal() * 0.05;

  let globalMean = 0;
  for (let i = 0; i < train.length; i++) globalMean += train[i].rating;
  globalMean /= train.length;

  const userFactors = tf.variable(tf.tensor2d(uData, [userRows, F]));
  const itemFactors = tf.variable(tf.tensor2d(iData, [itemRows, F]));
  HW6_MF_MODEL = { userFactors: userFactors, itemFactors: itemFactors, globalMean: globalMean };

  const optimizer = tf.train.adam(MF_CONFIG.lr);
  const order = new Int32Array(train.length);
  for (let i = 0; i < order.length; i++) order[i] = i;

  let bestVal = Infinity;
  let bestEpoch = 0;
  let bestU = null;
  let bestI = null;
  MF_INFO.history = [];

  try {
    for (let epoch = 1; epoch <= MF_CONFIG.epochs; epoch++) {
      for (let i = order.length - 1; i > 0; i--) {
        const j = Math.floor(rand() * (i + 1));
        const tmp = order[i];
        order[i] = order[j];
        order[j] = tmp;
      }
      for (let start = 0; start < order.length; start += MF_CONFIG.batchSize) {
        const end = Math.min(start + MF_CONFIG.batchSize, order.length);
        const size = end - start;
        const uIdx = new Int32Array(size);
        const iIdx = new Int32Array(size);
        const target = new Float32Array(size);
        for (let k = 0; k < size; k++) {
          const row = train[order[start + k]];
          uIdx[k] = row.userId;
          iIdx[k] = row.movieId;
          target[k] = row.rating;
        }
        const cost = optimizer.minimize(
          function () {
            return tf.tidy(function () {
              const u = tf.gather(userFactors, tf.tensor1d(uIdx, "int32"));
              const it = tf.gather(itemFactors, tf.tensor1d(iIdx, "int32"));
              const pred = tf.sum(tf.mul(u, it), 1).add(globalMean);
              const mse = tf.losses.meanSquaredError(tf.tensor1d(target, "float32"), pred);
              const l2 = userFactors.square().sum().add(itemFactors.square().sum()).mul(MF_CONFIG.reg / size);
              return mse.add(l2);
            });
          },
          true,
          [userFactors, itemFactors],
        );
        if (cost) cost.dispose();
      }

      const trainRmse = hw6MfRmse(train);
      const valRmse = hw6MfRmse(HW6_SPLIT.val);
      MF_INFO.history.push({ epoch: epoch, trainRMSE: trainRmse, valRMSE: valRmse });
      if (Number.isFinite(valRmse) && valRmse < bestVal) {
        bestVal = valRmse;
        bestEpoch = epoch;
        bestU = userFactors.dataSync().slice();
        bestI = itemFactors.dataSync().slice();
      }

      // Announce progress and hand control back to the browser so the UI paints
      // between epochs instead of blocking the main thread.
      hw6SetStatus(
        "Training MF reference: epoch " + epoch + "/" + MF_CONFIG.epochs +
        " \u2014 validation RMSE " + hw6Fmt(valRmse) + "\u2026",
      );
      if (run) {
        hw6UpdateProgress({
          stage: "Training MF reference",
          alpha: null, alphaIndex: epoch, alphaTotal: MF_CONFIG.epochs,
          method: "mf", processed: epoch, total: MF_CONFIG.epochs,
          elapsed: Date.now() - run.startedAt, backend: HW6_STATE.backend,
          iteration: null, active: null,
        });
      }
      if (run && run.cancelled) {
        // Checkpoint boundary only: the epoch is complete, so nothing partial is
        // baked into a model. Disposing here guarantees no half-trained baseline.
        hw6DisposeMf();
        MF_INFO.checkpoint = null;
        MF_INFO.validationRMSE = null;
        MF_INFO.trainedAt = null;
        throw hw6CancelledError("MF reference training");
      }
      if (typeof tf.nextFrame === "function") {
        try {
          await tf.nextFrame();
        } catch (error) {
          // Yielding is best-effort; training still completes without it.
        }
      }
    }

    if (bestU && bestI) {
      tf.tidy(function () {
        userFactors.assign(tf.tensor2d(bestU, [userRows, F]));
        itemFactors.assign(tf.tensor2d(bestI, [itemRows, F]));
      });
    }

    MF_INFO.factors = F;
    MF_INFO.epochs = MF_CONFIG.epochs;
    MF_INFO.learningRate = MF_CONFIG.lr;
    MF_INFO.regularization = MF_CONFIG.reg;
    MF_INFO.batchSize = MF_CONFIG.batchSize;
    MF_INFO.seed = MF_SEED;
    MF_INFO.backend = HW6_STATE.backend;
    MF_INFO.checkpoint = bestEpoch || null;
    MF_INFO.validationRMSE = Number.isFinite(bestVal) ? bestVal : null;
    MF_INFO.trainedAt = new Date().toISOString();
    return HW6_MF_MODEL;
  } catch (error) {
    hw6DisposeMf();
    MF_INFO.checkpoint = null;
    MF_INFO.validationRMSE = null;
    MF_INFO.trainedAt = null;
    if (hw6IsCancelledError(error)) {
      // The epoch-boundary cancel already disposed and cleared; rethrow so the
      // evaluation is marked cancelled rather than failed.
      throw error;
    }
    console.error("MF reference training failed:", error);
    return null;
  }
}

/**
 * Ensure the lazy MF reference is trained, reporting "not trained yet" honestly
 * when it cannot be (TensorFlow.js missing, data not ready, or a training
 * failure). This is the single entry point every MF need goes through; it never
 * trains on page load.
 *
 * @param {{cancelled:boolean, startedAt:number}} [run] Optional active-run token
 *   used to report MF progress and honour a cooperative Cancel.
 * @returns {Promise<MfModel|null>}
 */
async function hw6EnsureMfForNeed(run) {
  if (HW6_MF_MODEL && MF_INFO.trainedAt !== null) return HW6_MF_MODEL;
  if (typeof tf === "undefined") return null;
  if (!HW6_STATE.splitReady) return null;
  hw6SetStatus("MF reference is not trained yet \u2014 training it now\u2026");
  const model = await hw6EnsureMf(run);
  if (model && MF_INFO.checkpoint !== null) {
    hw6SetStatus(
      "MF reference trained: " + MF_INFO.factors + " factors, checkpoint epoch " + MF_INFO.checkpoint +
      ", validation RMSE " + hw6Fmt(MF_INFO.validationRMSE) + ".",
    );
  } else {
    hw6SetStatus("MF reference is not trained yet \u2014 its scores and recall are not meaningful until it is.");
  }
  return model;
}

/**
 * Provided MF scorer: predicted scores for the training-movie candidate pool.
 *
 * Scores are dot products plus the global training mean, on the same scale as
 * the previous assignment's model. They are NOT comparable to PPR walk
 * probabilities.
 *
 * @param {number} userId Raw user id.
 * @returns {Object<number, number>} `movieId -> score`; empty when the model is
 *   untrained or the user row does not exist.
 */
function mfScores(userId) {
  const out = {};
  const model = HW6_MF_MODEL;
  if (!model) return out;
  const uid = Math.round(Number(userId));
  if (!Number.isFinite(uid) || uid < 0 || uid >= model.userFactors.shape[0]) return out;
  const ids = HW6_STATE.candidateIds;
  if (!ids.length) return out;
  const F = model.userFactors.shape[1];
  return tf.tidy(function () {
    const idx = tf.tensor1d(Int32Array.from(ids), "int32");
    const items = tf.gather(model.itemFactors, idx);
    const userRow = model.userFactors.slice([uid, 0], [1, F]);
    const dot = tf.matMul(items, userRow, false, true).squeeze();
    const values = dot.add(model.globalMean).dataSync();
    const result = {};
    for (let i = 0; i < ids.length; i++) result[ids[i]] = values[i];
    return result;
  });
}

// ---------------------------------------------------------------------------
// Student stubs (TODO(hw6))
//
// Each stub below throws until you implement it. Keep the signature, fill in the
// body, and update the JSDoc contract notes if your design differs.
//
// Provided helpers you will want:
//   `HW6_STATE.candidateIds`  ascending training-movie ids (the candidate pool)
//   `HW6_STATE.candidateSet`  `Set` view of the pool
//   `hw6PopularityOf(id)`     training popularity of one movie
//   `hw6ExclusionSet(split, userId)` movies to exclude for that split
//   `hw6SharedCohort(split)`  the shared evaluation cohort for a split
//   `hw6ScoreVector(method, userId, alpha, split)` candidate-aligned scores
// ---------------------------------------------------------------------------

/**
 * Extract movie-node scores from a full propagation vector.
 *
 * TODO(hw6): implement this. Readme section 5 requires movie-node scores only.
 *
 * Contract:
 *  - `p` is the node-probability vector produced by `pageRank` /
 *    `personalizedPageRank`, indexed by graph node index (length = node count).
 *  - Return a `Float64Array` of length `graph.movieIds.length`, where entry `k`
 *    is the score of the movie node whose raw id is `graph.movieIds[k]`. Do NOT
 *    return user-node scores, and do NOT change the order.
 *  - Prefer `graph.movieNodes` (node indices for the movies, same order) when it
 *    exists; otherwise the convention is that user nodes occupy indices
 *    `[0, graph.userIds.length)` and movie nodes follow, so entry `k` is
 *    `p[userNodeCount + k]`.
 *  - Never add a padding node: the output length is exactly
 *    `graph.movieIds.length`.
 *  - Throw the `TODO(hw6)` error until implemented; do not return zeros, because
 *    a silent zero vector makes every downstream metric look valid when it is not.
 *
 * @param {ArrayLike<number>} p Node probabilities, indexed by graph node index.
 * @param {Object} graph        Graph object from `graph.js`.
 * @returns {Float64Array} Movie scores in `graph.movieIds` order.
 */
function movieScores(p, graph) {
  if (!p || typeof p.length !== "number") {
    throw new Error("movieScores: p must be the node-probability vector from pageRank/personalizedPageRank");
  }
  if (!graph) {
    throw new Error("movieScores: graph is required");
  }

  const ids = hw6MovieIds(graph);
  const count = ids.length;
  // No padding node: the output has exactly one entry per graph movie.
  const out = new Float64Array(count);

  if (Array.isArray(graph.movieNodes) && graph.movieNodes.length === count) {
    for (let k = 0; k < count; k++) out[k] = Number(p[graph.movieNodes[k]]);
    return out;
  }

  // Convention: user nodes are [0, nUsers) and movie nodes follow in order, so
  // movie k's score is p[nUsers + k]. This keeps the raw algorithm score as-is
  // (PageRank/PPR probability mass), never a rating or a normalised value.
  const offset = Number.isFinite(graph.nUsers) ? Number(graph.nUsers) : hw6UserIds(graph).length;
  for (let k = 0; k < count; k++) out[k] = Number(p[offset + k]);
  return out;
}

/**
 * Recommend the top-K movies for a user with one of the four methods.
 *
 * TODO(hw6): implement this. Readme section 5 defines the rules.
 *
 * Contract:
 *  - `method` is one of `"popularity"`, `"pagerank"`, `"ppr"`, `"mf"`.
 *    `"pagerank"` and `"ppr"` must use the alpha currently selected in the
 *    `#alpha` slider and the cached propagation results returned by
 *    `hw6RunPageRank` / `hw6RunPPR` (or `pageRank` / `personalizedPageRank`
 *    directly through the adapter). Do not re-run a sweep here.
 *  - Candidate pool: every movie observed in the training split, i.e.
 *    `HW6_STATE.candidateIds`. Movies outside it can never be recommended.
 *  - Exclusion: for `split === "test"` drop movies the user rated in training or
 *    validation regardless of the rating value; for `split === "val"` drop
 *    training-rated movies only. Use `hw6ExclusionSet(split, userId)`.
 *  - Ordering: score descending, and ties broken by ASCENDING raw movie id.
 *  - Return `RecommendedMovie[]` of length `<= K` (fewer when the pool is
 *    smaller) with `{ movieId, title, genres, score, popularity }`.
 *  - When fewer than `K` eligible candidates exist, still return what is
 *    available; the renderer reports the shortfall. Note in the panel that
 *    zero-score candidates included only to fill the list have not demonstrated
 *    relevance.
 *  - Return `[]` (never throw) for an unknown method, `K <= 0`, or a user with
 *    no eligible candidate.
 *  - PPR scores are walk probabilities: never present them as ratings.
 *
 * @param {"popularity"|"pagerank"|"ppr"|"mf"} method
 * @param {number} userId Raw user id.
 * @param {number} [K=10]
 * @param {"val"|"test"} [split="test"]
 * @returns {RecommendedMovie[]}
 */
function recommendTopK(method, userId, K, split) {
  if (method !== "popularity" && method !== "pagerank" && method !== "ppr" && method !== "mf") return [];
  const requested = K === undefined || K === null ? HW6_K : Number(K);
  if (!Number.isFinite(requested) || requested <= 0) return [];
  const limit = Math.floor(requested);
  const name = split === "val" ? "val" : "test";

  // Memoise the ranked list. Recall and mean-list-popularity both need the same
  // list for the same user, so without this the whole candidate set would be
  // scored and sorted twice per user. The key includes alpha for graph methods,
  // so an alpha change (or a graph rebuild, via `graphEpoch`) can never serve a
  // stale list.
  const alphaKey = method === "pagerank" || method === "ppr" ? hw6AlphaKey(hw6EvalAlpha()) : "na";
  const recKey = method + "|" + userId + "|" + limit + "|" + name + "|" + HW6_STATE.graphEpoch + "|" + alphaKey;
  const memo = hw6CacheGet(HW6_STATE.recCache, recKey);
  if (memo !== undefined) return memo;

  // Scores for every candidate movie, aligned to HW6_STATE.candidateIds. For
  // pagerank/ppr this routes through the cached propagation and `movieScores`;
  // a missing graph surfaces as an HW6_INTERFACE error (an integration gap),
  // which is intentionally not swallowed here.
  const scored = hw6ScoreVector(method, userId, hw6EvalAlpha(), name);
  const ids = scored.movieIds;
  const scores = scored.scores;
  if (!ids || !ids.length) return [];

  const excluded = hw6ExclusionSet(name, userId);
  const items = [];
  for (let i = 0; i < ids.length; i++) {
    const movieId = ids[i];
    if (excluded.has(movieId)) continue;
    const score = Number(scores[i]);
    // -Infinity marks a movie the method cannot place (e.g. an unscored MF row);
    // it must never appear in a top-K list.
    if (!Number.isFinite(score)) continue;
    items.push({ movieId: movieId, score: score });
  }

  // Score descending; ties broken by ASCENDING raw movie id (readme section 5).
  items.sort(function (a, b) {
    if (b.score !== a.score) return b.score - a.score;
    return a.movieId - b.movieId;
  });

  const top = items.slice(0, limit);
  const out = new Array(top.length);
  for (let i = 0; i < top.length; i++) {
    const movieId = top[i].movieId;
    out[i] = {
      movieId: movieId,
      title: hw6Title(movieId),
      genres: hw6Genres(movieId),
      score: top[i].score,
      popularity: hw6PopularityOf(movieId),
    };
  }
  hw6CacheSet(HW6_STATE.recCache, recKey, out, HW6_CACHE_LIMIT.rec);
  return out;
}

/**
 * Recall@K for one method over the shared evaluation cohort.
 *
 * TODO(hw6): implement this. Readme sections 7.1 and 7.2 define the protocol.
 *
 * Contract:
 *  - Cohort: use `hw6SharedCohort(split)` so all four methods report the exact
 *    same users (the shared intersection). Do not build a per-method cohort.
 *  - A movie is relevant when its rating in the evaluation split is `>= 4.0`,
 *    it belongs to the training candidate pool, and it passes that split's
 *    previously-rated exclusion rule.
 *  - Compute the ranked lists with `recommendTopK(method, userId, K, split)`.
 *    For `split === "val"` the candidate lists must be computed FRESH with the
 *    validation exclusion rule; never reuse the test lists.
 *  - Return `{ recall, cohort, excludedUnseen, excludedNoEligible, perUser }`:
 *      `recall` macro-average of `hit / relevant` over the cohort (`0` when the
 *      cohort is empty; never divide by zero);
 *      `cohort` ascending user ids;
 *      `excludedUnseen` count of relevant movies absent from the training pool;
 *      `excludedNoEligible` count of users with no eligible relevant movie;
 *      `perUser` `[{ userId, relevant, hit }]` for cohort users only.
 *  - Never compute rating RMSE here: PPR outputs are not ratings.
 *
 * @param {"popularity"|"pagerank"|"ppr"|"mf"} method
 * @param {number} [K=10]
 * @param {"val"|"test"} [split="test"]
 * @returns {RecallReport}
 */
function recallAtK(method, K, split) {
  const requested = K === undefined || K === null ? HW6_K : Number(K);
  const limit = Number.isFinite(requested) && requested > 0 ? Math.floor(requested) : 0;
  const name = split === "val" ? "val" : "test";

  // Exactly the shared cohort helper, so all four methods report the same users.
  const cohort = hw6SharedCohort(name);
  const grades = HW6_STATE.gradesBySplit[name];
  const perUser = [];
  let recallSum = 0;

  for (let i = 0; i < cohort.length; i++) {
    const userId = cohort[i];
    const excluded = hw6ExclusionSet(name, userId);
    const gradeMap = grades.get(userId) || new Map();
    // Relevant set: rating >= 4.0 in the split, in the training candidate pool,
    // and passing this split's previously-rated exclusion rule.
    const relevant = new Set();
    gradeMap.forEach(function (rating, movieId) {
      if (rating >= 4.0 && HW6_STATE.candidateSet.has(movieId) && !excluded.has(movieId)) {
        relevant.add(movieId);
      }
    });

    let hit = 0;
    if (relevant.size > 0 && limit > 0) {
      // Fresh per-split lists; never reuse the other split's recommendations.
      const recs = recommendTopK(method, userId, limit, name);
      for (let r = 0; r < recs.length; r++) {
        if (relevant.has(recs[r].movieId)) hit++;
      }
    }

    perUser.push({ userId: userId, relevant: relevant.size, hit: hit });
    if (relevant.size > 0) recallSum += hit / relevant.size;
  }

  // Macro-average over the shared cohort; never divide by zero.
  const recall = cohort.length > 0 ? recallSum / cohort.length : 0;
  const exclusions = hw6CohortExclusions(name);

  return {
    recall: recall,
    cohort: cohort,
    excludedUnseen: exclusions.excludedUnseen,
    excludedNoEligible: exclusions.excludedNoEligible,
    perUser: perUser,
  };
}

// ---------------------------------------------------------------------------
// graph.js integration adapter
//
// `graph.js` is authored independently against this page's contract. The shapes
// below are the ones this adapter understands. Anything it cannot recognise is
// raised as an `HW6_INTERFACE` error, which the harness reports as PENDING
// rather than FAIL: an unknown call signature is an integration gap, not a wrong
// number. Everything the adapter reads is documented in the handoff notes.
// ---------------------------------------------------------------------------

/**
 * The currently active graph, or `null`.
 *
 * Prefers `getGraph()` when available, then the `GRAPH` global.
 *
 * @returns {Object|null}
 */
function hw6ActiveGraph() {
  if (typeof getGraph === "function") {
    try {
      const graph = getGraph();
      if (graph) return graph;
    } catch (error) {
      // Fall through to the global below.
    }
  }
  return typeof GRAPH !== "undefined" && GRAPH ? GRAPH : null;
}

/**
 * Node count of a graph object, or `0` when unknown.
 *
 * Reads, in order: `n`, `nodeCount`, `numNodes`, then `userIds.length +
 * movieIds.length`.
 *
 * @param {Object} graph
 * @returns {number}
 */
function hw6NodeCount(graph) {
  if (!graph) return 0;
  if (Number.isFinite(graph.n)) return graph.n;
  if (Number.isFinite(graph.nodeCount)) return graph.nodeCount;
  if (Number.isFinite(graph.numNodes)) return graph.numNodes;
  const users = Array.isArray(graph.userIds) ? graph.userIds.length : 0;
  const moviesArr = Array.isArray(graph.movieIds) ? graph.movieIds.length : 0;
  if (users || moviesArr) return users + moviesArr;
  if (Array.isArray(graph.nodes)) return graph.nodes.length;
  return 0;
}

/**
 * Raw user ids of a graph object, or `[]`.
 *
 * @param {Object} graph
 * @returns {number[]}
 */
function hw6UserIds(graph) {
  if (!graph) return [];
  if (Array.isArray(graph.userIds)) return graph.userIds;
  if (Array.isArray(graph.nodes)) {
    return graph.nodes
      .filter(function (node) { return node && (node.kind === "user" || node.type === "user"); })
      .map(function (node) { return node.id !== undefined ? node.id : node.index; });
  }
  return [];
}

/**
 * Raw movie ids of a graph object, or `[]`.
 *
 * @param {Object} graph
 * @returns {number[]}
 */
function hw6MovieIds(graph) {
  if (!graph) return [];
  if (Array.isArray(graph.movieIds)) return graph.movieIds;
  if (Array.isArray(graph.nodes)) {
    return graph.nodes
      .filter(function (node) { return node && (node.kind === "movie" || node.type === "movie"); })
      .map(function (node) { return node.id !== undefined ? node.id : node.index; });
  }
  return [];
}

/**
 * Build `movieId -> local movie index` for a graph.
 *
 * This is the index space of the array `movieScores` must return: entry `k` is
 * the score of `graph.movieIds[k]`, i.e. a movie-node-relative index, NOT a
 * global node index. Prefers graph.js's `movieIndexOf`; otherwise the position
 * inside `graph.movieIds`.
 *
 * @param {Object} graph
 * @returns {Map<number, number>}
 */
function hw6MovieLocalIndex(graph) {
  const index = new Map();
  if (!graph) return index;
  if (graph.movieIndexOf instanceof Map) {
    graph.movieIndexOf.forEach(function (localIndex, movieId) {
      index.set(Number(movieId), Number(localIndex));
    });
    return index;
  }
  const ids = hw6MovieIds(graph);
  for (let i = 0; i < ids.length; i++) index.set(Number(ids[i]), i);
  return index;
}

/**
 * Build `movieId -> global node index` for a graph, for the graph-view score
 * accessor (node order is users first, then movies).
 *
 * @param {Object} graph
 * @returns {Map<number, number>}
 */
function hw6MovieNodeIndex(graph) {
  const index = new Map();
  if (!graph) return index;
  const offset = Number.isFinite(graph.nUsers) ? Number(graph.nUsers) : hw6UserIds(graph).length;
  hw6MovieLocalIndex(graph).forEach(function (localIndex, movieId) {
    index.set(movieId, offset + localIndex);
  });
  return index;
}

/**
 * Normalise whatever a propagation function returned into a `PropagationResult`.
 *
 * Accepts a bare vector or an object carrying `scores` / `p` / `probabilities`
 * (optionally with `iterations`, `residual` / `delta` / `l1`, `converged`).
 * Returns `null` when the shape or the numbers cannot be trusted, so the caller
 * raises an interface error instead of computing with garbage.
 *
 * @param {*} out
 * @param {number} n Expected node count (`<= 0` to accept the vector length).
 * @returns {PropagationResult|null}
 */
function hw6NormalizePropagation(out, n) {
  if (!out) return null;
  let vector = null;
  let iterations = null;
  let residual = null;
  let converged = null;

  if (Array.isArray(out) || ArrayBuffer.isView(out)) {
    vector = out;
  } else if (typeof out === "object") {
    vector = out.scores || out.p || out.probabilities || out.vector || null;
    const itRaw = out.iterations !== undefined ? out.iterations : out.iters;
    const resRaw = out.residual !== undefined ? out.residual : out.delta !== undefined ? out.delta : out.l1;
    iterations = Number.isFinite(Number(itRaw)) ? Number(itRaw) : null;
    residual = Number.isFinite(Number(resRaw)) ? Number(resRaw) : null;
    converged = typeof out.converged === "boolean" ? out.converged : null;
  }
  if (!vector || typeof vector.length !== "number") return null;

  const size = n > 0 ? n : vector.length;
  if (vector.length !== size) return null;
  for (let i = 0; i < size; i++) {
    if (!Number.isFinite(Number(vector[i]))) return null;
  }
  return {
    scores: vector,
    iterations: iterations,
    residual: residual,
    converged: converged,
    source: "graph.js",
  };
}

/**
 * Run `fn` with `graph` installed as the active graph, then restore the previous
 * one. Used only when a propagation function reads the module-level `GRAPH`
 * instead of taking a graph argument.
 *
 * @param {Object} graph
 * @param {() => any} fn
 * @returns {any}
 */
function hw6WithActiveGraph(graph, fn) {
  const canSwap = typeof setGraph === "function" && typeof getGraph === "function";
  const previous = canSwap ? getGraph() : null;
  if (canSwap && previous !== graph) setGraph(graph);
  try {
    return fn();
  } finally {
    if (canSwap && previous !== graph) {
      try {
        setGraph(previous);
      } catch (error) {
        // Restoring the previous graph is best-effort.
      }
    }
  }
}

/**
 * Invoke `pageRank` for a specific graph and alpha.
 *
 * The real `graph.js` signature is `pageRank(alpha, opts)` reading the active
 * `GRAPH`, so the module-global form is tried first; a graph-first
 * `pageRank(graph, alpha)` shape is also tolerated.
 *
 * A `TODO(hw6)` throw from an unimplemented stub is propagated unchanged (the
 * harness turns it into PENDING). Only a genuinely unrecognised call shape
 * becomes `HW6_INTERFACE`.
 *
 * @param {Object} graph
 * @param {number} alpha
 * @returns {PropagationResult}
 * @throws {Error} The stub's own `TODO(hw6)` error, or `HW6_INTERFACE`.
 */
function hw6AdapterPageRank(graph, alpha) {
  if (typeof pageRank !== "function") throw hw6InterfaceError("graph.js `pageRank` is unavailable");
  const n = hw6NodeCount(graph);
  const attempts = [
    function () { return hw6WithActiveGraph(graph, function () { return pageRank(alpha); }); },
  ];
  if (pageRank.length >= 2) {
    attempts.push(function () { return pageRank(graph, alpha); });
  }

  let lastMessage = "";
  for (let i = 0; i < attempts.length; i++) {
    try {
      const normalized = hw6NormalizePropagation(attempts[i](), n);
      if (normalized) return normalized;
    } catch (error) {
      const message = String(error && error.message ? error.message : error);
      // An unimplemented stub is not an interface problem: surface it as-is.
      if (message.indexOf(HW6_TODO_MARKER) !== -1) throw error;
      lastMessage = message;
    }
  }
  throw hw6InterfaceError("graph.js `pageRank` signature/return unrecognised" + (lastMessage ? " (" + lastMessage + ")" : ""));
}

/**
 * Invoke `personalizedPageRank` for a graph, seed node and alpha.
 *
 * The real `graph.js` signature is `personalizedPageRank(userId, alpha, opts)`
 * reading the active `GRAPH`, so the user-first form is tried first; a
 * graph-first `(graph, seed, alpha)` shape is also tolerated. The seed passed to
 * the user-first form is the raw user id (not a node index) when the graph
 * exposes one.
 *
 * A `TODO(hw6)` throw is propagated unchanged. Only an unrecognised call shape
 * becomes `HW6_INTERFACE`.
 *
 * @param {Object} graph
 * @param {number} seedNode Seed node index within the graph.
 * @param {number} alpha
 * @param {number} [seedUserId] Raw user id, when known; defaults to `seedNode`.
 * @returns {PropagationResult}
 * @throws {Error} The stub's own `TODO(hw6)` error, or `HW6_INTERFACE`.
 */
function hw6AdapterPPR(graph, seedNode, alpha, seedUserId) {
  if (typeof personalizedPageRank !== "function") {
    throw hw6InterfaceError("graph.js `personalizedPageRank` is unavailable");
  }
  const n = hw6NodeCount(graph);
  const users = hw6UserIds(graph);
  const rawUserId = seedUserId === undefined || seedUserId === null
    ? (seedNode >= 0 && seedNode < users.length ? users[seedNode] : seedNode)
    : seedUserId;

  const attempts = [
    function () { return hw6WithActiveGraph(graph, function () { return personalizedPageRank(rawUserId, alpha); }); },
  ];
  if (personalizedPageRank.length >= 3) {
    attempts.push(function () { return personalizedPageRank(graph, seedNode, alpha); });
  }
  attempts.push(function () { return hw6WithActiveGraph(graph, function () { return personalizedPageRank(rawUserId); }); });

  let lastMessage = "";
  for (let i = 0; i < attempts.length; i++) {
    try {
      const normalized = hw6NormalizePropagation(attempts[i](), n);
      if (normalized) return normalized;
    } catch (error) {
      const message = String(error && error.message ? error.message : error);
      if (message.indexOf(HW6_TODO_MARKER) !== -1) throw error;
      lastMessage = message;
    }
  }
  throw hw6InterfaceError("graph.js `personalizedPageRank` signature/return unrecognised" + (lastMessage ? " (" + lastMessage + ")" : ""));
}

/**
 * Node index of a raw user id.
 *
 * Prefers graph.js's own mapping (`nodeOfUser`, then `userIndexOf` or the
 * `userIndex` alias), then the position inside `graph.userIds`, then the
 * `[0, userIds.length)` convention.
 *
 * @param {Object} graph
 * @param {number} userId
 * @returns {number} Node index, or `-1`.
 */
function hw6UserNodeIndex(graph, userId) {
  if (!graph) return -1;
  if (typeof graph.nodeOfUser === "function") {
    try {
      const node = Number(graph.nodeOfUser(userId));
      if (Number.isFinite(node) && node >= 0) return node;
    } catch (error) {
      // Fall through to the map / array lookups below.
    }
  }
  const maps = [graph.userIndexOf, graph.userIndex];
  for (let i = 0; i < maps.length; i++) {
    if (maps[i] instanceof Map) {
      const mapped = maps[i].get(userId);
      if (mapped !== undefined) return Number(mapped);
    }
  }
  const users = hw6UserIds(graph);
  const position = users.indexOf(userId);
  if (position !== -1) return position;
  const numeric = Number(userId);
  return Number.isFinite(numeric) && numeric >= 0 && numeric < hw6NodeCount(graph) ? numeric : -1;
}

/**
 * Cached global PageRank for the active graph at `alpha`.
 *
 * @param {Object} graph
 * @param {number} alpha
 * @returns {PropagationResult}
 */
function hw6RunPageRank(graph, alpha) {
  const key = "pr|" + HW6_STATE.graphEpoch + "|" + hw6AlphaKey(alpha);
  const cached = hw6CacheGet(HW6_STATE.pageRankCache, key);
  if (cached && cached.graph === graph) return cached;
  const result = hw6AdapterPageRank(graph, alpha);
  const entry = { ...result, graph: graph, alpha: alpha };
  hw6CacheSet(HW6_STATE.pageRankCache, key, entry, HW6_CACHE_LIMIT.pageRank);
  return entry;
}

/**
 * Cached PPR for the active graph, restart user and `alpha`. The cache key
 * includes alpha (and the graph epoch), so changing alpha or rebuilding the
 * graph can never reuse a stale walk, while a 1 -> 2 -> 1 sequence reuses the
 * first user's walk instead of recomputing it.
 *
 * @param {Object} graph
 * @param {number} userId
 * @param {number} alpha
 * @returns {PropagationResult}
 */
function hw6RunPPR(graph, userId, alpha) {
  const key = "ppr|" + HW6_STATE.graphEpoch + "|" + userId + "|" + hw6AlphaKey(alpha);
  const cached = hw6CacheGet(HW6_STATE.pprCache, key);
  if (cached && cached.graph === graph) return cached;
  const seedNode = hw6UserNodeIndex(graph, userId);
  if (seedNode < 0) throw hw6InterfaceError("cannot map user " + userId + " to a graph node");
  const result = hw6AdapterPPR(graph, seedNode, alpha, userId);
  const entry = { ...result, graph: graph, userId: userId, alpha: alpha };
  hw6CacheSet(HW6_STATE.pprCache, key, entry, HW6_CACHE_LIMIT.ppr);
  return entry;
}

/**
 * Store one batched PPR result (from `graphPropagateBatch`) into the cache, in
 * the same shape `hw6RunPPR` returns, so the synchronous evaluation path treats
 * a pre-filled vector exactly like a freshly computed one.
 *
 * @param {Object} graph
 * @param {number} userId
 * @param {number} alpha
 * @param {{p:Float64Array, iterations:number, residual:number, converged:boolean}} prop
 * @returns {void}
 */
function hw6StorePprResult(graph, userId, alpha, prop) {
  const key = "ppr|" + HW6_STATE.graphEpoch + "|" + userId + "|" + hw6AlphaKey(alpha);
  const entry = {
    scores: prop.p,
    iterations: prop.iterations,
    residual: prop.residual,
    converged: prop.converged,
    source: "graph.js:batch",
    graph: graph,
    userId: userId,
    alpha: alpha,
  };
  hw6CacheSet(HW6_STATE.pprCache, key, entry, HW6_CACHE_LIMIT.ppr);
}

/**
 * Store the batched global PageRank result into the cache (same shape as
 * `hw6RunPageRank`).
 *
 * @param {Object} graph
 * @param {number} alpha
 * @param {{p:Float64Array, iterations:number, residual:number, converged:boolean}} prop
 * @returns {void}
 */
function hw6StorePageRankResult(graph, alpha, prop) {
  const key = "pr|" + HW6_STATE.graphEpoch + "|" + hw6AlphaKey(alpha);
  const entry = {
    scores: prop.p,
    iterations: prop.iterations,
    residual: prop.residual,
    converged: prop.converged,
    source: "graph.js:batch",
    graph: graph,
    alpha: alpha,
  };
  hw6CacheSet(HW6_STATE.pageRankCache, key, entry, HW6_CACHE_LIMIT.pageRank);
}

/**
 * Cache key for one user's PPR walk at one alpha, at the current graph epoch. The
 * epoch makes a graph rebuild miss every old key, and alpha/user are part of the
 * key, so a cache hit is only ever a genuinely reusable walk. Split is omitted
 * because a PPR vector is a property of the graph and the restart user, not of
 * the split it happens to be scored on.
 *
 * @param {number} userId
 * @param {number} alpha
 * @returns {string}
 */
function hw6PprCacheKey(userId, alpha) {
  return "ppr|" + HW6_STATE.graphEpoch + "|" + userId + "|" + hw6AlphaKey(alpha);
}

/**
 * Cache key for the global PageRank walk at one alpha, at the current epoch.
 *
 * @param {number} alpha
 * @returns {string}
 */
function hw6PageRankCacheKey(alpha) {
  return "pr|" + HW6_STATE.graphEpoch + "|" + hw6AlphaKey(alpha);
}

/**
 * Non-mutating lookup of a reusable PPR walk. Returns the cached entry only when
 * it exists for the current epoch and was computed on the same graph object, so
 * a rebuild can never serve a stale vector. Unlike `hw6CacheGet` it does not
 * refresh recency, which keeps a read-only prefill scan from reordering the LRU.
 *
 * @param {Object} graph
 * @param {number} userId
 * @param {number} alpha
 * @returns {PropagationResult|null}
 */
function hw6PeekPpr(graph, userId, alpha) {
  const entry = HW6_STATE.pprCache.get(hw6PprCacheKey(userId, alpha));
  return entry && entry.value && entry.value.graph === graph ? entry.value : null;
}

/**
 * Non-mutating lookup of the reusable global PageRank walk at `alpha`.
 *
 * @param {Object} graph
 * @param {number} alpha
 * @returns {PropagationResult|null}
 */
function hw6PeekPageRank(graph, alpha) {
  const entry = HW6_STATE.pageRankCache.get(hw6PageRankCacheKey(alpha));
  return entry && entry.value && entry.value.graph === graph ? entry.value : null;
}

/**
 * Drop every cached PPR vector for one alpha (across users) at the current graph
 * epoch. Used after a sweep alpha has been scored, so the sweep holds at most
 * one alpha of val vectors at a time instead of all four.
 *
 * @param {number} alpha
 * @returns {void}
 */
function hw6DropPprAlpha(alpha) {
  const suffix = "|" + hw6AlphaKey(alpha);
  const prefix = "ppr|" + HW6_STATE.graphEpoch + "|";
  const doomed = [];
  HW6_STATE.pprCache.forEach(function (entry, key) {
    if (key.indexOf(prefix) === 0 && key.slice(-suffix.length) === suffix) doomed.push(key);
  });
  for (let i = 0; i < doomed.length; i++) HW6_STATE.pprCache.delete(doomed[i]);
}

/**
 * Candidate-aligned score vector for one method (provided plumbing).
 *
 * Returns `{ movieIds, scores, propagation }` where `scores` is a `Float64Array`
 * aligned to `HW6_STATE.candidateIds`. Scores for movies the method cannot place
 * are `-Infinity`, so they sort last and never enter a top-K list by accident.
 *
 * `pagerank` / `ppr` route through `movieScores`, which is a student stub: this
 * helper therefore throws the `TODO(hw6)` error until the stub is implemented.
 *
 * @param {string} method
 * @param {number} userId
 * @param {number} alpha
 * @param {"val"|"test"} split
 * @returns {{movieIds:number[], scores:Float64Array, propagation:(PropagationResult|null)}}
 */
function hw6ScoreVector(method, userId, alpha, split) {
  const ids = HW6_STATE.candidateIds;
  const scores = new Float64Array(ids.length);
  if (!ids.length) return { movieIds: ids, scores: scores, propagation: null };

  if (method === "popularity") {
    for (let i = 0; i < ids.length; i++) scores[i] = hw6PopularityOf(ids[i]);
    return { movieIds: ids, scores: scores, propagation: null };
  }

  if (method === "mf") {
    const scored = mfScores(userId);
    for (let i = 0; i < ids.length; i++) {
      const value = scored[ids[i]];
      scores[i] = value === undefined ? Number.NEGATIVE_INFINITY : value;
    }
    return { movieIds: ids, scores: scores, propagation: null };
  }

  const graph = hw6ActiveGraph();
  if (!graph) throw hw6InterfaceError("no graph is active; press \"Build graph\" first");
  const propagation = method === "ppr" ? hw6RunPPR(graph, userId, alpha) : hw6RunPageRank(graph, alpha);
  // `movieScores` returns movie-node-relative entries, so the lookup here must
  // be the LOCAL movie index; using the global node index would skip the second
  // half of the movie array.
  const perMovie = movieScores(propagation.scores, graph);
  const index = hw6MovieLocalIndex(graph);
  for (let i = 0; i < ids.length; i++) {
    const position = index.get(ids[i]);
    const value = position === undefined ? NaN : Number(perMovie[position]);
    scores[i] = Number.isFinite(value) ? value : Number.NEGATIVE_INFINITY;
  }
  return { movieIds: ids, scores: scores, propagation: propagation };
}

// ---------------------------------------------------------------------------
// Provided runners
// ---------------------------------------------------------------------------

/**
 * Mean training popularity of the movies in one recommended list (readme 7.3).
 *
 * @param {RecommendedMovie[]} recs
 * @returns {number|null} Mean popularity, or `null` for an empty list.
 */
function hw6MeanPopularity(recs) {
  if (!recs || recs.length === 0) return null;
  let sum = 0;
  for (let i = 0; i < recs.length; i++) {
    const value = Number(recs[i].popularity);
    sum += Number.isFinite(value) ? value : hw6PopularityOf(recs[i].movieId);
  }
  return sum / recs.length;
}

/**
 * Record recommended rows for the recommendations CSV, de-duplicating on
 * `(userId, method, alpha, rank)` so repeated clicks do not duplicate rows.
 *
 * @param {number} userId
 * @param {string} method
 * @param {number|null} alpha
 * @param {RecommendedMovie[]} recs
 * @returns {void}
 */
function hw6RecordRecommendationRows(userId, method, alpha, recs) {
  for (let i = 0; i < recs.length; i++) {
    const rec = recs[i];
    const alphaKey = alpha === null || alpha === undefined ? "na" : Number(alpha).toFixed(2);
    const key = userId + "|" + method + "|" + alphaKey + "|" + (i + 1);
    if (HW6_STATE.recommendationKeys.has(key)) continue;
    HW6_STATE.recommendationKeys.add(key);
    HW6_STATE.recommendationRows.push({
      userId: userId,
      method: method,
      alpha: alpha === null || alpha === undefined ? "" : Number(alpha).toFixed(2),
      rank: i + 1,
      movieId: rec.movieId,
      title: rec.title,
      genres: Array.isArray(rec.genres) ? rec.genres.join(";") : "",
      score: rec.score,
      trainingPopularity: Number.isFinite(Number(rec.popularity)) ? rec.popularity : hw6PopularityOf(rec.movieId),
    });
  }
}

/**
 * Clear the recorded recommendation rows/keys. Called at the start of each
 * completed primary evaluation so the exported recommendations reflect exactly
 * the last completed run's settings instead of accumulating lists from unrelated
 * slider positions or earlier runs.
 *
 * @returns {void}
 */
function hw6ResetRecommendationRows() {
  HW6_STATE.recommendationRows.length = 0;
  HW6_STATE.recommendationKeys.clear();
}

/**
 * Deterministically record the assignment's required recommendation evidence for
 * the two fixed common-cohort users: their four-method Top-10 lists at the
 * selected primary alpha, plus the three controlled PPR settings (user A @ 0.85,
 * user B @ 0.85, user A @ 0.50). This is what makes `recommendations.csv`
 * non-empty after Evaluate alone, and it never depends on the demo slider.
 *
 * Each method's two lists are computed under an explicit alpha context so the
 * slider is not consulted or moved. Lists that cannot be produced (for example
 * MF before it is trained) are skipped rather than exported as invented rows.
 *
 * @param {number} alpha Selected primary alpha for PageRank / PPR.
 * @returns {{users:number[], methods:Array<Object>}} Summary for the snapshot.
 */
function hw6RecordCohortRecommendations(alpha) {
  const cohort = hw6SharedCohort("test");
  if (cohort.length < 2) return { users: cohort.slice(0, 2), methods: [] };
  const userA = cohort[0];
  const userB = cohort[1];
  const methods = ["popularity", "pagerank", "ppr", "mf"];
  const summary = [];

  const withContext = function (value, fn) {
    const previous = HW6_STATE.alphaContext;
    HW6_STATE.alphaContext = value;
    try {
      return fn();
    } finally {
      HW6_STATE.alphaContext = previous;
    }
  };

  for (let i = 0; i < methods.length; i++) {
    const method = methods[i];
    const usesAlpha = method === "pagerank" || method === "ppr";
    const methodAlpha = usesAlpha ? alpha : null;
    let listA = null; let listB = null;
    try {
      listA = usesAlpha ? withContext(alpha, function () { return recommendTopK(method, userA, HW6_K, "test"); }) : recommendTopK(method, userA, HW6_K, "test");
    } catch (e) { listA = null; }
    try {
      listB = usesAlpha ? withContext(alpha, function () { return recommendTopK(method, userB, HW6_K, "test"); }) : recommendTopK(method, userB, HW6_K, "test");
    } catch (e) { listB = null; }
    if (listA) hw6RecordRecommendationRows(userA, method, methodAlpha, listA);
    else throw new Error("failed to generate " + method + " for user " + userA);
    if (listB) hw6RecordRecommendationRows(userB, method, methodAlpha, listB);
    else throw new Error("failed to generate " + method + " for user " + userB);
    let common = null;
    if (listA && listB) {
      const set = new Set(listA.map(function (r) { return r.movieId; }));
      common = 0; listB.forEach(function (r) { if (set.has(r.movieId)) common += 1; });
    }
    summary.push({ method: method, alpha: methodAlpha, meanPopularityA: hw6MeanPopularity(listA), meanPopularityB: hw6MeanPopularity(listB), common: common, sizeA: listA.length, sizeB: listB.length });
  }

  // The readme section 6.2 controlled PPR settings, always at their fixed alphas.
  const controls = [
    { user: userA, alpha: 0.85, label: "baseline" },
    { user: userB, alpha: 0.85, label: "change seed only" },
    { user: userA, alpha: 0.50, label: "change alpha only" },
  ];
  controls.forEach(function (control) {
    try {
      const recs = withContext(control.alpha, function () { return recommendTopK("ppr", control.user, HW6_K, "test"); });
      if (recs && recs.length === HW6_K) hw6RecordRecommendationRows(control.user, "ppr", control.alpha, recs);
      else throw new Error("incomplete ppr controls");
    } catch (error) {
      throw error;
    }
  });

  return {
    users: [userA, userB],
    methods: summary,
    rows: hw6CloneRows(HW6_STATE.recommendationRows),
  };
}

/**
 * Run one propagation pass and store its convergence record for the panel.
 *
 * @param {"pagerank"|"ppr"} kind
 * @param {Object} graph
 * @param {number} alpha
 * @param {number} [userId] Seed user for PPR.
 * @returns {PropagationResult}
 */
function hw6RunAndRecordConvergence(kind, graph, alpha, userId) {
  const result = kind === "ppr" ? hw6RunPPR(graph, userId, alpha) : hw6RunPageRank(graph, alpha);
  HW6_STATE.convergence[kind] = {
    alpha: alpha,
    userId: kind === "ppr" ? userId : null,
    iterations: result.iterations,
    residual: result.residual,
    converged: result.converged,
    source: result.source,
    at: new Date().toISOString(),
  };
  return result;
}

/**
 * Derive a convergence verdict only when the propagation function did not
 * report one. An iteration cap is never treated as proof of convergence.
 *
 * @param {{iterations:(number|null), residual:(number|null), converged:(boolean|null)}} record
 * @returns {{value:boolean|null, derived:boolean}} Verdict and whether it was derived here.
 */
function hw6ConvergenceVerdict(record) {
  if (!record) return { value: null, derived: false };
  if (typeof record.converged === "boolean") return { value: record.converged, derived: false };
  if (Number.isFinite(record.residual)) return { value: record.residual < HW6_TOLERANCE, derived: true };
  if (Number.isFinite(record.iterations)) return { value: record.iterations < HW6_MAX_ITERATIONS, derived: true };
  return { value: null, derived: true };
}

/**
 * Four-method comparison runner on the shared cohort (readme sections 6 and 7).
 *
 * Recall@10 for popularity, global PageRank, PPR and MF, plus cohort size,
 * exclusion counts and mean recommended-list popularity. The alpha recorded for
 * PageRank and PPR is the selected validation alpha; MF keeps its own
 * validation-RMSE checkpoint, so the validation objectives differ and the
 * report must say so.
 *
 * The MF reference is trained lazily here (the comparison is the first place
 * that actually needs it). If it cannot be trained, the MF row keeps a `null`
 * recall plus an explicit note rather than a silent zero.
 *
 * @param {"test"|"val"} [split="test"] Evaluation split.
 * @returns {Promise<Array<Object>>} One row per method; rows carry an `error`
 *   field when the method could not be evaluated (unimplemented stub or
 *   interface gap).
 */
async function hw6RunComparison(split) {
  const name = split === "val" ? "val" : "test";
  const alpha = hw6EvalAlpha();
  const cohort = hw6SharedCohort(name);
  const exclusions = hw6CohortExclusions(name);
  const methods = ["popularity", "pagerank", "ppr", "mf"];
  const rows = [];

  // The four-method comparison needs MF, so this is a legitimate lazy trigger.
  await hw6EnsureMfForNeed();
  const mfReady = HW6_MF_MODEL !== null && MF_INFO.trainedAt !== null;

  for (let m = 0; m < methods.length; m++) {
    const method = methods[m];
    const usesAlpha = method === "pagerank" || method === "ppr";
    const row = {
      method: method,
      split: name,
      alpha: usesAlpha ? alpha : null,
      recall: null,
      cohortSize: cohort.length,
      excludedUnseen: exclusions.excludedUnseen,
      excludedNoEligible: exclusions.excludedNoEligible,
      meanPopularity: null,
      iterations: null,
      residual: null,
      converged: null,
      error: null,
      note: null,
    };
    try {
      if (method === "mf" && !mfReady) {
        // Never report a silent 0: state plainly that the reference is untrained.
        row.note = "MF not trained yet \u2014 its recall is not meaningful.";
      } else {
        const report = recallAtK(method, HW6_K, name);
        row.recall = Number(report.recall);
        if (Array.isArray(report.cohort)) row.cohortSize = report.cohort.length;
        if (Number.isFinite(Number(report.excludedUnseen))) row.excludedUnseen = Number(report.excludedUnseen);
        if (Number.isFinite(Number(report.excludedNoEligible))) row.excludedNoEligible = Number(report.excludedNoEligible);
        row.meanPopularity = hw6MeanListPopularityForCohort(method, cohort, name, alpha);
        const record = usesAlpha ? HW6_STATE.convergence[method === "ppr" ? "ppr" : "pagerank"] : null;
        if (record) {
          row.iterations = record.iterations;
          row.residual = record.residual;
          row.converged = hw6ConvergenceVerdict(record).value;
        }
      }
    } catch (error) {
      row.error = String(error && error.message ? error.message : error);
    }
    rows.push(row);
  }

  HW6_STATE.comparisonRows = rows;
  return rows;
}

/**
 * Mean over the cohort of each user's recommended-list mean popularity.
 *
 * @param {string} method
 * @param {number[]} cohort
 * @param {"test"|"val"} split
 * @param {number} alpha
 * @returns {number|null} `null` when no list could be produced.
 */
function hw6MeanListPopularityForCohort(method, cohort, split, alpha) {
  let total = 0;
  let users = 0;
  for (let i = 0; i < cohort.length; i++) {
    const recs = recommendTopK(method, cohort[i], HW6_K, split);
    const mean = hw6MeanPopularity(recs);
    if (mean === null) continue;
    total += mean;
    users++;
  }
  return users > 0 ? total / users : null;
}

// ---------------------------------------------------------------------------
// Asynchronous evaluation engine (progress + cancel)
//
// The synchronous runners above stay the harness-facing contract. This engine
// drives the UI: it pre-fills the propagation caches with batched, yielding
// walks (so the browser paints and Cancel works), then calls the very same
// synchronous runners, which become cache hits. Recall and mean-list popularity
// reuse one memoised list per user via `recommendTopK`'s cache.
// ---------------------------------------------------------------------------

/**
 * Build the cancellation error used by the async engine.
 *
 * @param {string} [where]
 * @returns {Error}
 */
function hw6CancelledError(where) {
  const error = new Error("HW6_RUN_CANCELLED" + (where ? ": " + where : ""));
  error.hw6Cancelled = true;
  return error;
}

/**
 * True when a thrown value is the engine's cancellation error.
 *
 * @param {*} error
 * @returns {boolean}
 */
function hw6IsCancelledError(error) {
  return !!(error && error.hw6Cancelled === true);
}

/** @returns {Object} The progress-panel elements (any may be `null`). */
function hw6ProgressElements() {
  return {
    panel: document.getElementById("hw6-progress"),
    stage: document.getElementById("hw6-progress-stage"),
    pct: document.getElementById("hw6-progress-pct"),
    bar: document.getElementById("hw6-progress-bar"),
    detail: document.getElementById("hw6-progress-detail"),
  };
}

/**
 * Paint the live progress panel from an engine status object.
 *
 * @param {{stage:(string|null), alpha:(number|null), alphaIndex:(number|null),
 *          alphaTotal:(number|null), method:(string|null), processed:number,
 *          total:(number|null), elapsed:(number|null), backend:(string|null)}} info
 * @returns {void}
 */
function hw6UpdateProgress(info) {
  const el = hw6ProgressElements();
  if (!el.panel) return;
  el.panel.classList.remove("hidden");

  const head = [];
  if (info.stage) head.push(info.stage);
  if (info.alpha !== null && info.alpha !== undefined) {
    let alphaText = "alpha " + Number(info.alpha).toFixed(2);
    if (info.alphaIndex && info.alphaTotal) alphaText += " (" + info.alphaIndex + "/" + info.alphaTotal + ")";
    head.push(alphaText);
  }
  if (info.method) head.push("method " + info.method);
  if (el.stage) el.stage.textContent = head.length ? head.join(" \u00b7 ") : "Working\u2026";

  const total = info.total;
  const processed = info.processed || 0;
  const determinate = typeof total === "number" && Number.isFinite(total) && total > 0;
  if (determinate) {
    const pct = Math.max(0, Math.min(100, (processed / total) * 100));
    if (el.bar) {
      el.bar.classList.remove("progress-bar--indeterminate");
      el.bar.style.width = pct.toFixed(1) + "%";
    }
    if (el.pct) el.pct.textContent = pct.toFixed(0) + "% (" + processed + "/" + total + ")";
  } else {
    if (el.bar) {
      el.bar.classList.add("progress-bar--indeterminate");
      el.bar.style.width = "100%";
    }
    if (el.pct) el.pct.textContent = "\u2014";
  }

  const detail = [];
  if (info.elapsed !== null && info.elapsed !== undefined) detail.push("elapsed " + (info.elapsed / 1000).toFixed(1) + " s");
  if (info.iteration !== null && info.iteration !== undefined) {
    detail.push("iteration " + info.iteration + (info.active !== null && info.active !== undefined ? " \u00b7 active " + info.active : ""));
  }
  if (info.backend) detail.push("backend " + info.backend);
  if (el.detail) el.detail.textContent = detail.join(" \u00b7 ");
}

/**
 * Throttled intra-chunk progress update, driven by the batched kernel's
 * `onIteration` hook. It reports the stage, the sweep alpha index, the number of
 * completed cohort users, the current iteration and the still-active column
 * count, so the panel keeps moving during a slow 64-user chunk instead of
 * freezing between chunk boundaries. Repaints are throttled to
 * `HW6_PROGRESS_MIN_REPAINT_MS` to avoid churning the DOM on every iteration.
 *
 * @param {Object} meta `{ stage, alphaIndex, alphaTotal, method }`
 * @param {number} alpha
 * @param {{cancelled:boolean, startedAt:number}} run
 * @param {number} processed Cohort users completed before this chunk.
 * @param {number} total     Cohort users in this stage.
 * @param {number} iteration Kernel iteration just finished.
 * @param {number} active    Columns still iterating.
 * @param {number} chunkSize Columns in this chunk.
 * @returns {void}
 */
function hw6UpdateProgressChunk(meta, alpha, run, processed, total, iteration, active, chunkSize) {
  const now = Date.now();
  if (HW6_STATE.lastProgressPaint !== 0 && now - HW6_STATE.lastProgressPaint < HW6_PROGRESS_MIN_REPAINT_MS) {
    return;
  }
  HW6_STATE.lastProgressPaint = now;
  hw6UpdateProgress({
    stage: meta.stage,
    alpha: alpha,
    alphaIndex: meta.alphaIndex,
    alphaTotal: meta.alphaTotal,
    method: meta.method,
    processed: processed,
    total: total,
    elapsed: now - run.startedAt,
    backend: HW6_STATE.backend,
    iteration: iteration,
    active: active,
    chunkSize: chunkSize,
  });
}

/**
 * Retain a final, readable run summary after the progress panel hides, so
 * cold/warm elapsed time and the backend are not lost when a run completes.
 *
 * @param {{stage:(string|null), elapsedMs:(number|null), backend:(string|null),
 *          detail:(string|null)}} summary
 * @returns {void}
 */
function hw6RecordRunSummary(summary) {
  HW6_STATE.lastRunSummary = summary;
  const el = document.getElementById("hw6-run-summary");
  if (!el) return;
  const parts = [];
  if (summary.stage) parts.push(summary.stage);
  if (summary.detail) parts.push(summary.detail);
  if (summary.elapsedMs !== null && summary.elapsedMs !== undefined) parts.push("elapsed " + (summary.elapsedMs / 1000).toFixed(1) + " s");
  if (summary.backend) parts.push("backend " + summary.backend);
  el.textContent = parts.join(" \u00b7 ");
}

/** @returns {void} */
function hw6HideProgress() {
  const el = hw6ProgressElements();
  if (el.panel) el.panel.classList.add("hidden");
}

/**
 * Toggle the "a run is in flight" UI state: conflicting buttons are disabled so
 * two evaluations cannot race, and the Cancel button is enabled only while a run
 * is active.
 *
 * @param {boolean} running
 * @returns {void}
 */
function hw6SetRunning(running) {
  if (running) {
    // Record each guard's own prior disabled state, then disable it. Restoring
    // the recorded value (rather than a blanket enable) preserves any control
    // that was already unavailable for an independent reason.
    HW6_STATE.guardPriorDisabled = {};
    for (let i = 0; i < HW6_RUN_GUARD_IDS.length; i++) {
      const id = HW6_RUN_GUARD_IDS[i];
      const button = document.getElementById(id);
      if (!button) continue;
      HW6_STATE.guardPriorDisabled[id] = !!button.disabled;
      button.disabled = true;
    }
    for (let i = 0; i < HW6_RUN_GUARD_INPUTS.length; i++) {
      const id = HW6_RUN_GUARD_INPUTS[i];
      const input = document.getElementById(id);
      if (!input) continue;
      HW6_STATE.guardPriorDisabled[id] = !!input.disabled;
      input.disabled = true;
    }
  } else {
    const prior = HW6_STATE.guardPriorDisabled || {};
    for (let i = 0; i < HW6_RUN_GUARD_IDS.length; i++) {
      const id = HW6_RUN_GUARD_IDS[i];
      const button = document.getElementById(id);
      if (button && Object.prototype.hasOwnProperty.call(prior, id)) button.disabled = prior[id];
    }
    for (let i = 0; i < HW6_RUN_GUARD_INPUTS.length; i++) {
      const id = HW6_RUN_GUARD_INPUTS[i];
      const input = document.getElementById(id);
      if (input && Object.prototype.hasOwnProperty.call(prior, id)) input.disabled = prior[id];
    }
    HW6_STATE.guardPriorDisabled = null;
  }
  const cancel = document.getElementById("hw6-cancel");
  if (cancel) cancel.disabled = !running;
  if (!running) HW6_STATE.activeRun = null;
}

/**
 * True while an evaluation is in flight. Used to refuse conflicting direct
 * actions (self-checks that swap the graph, a rebuild, a second comparison)
 * even if a caller bypasses the disabled buttons.
 *
 * @returns {boolean}
 */
function hw6EvaluationActive() {
  return !!HW6_STATE.activeRun;
}

/**
 * Convenience guard: when an evaluation is in flight, report why `what` is
 * unavailable and return `true` so the caller can bail early.
 *
 * @param {string} what Human label for the refused action.
 * @returns {boolean}
 */
function hw6RefuseIfBusy(what) {
  if (!hw6EvaluationActive()) return false;
  hw6SetStatus(what + " is disabled while an evaluation is running. Wait or press Cancel.");
  return true;
}

/** @returns {void} Request cancellation of the in-flight run, if any. */
function hw6RequestCancel() {
  const run = HW6_STATE.activeRun;
  if (!run) return;
  run.cancelled = true;
  hw6SetStatus("Cancelling\u2026 the run stops at the next yield point; its partial result is not presented as final.");
}

/**
 * Pre-fill the PPR cache for a whole cohort at one alpha using the batched
 * engine, reporting progress and honouring cancellation.
 *
 * @param {Object} graph
 * @param {number[]} cohort
 * @param {number} alpha
 * @param {Object} meta `{ stage, alphaIndex, alphaTotal, method }`
 * @param {{cancelled:boolean, startedAt:number}} run
 * @returns {Promise<void>}
 */
async function hw6PrefillPpr(graph, cohort, alpha, meta, run) {
  const n = hw6NodeCount(graph);
  if (!n) return;
  const total = cohort.length;
  // Only walk the users whose (epoch, user, alpha) key is not already cached. A
  // repeat of an identical evaluation therefore dispatches no propagation
  // kernels at all: every entry is a genuine cache hit on this graph.
  const pendingUsers = [];
  const pendingRestarts = [];
  let cachedCount = 0;
  for (let i = 0; i < cohort.length; i++) {
    const userId = cohort[i];
    if (hw6PeekPpr(graph, userId, alpha)) {
      cachedCount += 1;
      continue;
    }
    const seedNode = hw6UserNodeIndex(graph, userId);
    if (seedNode < 0) continue;
    const restart = new Float64Array(n);
    restart[seedNode] = 1;
    pendingUsers.push(userId);
    pendingRestarts.push(restart);
  }
  const pendingTotal = pendingRestarts.length;
  hw6UpdateProgress({ stage: meta.stage, alpha: alpha, alphaIndex: meta.alphaIndex, alphaTotal: meta.alphaTotal, method: meta.method, processed: cachedCount, total: total, elapsed: Date.now() - run.startedAt, backend: HW6_STATE.backend });
  for (let start = 0; start < pendingTotal; start += HW6_BATCH_SIZE) {
    if (run.cancelled) throw hw6CancelledError("pre-fill PPR");
    const chunkR = pendingRestarts.slice(start, start + HW6_BATCH_SIZE);
    const results = await graphPropagateBatch(chunkR, alpha, {
      tol: HW6_TOLERANCE,
      maxIter: HW6_MAX_ITERATIONS,
      maxYieldGapMs: HW6_YIELD_GAP_MS,
      shouldCancel: function () { return run.cancelled; },
      onIteration: function (iter, active) {
        const finished = cachedCount + Math.min(start, pendingTotal);
        const totalCount = total;
        hw6UpdateProgressChunk(meta, alpha, run, finished, totalCount, iter, active, chunkR.length);
      },
    });
    for (let k = 0; k < results.length; k++) {
      hw6StorePprResult(graph, pendingUsers[start + k], alpha, results[k]);
    }
    hw6UpdateProgress({ stage: meta.stage, alpha: alpha, alphaIndex: meta.alphaIndex, alphaTotal: meta.alphaTotal, method: meta.method, processed: cachedCount + Math.min(start + chunkR.length, pendingTotal), total: total, elapsed: Date.now() - run.startedAt, backend: HW6_STATE.backend });
  }
}

/**
 * Pre-fill the cached global PageRank for one alpha (a single uniform walk).
 *
 * @param {Object} graph
 * @param {number} alpha
 * @param {{cancelled:boolean, startedAt:number}} run
 * @param {Object} meta
 * @returns {Promise<void>}
 */
async function hw6PrefillGlobalPageRank(graph, alpha, run, meta) {
  const n = hw6NodeCount(graph);
  if (!n) return;
  // Reuse a walk already computed for this graph epoch and alpha.
  if (hw6PeekPageRank(graph, alpha)) {
    hw6UpdateProgress({ stage: meta.stage, alpha: alpha, alphaIndex: null, alphaTotal: null, method: meta.method, processed: 1, total: 1, elapsed: Date.now() - run.startedAt, backend: HW6_STATE.backend });
    return;
  }
  const restart = new Float64Array(n);
  const value = 1 / n;
  for (let i = 0; i < n; i++) restart[i] = value;
  hw6UpdateProgress({ stage: meta.stage, alpha: alpha, alphaIndex: meta.alphaIndex, alphaTotal: meta.alphaTotal, method: meta.method, processed: 0, total: 1, elapsed: Date.now() - run.startedAt, backend: HW6_STATE.backend });
  const results = await graphPropagateBatch([restart], alpha, {
    tol: HW6_TOLERANCE,
    maxIter: HW6_MAX_ITERATIONS,
    maxYieldGapMs: HW6_YIELD_GAP_MS,
    shouldCancel: function () { return run.cancelled; },
    onIteration: function (iter, active) {
      hw6UpdateProgressChunk(meta, alpha, run, 0, 1, iter, active, 1);
    },
  });
  hw6StorePageRankResult(graph, alpha, results[0]);
  hw6UpdateProgress({ stage: meta.stage, alpha: alpha, alphaIndex: null, alphaTotal: null, method: meta.method, processed: 1, total: 1, elapsed: Date.now() - run.startedAt, backend: HW6_STATE.backend });
}

/**
 * Async validation sweep: pre-fill each alpha's val PPR vectors, score the
 * synchronous recall (cache hits), then drop that alpha's vectors to bound peak
 * memory. Selection uses validation recall only; ties toward the smaller alpha.
 *
 * @param {Object} graph
 * @param {{cancelled:boolean, startedAt:number}} run
 * @returns {Promise<{rows:Array<Object>, best:(number|null), note:string}>}
 */
async function hw6AsyncAlphaSweep(graph, run) {
  const cohort = hw6SharedCohort("val");
  const rows = [];
  let best = null;
  let bestRecall = -Infinity;
  let note = "";

  for (let i = 0; i < ALPHA_GRID.length; i++) {
    if (run.cancelled) throw hw6CancelledError("validation sweep");
    const alpha = ALPHA_GRID[i];
    await hw6PrefillPpr(graph, cohort, alpha, { stage: "Validation sweep", alphaIndex: i + 1, alphaTotal: ALPHA_GRID.length, method: "ppr" }, run);
    if (run.cancelled) throw hw6CancelledError("validation sweep");

    const previousContext = HW6_STATE.alphaContext;
    HW6_STATE.alphaContext = alpha;
    const row = { alpha: alpha, recall: null, cohortSize: cohort.length, error: null };
    try {
      const report = recallAtK("ppr", HW6_K, "val");
      row.recall = Number(report.recall);
      if (Array.isArray(report.cohort)) row.cohortSize = report.cohort.length;
    } catch (error) {
      row.error = String(error && error.message ? error.message : error);
    } finally {
      HW6_STATE.alphaContext = previousContext;
    }
    rows.push(row);
    hw6DropPprAlpha(alpha);
  }

  // Atomic, complete selection: an alpha is only selected when EVERY row is
  // valid (four finite recalls, no error). A single failed alpha invalidates the
  // whole sweep so a partial sweep can never select a setting, cache itself as
  // reusable or license a completion claim. Ties keep the smaller alpha because
  // the comparison is strict `>` walking the ascending grid.
  let invalidReason = "";
  for (let i = 0; i < rows.length; i++) {
    if (rows[i].error) { invalidReason = "alpha " + hw6Fmt(rows[i].alpha, 2) + " failed (" + rows[i].error + ")"; break; }
    if (!Number.isFinite(rows[i].recall)) { invalidReason = "alpha " + hw6Fmt(rows[i].alpha, 2) + " produced no finite recall"; break; }
  }
  if (rows.length !== ALPHA_GRID.length && !invalidReason) {
    invalidReason = "evaluated " + rows.length + " of " + ALPHA_GRID.length + " alphas";
  }
  if (!invalidReason) {
    for (let i = 0; i < rows.length; i++) {
      if (rows[i].recall > bestRecall) {
        bestRecall = rows[i].recall;
        best = rows[i].alpha;
      }
    }
  } else {
    best = null;
    note = invalidReason;
  }

  HW6_STATE.alphaSweepRows = rows;
  HW6_STATE.alphaSweepNote = note;
  return { rows: rows, best: best, note: note };
}

/**
 * Async primary comparison on the TEST cohort at `alpha`. PageRank/PPR vectors
 * are pre-filled per cohort user, then the synchronous comparison reads them
 * from the cache; recall and mean popularity share one memoised list per user.
 *
 * @param {Object} graph
 * @param {number} alpha
 * @param {{cancelled:boolean, startedAt:number}} run
 * @returns {Promise<Array<Object>>}
 */
async function hw6AsyncComparison(graph, alpha, run) {
  const name = "test";
  const cohort = hw6SharedCohort(name);
  const exclusions = hw6CohortExclusions(name);
  const methods = ["popularity", "pagerank", "ppr", "mf"];
  const rows = [];

  // The four-method comparison needs MF, so this is a legitimate lazy trigger.
  // The run token lets MF report epoch progress and honour a Cancel at an epoch
  // boundary (the algorithm, seed and checkpoint rule are unchanged).
  await hw6EnsureMfForNeed(run);
  const mfReady = HW6_MF_MODEL !== null && MF_INFO.trainedAt !== null;
  const seedUser = hw6ReadNumber("seed-user-select", Number.NaN);
  const seed = Number.isFinite(seedUser) ? seedUser : cohort.length ? cohort[0] : null;

  const previousContext = HW6_STATE.alphaContext;
  HW6_STATE.alphaContext = alpha;
  try {
    for (let m = 0; m < methods.length; m++) {
      if (run.cancelled) throw hw6CancelledError("comparison");
      const method = methods[m];
      const usesAlpha = method === "pagerank" || method === "ppr";
      const row = {
        method: method, split: name, alpha: usesAlpha ? alpha : null, recall: null,
        cohortSize: cohort.length, excludedUnseen: exclusions.excludedUnseen,
        excludedNoEligible: exclusions.excludedNoEligible, meanPopularity: null,
        iterations: null, residual: null, converged: null, error: null, note: null,
      };
      try {
        if (method === "mf" && !mfReady) {
          row.note = "MF not trained yet \u2014 its recall is not meaningful.";
        } else {
          if (usesAlpha && graph) {
            if (method === "pagerank") {
              await hw6PrefillGlobalPageRank(graph, alpha, run, { stage: "Primary comparison", method: "pagerank" });
              hw6RunAndRecordConvergence("pagerank", graph, alpha);
            } else {
              await hw6PrefillPpr(graph, cohort, alpha, { stage: "Primary comparison", method: "ppr" }, run);
              if (seed !== null) hw6RunAndRecordConvergence("ppr", graph, alpha, seed);
            }
          }
          if (run.cancelled) throw hw6CancelledError("comparison");
          const report = recallAtK(method, HW6_K, name);
          row.recall = Number(report.recall);
          if (Array.isArray(report.cohort)) row.cohortSize = report.cohort.length;
          if (Number.isFinite(Number(report.excludedUnseen))) row.excludedUnseen = Number(report.excludedUnseen);
          if (Number.isFinite(Number(report.excludedNoEligible))) row.excludedNoEligible = Number(report.excludedNoEligible);
          row.meanPopularity = hw6MeanListPopularityForCohort(method, cohort, name, alpha);
          const record = usesAlpha ? HW6_STATE.convergence[method === "ppr" ? "ppr" : "pagerank"] : null;
          if (record) {
            row.iterations = record.iterations;
            row.residual = record.residual;
            row.converged = hw6ConvergenceVerdict(record).value;
          }
        }
      } catch (error) {
        if (hw6IsCancelledError(error)) throw error;
        row.error = String(error && error.message ? error.message : error);
      }
      rows.push(row);
    }
  } finally {
    HW6_STATE.alphaContext = previousContext;
  }

  HW6_STATE.comparisonRows = rows;
  HW6_STATE.comparisonConfig = {
    graphEpoch: HW6_STATE.graphEpoch,
    selectedAlpha: HW6_STATE.selectedAlpha,
    alpha: alpha,
    split: name,
    cohortSize: cohort.length,
  };
  return rows;
}

/**
 * Render an honest "cancelled" note into the evaluation panels so a partial run
 * is never mistaken for a final result.
 *
 * @returns {void}
 */
function hw6RenderCancelledNote() {
  const note = '<p class="empty-state">The last evaluation was cancelled before it finished; no result from it is presented as final. Press \u201cEvaluate\u201d again to run to completion.</p>';
  const targets = ["comparison-results", "recall-results"];
  for (let i = 0; i < targets.length; i++) {
    const el = document.getElementById(targets[i]);
    if (el) el.innerHTML = note;
  }
}

/**
 * True only when a validation sweep is COMPLETE: it has a selected alpha and
 * every grid alpha produced a finite recall with no error. A sweep that is
 * missing an alpha, or whose any alpha failed, is never reusable and never
 * selects a setting.
 *
 * @param {*} sweep
 * @returns {boolean}
 */
function hw6SweepComplete(sweep) {
  if (!sweep || !Array.isArray(sweep.rows) || sweep.rows.length === 0) return false;
  if (sweep.best === null || sweep.best === undefined) return false;
  if (typeof sweep.best !== "number" || !Number.isFinite(sweep.best)) return false;
  if (sweep.rows.length !== ALPHA_GRID.length) return false;
  const seen = {};
  for (let i = 0; i < sweep.rows.length; i++) {
    const r = sweep.rows[i];
    if (!r || r.error) return false;
    const recallRaw = r.recall;
    if (typeof recallRaw !== "number" || !Number.isFinite(recallRaw) || recallRaw < 0 || recallRaw > 1) return false;
    const a = r.alpha;
    if (typeof a !== "number" || !Number.isFinite(a)) return false;
    const k = hw6AlphaKey(a);
    if (seen[k]) return false;
    seen[k] = true;
    if (HW6_STATE.splitReady) {
      const vcoh = hw6SharedCohort("val");
      if (typeof r.cohortSize !== "number" || !Number.isFinite(r.cohortSize) || r.cohortSize !== vcoh.length) return false;
    }
  }
  return true;
}

/**
 * True only when every four-method comparison row carries a finite recall and no
 * error. Used to reject an incomplete/errored evaluation as a "complete final
 * run" so a partial comparison can never be cached as the completed result.
 *
 * @param {Array<Object>} rows
 * @returns {boolean}
 */
function hw6ComparisonComplete(rows) {
  if (!Array.isArray(rows) || rows.length === 0) return false;
  if (rows.length !== 4) return false;
  const required = { popularity: true, pagerank: true, ppr: true, mf: true };
  for (let i = 0; i < rows.length; i++) {
    const r = rows[i];
    if (!r || r.error) return false;
    const recallRaw = r.recall;
    if (typeof recallRaw !== "number" || !Number.isFinite(recallRaw) || recallRaw < 0 || recallRaw > 1) return false;
    const m = String(r.method);
    if (!required.hasOwnProperty(m)) return false;
    required[m] = false;
    if (r.split !== "test") return false;
    if (HW6_STATE.splitReady) {
      const tcoh = hw6SharedCohort("test");
      if (typeof r.cohortSize !== "number" || !Number.isFinite(r.cohortSize) || r.cohortSize !== tcoh.length) return false;
    }
    if (m === "pagerank" || m === "ppr") {
      if (r.alpha === null || r.alpha === undefined) return false;
      if (typeof r.alpha !== "number" || !Number.isFinite(r.alpha)) return false;
      if (typeof HW6_STATE.selectedAlpha === "number" && Number.isFinite(HW6_STATE.selectedAlpha) &&
          Math.abs(r.alpha - HW6_STATE.selectedAlpha) > 1e-9) return false;
    } else {
      if (r.alpha !== null && r.alpha !== undefined) return false;
    }
  }
  return required.popularity === false && required.pagerank === false && required.ppr === false && required.mf === false;
}

/**
 * Shallow-copy an array of plain result rows so a later live-state reset cannot
 * mutate an already-captured snapshot (rows are replaced, never edited in place).
 *
 * @param {Array<Object>|null} rows
 * @returns {Array<Object>}
 */
function hw6CloneRows(rows) {
  if (!Array.isArray(rows)) return [];
  return rows.map(function (row) { return Object.assign({}, row); });
}

/**
 * Build the immutable completed-run snapshot from the just-finished live state.
 * A new object graph is produced (rows copied, convergence and recallReport
 * cloned) so nothing captured here shares mutable structure with `HW6_STATE`.
 *
 * @param {Object} params `{ identity, alpha, comparisonRows, alphaSweepRows,
 *   selectedAlpha, convergence, recallReport, recommendations, config }`
 * @returns {Object}
 */
function hw6BuildCompletedSnapshot(params) {
  const convergence = {};
  ["pagerank", "ppr"].forEach(function (key) {
    const record = params.convergence ? params.convergence[key] : null;
    convergence[key] = record ? Object.assign({}, record) : null;
  });
  return {
    identity: params.identity,
    completedAt: new Date().toISOString(),
    graphEpoch: HW6_STATE.graphEpoch,
    backend: HW6_STATE.backend,
    alpha: params.alpha,
    selectedAlpha: params.selectedAlpha,
    config: Object.assign({}, params.config),
    comparisonRows: hw6CloneRows(params.comparisonRows),
    alphaSweepRows: hw6CloneRows(params.alphaSweepRows),
    convergence: convergence,
    recallReport: params.recallReport ? Object.assign({}, params.recallReport) : null,
    recommendations: params.recommendations ? JSON.parse(JSON.stringify(params.recommendations)) : null,
  };
}

/**
 * The run identity/status columns appended to exports so a CSV row set is never
 * mistaken for the current run when it is actually the last completed one.
 *
 * @returns {{id:(string|null), status:string}}
 */
function hw6ExportRunIdentity() {
  const snapshot = HW6_STATE.completedSnapshot;
  return {
    id: snapshot && snapshot.identity ? snapshot.identity : "",
    status: HW6_STATE.runState,
  };
}

/**
 * Run the evaluation asynchronously with progress and cancellation.
 *
 * Order is fixed by the assignment: the PPR alpha is chosen by a validation-only
 * sweep first; only then are the primary TEST PageRank/PPR (and the popularity /
 * MF baselines) evaluated at that selected alpha. A completed sweep for the
 * current graph is reused by a repeat run. If the sweep cannot select an alpha,
 * the primary comparison is deliberately not run: a test number must never be
 * reported as if a failed sweep had selected it.
 *
 * @param {{stages?:"full"|"sweep"|"comparison"}} [opts]
 * @returns {Promise<void>}
 */
async function hw6RunEvaluationAsync(opts) {
  const options = opts || {};
  const mode = options.stages || "full";
  if (HW6_STATE.activeRun) {
    hw6SetStatus("An evaluation is already running \u2014 wait for it to finish or press Cancel.");
    return;
  }
  const run = { cancelled: false, startedAt: Date.now() };
  HW6_STATE.activeRun = run;
  HW6_STATE.runState = "active";
  hw6SetRunning(true);
  const hasBatch = typeof graphPropagateBatch === "function";
  hw6ShowProgressInitial();

  try {
    const graph = hw6ActiveGraph();

    if (mode !== "comparison") {
      let sweep;
      // Reuse a COMPLETE sweep for the current epoch, for both the full and the
      // sweep-only entry points: an identical repeat must not recompute walks.
      const reusable = hw6SweepComplete(HW6_STATE.sweep) && HW6_STATE.sweep.epoch === HW6_STATE.graphEpoch;
      if (reusable) {
        sweep = HW6_STATE.sweep;
      } else if (graph && hw6GraphJsReady() && hasBatch) {
        sweep = await hw6AsyncAlphaSweep(graph, run);
        HW6_STATE.sweep = hw6SweepComplete(sweep)
          ? { epoch: HW6_STATE.graphEpoch, rows: sweep.rows, best: sweep.best, note: sweep.note }
          : null;
      } else if (graph && hw6GraphJsReady()) {
        sweep = hw6RunAlphaSweep();
        HW6_STATE.sweep = hw6SweepComplete(sweep)
          ? { epoch: HW6_STATE.graphEpoch, rows: sweep.rows, best: sweep.best, note: sweep.note }
          : null;
      } else {
        sweep = { rows: [], best: null, note: graph ? "graph propagation is unavailable" : "no graph is built" };
      }
      hw6RenderAlphaSweep(sweep);
      if (run.cancelled) throw hw6CancelledError("after validation sweep");

      if (mode === "sweep") {
        if (hw6SweepComplete(sweep)) {
          HW6_STATE.runState = "completed";
          hw6SetStatus("Validation sweep complete \u2014 selected alpha " + hw6Fmt(sweep.best, 2) + ".");
        } else {
          HW6_STATE.runState = "failed";
          hw6SetStatus("Validation sweep incomplete \u2014 no alpha selected" + (sweep.note ? " (" + sweep.note + ")" : "") + ". A partial sweep never selects a setting.");
        }
        const elapsed = Date.now() - run.startedAt;
        hw6RecordRunSummary({ stage: "Validation sweep", detail: hw6SweepComplete(sweep) ? "selected alpha " + hw6Fmt(sweep.best, 2) : "no alpha selected", elapsedMs: elapsed, backend: HW6_STATE.backend });
        return;
      }

      if (graph && sweep.best === null) {
        HW6_STATE.runState = "failed";
        hw6SetStatus("The validation sweep did not select an alpha, so the primary comparison was not run \u2014 a failed sweep must not select a test setting.");
        hw6RecordRunSummary({ stage: "Evaluation", detail: "blocked: no alpha selected", elapsedMs: Date.now() - run.startedAt, backend: HW6_STATE.backend });
        return;
      }

      const alpha = sweep.best === null ? hw6ReadAlpha() : sweep.best;
      HW6_STATE.selectedAlpha = sweep.best;
      const comparison = hasBatch
        ? await hw6AsyncComparison(graph, alpha, run)
        : await hw6RunComparisonSyncAt(alpha);
      if (run.cancelled) throw hw6CancelledError("after comparison");
      hw6RenderComparison(comparison);
      hw6RenderConvergence();

      const method = hw6ReadMethod();
      const recallTarget = document.getElementById("recall-results");
      let report = null;
      if (recallTarget) {
        const previousContext = HW6_STATE.alphaContext;
        HW6_STATE.alphaContext = alpha;
        try {
          report = recallAtK(method, HW6_K, "test");
          HW6_STATE.recallReport = report;
          hw6RenderRecall(report, method, "test");
        } catch (error) {
          recallTarget.innerHTML = hw6UnavailableHtml("Retrieval evaluation", error);
        } finally {
          HW6_STATE.alphaContext = previousContext;
        }
      }

      // A run only counts as completed — and only then becomes the immutable
      // export snapshot — when the sweep AND every comparison method succeeded.
      if (!hw6ComparisonComplete(comparison)) {
        HW6_STATE.runState = "failed";
        hw6SetStatus("Evaluation incomplete: at least one method produced no finite result; this run is not cached as a completed result and does not replace the previous completed evidence.");
        hw6RecordRunSummary({ stage: "Evaluation", detail: "incomplete \u2014 not cached", elapsedMs: Date.now() - run.startedAt, backend: HW6_STATE.backend });
        return;
      }

      const identity = "run-" + HW6_STATE.graphEpoch + "-" + (HW6_STATE.runCounter += 1);
      hw6ResetRecommendationRows();
      const recommendations = hw6RecordCohortRecommendations(alpha);
      HW6_STATE.completedSnapshot = hw6BuildCompletedSnapshot({
        identity: identity,
        alpha: alpha,
        comparisonRows: comparison,
        alphaSweepRows: sweep.rows,
        selectedAlpha: sweep.best,
        convergence: HW6_STATE.convergence,
        recallReport: report,
        recommendations: recommendations,
        config: { selectedAlpha: sweep.best, alpha: alpha, split: "test", cohortSize: comparison.length ? comparison[0].cohortSize : null, backend: HW6_STATE.backend },
      });
      HW6_STATE.runState = "completed";
      hw6SetStatus("Evaluation run complete \u2014 selected " + (sweep.best === null ? "alpha " + hw6Fmt(alpha, 2) + " (no sweep)" : "alpha " + hw6Fmt(sweep.best, 2)) + "; PageRank and PPR evaluated at that alpha.");
      hw6RecordRunSummary({ stage: "Evaluation", detail: "selected alpha " + hw6Fmt(alpha, 2), elapsedMs: Date.now() - run.startedAt, backend: HW6_STATE.backend });
      return;
    }

    // Comparison-only mode.
    const alpha = HW6_STATE.selectedAlpha !== null ? HW6_STATE.selectedAlpha : hw6ReadAlpha();
    const comparison = hasBatch ? await hw6AsyncComparison(graph, alpha, run) : await hw6RunComparisonSyncAt(alpha);
    if (run.cancelled) throw hw6CancelledError("after comparison");
    hw6RenderComparison(comparison);
    hw6RenderConvergence();
    if (!hw6ComparisonComplete(comparison)) {
      HW6_STATE.runState = "failed";
      hw6SetStatus("Comparison incomplete: at least one method produced no finite result; not cached as a completed result.");
      hw6RecordRunSummary({ stage: "Comparison", detail: "incomplete \u2014 not cached", elapsedMs: Date.now() - run.startedAt, backend: HW6_STATE.backend });
      return;
    }
    const identity = "run-" + HW6_STATE.graphEpoch + "-" + (HW6_STATE.runCounter += 1);
    hw6ResetRecommendationRows();
    const recommendations = hw6RecordCohortRecommendations(alpha);
    HW6_STATE.completedSnapshot = hw6BuildCompletedSnapshot({
      identity: identity,
      alpha: alpha,
      comparisonRows: comparison,
      alphaSweepRows: HW6_STATE.alphaSweepRows,
      selectedAlpha: HW6_STATE.selectedAlpha,
      convergence: HW6_STATE.convergence,
      recallReport: HW6_STATE.recallReport,
      recommendations: recommendations,
      config: { selectedAlpha: HW6_STATE.selectedAlpha, alpha: alpha, split: "test", cohortSize: comparison.length ? comparison[0].cohortSize : null, backend: HW6_STATE.backend },
    });
    HW6_STATE.runState = "completed";
    hw6SetStatus("Comparison complete at alpha " + hw6Fmt(alpha, 2) + ".");
    hw6RecordRunSummary({ stage: "Comparison", detail: "alpha " + hw6Fmt(alpha, 2), elapsedMs: Date.now() - run.startedAt, backend: HW6_STATE.backend });
  } catch (error) {
    if (hw6IsCancelledError(error)) {
      HW6_STATE.runState = "cancelled";
      hw6RenderCancelledNote();
      hw6SetStatus("Evaluation cancelled \u2014 no result from this run is presented as final. Press \u201cEvaluate\u201d again to restart.");
      hw6RecordRunSummary({ stage: "Evaluation", detail: "cancelled \u2014 previous completed evidence retained", elapsedMs: Date.now() - run.startedAt, backend: HW6_STATE.backend });
    } else {
      HW6_STATE.runState = "failed";
      hw6SetStatus("Evaluation failed: " + String(error && error.message ? error.message : error));
      hw6RecordRunSummary({ stage: "Evaluation", detail: "failed \u2014 previous completed evidence retained", elapsedMs: Date.now() - run.startedAt, backend: HW6_STATE.backend });
    }
  } finally {
    HW6_STATE.activeRun = null;
    hw6SetRunning(false);
    hw6HideProgress();
  }
}

/**
 * Show the progress panel in an indeterminate "starting" state before the first
 * concrete update arrives.
 *
 * @returns {void}
 */
function hw6ShowProgressInitial() {
  const el = hw6ProgressElements();
  if (!el.panel) return;
  el.panel.classList.remove("hidden");
  if (el.stage) el.stage.textContent = "Starting\u2026";
  if (el.pct) el.pct.textContent = "\u2014";
  if (el.bar) {
    el.bar.classList.add("progress-bar--indeterminate");
    el.bar.style.width = "100%";
  }
}

/**
 * Synchronous comparison at an explicit alpha, used only when the batched kernel
 * is unavailable. The alpha context makes the synchronous runners agree on the
 * selected alpha without touching the slider.
 *
 * @param {number} alpha
 * @returns {Promise<Array<Object>>}
 */
async function hw6RunComparisonSyncAt(alpha) {
  const previousContext = HW6_STATE.alphaContext;
  HW6_STATE.alphaContext = alpha;
  try {
    return await hw6RunComparison("test");
  } finally {
    HW6_STATE.alphaContext = previousContext;
  }
}

/**
 * Validation-only alpha sweep for PPR (readme section 6.1).
 *
 * Recall@10 is measured on the validation split for each alpha in `ALPHA_GRID`.
 * The selected alpha maximises validation recall; exact ties are broken toward
 * the SMALLER alpha. Test outcomes are never consulted here.
 *
 * @returns {{rows:Array<Object>, best:(number|null), note:string}}
 */
function hw6RunAlphaSweep() {
  const rows = [];
  let best = null;
  let bestRecall = -Infinity;
  let note = "";

  // Evaluate each alpha through the programmatic context so the user's demo
  // slider is never moved, and the propagation caches stay keyed by the swept
  // alpha (which is what makes repeated evaluations reuse the walks).
  const previousContext = HW6_STATE.alphaContext;

  for (let i = 0; i < ALPHA_GRID.length; i++) {
    const alpha = ALPHA_GRID[i];
    const row = { alpha: alpha, recall: null, cohortSize: hw6SharedCohort("val").length, error: null };
    HW6_STATE.alphaContext = alpha;
    try {
      const report = recallAtK("ppr", HW6_K, "val");
      row.recall = Number(report.recall);
      if (Array.isArray(report.cohort)) row.cohortSize = report.cohort.length;
      // Strict `>`: an exact tie keeps the earlier (smaller) alpha.
      if (Number.isFinite(row.recall) && row.recall > bestRecall) {
        bestRecall = row.recall;
        best = alpha;
      }
    } catch (error) {
      row.error = String(error && error.message ? error.message : error);
      note = row.error;
    }
    rows.push(row);
  }

  HW6_STATE.alphaContext = previousContext;
  HW6_STATE.alphaSweepRows = rows;
  HW6_STATE.alphaSweepNote = note;
  return { rows: rows, best: best, note: note };
}

/**
 * Run `fn` with the alpha slider temporarily set to `value`, restoring the
 * previous value afterwards. The slider's own `input` event is never dispatched,
 * so this cannot recurse.
 *
 * @param {number} value
 * @param {() => any} fn
 * @returns {any}
 */
function hw6WithAlpha(value, fn) {
  const slider = document.getElementById("alpha");
  const previous = slider ? Number(slider.value) : 0.85;
  hw6SetAlpha(value);
  try {
    return fn();
  } finally {
    hw6SetAlpha(previous);
  }
}

/**
 * Two-user, one-factor-at-a-time PPR demonstration (readme section 6.2).
 *
 * Both users are drawn deterministically from the shared TEST cohort (the two
 * smallest ids) and are displayed. Three settings are shown: baseline
 * (user A, alpha 0.85), change the seed only (user B, alpha 0.85) and change
 * alpha only (user A, alpha 0.50). Exactly one factor changes per comparison.
 *
 * @returns {Object} `{ error, users, settings, common }`; `error` is set when
 *   the demonstration cannot run (unimplemented stub or a missing cohort).
 */
function hw6RunTwoUserDemo() {
  const cohort = hw6SharedCohort("test");
  if (cohort.length < 2) {
    return { error: "Need at least two users in the shared test cohort.", users: cohort, settings: [], common: [] };
  }
  const userA = cohort[0];
  const userB = cohort[1];
  const settings = [];
  try {
    settings.push({
      label: "baseline",
      user: userA,
      alpha: 0.85,
      changed: "nothing (reference setting)",
      recs: hw6WithAlpha(0.85, function () { return recommendTopK("ppr", userA, HW6_K, "test"); }),
    });
    settings.push({
      label: "change seed only",
      user: userB,
      alpha: 0.85,
      changed: "restart user only",
      recs: hw6WithAlpha(0.85, function () { return recommendTopK("ppr", userB, HW6_K, "test"); }),
    });
    settings.push({
      label: "change alpha only",
      user: userA,
      alpha: 0.50,
      changed: "alpha only",
      recs: hw6WithAlpha(0.50, function () { return recommendTopK("ppr", userA, HW6_K, "test"); }),
    });
  } catch (error) {
    return {
      error: String(error && error.message ? error.message : error),
      users: [userA, userB],
      settings: settings,
      common: [],
    };
  }

  const common = [];
  for (let i = 0; i < settings.length; i++) {
    for (let j = i + 1; j < settings.length; j++) {
      const left = settings[i];
      const right = settings[j];
      const set = new Set(left.recs.map(function (rec) { return rec.movieId; }));
      let shared = 0;
      right.recs.forEach(function (rec) { if (set.has(rec.movieId)) shared++; });
      common.push({ pair: left.label + " vs " + right.label, shared: shared, of: HW6_K });
    }
  }

  settings.forEach(function (setting) {
    hw6RecordRecommendationRows(setting.user, "ppr", setting.alpha, setting.recs);
  });

  const demo = { error: null, users: [userA, userB], settings: settings, common: common };
  HW6_STATE.demo = demo;
  return demo;
}

/**
 * Recommendations for the current controls (method, user, split), recorded for
 * export. Always uses the TEST split for the on-screen panel.
 *
 * @returns {{recs:RecommendedMovie[]}|{error:string}}
 */
function hw6CurrentRecommendations() {
  const method = hw6ReadMethod();
  const userId = hw6ReadNumber("user-select", Number.NaN);
  const alpha = hw6ReadAlpha();
  if (!Number.isFinite(userId)) return { error: "No target user selected." };
  try {
    const recs = recommendTopK(method, userId, HW6_K, "test");
    hw6RecordRecommendationRows(userId, method, method === "pagerank" || method === "ppr" ? alpha : null, recs);
    return { recs: recs };
  } catch (error) {
    return { error: String(error && error.message ? error.message : error) };
  }
}

// ---------------------------------------------------------------------------
// Panel renderers (provided)
// ---------------------------------------------------------------------------

/**
 * Render the dataset and split summary into `#dataset-summary-body`.
 *
 * Reports users, movies, ratings, sparsity, positive training interactions,
 * zero-degree users and movies, and the split record counts. Missing ratings are
 * unobserved, never zeros.
 *
 * @returns {void}
 */
function hw6RenderDatasetSummary() {
  const target = document.getElementById("dataset-summary-body");
  if (!target) return;
  if (!HW6_STATE.splitReady) {
    target.innerHTML = '<p class="empty-state">Loading dataset\u2026</p>';
    return;
  }

  const train = HW6_SPLIT.train;
  const val = HW6_SPLIT.val;
  const test = HW6_SPLIT.test;
  const total = train.length + val.length + test.length;
  const users = HW6_STATE.trainUserIds.length;
  const movieCount = HW6_STATE.candidateIds.length;
  const cells = (numUsers + 1) * (HW6_MAX_MOVIE_ID + 1);
  const sparsity = cells > 0 ? 1 - total / cells : 0;
  const exclusions = typeof SPLIT_EXCLUSIONS === "undefined" ? [] : SPLIT_EXCLUSIONS;

  let zeroDegreeUsers = 0;
  for (let i = 0; i < HW6_STATE.trainUserIds.length; i++) {
    if (!HW6_STATE.usersWithPositiveTrain.has(HW6_STATE.trainUserIds[i])) zeroDegreeUsers++;
  }
  let zeroDegreeMovies = 0;
  for (let i = 0; i < HW6_STATE.candidateIds.length; i++) {
    if (hw6PopularityOf(HW6_STATE.candidateIds[i]) === 0) zeroDegreeMovies++;
  }

  target.innerHTML =
    '<dl class="summary-list">' +
    "<dt>Users observed in training</dt><dd>" + users.toLocaleString("en-US") + "</dd>" +
    "<dt>Movies observed in training</dt><dd>" + movieCount.toLocaleString("en-US") + "</dd>" +
    "<dt>Ratings loaded</dt><dd>" + ratings.length.toLocaleString("en-US") + "</dd>" +
    "<dt>Split: train / validation / test</dt><dd>" +
    train.length.toLocaleString("en-US") + " / " + val.length.toLocaleString("en-US") + " / " + test.length.toLocaleString("en-US") +
    "</dd>" +
    "<dt>Positive training interactions</dt><dd>" + HW6_STATE.positivePairs.toLocaleString("en-US") + "</dd>" +
    "<dt>Zero-degree users</dt><dd>" + zeroDegreeUsers.toLocaleString("en-US") + "</dd>" +
    "<dt>Zero-degree movies</dt><dd>" + zeroDegreeMovies.toLocaleString("en-US") + "</dd>" +
    "<dt>Matrix sparsity</dt><dd>" + (sparsity * 100).toFixed(2) + "%</dd>" +
    "</dl>" +
    '<p class="hint">' +
    exclusions.length + " user(s) skipped while splitting. The split is chronological within each user, not globally time-consistent; " +
    "MovieLens records rating times, not viewing sessions. An unobserved rating is missing data, not a zero." +
    "</p>";
}

/**
 * Render the graph panel into `#graph-summary-body`.
 *
 * Ownership: `graph.js` owns the `Graph` shape and renders this panel itself
 * whenever a graph is installed (`setGraph` -> its own summary renderer), so
 * once a graph is active this function leaves that render to stand and does not
 * overwrite it with a second, differently-derived set of counts.
 *
 * Before a graph exists (and if `graph.js` is unavailable) it renders the node
 * universe the training split implies - observed users, observed movies,
 * positive training edges and zero-degree counts - which is exactly the
 * expectation the built graph must match.
 *
 * @returns {void}
 */
function hw6RenderGraphSummary() {
  const target = document.getElementById("graph-summary-body");
  if (!target) return;
  const graph = hw6ActiveGraph();
  if (graph && typeof setGraph === "function") return;

  const observedUsers = HW6_STATE.trainUserIds.length;
  const observedMovies = HW6_STATE.candidateIds.length;
  if (!HW6_STATE.splitReady) {
    target.innerHTML = '<p class="empty-state">Loading dataset\u2026</p>';
    return;
  }
  let zeroDegreeUsers = 0;
  for (let i = 0; i < HW6_STATE.trainUserIds.length; i++) {
    if (!HW6_STATE.usersWithPositiveTrain.has(HW6_STATE.trainUserIds[i])) zeroDegreeUsers++;
  }
  let zeroDegreeMovies = 0;
  for (let i = 0; i < HW6_STATE.candidateIds.length; i++) {
    if (hw6PopularityOf(HW6_STATE.candidateIds[i]) === 0) zeroDegreeMovies++;
  }

  if (!graph) {
    target.innerHTML =
      '<p class="empty-state">No graph yet. Press \u201cBuild graph\u201d to report nodes, edges, ' +
      "zero-degree users, zero-degree movies and positive training interactions.</p>" +
      '<p class="hint">Expected node universe from the training split: ' +
      (observedUsers + observedMovies).toLocaleString("en-US") +
      " nodes (" + observedUsers.toLocaleString("en-US") + " users + " + observedMovies.toLocaleString("en-US") + " movies), " +
      HW6_STATE.positivePairs.toLocaleString("en-US") + " positive training edge(s), " +
      zeroDegreeUsers.toLocaleString("en-US") + " zero-degree user(s), " +
      zeroDegreeMovies.toLocaleString("en-US") + " zero-degree movie(s). No padding node.</p>";
    return;
  }

  const nodes = hw6NodeCount(graph) || hw6UserIds(graph).length + hw6MovieIds(graph).length;
  const positiveEdges = Number.isFinite(graph.positiveEdges) ? Number(graph.positiveEdges) : HW6_STATE.positivePairs;
  const edgesText =
    (positiveEdges * 2).toLocaleString("en-US") + " directed (" +
    positiveEdges.toLocaleString("en-US") + " positive bipartite edge(s))";
  const zeroUsers = Array.isArray(graph.zeroDegreeUsers) ? graph.zeroDegreeUsers.length : zeroDegreeUsers;
  const zeroMovies = Array.isArray(graph.zeroDegreeMovies) ? graph.zeroDegreeMovies.length : zeroDegreeMovies;

  target.innerHTML =
    '<dl class="summary-list">' +
    "<dt>Graph nodes</dt><dd>" + nodes.toLocaleString("en-US") + "</dd>" +
    "<dt>Graph edges</dt><dd>" + edgesText + "</dd>" +
    "<dt>User nodes (training)</dt><dd>" + observedUsers.toLocaleString("en-US") + "</dd>" +
    "<dt>Movie nodes (training)</dt><dd>" + observedMovies.toLocaleString("en-US") + "</dd>" +
    "<dt>Positive training interactions</dt><dd>" + positiveEdges.toLocaleString("en-US") + "</dd>" +
    "<dt>Zero-degree users</dt><dd>" + zeroUsers.toLocaleString("en-US") + "</dd>" +
    "<dt>Zero-degree movies</dt><dd>" + zeroMovies.toLocaleString("en-US") + "</dd>" +
    "</dl>" +
    '<p class="hint">The graph is built from positive training ratings only; users and movies with no positive ' +
    "training edge are zero-degree nodes. The dangling-mass policy is the same rule for global PageRank and PPR; " +
    "no padding node is added.</p>";
}

/**
 * Render the convergence panel (readme section 3.4) into `#convergence-results`.
 *
 * Reports iterations, the final L1 residual and whether convergence was reached
 * for global PageRank and PPR. Never claims convergence when the iteration cap
 * was hit. `graph.js`'s 4-node teaching demo also writes this panel from
 * `fixtureStep` / `fixtureReset`; its current state is included as a separate,
 * clearly-labelled row so that information is not lost when this renderer runs.
 *
 * @returns {void}
 */
function hw6RenderConvergence() {
  const target = document.getElementById("convergence-results");
  if (!target) return;
  const rows = [];

  const viewState = typeof GRAPH_VIEW_STATE !== "undefined" ? GRAPH_VIEW_STATE : null;
  const fixture = viewState && viewState.fixture ? viewState.fixture : null;
  if (fixture) {
    const residual = Number.isFinite(Number(fixture.residual)) ? hw6Fmt(fixture.residual, 10) : "not run";
    rows.push(
      "<tr>" +
      "<td>Fixture demo (4-node reference graph)</td>" +
      '<td class="num">' + hw6Fmt(fixture.alpha === undefined ? hw6ReadAlpha() : fixture.alpha, 2) + "</td>" +
      '<td class="num">' + (fixture.iteration === undefined || fixture.iteration === null ? "\u2014" : fixture.iteration) + "</td>" +
      '<td class="num">' + residual + "</td>" +
      "<td>\u2014 (teaching demo)</td>" +
      '<td class="num">\u2014</td>' +
      "</tr>",
    );
  }

  const keys = ["pagerank", "ppr"];
  for (let i = 0; i < keys.length; i++) {
    const key = keys[i];
    const record = HW6_STATE.convergence[key];
    if (!record) {
      rows.push("<tr><td>" + (key === "ppr" ? "PPR" : "global PageRank") + "</td><td colspan=\"5\">not run yet</td></tr>");
      continue;
    }
    const verdict = hw6ConvergenceVerdict(record);
    const convergedText =
      verdict.value === null
        ? "not reported"
        : verdict.value
          ? "yes" + (verdict.derived ? " (derived from residual)" : "")
          : "no \u2014 cap or tolerance not met" + (verdict.derived ? " (derived)" : "");
    rows.push(
      "<tr>" +
      "<td>" + (key === "ppr" ? "PPR (user " + record.userId + ")" : "global PageRank") + "</td>" +
      '<td class="num">' + hw6Fmt(record.alpha, 2) + "</td>" +
      '<td class="num">' + (record.iterations === null ? "\u2014" : record.iterations) + "</td>" +
      '<td class="num">' + (record.residual === null ? "\u2014" : hw6Fmt(record.residual, 10)) + "</td>" +
      "<td>" + convergedText + "</td>" +
      '<td class="num">' + (record.iterations === null ? "\u2014" : record.iterations + " / " + HW6_MAX_ITERATIONS) + "</td>" +
      "</tr>",
    );
  }

  target.innerHTML =
    '<table class="data-table">' +
    "<thead><tr><th scope=\"col\">Run</th><th scope=\"col\">Alpha</th><th scope=\"col\">Iterations</th>" +
    "<th scope=\"col\">Final L1 residual</th><th scope=\"col\">Converged</th><th scope=\"col\">Cap</th></tr></thead>" +
    "<tbody>" + rows.join("") + "</tbody></table>" +
    '<p class="hint">Tolerance ' + HW6_TOLERANCE + ", cap " + HW6_MAX_ITERATIONS +
    " iterations. Reaching the cap is not proof of convergence. PageRank is an iterative calculation, not neural-network training.</p>";
}

/**
 * Render the alpha sweep panel (readme section 6.1).
 *
 * @param {{rows:Array<Object>, best:(number|null), note:string}} [sweep]
 * @returns {void}
 */
function hw6RenderAlphaSweep(sweep) {
  const target = document.getElementById("alpha-sweep-results");
  if (!target) return;
  const data = sweep || { rows: HW6_STATE.alphaSweepRows, best: null, note: HW6_STATE.alphaSweepNote || "" };
  if (!data || !data.rows || data.rows.length === 0) {
    target.innerHTML =
      '<p class="empty-state">Validation Recall@10 for alpha in {0.50, 0.70, 0.85, 0.95} is reported here; ' +
      "exact ties are broken by the smaller alpha.</p>";
    return;
  }
  const failed = data.rows.some(function (row) { return !!row.error; });
  if (failed) {
    const first = data.rows.find(function (row) { return !!row.error; });
    target.innerHTML = hw6UnavailableHtml("Alpha sweep", first.error);
    return;
  }

  const body = data.rows
    .map(function (row) {
      const isBest = data.best !== null && Math.abs(row.alpha - data.best) < 1e-12;
      return (
        "<tr" + (isBest ? ' class="best-row"' : "") + ">" +
        '<td class="num">' + hw6Fmt(row.alpha, 2) + "</td>" +
        '<td class="num">' + hw6Fmt(row.recall) + "</td>" +
        '<td class="num">' + row.cohortSize.toLocaleString("en-US") + "</td>" +
        "<td>" + (isBest ? "selected (validation)" : "") + "</td>" +
        "</tr>"
      );
    })
    .join("");

  target.innerHTML =
    '<table class="data-table">' +
    "<thead><tr><th scope=\"col\">Alpha</th><th scope=\"col\">Validation Recall@10</th>" +
    "<th scope=\"col\">Cohort size</th><th scope=\"col\">Selection</th></tr></thead>" +
    "<tbody>" + body + "</tbody></table>" +
    '<p class="hint">Selected alpha: <strong>' +
    (data.best === null ? "none" : hw6Fmt(data.best, 2)) +
    "</strong> \u2014 chosen on validation only, exact ties toward the smaller alpha. The same alpha is used for the " +
    "primary global PageRank comparison (a disclosed control, not an independent optimisation). MF keeps the previous " +
    "validation-RMSE checkpoint rule, so MF and PPR use different validation objectives.</p>";
}

/**
 * Render the four-method comparison panel.
 *
 * @param {Array<Object>} [rows]
 * @returns {void}
 */
function hw6RenderComparison(rows) {
  const target = document.getElementById("comparison-results");
  if (!target) return;
  const data = rows || HW6_STATE.comparisonRows;
  if (!data || data.length === 0) {
    target.innerHTML =
      '<p class="empty-state">Recall@10 for the shared cohort across popularity, global PageRank, PPR and MF is ' +
      "compared here.</p>";
    return;
  }
  const failedRow = data.find(function (row) { return !!row.error; });
  if (failedRow) {
    target.innerHTML = hw6UnavailableHtml("Method comparison", failedRow.error);
    return;
  }

  const body = data
    .map(function (row) {
      return (
        "<tr>" +
        "<td>" + hw6EscapeHtml(row.method) +
        (row.note ? '<br><span class="hint">' + hw6EscapeHtml(row.note) + "</span>" : "") +
        "</td>" +
        '<td class="num">' + (row.alpha === null ? "\u2014" : hw6Fmt(row.alpha, 2)) + "</td>" +
        '<td class="num">' + hw6Fmt(row.recall) + "</td>" +
        '<td class="num">' + row.cohortSize.toLocaleString("en-US") + "</td>" +
        '<td class="num">' + row.excludedUnseen.toLocaleString("en-US") + "</td>" +
        '<td class="num">' + row.excludedNoEligible.toLocaleString("en-US") + "</td>" +
        '<td class="num">' + hw6Fmt(row.meanPopularity, 2) + "</td>" +
        "</tr>"
      );
    })
    .join("");

  const notes = data
    .filter(function (row) { return !!row.note; })
    .map(function (row) { return row.method + ": " + row.note; })
    .join(" ");

  target.innerHTML =
    '<table class="data-table">' +
    "<thead><tr><th scope=\"col\">Method</th><th scope=\"col\">Alpha</th><th scope=\"col\">Recall@10</th>" +
    "<th scope=\"col\">Cohort</th><th scope=\"col\">Unseen excluded</th><th scope=\"col\">No eligible user</th>" +
    "<th scope=\"col\">Mean popularity</th></tr></thead>" +
    "<tbody>" + body + "</tbody></table>" +
    (notes ? '<p class="hint">' + hw6EscapeHtml(notes) + "</p>" : "") +
    '<p class="hint">All four methods use the same candidate rules and the same users. PPR scores are long-run ' +
    "graph-walk probabilities, not purchase probabilities and not predicted ratings.</p>";
}

/**
 * HTML body for the readme section 6.2 two-user demonstration.
 *
 * Pure renderer: it renders the demo object it is given (or the last cached one)
 * and never runs the demo itself, so nothing is written outside the dedicated
 * `#two-user-demo-results` container.
 *
 * @param {Object} [demo] Result of `hw6RunTwoUserDemo()`; defaults to the cached one.
 * @returns {string} HTML for the demo body (`""` when there is nothing to show).
 */
function hw6RunTwoUserDemoHtml(demo) {
  const data = demo || HW6_STATE.demo;
  if (!data) return "";
  if (data.error) {
    return hw6UnavailableHtml("Two-user demo", data.error);
  }
  if (!data.settings || data.settings.length === 0) {
    return '<p class="empty-state">Need two users in the shared test cohort before the demonstration can run.</p>';
  }

  const columns = data.settings
    .map(function (setting) {
      const items =
        setting.recs && setting.recs.length
          ? setting.recs
              .map(function (rec, i) {
                return (
                  "<li>" +
                  hw6EscapeHtml(rec.title) +
                  ' <span class="rec-score">id ' + rec.movieId +
                  ", score " + hw6Fmt(rec.score, 4) +
                  ", pop " + hw6PopularityOf(rec.movieId) +
                  (i === 0 ? "" : "") +
                  "</span></li>"
                );
              })
              .join("")
          : '<li class="empty-state">no list produced</li>';
      const count = setting.recs ? setting.recs.length : 0;
      return (
        "<div>" +
        "<h4>" + hw6EscapeHtml(setting.label) + "</h4>" +
        '<p class="hint">user <strong>' + setting.user + "</strong>, alpha <strong>" + hw6Fmt(setting.alpha, 2) +
        "</strong>; changed: " + hw6EscapeHtml(setting.changed) + "</p>" +
        '<ol class="recommendation-list">' + items + "</ol>" +
        (count < HW6_K
          ? '<p class="hint">Only ' + count + " eligible candidates (fewer than " + HW6_K + ").</p>"
          : "") +
        "</div>"
      );
    })
    .join("");

  const common = data.common
    .map(function (row) { return row.pair + ": " + row.shared + " common of top " + row.of; })
    .join("; ");

  return (
    "<h3>One-factor-at-a-time demonstration (readme 6.2)</h3>" +
    '<p class="hint">Users fixed and shown: ' + data.users.join(", ") +
    ". Exactly one factor changes per comparison \u2014 never attribute a difference to a factor that was also changed.</p>" +
    '<div class="control-grid control-grid--two">' + columns + "</div>" +
    '<p class="hint">Common movies between lists: ' + hw6EscapeHtml(common) +
    ". Descriptive evidence only, not proof of better personalization.</p>"
  );
}

/**
 * Render the Top-10 recommendations table for the current controls.
 *
 * When the selected method is `mf` and the lazy reference has not been trained
 * yet, it is trained now (the first need for MF scores) before the list is
 * built. If it still cannot be produced, the panel says so explicitly.
 *
 * @returns {Promise<void>}
 */
async function hw6RenderRecommendations() {
  const target = document.getElementById("recommendations");
  if (!target) return;
  if (hw6RefuseIfBusy("Recommend Top-10")) return;
  const method = hw6ReadMethod();
  const alpha = hw6ReadAlpha();
  if (method === "mf" && (HW6_MF_MODEL === null || MF_INFO.trainedAt === null)) {
    await hw6EnsureMfForNeed();
  }
  if (method === "mf" && (HW6_MF_MODEL === null || MF_INFO.trainedAt === null)) {
    target.innerHTML =
      '<p class="empty-state">The provided MF reference is not trained yet and could not be trained, so its scores ' +
      "are unavailable. Check the TensorFlow.js status above.</p>";
    return;
  }
  const outcome = hw6CurrentRecommendations();
  if (outcome.error) {
    target.innerHTML = hw6UnavailableHtml("Recommendations", outcome.error);
    return;
  }
  const recs = outcome.recs;
  if (!recs || recs.length === 0) {
    target.innerHTML = '<p class="empty-state">No eligible recommendations for this user.</p>';
    return;
  }

  const body = recs
    .map(function (rec, i) {
      const genres = Array.isArray(rec.genres) && rec.genres.length ? rec.genres.join(", ") : "\u2014";
      return (
        "<tr>" +
        '<td class="num">' + (i + 1) + "</td>" +
        '<td class="num">' + rec.movieId + "</td>" +
        "<td>" + hw6EscapeHtml(rec.title) + "</td>" +
        "<td>" + hw6EscapeHtml(genres) + "</td>" +
        '<td class="num">' + hw6Fmt(rec.score) + "</td>" +
        '<td class="num">' + hw6PopularityOf(rec.movieId) + "</td>" +
        "</tr>"
      );
    })
    .join("");

  const zeroScore = recs.some(function (rec) { return !(Number(rec.score) > 0); });

  target.innerHTML =
    '<p class="hint">Method <strong>' + hw6EscapeHtml(method) + "</strong>, alpha <strong>" + hw6Fmt(alpha, 2) +
    "</strong>, test-split exclusions (training + validation rated movies).</p>" +
    '<table class="data-table">' +
    "<thead><tr><th scope=\"col\">Rank</th><th scope=\"col\">Movie id</th><th scope=\"col\">Title</th>" +
    "<th scope=\"col\">Genres</th><th scope=\"col\">Score</th><th scope=\"col\">Training popularity</th></tr></thead>" +
    "<tbody>" + body + "</tbody></table>" +
    (recs.length < HW6_K
      ? '<p class="hint">Fewer than ' + HW6_K + " eligible candidates exist for this user (" + recs.length + ").</p>"
      : "") +
    (zeroScore
      ? '<p class="hint">Some listed candidates have a zero score: they were included only to fill the list and have ' +
        "not demonstrated relevance.</p>"
      : "") +
    (method === "ppr" || method === "pagerank"
      ? '<p class="hint">These are graph-walk probabilities, not predicted ratings.</p>'
      : "");
}

/**
 * Render the retrieval-evaluation panel from a recall report.
 *
 * @param {RecallReport} report
 * @param {string} method
 * @param {string} split
 * @returns {void}
 */
function hw6RenderRecall(report, method, split) {
  const target = document.getElementById("recall-results");
  if (!target) return;
  const perUser = report.perUser || [];
  const rows = perUser
    .slice(0, 50)
    .map(function (row) {
      return (
        "<tr>" +
        "<td>" + row.userId + "</td>" +
        '<td class="num">' + row.relevant + "</td>" +
        '<td class="num">' + row.hit + "</td>" +
        '<td class="num">' + (row.relevant ? hw6Fmt(row.hit / row.relevant) : "\u2014") + "</td>" +
        "</tr>"
      );
    })
    .join("");

  target.innerHTML =
    "<p><strong>Recall@" + HW6_K + " (" + hw6EscapeHtml(method) + ", " + hw6EscapeHtml(split) + "):</strong> " +
    '<span class="num">' + hw6Fmt(report.recall) + "</span> over " + (report.cohort ? report.cohort.length : 0) + " user(s).</p>" +
    '<table class="data-table">' +
    "<thead><tr><th scope=\"col\">User</th><th scope=\"col\">Relevant</th><th scope=\"col\">Hits</th>" +
    "<th scope=\"col\">Recall</th></tr></thead>" +
    "<tbody>" + (rows || '<tr><td colspan="4">No evaluable users.</td></tr>') + "</tbody></table>" +
    (perUser.length > 50 ? '<p class="hint">Showing the first 50 of ' + perUser.length + " cohort users.</p>" : "") +
    '<p class="hint">Excluded: ' + (report.excludedUnseen || 0) + " relevant movie(s) absent from the training candidate pool; " +
    (report.excludedNoEligible || 0) + " user(s) without an eligible relevant movie. This is a warm-user evaluation and " +
    "does not establish cold-start performance. No rating RMSE is computed for PPR.</p>";
}

/**
 * Populate `#user-select` and `#seed-user-select` from the shared test cohort.
 *
 * @returns {void}
 */
function hw6PopulateUserSelects() {
  const cohort = hw6SharedCohort("test");
  const options = cohort.length
    ? cohort.map(function (id) { return '<option value="' + id + '">User ' + id + "</option>"; }).join("")
    : '<option value="">No cohort users</option>';
  const userSelect = document.getElementById("user-select");
  if (userSelect) userSelect.innerHTML = options;
  const seedSelect = document.getElementById("seed-user-select");
  if (seedSelect) {
    seedSelect.innerHTML = options;
    if (cohort.length > 1) seedSelect.value = String(cohort[1]);
  }
}

// ---------------------------------------------------------------------------
// Verification helpers for the self-checks
//
// These helpers exist only to verify the readme's stated propagation properties
// against `REFERENCE_GRAPH`. The product path never uses them; it calls
// `pageRank` / `personalizedPageRank` through the adapter above.
// ---------------------------------------------------------------------------

/**
 * True when `graph.js`'s propagation helpers are actually implemented.
 *
 * Probes `rowNormalize(REFERENCE_GRAPH)` once and requires a usable
 * `{ Q: number[][], danglingMask: boolean[] }`. An unimplemented `TODO(hw6)`
 * stub, a throw, or an unusable return all count as "not ready", which makes the
 * dependent checks PENDING. This is what stops the harness from reporting PASS
 * for a formula it verified itself while the application's own implementation is
 * still a stub.
 *
 * @returns {boolean}
 */
function hw6GraphJsReady() {
  if (typeof rowNormalize !== "function") return false;
  try {
    // A valid Graph in graph.js's documented shape. REFERENCE_GRAPH cannot be
    // used here: it is the 4-node teaching fixture with `edges` but no `R`.
    const probe = hw6MakeSyntheticGraph([[1, 1]]);
    const result = rowNormalize(probe);
    return !!(result && Array.isArray(result.Q) && result.Q.length === probe.nodeCount);
  } catch (error) {
    return false;
  }
}

/**
 * Run `fn` with a synthetic bipartite graph installed as the active graph.
 *
 * The real `pageRank` / `personalizedPageRank` read the module-level `GRAPH`, so
 * verifying the student's implementation on a small, cheap graph means swapping
 * the active graph for the duration and restoring it afterwards.
 *
 * @param {number[][]} R `R[u][i]` 0/1.
 * @param {() => any} fn
 * @returns {any}
 */
function hw6WithSyntheticGraph(R, fn) {
  if (typeof setGraph !== "function" || typeof getGraph !== "function") {
    throw hw6InterfaceError("graph.js `setGraph` / `getGraph` are unavailable");
  }
  // Swapping the graph disposes the transition tensors; doing that while a
  // batched evaluation holds them would corrupt the live run. Refuse rather than
  // silently break it (the caller surfaces this as a check failure/skip).
  if (typeof hw6EvaluationActive === "function" && hw6EvaluationActive()) {
    throw new Error("HW6_EVALUATION_ACTIVE: cannot replace the graph while an evaluation is running");
  }
  const previous = getGraph();
  setGraph(hw6MakeSyntheticGraph(R));
  try {
    return fn();
  } finally {
    setGraph(previous);
  }
}

/**
 * Iterate `graph.js`'s provided 4-node fixture propagation to convergence.
 *
 * Kept for the provided step demo / diagnostics. The readme section 4.1 harness
 * checks NO LONGER call this: they exercise the student's own `pageRank` /
 * `personalizedPageRank` through `REFERENCE_GRAPH.transition`, so an untouched
 * starter cannot pass a check it did not implement.
 *
 * Uses the provided `graphFixturePropagate(alpha, restart, pPrev)` helper when
 * present (fast, no rendering) and otherwise the public fixture demo
 * (`GRAPH_VIEW_STATE.fixture.restart` + `fixtureReset` / `fixtureStep`).
 *
 * @param {string} restartName `"uniform"` or a node label such as `"A"`.
 * @param {number} alpha
 * @returns {PropagationResult|null}
 */
function hw6FixtureIterate(restartName, alpha) {
  const nodeCount = 4;
  if (typeof graphFixturePropagate === "function") {
    try {
      let p = [0.25, 0.25, 0.25, 0.25];
      if (restartName !== "uniform" && typeof graphFixtureRestartVector === "function") {
        p = Array.from(graphFixtureRestartVector(restartName)).map(Number);
      } else if (restartName !== "uniform") {
        const order = typeof REFERENCE_GRAPH !== "undefined" && Array.isArray(REFERENCE_GRAPH.nodeOrder)
          ? REFERENCE_GRAPH.nodeOrder
          : ["A", "B", "C", "D"];
        const seed = order.indexOf(restartName);
        p = [0, 0, 0, 0];
        if (seed >= 0) p[seed] = 1;
      }
      let iterations = 0;
      let residual = Infinity;
      let converged = false;
      while (iterations < HW6_MAX_ITERATIONS) {
        const step = graphFixturePropagate(alpha, restartName, p);
        if (!step || !step.p || step.p.length !== nodeCount) return null;
        let l1 = 0;
        for (let i = 0; i < nodeCount; i++) l1 += Math.abs(Number(step.p[i]) - p[i]);
        p = Array.from(step.p).map(Number);
        iterations++;
        residual = l1;
        if (l1 < HW6_TOLERANCE) {
          converged = true;
          break;
        }
      }
      return { scores: p, iterations: iterations, residual: residual, converged: converged, source: "graph.js graphFixturePropagate" };
    } catch (error) {
      // Fall through to the public fixture demo below.
    }
  }

  if (typeof fixtureReset !== "function" || typeof fixtureStep !== "function") return null;
  const viewState = typeof GRAPH_VIEW_STATE !== "undefined" ? GRAPH_VIEW_STATE : null;
  if (!viewState || !viewState.fixture) return null;
  const fixture = viewState.fixture;
  const previousRestart = fixture.restart;
  const previousMode = viewState.mode;
  const previousAlpha = hw6ReadAlpha();
  try {
    fixture.restart = restartName;
    hw6SetAlpha(alpha);
    fixtureReset();
    let iterations = 0;
    let residual = Infinity;
    let converged = false;
    while (iterations < HW6_MAX_ITERATIONS) {
      const before = Array.from(fixture.p || []).map(Number);
      fixtureStep();
      const after = Array.from(fixture.p || []).map(Number);
      if (after.length !== nodeCount || before.length !== nodeCount) return null;
      let l1 = 0;
      for (let i = 0; i < nodeCount; i++) l1 += Math.abs(after[i] - before[i]);
      iterations++;
      residual = l1;
      if (l1 < HW6_TOLERANCE) {
        converged = true;
        break;
      }
    }
    const scores = Array.from(fixture.p || []).map(Number);
    if (scores.length !== nodeCount) return null;
    return { scores: scores, iterations: iterations, residual: residual, converged: converged, source: "graph.js fixtureStep" };
  } catch (error) {
    return null;
  } finally {
    try {
      fixture.restart = previousRestart;
      fixtureReset();
    } catch (error) {
      // Restoring the demo is best-effort.
    }
    viewState.mode = previousMode;
    hw6SetAlpha(previousAlpha);
  }
}

/**
 * Read `graph.js`'s row-normalized transition for a graph object and convert it
 * to the `{ rows, degrees }` form the verification step uses.
 *
 * `degrees[i]` is 0 exactly when `danglingMask[i]` is true, so the readme's
 * dangling-mass correction `alpha * (Q^T p + d * v) + (1 - alpha) * v` applies
 * unchanged.
 *
 * @param {Object} graph
 * @returns {{rows:Array<Float64Array>, degrees:Float64Array, danglingMask:boolean[]}|null}
 *   `null` when `rowNormalize` is unavailable or returns an unusable shape.
 */
function hw6NormalizedFromGraph(graph) {
  if (!graph || typeof rowNormalize !== "function") return null;
  const n = hw6NodeCount(graph);
  if (!n) return null;
  let result = null;
  try {
    result = rowNormalize(graph);
  } catch (error) {
    return null;
  }
  if (!result || !Array.isArray(result.Q) || result.Q.length !== n) return null;
  const mask = Array.isArray(result.danglingMask) ? result.danglingMask : null;
  const rows = new Array(n);
  const degrees = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    const source = result.Q[i];
    const row = new Float64Array(n);
    let sum = 0;
    for (let j = 0; j < n; j++) {
      const raw = source && source[j] !== undefined ? Number(source[j]) : 0;
      row[j] = Number.isFinite(raw) ? raw : 0;
      sum += row[j];
    }
    rows[i] = row;
    degrees[i] = mask ? (mask[i] ? 0 : 1) : sum > 0 ? 1 : 0;
  }
  return { rows: rows, degrees: degrees, danglingMask: mask };
}

/**
 * Build a minimal bipartite graph in `graph.js`'s documented `Graph` shape, so a
 * check can exercise the real `rowNormalize` on a fixture with known zero-degree
 * nodes instead of on the readme reference graph only.
 *
 * Node order is users first, then movies (ungrouped `nUsers` / `nMovies`).
 *
 * @param {number[][]} R `R[u][i]` 0/1, shape `[nUsers][nMovies]`.
 * @returns {Object} A `Graph`-shaped object sufficient for `rowNormalize`.
 */
function hw6MakeSyntheticGraph(R) {
  const nUsers = R.length;
  const nMovies = nUsers > 0 ? R[0].length : 0;
  const nodeCount = nUsers + nMovies;
  const userIds = [];
  const movieIds = [];
  for (let u = 0; u < nUsers; u++) userIds.push(u + 1);
  for (let i = 0; i < nMovies; i++) movieIds.push(i + 1);
  const userIndexOf = new Map();
  userIds.forEach(function (id, u) { userIndexOf.set(id, u); });
  const movieIndexOf = new Map();
  movieIds.forEach(function (id, i) { movieIndexOf.set(id, i); });

  const degree = new Float64Array(nodeCount);
  for (let u = 0; u < nUsers; u++) {
    for (let i = 0; i < nMovies; i++) {
      if (R[u][i]) {
        degree[u]++;
        degree[nUsers + i]++;
      }
    }
  }
  const zeroDegreeUsers = [];
  for (let u = 0; u < nUsers; u++) if (degree[u] === 0) zeroDegreeUsers.push(userIds[u]);
  const zeroDegreeMovies = [];
  for (let i = 0; i < nMovies; i++) if (degree[nUsers + i] === 0) zeroDegreeMovies.push(movieIds[i]);
  let positiveEdges = 0;
  for (let u = 0; u < nUsers; u++) {
    for (let i = 0; i < nMovies; i++) if (R[u][i]) positiveEdges++;
  }

  return {
    nUsers: nUsers,
    nMovies: nMovies,
    userIds: userIds,
    movieIds: movieIds,
    userIndexOf: userIndexOf,
    movieIndexOf: movieIndexOf,
    nodeCount: nodeCount,
    nodeOfUser: function (userId) {
      const local = userIndexOf.get(Number(userId));
      return local === undefined ? -1 : local;
    },
    movieOfNode: function (node) {
      const local = Number(node) - nUsers;
      return local >= 0 && local < nMovies ? movieIds[local] : -1;
    },
    isUserNode: function (node) { return Number(node) < nUsers; },
    degree: degree,
    zeroDegreeUsers: zeroDegreeUsers,
    zeroDegreeMovies: zeroDegreeMovies,
    positiveEdges: positiveEdges,
    R: R,
  };
}

/**
 * One readme-formula update: `p_next = alpha * (Q^T p + d * v) + (1 - alpha) * v`.
 *
 * @param {number} alpha
 * @param {{rows:Array<Float64Array>, degrees:Float64Array}} normalized
 * @param {ArrayLike<number>} v Restart distribution.
 * @param {ArrayLike<number>} p Current distribution.
 * @returns {Float64Array}
 */
function hw6Step(alpha, normalized, v, p) {
  const n = p.length;
  const out = new Float64Array(n);
  let dangling = 0;
  for (let i = 0; i < n; i++) {
    if (normalized.degrees[i] <= 0) dangling += p[i];
  }
  for (let i = 0; i < n; i++) {
    if (normalized.degrees[i] <= 0) continue;
    const row = normalized.rows[i];
    const weighted = alpha * p[i];
    for (let j = 0; j < n; j++) {
      if (row[j] !== 0) out[j] += weighted * row[j];
    }
  }
  const base = 1 - alpha + alpha * dangling;
  for (let j = 0; j < n; j++) out[j] += base * v[j];
  return out;
}

/**
 * Reference power iteration on a normalized dense matrix (readme section 3.4).
 *
 * @param {{rows:Array<Float64Array>, degrees:Float64Array}} normalized
 * @param {number} alpha
 * @param {ArrayLike<number>} restart
 * @returns {PropagationResult}
 */
function hw6ReferenceIterate(normalized, alpha, restart) {
  const n = restart.length;
  let p = Float64Array.from(restart);
  let iterations = 0;
  let residual = Infinity;
  let converged = false;
  while (iterations < HW6_MAX_ITERATIONS) {
    const next = hw6Step(alpha, normalized, restart, p);
    let l1 = 0;
    for (let i = 0; i < n; i++) l1 += Math.abs(next[i] - p[i]);
    p = next;
    iterations++;
    residual = l1;
    if (l1 < HW6_TOLERANCE) {
      converged = true;
      break;
    }
  }
  return { scores: p, iterations: iterations, residual: residual, converged: converged, source: "reference" };
}

/**
 * Uniform restart vector over `n` nodes.
 *
 * @param {number} n
 * @returns {Float64Array}
 */
function hw6UniformRestart(n) {
  const v = new Float64Array(n);
  if (n <= 0) return v;
  for (let i = 0; i < n; i++) v[i] = 1 / n;
  return v;
}

/**
 * One-hot restart vector at `seed`.
 *
 * @param {number} n
 * @param {number} seed
 * @returns {Float64Array}
 */
function hw6OneHotRestart(n, seed) {
  const v = new Float64Array(n);
  if (seed >= 0 && seed < n) v[seed] = 1;
  return v;
}

/**
 * Node index of a label inside a graph object, or `-1`.
 *
 * Reads `graph.nodeOrder` (the array `REFERENCE_GRAPH` uses) and then the
 * `label` field of `graph.nodes`.
 *
 * @param {Object} graph
 * @param {string} label
 * @returns {number}
 */
function hw6LabelIndex(graph, label) {
  if (!graph) return -1;
  if (Array.isArray(graph.nodeOrder)) {
    const position = graph.nodeOrder.indexOf(label);
    if (position !== -1) return position;
  }
  if (Array.isArray(graph.nodes)) {
    for (let i = 0; i < graph.nodes.length; i++) {
      const node = graph.nodes[i];
      if (!node) continue;
      if (node.label === label) return Number.isFinite(node.index) ? node.index : i;
    }
  }
  return -1;
}

/**
 * Run the STUDENT's `pageRank` on the 4-node reference fixture through the
 * frozen `opts.transition` interface.
 *
 * This is what makes the readme section 4.1 checks exercise the student's own
 * implementation instead of a provided fixture propagation. The provided
 * `graph.js` fixture propagation stays available for the step demo, but the
 * harness must not use it here.
 *
 * A `TODO(hw6)` throw from the stub propagates unchanged, so the harness reports
 * PENDING; a missing `REFERENCE_GRAPH.transition` is an integration gap and is
 * raised as `HW6_INTERFACE` (also PENDING), never FAIL.
 *
 * @param {number} alpha Link-following probability.
 * @returns {PropagationResult} Length-4 vector with `source` naming the student call.
 */
function hw6StudentReferencePageRank(alpha) {
  if (typeof pageRank !== "function") throw hw6InterfaceError("graph.js `pageRank` is unavailable");
  const reference = typeof REFERENCE_GRAPH !== "undefined" ? REFERENCE_GRAPH : null;
  if (!reference) throw hw6InterfaceError("REFERENCE_GRAPH is not defined");
  if (!reference.transition) throw hw6InterfaceError("REFERENCE_GRAPH.transition is not available yet");
  const out = pageRank(alpha, { transition: reference.transition });
  const result = hw6NormalizePropagation(out, 4);
  if (!result) throw hw6InterfaceError("graph.js `pageRank` did not return a length-4 vector for REFERENCE_GRAPH.transition");
  result.source = "student pageRank(alpha, {transition: REFERENCE_GRAPH.transition})";
  return result;
}

/**
 * Run the STUDENT's `personalizedPageRank` on the 4-node reference fixture
 * through the frozen `opts.transition` interface, seeding the given node index.
 *
 * @param {number} seedNode Seed node index (A = 0).
 * @param {number} alpha    Link-following probability.
 * @returns {PropagationResult} Length-4 vector with `source` naming the student call.
 */
function hw6StudentReferencePPR(seedNode, alpha) {
  if (typeof personalizedPageRank !== "function") {
    throw hw6InterfaceError("graph.js `personalizedPageRank` is unavailable");
  }
  const reference = typeof REFERENCE_GRAPH !== "undefined" ? REFERENCE_GRAPH : null;
  if (!reference) throw hw6InterfaceError("REFERENCE_GRAPH is not defined");
  if (!reference.transition) throw hw6InterfaceError("REFERENCE_GRAPH.transition is not available yet");
  const out = personalizedPageRank(seedNode, alpha, { transition: reference.transition });
  const result = hw6NormalizePropagation(out, 4);
  if (!result) {
    throw hw6InterfaceError("graph.js `personalizedPageRank` did not return a length-4 vector for REFERENCE_GRAPH.transition");
  }
  result.source = "student personalizedPageRank(seedNode, alpha, {transition: REFERENCE_GRAPH.transition})";
  return result;
}

/**
 * Run the automated self-checks and write the log into `#self-checks`.
 *
 * Checks that reach an unimplemented stub, or that cannot resolve the `graph.js`
 * interface, are reported as PENDING rather than FAIL, so the harness is useful
 * both before and after the three stubs are implemented. The harness never
 * throws.
 *
 * @param {HTMLElement|null} [logElement] Element that receives the text log.
 * @returns {{passed:number, failed:number, pending:number, checks:Array<Object>}}
 */
function runSelfChecks(logElement) {
  const checks = [];
  const close = function (a, b, eps) {
    const tolerance = eps === undefined ? 1e-6 : eps;
    return Math.abs(Number(a) - Number(b)) < tolerance;
  };

  const pass = function (name, detail) { checks.push({ name: name, status: "PASS", detail: detail }); };
  const fail = function (name, detail) { checks.push({ name: name, status: "FAIL", detail: detail }); };
  const pending = function (name, detail) { checks.push({ name: name, status: "PENDING", detail: detail }); };

  const check = function (name, fn) {
    try {
      const outcome = fn();
      if (outcome && outcome.pending) pending(name, outcome.pending);
      else pass(name, outcome === undefined ? "" : String(outcome));
    } catch (error) {
      const message = String(error && error.message ? error.message : error);
      if (message.indexOf(HW6_TODO_MARKER) !== -1) pending(name, "TODO(hw6): not implemented yet.");
      else if (message.indexOf(HW6_INTERFACE_MARK) !== -1) pending(name, "interface unavailable \u2014 " + message);
      else fail(name, message);
    }
  };

  const reference = typeof REFERENCE_GRAPH !== "undefined" ? REFERENCE_GRAPH : null;
  const UNIFORM_REF = [0.3725268513, 0.1958239118, 0.3941492369, 0.0375];
  const RESTART_A_REF = [0.4522328999, 0.1921989825, 0.3555681176, 0];

  // 1. Reference graph, uniform restart (readme 4.1) — the STUDENT's pageRank.
  check("reference graph: uniform alpha=0.85 matches readme 4.1 values", function () {
    if (!reference) throw hw6InterfaceError("REFERENCE_GRAPH is not defined");
    const readme = reference.reference && reference.reference.uniform ? reference.reference.uniform : null;
    const expected = readme && Array.isArray(readme.p) ? readme.p : UNIFORM_REF;
    const expectedRanking = readme && Array.isArray(readme.ranking) ? readme.ranking.join(",") : "2,0,1,3";
    // Exercise the student's own implementation through the frozen interface.
    // There is deliberately NO fixture fallback here: a provided propagation
    // would let an untouched starter pass a readme 4.1 check it did not implement.
    const result = hw6StudentReferencePageRank(0.85);

    for (let i = 0; i < expected.length; i++) {
      if (!close(result.scores[i], expected[i], 1e-6)) {
        throw new Error("node " + i + " = " + Number(result.scores[i]).toFixed(10) + ", expected " + expected[i]);
      }
    }
    const order = [0, 1, 2, 3].slice().sort(function (a, b) { return Number(result.scores[b]) - Number(result.scores[a]); });
    if (order.join(",") !== expectedRanking) throw new Error("ranking = " + order.join(",") + ", expected " + expectedRanking);
    return "C>A>B>D; produced by " + result.source + ", iterations " + result.iterations;
  });

  // 2. Reference graph, restart at A (readme 4.1) — the STUDENT's personalizedPageRank.
  check("reference graph: restart at A alpha=0.85 matches readme 4.1 values", function () {
    if (!reference) throw hw6InterfaceError("REFERENCE_GRAPH is not defined");
    const readme = reference.reference && reference.reference.restartAtA ? reference.reference.restartAtA : null;
    const expected = readme && Array.isArray(readme.p) ? readme.p : RESTART_A_REF;
    const expectedRanking = readme && Array.isArray(readme.ranking) ? readme.ranking.join(",") : "0,2,1,3";
    const seed = hw6LabelIndex(reference, "A");
    if (seed < 0) throw hw6InterfaceError("node index of A is unavailable on REFERENCE_GRAPH");
    const result = hw6StudentReferencePPR(seed, 0.85);

    for (let i = 0; i < expected.length; i++) {
      if (!close(result.scores[i], expected[i], 1e-6)) {
        throw new Error("node " + i + " = " + Number(result.scores[i]).toFixed(10) + ", expected " + expected[i]);
      }
    }
    const order = [0, 1, 2, 3].slice().sort(function (a, b) { return Number(result.scores[b]) - Number(result.scores[a]); });
    if (order.join(",") !== expectedRanking) throw new Error("ranking = " + order.join(",") + ", expected " + expectedRanking);
    return "A>C>B>D (D stays 0); produced by " + result.source;
  });

  // 3. Row normalization (graph.js `rowNormalize(graph)` -> { Q, danglingMask }).
  check("row normalization: every nonzero-degree row sums to 1", function () {
    if (!hw6GraphJsReady()) return { pending: "graph.js `rowNormalize` is still a " + HW6_TODO_MARKER + " stub" };
    // Two fixtures: a star with no zero-degree nodes, and one with two of them.
    const fixtures = [
      { R: [[1, 1]], dangling: [false, false, false] },
      { R: [[1, 1, 0], [0, 0, 0]], dangling: [false, true, false, false, true] },
    ];
    let positive = 0;
    let dangling = 0;
    for (let f = 0; f < fixtures.length; f++) {
      const synthetic = hw6MakeSyntheticGraph(fixtures[f].R);
      const result = rowNormalize(synthetic);
      const n = synthetic.nodeCount;
      if (!result || !Array.isArray(result.Q) || result.Q.length !== n || !Array.isArray(result.danglingMask)) {
        throw hw6InterfaceError("rowNormalize did not return { Q, danglingMask } for a synthetic graph");
      }
      for (let i = 0; i < n; i++) {
        let sum = 0;
        for (let j = 0; j < n; j++) sum += Number((result.Q[i] && result.Q[i][j]) || 0);
        if (!!result.danglingMask[i] !== fixtures[f].dangling[i]) {
          throw new Error("danglingMask[" + i + "] = " + result.danglingMask[i] + ", expected " + fixtures[f].dangling[i]);
        }
        if (fixtures[f].dangling[i]) {
          if (!close(sum, 0, 1e-12)) throw new Error("dangling row " + i + " is not zero (sum " + sum + ")");
          dangling++;
        } else {
          if (!close(sum, 1, 1e-9)) throw new Error("row " + i + " sums to " + sum);
          positive++;
        }
      }
    }
    return positive + " positive-degree row(s) sum to 1; " + dangling + " dangling row(s) are zero";
  });

  // 4. Restart distributions.
  check("restart distributions: nonnegative and summing to 1", function () {
    if (!reference) throw hw6InterfaceError("REFERENCE_GRAPH is not defined");
    const n = hw6NodeCount(reference);
    const vectors = [hw6UniformRestart(n), hw6OneHotRestart(n, 0), hw6OneHotRestart(n, n - 1)];
    for (let k = 0; k < vectors.length; k++) {
      let sum = 0;
      for (let i = 0; i < n; i++) {
        if (vectors[k][i] < 0) throw new Error("restart vector " + k + " has a negative entry");
        sum += vectors[k][i];
      }
      if (!close(sum, 1, 1e-12)) throw new Error("restart vector " + k + " sums to " + sum);
    }
    return "uniform + one-hot over " + n + " nodes";
  });

  // 5. Probability preservation, on the application's own pageRank.
  check("iterations preserve total probability (sum approx 1)", function () {
    // 1 user x 2 movies: user 1 rates both movies; both movie nodes point back.
    const outcome = hw6WithSyntheticGraph([[1, 1]], function () {
      const graph = hw6ActiveGraph();
      return { graph: graph, n: hw6NodeCount(graph), result: hw6AdapterPageRank(graph, 0.85) };
    });
    let sum = 0;
    for (let i = 0; i < outcome.result.scores.length; i++) sum += Number(outcome.result.scores[i]);
    if (!close(sum, 1, 1e-6)) throw new Error("total mass = " + sum);
    return "sum = " + sum.toFixed(10) + " over " + outcome.n + " synthetic nodes (" + outcome.result.source + ")";
  });

  // 6. Nonnegativity, on the application's own pageRank.
  check("stationary scores are nonnegative", function () {
    const outcome = hw6WithSyntheticGraph([[1, 1]], function () {
      const graph = hw6ActiveGraph();
      return { result: hw6AdapterPageRank(graph, 0.85) };
    });
    for (let i = 0; i < outcome.result.scores.length; i++) {
      if (!(Number(outcome.result.scores[i]) >= 0)) throw new Error("node " + i + " = " + outcome.result.scores[i]);
    }
    return outcome.result.scores.length + " scores nonnegative (" + outcome.result.source + ")";
  });

  // 7. Stationarity of one extra update, on the application's own pageRank.
  check("one extra update produces approximately the same distribution", function () {
    if (!hw6GraphJsReady()) return { pending: "graph.js `rowNormalize` is still a " + HW6_TODO_MARKER + " stub" };
    const synthetic = hw6MakeSyntheticGraph([[1, 1]]);
    const normalized = hw6NormalizedFromGraph(synthetic);
    if (!normalized) throw hw6InterfaceError("rowNormalize did not return a usable { Q, danglingMask }");
    const result = hw6WithSyntheticGraph([[1, 1]], function () {
      const graph = hw6ActiveGraph();
      return hw6AdapterPageRank(graph, 0.85);
    });
    const n = synthetic.nodeCount;
    const next = hw6Step(0.85, normalized, hw6UniformRestart(n), result.scores);
    let l1 = 0;
    for (let i = 0; i < n; i++) l1 += Math.abs(next[i] - Number(result.scores[i]));
    if (!(l1 < 1e-6)) throw new Error("L1 change after one update = " + l1);
    return "L1 = " + l1.toExponential(3);
  });

  // 8. Zero-degree node fixture preserves mass.
  check("zero-degree node fixture preserves probability mass", function () {
    if (!hw6GraphJsReady()) {
      return { pending: "graph.js `rowNormalize` is still a " + HW6_TODO_MARKER + " stub" };
    }
    // 2 users x 3 movies: user 1 rates movies 1 and 2; user 2 has no positives;
    // movie 3 has no positives. Node order users-then-movies gives exactly two
    // zero-degree nodes (node 1 and node 4) and no padding node.
    const R = [[1, 1, 0], [0, 0, 0]];
    const synthetic = hw6MakeSyntheticGraph(R);
    const normalized = hw6NormalizedFromGraph(synthetic);
    if (!normalized) throw hw6InterfaceError("rowNormalize did not return a usable { Q, danglingMask }");
    const expectedDangling = [false, true, false, false, true];
    for (let i = 0; i < synthetic.nodeCount; i++) {
      if (!!normalized.danglingMask[i] !== expectedDangling[i]) {
        throw new Error("danglingMask[" + i + "] = " + normalized.danglingMask[i] + ", expected " + expectedDangling[i]);
      }
    }
    // The application's own PageRank and PPR must keep the total mass at 1 even
    // when a zero-degree node is seeded.
    const outcome = hw6WithSyntheticGraph(R, function () {
      const graph = hw6ActiveGraph();
      return {
        global: hw6AdapterPageRank(graph, 0.85),
        seeded: hw6AdapterPPR(graph, 1, 0.85, 2),
      };
    });
    let globalSum = 0;
    let seededSum = 0;
    for (let i = 0; i < synthetic.nodeCount; i++) {
      globalSum += Number(outcome.global.scores[i]);
      seededSum += Number(outcome.seeded.scores[i]);
    }
    if (!close(globalSum, 1, 1e-6)) throw new Error("uniform-restart total mass = " + globalSum);
    if (!close(seededSum, 1, 1e-6)) throw new Error("zero-degree-seed total mass = " + seededSum);
    return "danglingMask [" + expectedDangling.join(",") + "]; mass " + globalSum.toFixed(8) + " / " + seededSum.toFixed(8);
  });

  // 9. Matrix orientation / graph direction.
  check("matrix orientation: probability collects at the pointed-to node, not the source", function () {
    if (!hw6GraphJsReady()) {
      return { pending: "graph.js `rowNormalize` is still a " + HW6_TODO_MARKER + " stub" };
    }
    // 1 user x 2 movies, both movies pointed to by the user and both pointing
    // back to it. The user node is the hub, so P^T must give it the largest
    // stationary mass; an untransposed update would make the movies win.
    const synthetic = hw6MakeSyntheticGraph([[1, 1]]);
    const normalized = hw6NormalizedFromGraph(synthetic);
    if (!normalized) throw hw6InterfaceError("rowNormalize did not return a usable { Q, danglingMask }");
    const n = synthetic.nodeCount;
    const fromApp = hw6WithSyntheticGraph([[1, 1]], function () {
      const graph = hw6ActiveGraph();
      return Array.from(hw6AdapterPageRank(graph, 0.85).scores).map(Number);
    });
    const fromReference = Array.from(hw6ReferenceIterate(normalized, 0.85, hw6UniformRestart(n)).scores);
    const pairs = [fromApp, fromReference];
    for (let k = 0; k < pairs.length; k++) {
      const p = pairs[k];
      if (!(p[0] > p[1])) {
        throw new Error(
          (k === 0 ? "pageRank" : "reference") + " hub mass " + p[0].toFixed(6) +
          " should exceed movie mass " + p[1].toFixed(6),
        );
      }
      if (!close(p[1], p[2], 1e-9)) throw new Error("the two symmetric movies must tie: " + p[1] + " vs " + p[2]);
    }
    let sum = 0;
    for (let i = 0; i < n; i++) sum += fromApp[i];
    if (!close(sum, 1, 1e-6)) throw new Error("total mass = " + sum);
    return "p = [" + fromApp.map(function (x) { return x.toFixed(6); }).join(", ") + "]";
  });

  // 10. Recommendation filtering and tie-break.
  check("recommendTopK: excludes previously rated movies and breaks ties by ascending id", function () {
    if (!HW6_STATE.splitReady) throw hw6InterfaceError("the training split is not available");
    const cohort = hw6SharedCohort("test");
    if (cohort.length === 0) {
      return { pending: "the shared test cohort is empty for this dataset, so no user can be checked" };
    }
    const userId = cohort[0];
    const recs = recommendTopK("popularity", userId, HW6_K, "test");
    if (!Array.isArray(recs)) throw new Error("expected an array of recommendations");
    if (recs.length > HW6_K) throw new Error("returned " + recs.length + " items for K=" + HW6_K);
    const excluded = hw6ExclusionSet("test", userId);
    for (let i = 0; i < recs.length; i++) {
      if (excluded.has(recs[i].movieId)) throw new Error("movie " + recs[i].movieId + " was rated by the user and must be excluded");
      if (!HW6_STATE.candidateSet.has(recs[i].movieId)) throw new Error("movie " + recs[i].movieId + " is outside the training candidate pool");
      if (i > 0) {
        const prev = recs[i - 1];
        if (Number(prev.score) < Number(recs[i].score)) throw new Error("results are not sorted by descending score");
        if (Number(prev.score) === Number(recs[i].score) && !(prev.movieId < recs[i].movieId)) {
          throw new Error("tie between " + prev.movieId + " and " + recs[i].movieId + " is not broken by ascending id");
        }
      }
    }
    return recs.length + " eligible items for user " + userId;
  });

  // 11. No padding node.
  check("graph mapping: node count equals observed users plus observed movies", function () {
    const graph = hw6ActiveGraph();
    if (!graph) throw hw6InterfaceError("no graph is active; press \"Build graph\" first");
    if (!HW6_STATE.splitReady) throw hw6InterfaceError("the training split is not available");
    const observedUsers = HW6_STATE.trainUserIds.length;
    const observedMovies = HW6_STATE.candidateIds.length;
    const observed = observedUsers + observedMovies;
    const nodes = hw6NodeCount(graph);
    if (!nodes) throw hw6InterfaceError("the graph does not report a node count");
    if (nodes > observed) {
      throw new Error("graph reports " + nodes + " nodes but observed users + movies = " + observed + " (a padding node?)");
    }
    if (nodes < observed) {
      // A fixture / step view legitimately activates a smaller reference graph;
      // that is not evidence of a padding node, so this is not a failure.
      return { pending: "the active graph has " + nodes + " nodes, fewer than the " + observed + " dataset nodes (a fixture view may be active)" };
    }
    const graphMovies = hw6MovieIds(graph).length;
    const graphUsers = hw6UserIds(graph).length;
    if (graphMovies && graphUsers && (graphMovies !== observedMovies || graphUsers !== observedUsers)) {
      throw new Error(
        "graph reports " + graphUsers + " users / " + graphMovies + " movies but the training split observed " +
        observedUsers + " / " + observedMovies,
      );
    }
    return observed + " nodes = " + observedUsers + " users + " + observedMovies + " movies";
  });

  // 12. Public surface.
  check("public surface: every required name is defined", function () {
    const required = [
      "movieScores", "recommendTopK", "recallAtK", "popularityScores", "mfScores",
      "serializeVerification", "serializeResults", "serializeRecommendations", "download", "runSelfChecks",
    ];
    const missing = required.filter(function (name) { return typeof window[name] !== "function"; });
    if (missing.length) throw new Error("missing top-level functions: " + missing.join(", "));
    if (!Array.isArray(ALPHA_GRID) || ALPHA_GRID.length !== 4) throw new Error("ALPHA_GRID is not the four-value grid");
    if (!MF_INFO || typeof MF_INFO !== "object") throw new Error("MF_INFO is not exposed");
    return required.length + " functions + ALPHA_GRID + MF_INFO";
  });

  const passed = checks.filter(function (c) { return c.status === "PASS"; }).length;
  const failed = checks.filter(function (c) { return c.status === "FAIL"; }).length;
  const pendingCount = checks.filter(function (c) { return c.status === "PENDING"; }).length;

  const lines = [
    "HW6 self-checks \u2014 pass " + passed + ", fail " + failed + ", pending " + pendingCount + " (of " + checks.length + ")",
    "backend: " + (HW6_STATE.backend || "unknown"),
    "MF reference: " + (MF_INFO.checkpoint === null ? "not trained yet" : "factors " + MF_INFO.factors + ", checkpoint epoch " + MF_INFO.checkpoint + ", val RMSE " + hw6Fmt(MF_INFO.validationRMSE)),
    "",
  ];
  checks.forEach(function (c) {
    lines.push(c.status + "  " + c.name + (c.detail ? " \u2014 " + c.detail : ""));
  });
  const text = lines.join("\n");

  const target = logElement || document.getElementById("self-checks");
  if (target) target.textContent = text;

  const summary = { passed: passed, failed: failed, pending: pendingCount, checks: checks };
  HW6_STATE.lastVerification = summary;
  return summary;
}

// ---------------------------------------------------------------------------
// Exporters (readme section 9)
// ---------------------------------------------------------------------------

/**
 * Machine-readable self-check export.
 *
 * @returns {string} JSON text with `backend`, `generated_at`, `checks`,
 *   `passed`, `failed` and `pending`. Runs the harness first when it has not
 *   been run in this session, so the exported file is never empty.
 */
function serializeVerification() {
  const result = HW6_STATE.lastVerification || runSelfChecks();
  let backend = HW6_STATE.backend;
  if (!backend && typeof tf !== "undefined" && typeof tf.getBackend === "function") {
    try {
      backend = tf.getBackend();
    } catch (error) {
      backend = null;
    }
  }
  const payload = {
    backend: backend,
    generated_at: new Date().toISOString(),
    checks: result.checks.map(function (c) {
      return { name: c.name, status: c.status, detail: c.detail };
    }),
    passed: result.passed,
    failed: result.failed,
    pending: result.pending,
  };
  return JSON.stringify(payload, null, 2);
}

/**
 * Exporter for the retrieval metrics, as CSV.
 *
 * Columns, in order: `method, alpha, recall_at_10, cohort_size,
 * excluded_unseen_ids, excluded_no_eligible_user, mean_popularity, iterations,
 * residual, converged`, then evidence columns (`split`, `excluded_unseen_id_list`,
 * `run_id`, `run_status`, `generated_at`) that do not disturb the required prefix.
 *
 * Rows come from the immutable last-COMPLETED snapshot only. A cancelled or
 * failed in-flight run never contributes rows and never mixes with earlier
 * evidence: the exported file is always one internally consistent completed run,
 * labelled by `run_id`/`run_status`. Changing the demo slider cannot alter it.
 * Before any completed run the file is header-only rather than a partial mix.
 *
 * @returns {string} CSV text (RFC 4180 quoting).
 */
function serializeResults() {
  const header = [
    "method", "alpha", "recall_at_10", "cohort_size", "excluded_unseen_ids",
    "excluded_no_eligible_user", "mean_popularity", "iterations", "residual",
    "converged", "split", "excluded_unseen_id_list", "run_id", "run_status",
    "generated_at",
  ];
  const rows = [];
  const generatedAt = new Date().toISOString();
  const snapshot = HW6_STATE.completedSnapshot;
  const identity = hw6ExportRunIdentity();

  const exclusions = HW6_STATE.splitReady ? hw6CohortExclusions("test") : null;
  const unseenList = exclusions && exclusions.unseenIds ? exclusions.unseenIds.join(";") : "";

  if (!snapshot) return hw6CsvDocument(header, rows);

  (snapshot.comparisonRows || []).forEach(function (row) {
    rows.push([
      row.method,
      row.alpha === null ? "" : Number(row.alpha).toFixed(2),
      row.recall === null ? "" : row.recall,
      row.cohortSize,
      row.excludedUnseen,
      row.excludedNoEligible,
      row.meanPopularity === null ? "" : row.meanPopularity,
      row.iterations === null ? "" : row.iterations,
      row.residual === null ? "" : row.residual,
      row.converged === null ? "" : String(row.converged),
      row.split,
      unseenList,
      identity.id,
      identity.status,
      generatedAt,
    ]);
  });

  (snapshot.alphaSweepRows || []).forEach(function (row) {
    rows.push([
      "ppr",
      Number(row.alpha).toFixed(2),
      row.recall === null ? "" : row.recall,
      row.cohortSize,
      "",
      "",
      "",
      "",
      "",
      "",
      "val",
      "",
      identity.id,
      identity.status,
      generatedAt,
    ]);
  });

  ["pagerank", "ppr"].forEach(function (key) {
    const record = snapshot.convergence[key];
    if (!record) return;
    const verdict = hw6ConvergenceVerdict(record);
    rows.push([
      key,
      hw6Fmt(record.alpha, 2),
      "",
      hw6SharedCohort("test").length,
      "",
      "",
      "",
      record.iterations === null ? "" : record.iterations,
      record.residual === null ? "" : record.residual,
      verdict.value === null ? "" : String(verdict.value),
      "test",
      "",
      identity.id,
      identity.status,
      generatedAt,
    ]);
  });

  return hw6CsvDocument(header, rows);
}

/**
 * Exporter for the recommendation lists, as CSV.
 *
 * Columns: `user_id, method, alpha, rank, movie_id, title, genres, score,
 * training_popularity`. Rows are recorded whenever a list is produced on screen
 * (the Top-10 panel or the two-user demonstration), so the exported file matches
 * the last tables actually rendered.
 *
 * @returns {string} CSV text (RFC 4180 quoting; commas inside titles are quoted).
 */
function serializeRecommendations() {
  const snapshot = HW6_STATE.completedSnapshot;
  const header = [
    "user_id", "method", "alpha", "rank", "movie_id", "title", "genres",
    "score", "training_popularity", "list_mean_popularity", "run_id", "run_status",
  ];
  const snapRecs = snapshot && snapshot.recommendations
    ? (Array.isArray(snapshot.recommendations.rows) ? snapshot.recommendations.rows : [])
    : null;
  const sourceRows = snapRecs !== null ? snapRecs : HW6_STATE.recommendationRows;
  const identity = snapRecs !== null
    ? { id: snapshot.identity || "", status: HW6_STATE.runState }
    : hw6ExportRunIdentity();
  const groupTotals = {};
  sourceRows.forEach(function (row) {
    const key = row.userId + "|" + row.method + "|" + row.alpha;
    if (!groupTotals[key]) groupTotals[key] = { sum: 0, n: 0 };
    groupTotals[key].sum += Number(row.trainingPopularity) || 0;
    groupTotals[key].n += 1;
  });
  const rows = sourceRows.map(function (row) {
    const key = row.userId + "|" + row.method + "|" + row.alpha;
    const group = groupTotals[key];
    const mean = group && group.n ? group.sum / group.n : "";
    return [
      row.userId,
      row.method,
      row.alpha,
      row.rank,
      row.movieId,
      row.title,
      row.genres,
      row.score,
      row.trainingPopularity,
      mean === "" ? "" : hw6Fmt(mean),
      identity.id,
      identity.status,
    ];
  });
  return hw6CsvDocument(header, rows);
}

/**
 * Wire the three exporter buttons to `download`.
 *
 * @returns {void}
 */
function hw6WireExports() {
  const verification = document.getElementById("export-verification");
  if (verification) {
    verification.addEventListener("click", function () {
      if (hw6RefuseIfBusy("Exporting verification")) return;
      download("verification.json", serializeVerification(), "application/json");
    });
  }
  const results = document.getElementById("export-results");
  if (results) {
    results.addEventListener("click", function () {
      if (hw6RefuseIfBusy("Exporting results")) return;
      download("results.csv", serializeResults(), "text/csv");
    });
  }
  const recommendations = document.getElementById("export-recommendations");
  if (recommendations) {
    recommendations.addEventListener("click", function () {
      if (hw6RefuseIfBusy("Exporting recommendations")) return;
      download("recommendations.csv", serializeRecommendations(), "text/csv");
    });
  }
}

// ---------------------------------------------------------------------------
// graph.js call sites (provided)
// ---------------------------------------------------------------------------

/**
 * Build the graph from the training split, then refresh the graph summary and
 * view.
 *
 * The real `graph.js` signature is `buildGraph(train)`, so the training-record
 * array is passed; the arity dispatch below keeps a 2-argument
 * `(train, movies)` or 0-argument form working too. `setGraph` is preferred over
 * assigning `GRAPH`, because `graph.js` re-renders the summary and the graph
 * view from inside it.
 *
 * @returns {void}
 */
function hw6HandleBuildGraph() {
  if (hw6EvaluationActive()) {
    hw6SetStatus("A graph rebuild is disabled while an evaluation is running \u2014 it would replace the graph the live kernel is using. Wait or press Cancel.");
    return;
  }
  if (typeof buildGraph !== "function") {
    hw6SetStatus("graph.js `buildGraph` is not available on this page.");
    hw6RenderGraphSummary();
    return;
  }
  try {
    const arity = buildGraph.length;
    let graph;
    if (arity >= 2) graph = buildGraph(HW6_SPLIT.train, movies);
    else if (arity === 1) graph = buildGraph(HW6_SPLIT.train);
    else graph = buildGraph();
    if (graph && typeof setGraph === "function") {
      setGraph(graph);
    }
    hw6InvalidateCaches();
    hw6SafeRenderGraphView();
    const nodes = hw6NodeCount(graph);
    hw6SetStatus("Graph built: " + (nodes ? nodes.toLocaleString("en-US") + " nodes reported." : "node count not reported."));
  } catch (error) {
    const message = String(error && error.message ? error.message : error);
    hw6SetStatus("Graph build failed: " + message);
    if (message.indexOf(HW6_TODO_MARKER) === -1) {
      const target = document.getElementById("graph-summary-body");
      if (target) target.innerHTML = hw6UnavailableHtml("Graph summary", error);
    } else {
      hw6RenderGraphSummary();
    }
  }
}

/**
 * Candidate node-score vector for the graph view's "score" colouring.
 *
 * Returns a `Float64Array` indexed by graph node index (users first, then
 * movies): user nodes contribute 0, movie nodes carry the selected method's
 * score. Returns `null` when the scores cannot be produced (for example while
 * `movieScores` is still a `TODO(hw6)` stub), so the caller can fall back to
 * degree colouring without throwing.
 *
 * @param {Object} graph
 * @param {number} seedUserId
 * @returns {Float64Array|null}
 */
function hw6NodeScoresForView(graph, seedUserId) {
  if (!graph) return null;
  const n = hw6NodeCount(graph);
  if (!n) return null;
  const method = hw6ReadMethod();
  const alpha = hw6ReadAlpha();
  try {
    if (method === "pagerank") return Float64Array.from(hw6RunPageRank(graph, alpha).scores);
    if (method === "ppr") return Float64Array.from(hw6RunPPR(graph, seedUserId, alpha).scores);
    const vector = new Float64Array(n);
    // The view accessor takes a GLOBAL node index.
    const index = hw6MovieNodeIndex(graph);
    if (method === "popularity") {
      index.forEach(function (node, movieId) { vector[node] = hw6PopularityOf(movieId); });
      return vector;
    }
    const scored = mfScores(seedUserId);
    index.forEach(function (node, movieId) {
      const value = scored[movieId];
      vector[node] = value === undefined ? 0 : value;
    });
    return vector;
  } catch (error) {
    return null;
  }
}

/**
 * Render the graph view with the current controls.
 *
 * Uses the real `graph.js` options object (`renderGraphView({ mode, seedUserId,
 * colorBy, maxNodes, getScore })`): after a graph is built the view switches to
 * the seed user's ego subgraph coloured by the selected method's scores. Falls
 * back to a bare call if that shape is rejected.
 *
 * @returns {void}
 */
function hw6SafeRenderGraphView() {
  if (typeof renderGraphView !== "function") return;
  const maxNodes = Math.max(10, Math.round(hw6ReadNumber("max-nodes", 120)));
  const graph = hw6ActiveGraph();
  const seedUserId = hw6ReadNumber("seed-user-select", hw6ReadNumber("user-select", Number.NaN));

  const opts = { maxNodes: maxNodes };
  let scores = null;
  if (graph && Number.isFinite(seedUserId)) {
    scores = hw6NodeScoresForView(graph, seedUserId);
    opts.mode = "ego";
    opts.seedUserId = seedUserId;
    if (scores) {
      opts.colorBy = "score";
      opts.getScore = function (nodeIndex) {
        const value = Number(scores[nodeIndex]);
        return Number.isFinite(value) ? value : 0;
      };
    } else {
      opts.colorBy = "degree";
    }
  }

  try {
    renderGraphView(opts);
    return;
  } catch (error) {
    // Fall through to the no-argument form.
  }
  try {
    renderGraphView();
  } catch (error) {
    hw6SetStatus("Graph view could not be rendered: " + String(error && error.message ? error.message : error));
  }
}

/**
 * Run global PageRank at the selected alpha and record convergence.
 *
 * @returns {void}
 */
function hw6HandleRunPageRank() {
  if (hw6RefuseIfBusy("Running PageRank")) return;
  const graph = hw6ActiveGraph();
  const target = document.getElementById("convergence-results");
  if (!graph) {
    hw6SetStatus("Build the graph before running PageRank.");
    if (target) target.innerHTML = '<p class="empty-state">No graph yet \u2014 press \u201cBuild graph\u201d first.</p>';
    return;
  }
  try {
    hw6RunAndRecordConvergence("pagerank", graph, hw6ReadAlpha());
    hw6RenderConvergence();
    hw6SafeRenderGraphView();
    const record = HW6_STATE.convergence.pagerank;
    const verdict = hw6ConvergenceVerdict(record);
    hw6SetStatus(
      "Global PageRank: " + record.iterations + " iteration(s), final L1 " + hw6Fmt(record.residual, 8) +
      ", converged " + (verdict.value ? "yes" : "no") + (verdict.derived ? " (derived)" : "") + ".",
    );
  } catch (error) {
    if (target) target.innerHTML = hw6UnavailableHtml("Convergence", error);
    hw6SetStatus("PageRank unavailable: " + String(error && error.message ? error.message : error));
  }
}

/**
 * Run PPR from the restart seed user at the selected alpha and record convergence.
 *
 * @returns {void}
 */
function hw6HandleRunPPR() {
  if (hw6RefuseIfBusy("Running PPR")) return;
  const graph = hw6ActiveGraph();
  const target = document.getElementById("convergence-results");
  if (!graph) {
    hw6SetStatus("Build the graph before running PPR.");
    if (target) target.innerHTML = '<p class="empty-state">No graph yet \u2014 press \u201cBuild graph\u201d first.</p>';
    return;
  }
  const seedUserId = hw6ReadNumber("seed-user-select", hw6ReadNumber("user-select", Number.NaN));
  if (!Number.isFinite(seedUserId)) {
    hw6SetStatus("No restart seed user selected.");
    return;
  }
  try {
    hw6RunAndRecordConvergence("ppr", graph, hw6ReadAlpha(), seedUserId);
    hw6RenderConvergence();
    hw6SafeRenderGraphView();
    const record = HW6_STATE.convergence.ppr;
    const verdict = hw6ConvergenceVerdict(record);
    hw6SetStatus(
      "PPR from user " + seedUserId + ": " + record.iterations + " iteration(s), final L1 " + hw6Fmt(record.residual, 8) +
      ", converged " + (verdict.value ? "yes" : "no") + (verdict.derived ? " (derived)" : "") + ".",
    );
  } catch (error) {
    if (target) target.innerHTML = hw6UnavailableHtml("Convergence", error);
    hw6SetStatus("PPR unavailable: " + String(error && error.message ? error.message : error));
  }
}

/**
 * Run the whole retrieval evaluation: four-method comparison, validation alpha
 * sweep and the recall detail for the selected method. Every panel degrades to
 * a readable state while the stubs are unimplemented.
 *
 * The two-user demonstration is NOT part of this run: it has its own control
 * (`#two-user-demo-btn`) and its own output container (`#two-user-demo-results`).
 *
 * @returns {Promise<void>}
 */
async function hw6HandleRecall() {
  // Fixed order: validation sweep selects the alpha, then the primary TEST
  // comparison runs at that alpha. A completed sweep is reused by a repeat run.
  await hw6RunEvaluationAsync({ stages: "full" });
}

/**
 * Run the validation-only PPR alpha sweep (readme section 6.1) and render it
 * into `#alpha-sweep-results`, reporting the selected alpha (exact ties toward
 * the smaller alpha). Degrades to a readable explanation while the stubs throw.
 *
 * @returns {Promise<void>}
 */
async function hw6HandleAlphaSweep() {
  // Refresh the validation sweep only (no primary comparison) with progress and
  // cancellation. Uses the same shared ordering rules as the full evaluation.
  await hw6RunEvaluationAsync({ stages: "sweep" });
}

/**
 * Run the readme section 6.2 one-factor-at-a-time two-user demonstration and
 * render it into `#two-user-demo-results` (its only output target). Degrades to
 * a readable explanation while the stubs throw.
 *
 * @returns {void}
 */
function hw6HandleTwoUserDemo() {
  const target = document.getElementById("two-user-demo-results");
  if (hw6EvaluationActive()) {
    hw6SetStatus("The two-user demonstration is disabled while an evaluation is running. Wait or press Cancel.");
    return;
  }
  let demo = null;
  try {
    demo = hw6RunTwoUserDemo();
  } catch (error) {
    if (target) target.innerHTML = hw6UnavailableHtml("Two-user demo", error);
    hw6SetStatus("Two-user demonstration unavailable: " + String(error && error.message ? error.message : error));
    return;
  }
  if (target) {
    target.innerHTML =
      hw6RunTwoUserDemoHtml(demo) ||
      '<p class="empty-state">The two-user demonstration produced no output.</p>';
  }
  if (demo && demo.error) {
    hw6SetStatus("Two-user demonstration unavailable: " + demo.error);
  } else if (demo && demo.users && demo.users.length >= 2) {
    hw6SetStatus("Two-user demonstration complete \u2014 users " + demo.users[0] + " and " + demo.users[1] +
      "; one factor changes per comparison.");
  } else {
    hw6SetStatus("Two-user demonstration could not select two cohort users.");
  }
}

/**
 * Run the self-checks and report the summary in the status line.
 *
 * @returns {void}
 */
function hw6HandleRunTests() {
  // Self-checks temporarily replace the active graph (`setGraph`), which disposes
  // the tensors the live batched kernel still holds. Refuse while a run is in
  // flight so the two cannot race.
  if (hw6EvaluationActive()) {
    hw6SetStatus("Self-checks are disabled while an evaluation is running \u2014 they temporarily replace the graph. Wait or press Cancel.");
    return;
  }
  const summary = runSelfChecks();
  hw6SetStatus(
    "Self-checks: pass " + summary.passed + ", fail " + summary.failed + ", pending " + summary.pending +
    ". Unimplemented TODO(hw6) paths are reported PENDING, not failed.",
  );
}

// ---------------------------------------------------------------------------
// UI wiring (provided)
// ---------------------------------------------------------------------------

/**
 * Wire every control once. Safe to call a single time from the bootstrap; all
 * lookups are null-guarded so a missing element cannot throw at load time.
 *
 * @returns {void}
 */
function hw6WireControls() {
  const alpha = document.getElementById("alpha");
  if (alpha) {
    alpha.addEventListener("input", function () {
      const readout = document.getElementById("alpha-value");
      const value = hw6ReadAlpha();
      if (readout) readout.textContent = value.toFixed(2);
      // Caches are keyed by alpha, so no result is dropped here; the next run at
      // this alpha reuses any walk already computed for it.
      hw6SetStatus("Alpha set to " + value.toFixed(2) + ". Press a run button to (re)compute at this alpha.");
    });
  }

  const method = document.getElementById("method");
  if (method) {
    // Selecting `mf` is the first need for MF scores: train the lazy reference
    // now (with progress reported and the browser painting between epochs).
    method.addEventListener("change", async function () {
      const value = hw6ReadMethod();
      if (hw6RefuseIfBusy("Changing the method")) {
        return;
      }
      if (value === "mf") {
        await hw6EnsureMfForNeed();
      }
      hw6SetStatus("Method set to " + value + ". Press \u201cRecommend Top-10\u201d to refresh the list.");
    });
  }

  const buildButton = document.getElementById("build-graph");
  if (buildButton) buildButton.addEventListener("click", hw6HandleBuildGraph);

  const pagerankButton = document.getElementById("run-pagerank");
  if (pagerankButton) pagerankButton.addEventListener("click", hw6HandleRunPageRank);

  const pprButton = document.getElementById("run-ppr");
  if (pprButton) pprButton.addEventListener("click", hw6HandleRunPPR);

  const recommendButton = document.getElementById("recommend-btn");
  if (recommendButton) recommendButton.addEventListener("click", hw6RenderRecommendations);

  const recallButton = document.getElementById("recall-btn");
  if (recallButton) recallButton.addEventListener("click", hw6HandleRecall);

  const alphaSweepButton = document.getElementById("alpha-sweep-btn");
  if (alphaSweepButton) alphaSweepButton.addEventListener("click", hw6HandleAlphaSweep);

  const twoUserDemoButton = document.getElementById("two-user-demo-btn");
  if (twoUserDemoButton) twoUserDemoButton.addEventListener("click", hw6HandleTwoUserDemo);

  const testButton = document.getElementById("run-tests");
  if (testButton) testButton.addEventListener("click", hw6HandleRunTests);

  const cancelButton = document.getElementById("hw6-cancel");
  if (cancelButton) {
    cancelButton.disabled = true;
    cancelButton.addEventListener("click", hw6RequestCancel);
  }

  hw6WireExports();
}

// ---------------------------------------------------------------------------
// Bootstrap
// ---------------------------------------------------------------------------

/**
 * Select the backend, load the data, split it chronologically, build the
 * provided indices, render the summaries and wire the controls.
 *
 * NOTHING MF-related is computed here. The provided MF reference is trained
 * lazily, on the first need only (switching the method to `mf`, pressing
 * "Recommend Top-10" with `mf` selected, or running the four-method comparison),
 * with progress reported in the status line and the browser painting between
 * epochs. The load path therefore must not claim the model was trained.
 *
 * Resolves without throwing; every failure is reported in the status line.
 *
 * @returns {Promise<void>}
 */
async function hw6Bootstrap() {
  const backend = await hw6InitTf();
  if (!backend) return;

  hw6SetStatus("Loading data\u2026");
  try {
    await loadData();
    const split = splitByTimestamp();
    HW6_SPLIT.train = split.train;
    HW6_SPLIT.val = split.val;
    HW6_SPLIT.test = split.test;
    HW6_MAX_MOVIE_ID = movies.reduce(function (max, movie) { return Math.max(max, movie.id); }, 0);
    hw6BuildSplitIndices(HW6_SPLIT);
    hw6PopulateUserSelects();
    hw6RenderDatasetSummary();
    hw6RenderGraphSummary();
    hw6WireControls();
    HW6_STATE.loaded = true;
    hw6SetStatus(
      "Dataset ready: " + HW6_STATE.trainUserIds.length.toLocaleString("en-US") + " training users, " +
      HW6_STATE.candidateIds.length.toLocaleString("en-US") + " candidate movies on " + backend +
      ". The MF reference is not trained yet (it trains on first use). Implement the TODO(hw6) functions, " +
      "then build the graph.",
    );
  } catch (error) {
    console.error("Initialization failed:", error);
    hw6SetStatus("Error: " + String(error && error.message ? error.message : error));
  }
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", function () {
    hw6Bootstrap();
  });
} else {
  hw6Bootstrap();
}

