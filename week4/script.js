/**
 * week4/script.js — Association-rule mining starter (HW4).
 *
 * This is a plain (classic) script, NOT an ES module. `week4/transactions.js`
 * (loaded first, in a regular <script> tag) assigns the dictionary-encoded UCI
 * Online Retail baskets to `window.HW4`. This file reads that global, decodes
 * the integer-index baskets back into `{ stock, description }` items, renders a
 * dataset summary, and wires two slider controls ("minimum support" and
 * "minimum confidence") plus a "Run rules" button. Because nothing is fetched,
 * the page works from a `file://` URL with no server.
 *
 * WHAT YOU MUST IMPLEMENT (`TODO(hw4)` — each stub throws until you write it):
 *   1. `dedupeBasket`         — unique stock codes in a basket, first-appearance order.
 *   2. `countItemset`         — baskets containing every requested stock (`0` if any is absent).
 *   3. `computeSupport`       — support = count(A union B) / N, guarded when N = 0.
 *   4. `computeConfidence`    — confidence = count(A union B) / count(A), guarded when count(A) = 0.
 *   5. `computeLift`          — lift = confidence / (count(B) / N), guarded.
 *   6. `findFrequentItemsets` — mine frequent itemsets (Apriori or equivalent).
 *   7. `generateRules`        — turn frequent itemsets into both-direction rules.
 *
 * PROVIDED for you (scaffolding): dataset loading/decoding from `window.HW4`, the
 * inverted-index builder (`buildIndex` / `asIndex` / `indexCache`), the thin
 * counting wrappers `countItem` / `countPair` (they call your `countItemset` and
 * therefore also throw until it is implemented), threshold validation and slider
 * readout, the DOM wiring, all formatting and rendering helpers, and the test
 * harness. Leave the provided code as-is and implement only the stubs above.
 *
 * Metrics (see week4/readme.md for definitions):
 *   support(A -> B)    = count(A union B) / N
 *   confidence(A -> B) = count(A union B) / count(A)
 *   lift(A -> B)       = confidence(A -> B) / (count(B) / N)
 *
 * @module week4/script
 */

// Defensive guard: `transactions.js` must run before this file. If `window.HW4`
// is missing (for example because `transactions.js` was not copied next to
// `index.html`), fail loudly and visibly instead of throwing an opaque error.
if (typeof window === "undefined" || typeof window.HW4 === "undefined") {
  if (typeof document !== "undefined") {
    document.body.innerHTML =
      '<p style="padding:2rem;font-family:sans-serif">' +
      "Failed to load `transactions.js`. Make sure it sits next to `index.html` " +
      "and is loaded before `script.js`.</p>";
  }
  throw new Error(
    "window.HW4 is not defined — load transactions.js before script.js.",
  );
}

/** The raw dataset global assigned by `week4/transactions.js`. */
const data = window.HW4;

/**
 * Baskets decoded from the dictionary-encoded arrays in `window.HW4`. Each
 * basket is an array of `{ stock, description }` items, matching the fixture
 * shape used by the tests.
 *
 * @type {Array<Array<Item>>}
 */
const TRANSACTIONS = data.baskets.map((basket) =>
  basket.map((stockIndex) => ({
    stock: data.stocks[stockIndex],
    description: data.descriptions[stockIndex],
  })),
);

/**
 * Dataset-wide inverted index, built once at startup. `null` until `init()` has
 * run, so the UI can tell "still loading" from "loaded".
 *
 * @type {BasketIndex|null}
 */
let DATASET_INDEX = null;

/** Number of baskets (the `N` used by support and lift). */
let N = data.N_BASKETS;

/**
 * The most recent completed pipeline run. Cached so that changing the ranking
 * re-sorts the rules already in hand instead of re-mining the dataset, and so
 * that every panel on the page is rendered from one consistent snapshot.
 *
 * @type {Object|null}
 */
let LAST_RESULT = null;

/**
 * @typedef {Object} Item
 * @property {string} stock       Stock code (the item identity).
 * @property {string} description Human-readable product description.
 */

/**
 * @typedef {Object} Rule
 * @property {string[]} antecedent    Item identities on the left-hand side (A).
 * @property {string[]} consequent    Item identities on the right-hand side (B).
 * @property {number} jointCount      count(A union B).
 * @property {number} antecedentCount count(A).
 * @property {number} consequentCount count(B).
 * @property {number} support         jointCount / N.
 * @property {number} confidence      jointCount / antecedentCount.
 * @property {number} lift            confidence / (consequentCount / N).
 */

/**
 * @typedef {Object} BasketIndex
 * @property {number} n                              Number of baskets.
 * @property {Map<string, Set<number>>} byStock      Stock code -> basket ids.
 */

// ---------------------------------------------------------------------------
// Provided helpers and student stubs
//
// Functions carrying a `TODO(hw4)` marker are stubs you must implement; every
// other function in this file is scaffolding and should be left as-is.
// ---------------------------------------------------------------------------

/**
 * Read the item identity out of a basket entry. Accepts either a plain string or
 * an object with a `stock` property so the helpers work with both the real
 * dataset and the small fixtures used by the tests.
 *
 * @param {Item|string} item
 * @returns {string}
 */
function stockOf(item) {
  return typeof item === "string" ? item : item.stock;
}

/**
 * Build an inverted index from basket id to the set of baskets containing each
 * stock code. Counting a candidate itemset then becomes a set intersection.
 *
 * @param {Array<Array<Item|string>>} baskets
 * @returns {BasketIndex}
 */
function buildIndex(baskets) {
  /** @type {Map<string, Set<number>>} */
  const byStock = new Map();
  baskets.forEach((basket, basketId) => {
    for (const item of basket) {
      const stock = stockOf(item);
      let posting = byStock.get(stock);
      if (!posting) {
        posting = new Set();
        byStock.set(stock, posting);
      }
      posting.add(basketId);
    }
  });
  return { n: baskets.length, byStock };
}

/** Cache of lazily built indexes, keyed by the baskets array. */
const indexCache = new WeakMap();

/**
 * Accept either a ready-made index or a raw basket array and return an index.
 *
 * @param {BasketIndex|Array<Array<Item|string>>} basketsOrIndex
 * @returns {BasketIndex}
 */
function asIndex(basketsOrIndex) {
  if (basketsOrIndex && !Array.isArray(basketsOrIndex) && basketsOrIndex.byStock) {
    return basketsOrIndex;
  }
  let index = indexCache.get(basketsOrIndex);
  if (!index) {
    index = buildIndex(basketsOrIndex);
    indexCache.set(basketsOrIndex, index);
  }
  return index;
}

/**
 * Count the baskets that contain every stock code in `stocks`.
 *
 * TODO(hw4): build (or reuse) a basket -> stock inverted index, intersect the
 * posting lists of the requested stocks, and return the size of the
 * intersection.
 *
 * Contract:
 *  - Accept either a ready-made index or a raw basket array. The provided
 *    `asIndex` helper returns an index for either input.
 *  - Return `0` when `stocks` is empty, and also when any requested stock is
 *    absent from the dataset (never throw for an unknown stock).
 *  - A stock repeated within one basket must be counted at most once. Dedupe the
 *    request before intersecting.
 *
 * @param {BasketIndex|Array<Array<Item|string>>} basketsOrIndex
 * @param {Array<string|Item>} stocks
 * @returns {number} count(A) for a single-element `stocks`, count(A union B) for two.
 */
function countItemset(basketsOrIndex, stocks) {
  const index = asIndex(basketsOrIndex);

  // An empty request can never be satisfied.
  if (!stocks || stocks.length === 0) return 0;

  // A basket is a set of items, so a stock listed twice in the request must not
  // narrow the intersection twice.
  const wanted = dedupeBasket(stocks);
  if (wanted.length === 0) return 0;

  // Gather every posting list up front. A stock that never occurs cannot be part
  // of any basket together with the others, so the answer is 0 -- never an error.
  const postings = [];
  for (const stock of wanted) {
    const posting = index.byStock.get(stock);
    if (!posting) return 0;
    postings.push(posting);
  }

  // Intersect smallest-first so the working set shrinks as quickly as possible.
  // Only fresh Sets are written to, so the cached index is never mutated.
  postings.sort((a, b) => a.size - b.size);
  let common = postings[0];
  for (let i = 1; i < postings.length && common.size > 0; i += 1) {
    const next = postings[i];
    const acc = new Set();
    for (const basketId of common) {
      if (next.has(basketId)) acc.add(basketId);
    }
    common = acc;
  }
  return common.size;
}

/**
 * Count the baskets containing a single stock code.
 *
 * @param {BasketIndex|Array<Array<Item|string>>} basketsOrIndex
 * @param {string|Item} stock
 * @returns {number}
 */
function countItem(basketsOrIndex, stock) {
  return countItemset(basketsOrIndex, [stock]);
}

/**
 * Count the baskets containing both `stockA` and `stockB`.
 *
 * @param {BasketIndex|Array<Array<Item|string>>} basketsOrIndex
 * @param {string|Item} stockA
 * @param {string|Item} stockB
 * @returns {number}
 */
function countPair(basketsOrIndex, stockA, stockB) {
  return countItemset(basketsOrIndex, [stockA, stockB]);
}

/**
 * Remove repeated item identities from a raw basket, keeping first-appearance
 * order. A basket is a set of items, so duplicates must not be counted twice.
 *
 * TODO(hw4): walk the input once, map each entry to its stock code with the
 * provided `stockOf` helper, and return each distinct code the first time it
 * appears.
 *
 * Contract:
 *  - An empty input returns `[]`.
 *  - The input may mix plain strings and `{ stock, description }` objects.
 *  - Only the stock code is returned; the description is dropped.
 *
 * @param {Array<string|Item>} rawItems
 * @returns {Array<string>} unique stock codes, in first-appearance order.
 */
function dedupeBasket(rawItems) {
  if (!rawItems || rawItems.length === 0) return [];

  const seen = new Set();
  const unique = [];
  for (const item of rawItems) {
    const stock = stockOf(item);
    if (!seen.has(stock)) {
      seen.add(stock);
      unique.push(stock);
    }
  }
  return unique;
}

/**
 * Compute support as `jointCount / n`.
 *
 * TODO(hw4): return the fraction and flag the `n === 0` case.
 *
 * Contract:
 *  - `defined: true` with `value = jointCount / n` when `n > 0`.
 *  - `defined: false` with `value = 0` when `n === 0` (never divide by zero).
 *
 * @param {number} jointCount count(A union B)
 * @param {number} n          number of baskets
 * @returns {{value: number, defined: boolean}} `defined` is false when `n === 0`.
 */
function computeSupport(jointCount, n) {
  // Guard every non-positive or non-finite denominator so no NaN/Infinity escapes.
  if (!Number.isFinite(n) || n <= 0) {
    return { value: 0, defined: false };
  }
  const value = jointCount / n;
  if (!Number.isFinite(value)) {
    return { value: 0, defined: false };
  }
  return { value, defined: true };
}

/**
 * Compute confidence as `jointCount / antecedentCount`.
 *
 * TODO(hw4): return the fraction and flag the `antecedentCount === 0` case.
 *
 * Contract:
 *  - `defined: true` with `value = jointCount / antecedentCount` when count(A) > 0.
 *  - `defined: false` with `value = 0` when count(A) === 0: a rule whose
 *    left-hand side never occurs has no confidence.
 *
 * @param {number} jointCount      count(A union B)
 * @param {number} antecedentCount count(A)
 * @returns {{value: number, defined: boolean}}
 */
function computeConfidence(jointCount, antecedentCount) {
  if (!Number.isFinite(antecedentCount) || antecedentCount <= 0) {
    return { value: 0, defined: false };
  }
  const value = jointCount / antecedentCount;
  if (!Number.isFinite(value)) {
    return { value: 0, defined: false };
  }
  return { value, defined: true };
}

/**
 * Compute lift as `confidence / (consequentCount / n)`.
 *
 * TODO(hw4): divide the incoming confidence by the consequent's baseline rate.
 *
 * Contract:
 *  - `defined: true` with `value = confidence.value / (consequentCount / n)`
 *    when every input is usable.
 *  - `defined: false` with `value = 0` when the incoming confidence is missing or
 *    undefined, when `n === 0`, or when the baseline `consequentCount / n` is 0
 *    (the consequent never occurs).
 *
 * @param {{value: number, defined: boolean}} confidence confidence(A -> B)
 * @param {number} consequentCount count(B)
 * @param {number} n               number of baskets
 * @returns {{value: number, defined: boolean}}
 */
function computeLift(confidence, consequentCount, n) {
  // `confidence` is the { value, defined } wrapper returned by
  // computeConfidence -- NOT a bare number. Require a well-formed, usable
  // wrapper so that passing `confidence.value` here degrades to "undefined"
  // instead of silently yielding NaN.
  if (
    !confidence ||
    typeof confidence !== "object" ||
    confidence.defined !== true ||
    !Number.isFinite(confidence.value)
  ) {
    return { value: 0, defined: false };
  }

  if (!Number.isFinite(n) || n <= 0) {
    return { value: 0, defined: false };
  }

  // The consequent's baseline occurrence rate; zero means the consequent never
  // occurs, which makes lift undefined rather than infinite.
  if (!Number.isFinite(consequentCount) || consequentCount <= 0) {
    return { value: 0, defined: false };
  }
  const baseline = consequentCount / n;
  if (!Number.isFinite(baseline) || baseline === 0) {
    return { value: 0, defined: false };
  }

  const value = confidence.value / baseline;
  if (!Number.isFinite(value)) {
    return { value: 0, defined: false };
  }
  return { value, defined: true };
}

/**
 * Validate the two slider values, expressed as fractions in `(0, 1]`.
 *
 * @param {number} minSupport    minimum support fraction
 * @param {number} minConfidence minimum confidence fraction
 * @returns {{ok: boolean, errors: string[]}}
 */
function validateThresholds(minSupport, minConfidence) {
  const errors = [];
  for (const [label, value] of [
    ["Minimum support", minSupport],
    ["Minimum confidence", minConfidence],
  ]) {
    if (typeof value !== "number" || Number.isNaN(value)) {
      errors.push(`${label} must be a number.`);
    } else if (value <= 0 || value > 1) {
      errors.push(`${label} must be greater than 0 and at most 1.`);
    }
  }
  return { ok: errors.length === 0, errors };
}

/**
 * Mine all frequent itemsets whose support is at least `minSupport`.
 *
 * TODO(hw4): implement Apriori (level-wise candidate generation with a
 * downward-closure pruning step) or any equivalent frequent-itemset miner.
 *
 * Contract:
 *  - Return one entry per frequent itemset: `{ items, count, support }`, where
 *    `items` is the (deduplicated) set of stock codes, `count` is the number of
 *    baskets containing every item, and `support = count / N`.
 *  - Every returned itemset must satisfy `support >= minSupport`.
 *  - You may call `countItemset` while developing, but a per-candidate scan over
 *    17,080 baskets is slow — build your own occurrence/index structures.
 *  - `generateRules` consumes this exact shape, so keep the field names stable.
 *
 * @param {Array<Array<Item|string>>} transactions baskets
 * @param {number} minSupport minimum support fraction in `(0, 1]`
 * @returns {Array<{items: string[], count: number, support: number}>} frequent itemsets
 */
function findFrequentItemsets(transactions, minSupport) {
  const index = asIndex(transactions);
  const n = index.n;

  // A non-positive or non-finite threshold is not a meaningful constraint, and an
  // empty dataset has nothing to mine. Bail out instead of returning noise.
  if (!Number.isFinite(minSupport) || minSupport <= 0) return [];
  if (n <= 0) return [];

  // Compare integer counts instead of floating-point ratios so that an itemset
  // sitting exactly on the threshold is never lost to rounding error.
  const minCount = Math.max(1, Math.ceil(minSupport * n - 1e-9));

  // Stock codes are the item identity. Convert each posting list to a sorted
  // array once, so every count below is a linear merge instead of a hash probe
  // per element. Basket ids are inserted in ascending order, so this is cheap.
  const postings = new Map();
  for (const [stock, basketIds] of index.byStock) {
    const list = Array.from(basketIds);
    list.sort((a, b) => a - b);
    postings.set(stock, list);
  }

  // Separator that cannot occur inside a stock code, making a joined string an
  // unambiguous itemset identity.
  const SEP = "\u0000";

  /**
   * Count the baskets present in every posting list, giving up as soon as the
   * running intersection can no longer reach `limit`. Lists are intersected
   * smallest-first so the working set shrinks as quickly as possible.
   *
   * The early exit only ever rejects: an intermediate intersection can still
   * shrink further, so it is safe to give up once it drops below `limit`, but
   * never safe to stop at `limit`. The value returned for an accepted candidate
   * is therefore the exact count.
   *
   * @param {Array<Array<number>>} lists ascending basket-id lists
   * @param {number} limit rejection threshold
   * @returns {number} exact count, or `0` when it cannot reach `limit`
   */
  const intersectCount = (lists, limit) => {
    let acc = lists[0];
    for (let i = 1; i < lists.length; i += 1) {
      const other = lists[i];
      const merged = [];
      let p = 0;
      let q = 0;
      while (p < acc.length && q < other.length) {
        const a = acc[p];
        const b = other[q];
        if (a === b) {
          merged.push(a);
          p += 1;
          q += 1;
        } else if (a < b) {
          p += 1;
        } else {
          q += 1;
        }
      }
      acc = merged;
      // Another list still has to be intersected, so this cannot recover.
      if (acc.length === 0) return 0;
      if (i < lists.length - 1 && acc.length < limit) return 0;
    }
    return acc.length;
  };

  /** @type {Array<{items: string[], count: number, support: number}>} */
  const frequent = [];

  // L1: a singleton's count is already the length of its posting list, so the
  // frequent items are read straight off the index with no counting at all.
  let level = [];
  for (const [stock, list] of postings) {
    if (list.length >= minCount) {
      level.push({ items: [stock], count: list.length, support: list.length / n });
    }
  }
  let frequentKeys = new Set();
  for (const entry of level) {
    frequentKeys.add(entry.items.join(SEP));
    frequent.push(entry);
  }

  for (let k = 2; level.length > 0; k += 1) {
    // Join step. Two (k-1)-itemsets can only extend the same (k-2)-prefix, so
    // grouping by that prefix replaces the all-pairs join with far fewer
    // candidate pairs -- the difference between millions of candidates and a
    // few thousand on the real dataset.
    const groups = new Map();
    for (const entry of level) {
      const prefix = entry.items.slice(0, -1).join(SEP);
      let group = groups.get(prefix);
      if (!group) {
        group = [];
        groups.set(prefix, group);
      }
      group.push(entry);
    }

    const next = [];
    for (const group of groups.values()) {
      // Order by trailing item so that i < j implies items[i].last < items[j].last,
      // which keeps every generated candidate in ascending item order.
      group.sort((a, b) => {
        const x = a.items[a.items.length - 1];
        const y = b.items[b.items.length - 1];
        return x < y ? -1 : x > y ? 1 : 0;
      });
      for (let i = 0; i < group.length; i += 1) {
        for (let j = i + 1; j < group.length; j += 1) {
          // Items are ascending within each parent and distinct across the group,
          // so this concatenation is already sorted and free of repeats.
          const items = group[i].items.concat(group[j].items[group[j].items.length - 1]);

          // Prune step (downward closure). The two parents that generated this
          // candidate are the itemset without its last item and the one without
          // its second-to-last item; both are frequent by construction. Every
          // other (k-1)-subset drops a leading item and must be tested.
          let closed = true;
          for (let drop = 0; drop < items.length - 2; drop += 1) {
            const subset = items.slice(0, drop).concat(items.slice(drop + 1));
            if (!frequentKeys.has(subset.join(SEP))) {
              closed = false;
              break;
            }
          }
          if (!closed) continue;

          const lists = items.map((stock) => postings.get(stock));
          lists.sort((a, b) => a.length - b.length);

          // intersectCount only ever returns an exact count or rejects outright,
          // so a single pass yields both the verdict and the reported count.
          const count = intersectCount(lists, minCount);
          if (count < minCount) continue;

          next.push({ items, count, support: count / n });
        }
      }
    }

    // Stop as soon as a level yields nothing: by downward closure no larger
    // itemset can be frequent either.
    if (next.length === 0) break;

    // The next level prunes against this level's keys, not the whole history.
    const nextKeys = new Set();
    // Appended in a loop: spreading a large level as call arguments overflows
    // the stack once a level reaches ~100k itemsets.
    for (const entry of next) {
      nextKeys.add(entry.items.join(SEP));
      frequent.push(entry);
    }
    frequentKeys = nextKeys;
    level = next;
  }

  // Stable canonical output: ascending by size, then lexicographic by item code.
  frequent.sort((a, b) => {
    if (a.items.length !== b.items.length) return a.items.length - b.items.length;
    const x = a.items.join(SEP);
    const y = b.items.join(SEP);
    return x < y ? -1 : x > y ? 1 : 0;
  });
  return frequent;
}

/**
 * Turn frequent itemsets into association rules and keep the ones whose
 * confidence is at least `minConfidence`.
 *
 * TODO(hw4): for each frequent itemset, split it into a non-empty antecedent `A`
 * and a non-empty, disjoint consequent `B` in BOTH directions, compute the
 * confidence for each direction, and keep the rules that pass the threshold.
 *
 * Contract:
 *  - Each returned rule follows the `Rule` shape documented at the top of this
 *    module. At minimum it carries `antecedent` and `consequent`; the renderer
 *    fills in the counts and metrics with `enrichRule`, but returning them
 *    yourself is fine and faster.
 *  - Generate both `A -> B` and `B -> A`: they are separate rules with (usually)
 *    different confidence. Skip a direction whose consequent is empty.
 *  - Keep only rules with `confidence >= minConfidence`. `count(A)` is non-zero
 *    for every generated rule, so the confidence is always defined.
 *
 * @param {Array<{items: string[], count: number, support: number}>} frequentItemsets
 * @param {number} minConfidence minimum confidence fraction in `(0, 1]`
 * @returns {Rule[]}
 */
function generateRules(frequentItemsets, minConfidence) {
  if (!Array.isArray(frequentItemsets) || frequentItemsets.length === 0) return [];

  // Identity is the StockCode, so a joined, ascending item list is an
  // unambiguous key. `findFrequentItemsets` already emits sorted items; sorting
  // again keeps the lookup correct for a hand-built or unsorted input.
  const SEP = "\u0000";
  const keyOf = (codes) => codes.slice().sort().join(SEP);

  /** @type {Map<string, {count: number, support: number}>} */
  const lookup = new Map();
  let maxCount = 0;
  for (const entry of frequentItemsets) {
    if (!entry || !Array.isArray(entry.items) || entry.items.length === 0) continue;
    if (!Number.isFinite(entry.count)) continue;
    lookup.set(keyOf(entry.items), {
      count: entry.count,
      support: entry.support,
    });
    if (entry.count > maxCount) maxCount = entry.count;
  }
  if (lookup.size === 0) return [];

  // The signature carries no basket count, and the module-level `N` describes the
  // real dataset rather than whatever itemsets were handed in, so recover N from
  // the data itself: `support === count / N`, so `N === count / support`.
  let n = 0;
  for (const entry of lookup.values()) {
    if (!Number.isFinite(entry.support) || entry.support <= 0) continue;
    const candidate = entry.count / entry.support;
    const rounded = Math.round(candidate);
    if (rounded > 0 && Math.abs(candidate - rounded) < 1e-6) {
      n = rounded;
      break;
    }
  }
  // Fall back to the largest observed count if no usable support was supplied.
  if (n <= 0) n = maxCount;
  if (n <= 0) return [];

  // A rule sitting exactly on the threshold must not be lost to floating point.
  const threshold = minConfidence - 1e-9;

  /** @type {Rule[]} */
  const rules = [];
  for (const entry of frequentItemsets) {
    if (!entry || !Array.isArray(entry.items) || entry.items.length < 2) continue;
    if (!Number.isFinite(entry.count)) continue;

    const items = entry.items.slice().sort();
    const size = items.length;
    const jointCount = entry.count;

    // Every element goes to A, to B, or to neither. Neither is excluded here
    // because A union B must equal the itemset; only the two degenerate splits
    // (A empty, B empty) are skipped. Enumerating by antecedent mask yields each
    // rule exactly once and covers both directions, since the complement of A is
    // enumerated in its own right.
    const span = 1 << size;
    for (let mask = 1; mask < span - 1; mask += 1) {
      const antecedent = [];
      const consequent = [];
      for (let i = 0; i < size; i += 1) {
        if (mask & (1 << i)) antecedent.push(items[i]);
        else consequent.push(items[i]);
      }

      // Downward closure means both halves are frequent, so both counts are in
      // the lookup. Skip the split anyway if either is missing rather than
      // emitting a rule built on a guessed count.
      const antecedentEntry = lookup.get(keyOf(antecedent));
      const consequentEntry = lookup.get(keyOf(consequent));
      if (!antecedentEntry || !consequentEntry) continue;

      const antecedentCount = antecedentEntry.count;
      const consequentCount = consequentEntry.count;
      if (antecedentCount <= 0) continue;

      // computeConfidence returns the { value, defined } wrapper, which is also
      // exactly what computeLift expects. Never unwrap it in between.
      const confidence = computeConfidence(jointCount, antecedentCount);
      if (!confidence.defined) continue;
      if (confidence.value < threshold) continue;

      const support = computeSupport(jointCount, n);
      const lift = computeLift(confidence, consequentCount, n);

      rules.push({
        antecedent,
        consequent,
        jointCount,
        antecedentCount,
        consequentCount,
        support: support.value,
        confidence: confidence.value,
        lift: lift.value,
      });
    }
  }
  return rules;
}

// ---------------------------------------------------------------------------
// Tiny worked example (used by the automated tests and the readout panel)
// ---------------------------------------------------------------------------

/**
 * A five-basket fixture with hand-computed support / confidence / lift values.
 *
 * The fixture is deliberately small enough to check with a pencil:
 *
 *   T1: bread, milk, jam, ham
 *   T2: bread, milk, jam
 *   T3: bread, milk
 *   T4: bread, jam, eggs
 *   T5: bread, eggs
 *
 * Item counts: bread = 5, milk = 3, jam = 3, eggs = 2, ham = 1 (N = 5).
 *
 * Because `bread` appears in every basket, every rule with `bread` on either
 * side has lift exactly 1 — that is the teaching point of the fixture.
 *
 * @returns {{n: number, baskets: string[][], itemCounts: Object<string, number>, rules: Array<Object>}}
 */
function tinyWorkedExample() {
  const baskets = [
    ["bread", "milk", "jam", "ham"],
    ["bread", "milk", "jam"],
    ["bread", "milk"],
    ["bread", "jam", "eggs"],
    ["bread", "eggs"],
  ];
  const n = baskets.length;
  return {
    n,
    baskets,
    itemCounts: { bread: 5, milk: 3, jam: 3, eggs: 2, ham: 1 },
    rules: [
      // lift == 1: bread is in every basket, so confidence equals support(B).
      {
        antecedent: ["bread"],
        consequent: ["milk"],
        jointCount: 3,
        antecedentCount: 5,
        consequentCount: 3,
        support: 3 / n,
        confidence: 3 / 5,
        lift: 1,
      },
      // lift > 1: milk and jam co-occur more than independence predicts.
      {
        antecedent: ["milk"],
        consequent: ["jam"],
        jointCount: 2,
        antecedentCount: 3,
        consequentCount: 3,
        support: 2 / n,
        confidence: 2 / 3,
        lift: (2 / 3) / (3 / 5),
      },
      // lift < 1 (and not degenerate): jam and eggs co-occur less than expected.
      {
        antecedent: ["jam"],
        consequent: ["eggs"],
        jointCount: 1,
        antecedentCount: 3,
        consequentCount: 2,
        support: 1 / n,
        confidence: 1 / 3,
        lift: (1 / 3) / (2 / 5),
      },
    ],
  };
}

// ---------------------------------------------------------------------------
// Rule normalisation
// ---------------------------------------------------------------------------

/**
 * Fill in any missing counts / metrics on a rule using the dataset index, so the
 * table can render a rule even when the student returns only `antecedent` and
 * `consequent`.
 *
 * @param {Partial<Rule>} rule
 * @param {BasketIndex} index
 * @returns {Rule}
 */
function enrichRule(rule, index) {
  const antecedent = (rule.antecedent || []).map(stockOf);
  const consequent = (rule.consequent || []).map(stockOf);
  const n = index.n || N;
  const jointCount =
    typeof rule.jointCount === "number"
      ? rule.jointCount
      : countItemset(index, [...antecedent, ...consequent]);
  const antecedentCount =
    typeof rule.antecedentCount === "number"
      ? rule.antecedentCount
      : countItemset(index, antecedent);
  const consequentCount =
    typeof rule.consequentCount === "number"
      ? rule.consequentCount
      : countItemset(index, consequent);
  const support = computeSupport(jointCount, n);
  const confidence = computeConfidence(jointCount, antecedentCount);
  const lift = computeLift(confidence, consequentCount, n);
  return {
    antecedent,
    consequent,
    jointCount,
    antecedentCount,
    consequentCount,
    support: typeof rule.support === "number" ? rule.support : support.value,
    confidence:
      typeof rule.confidence === "number" ? rule.confidence : confidence.value,
    lift: typeof rule.lift === "number" ? rule.lift : lift.value,
    supportDefined: support.defined,
    confidenceDefined: confidence.defined,
    liftDefined: lift.defined,
  };
}

/**
 * Swap the antecedent and consequent of a rule and recompute the metrics.
 *
 * Confidence is not symmetric, so `B -> A` usually has a different confidence
 * and support value from `A -> B` even though lift is unchanged.
 *
 * @param {Rule} rule
 * @param {BasketIndex} index
 * @returns {Rule}
 */
function reverseRule(rule, index) {
  return enrichRule(
    {
      antecedent: rule.consequent,
      consequent: rule.antecedent,
    },
    index,
  );
}

// ---------------------------------------------------------------------------
// Business analysis layer
//
// Everything in this section is presentation logic built *on top of* the rules.
// `generateRules` keeps its documented eight-field contract and is never touched
// from here: business metrics are derived by `enrichBusinessRule` on the way to
// the UI, and a caller that only wants raw rules can simply skip that call.
//
// One caveat deserves emphasis, because it is the easiest mistake to make with
// this kind of dashboard. `potential_attach_baskets` equals `A_without_B`: the
// number of baskets that already contain A and do *not* contain B. It measures
// how large the currently reachable audience is. It is NOT a forecast of
// incremental units. Association rules say nothing about causal response, and
// the only honest way to estimate an increment is the A/B test at the bottom of
// the page.
// ---------------------------------------------------------------------------

/**
 * Return `value` when it is a usable finite number, otherwise `null`.
 *
 * Every business metric routes through this so the UI can print `n/a` instead of
 * leaking a `NaN` or an `Infinity` into the page.
 *
 * @param {unknown} value
 * @returns {number|null}
 */
function finiteOrNull(value) {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

/**
 * Derive the commercial metrics of one rule `A -> B` from its counts.
 *
 * This is a pure function: it reads only the fields already present on `rule`
 * plus the basket count, and returns a new object. The original rule is not
 * mutated, so a caller can enrich for display and still keep the raw rule.
 *
 * The metrics, for rule A -> B with basket count `n`:
 *   countA                       = rule.antecedentCount
 *   countB                       = rule.consequentCount
 *   countAB                      = rule.jointCount
 *   A_without_B                  = countA - countAB
 *   attach_rate                  = rule.confidence
 *   headroom                     = 1 - attach_rate
 *   baseline_B                   = countB / n
 *   expected_AB_if_independent   = countA * countB / n
 *   incremental_cooccurrence     = countAB - expected_AB_if_independent
 *   potential_attach_baskets     = A_without_B   (reachable audience, NOT a forecast)
 *
 * A missing or non-finite input yields `null` for that metric rather than
 * `NaN`, so the renderers can degrade gracefully instead of printing garbage.
 *
 * @param {Rule} rule
 * @param {number} n number of baskets
 * @returns {Object|null} the rule with business metrics attached
 */
function enrichBusinessRule(rule, n) {
  if (!rule) return null;

  const countA = finiteOrNull(rule.antecedentCount);
  const countB = finiteOrNull(rule.consequentCount);
  const countAB = finiteOrNull(rule.jointCount);
  const attachRate = finiteOrNull(rule.confidence);
  const baskets = finiteOrNull(n);

  const aWithoutB =
    countA === null || countAB === null ? null : countA - countAB;
  const headroom = attachRate === null ? null : 1 - attachRate;
  const baselineB =
    countB === null || baskets === null || baskets <= 0 ? null : countB / baskets;
  const expectedAB =
    countA === null || countB === null || baskets === null || baskets <= 0
      ? null
      : (countA * countB) / baskets;
  const incremental =
    countAB === null || expectedAB === null ? null : countAB - expectedAB;

  return {
    ...rule,
    countA,
    countB,
    countAB,
    A_without_B: aWithoutB,
    attach_rate: attachRate,
    headroom,
    baseline_B: baselineB,
    expected_AB_if_independent: expectedAB,
    incremental_cooccurrence: incremental,
    potential_attach_baskets: aWithoutB,
  };
}

/**
 * The ranking views offered by the UI. Each one is a plain descending sort on a
 * single named metric — deliberately no composite score, so the ordering can
 * always be explained ("this is the second-largest reachable audience, and that
 * is all it claims").
 *
 * `value` is the accessor for the metric the ranking sorts on.
 *
 * @type {ReadonlyArray<{id: string, label: string, value: (r: Object) => number|null}>}
 */
const BUSINESS_RANKINGS = Object.freeze([
  {
    id: "a-without-b",
    label: "A without B — cross-sell reach",
    value: (rule) => rule.A_without_B,
  },
  {
    id: "lift",
    label: "Lift — strongest association",
    value: (rule) => rule.lift,
  },
  {
    id: "confidence",
    label: "Confidence — attach rate",
    value: (rule) => rule.confidence,
  },
  {
    id: "count-ab",
    label: "A+B baskets — co-occurrence volume",
    value: (rule) => rule.countAB,
  },
  {
    id: "incremental",
    label: "Incremental co-occurrence",
    value: (rule) => rule.incremental_cooccurrence,
  },
]);

/** Ranking the table starts on: commercial reach, not statistical strength. */
const DEFAULT_RANKING_ID = "a-without-b";

/**
 * A rule counts as "tiny remaining audience" for the misleading-rule panel at or
 * below this many reachable baskets.
 */
const MISLEADING_REACH_LIMIT = 20;

/**
 * Compare two business rules on one metric, descending.
 *
 * Rules with an undefined metric sort last. Ties fall back through lift, then
 * confidence, then the rule text, so that repeated runs of the same thresholds
 * always produce the same order — symmetric rules such as `a -> b` and `b -> a`
 * share a lift and would otherwise be ordered arbitrarily.
 *
 * @param {Object} a
 * @param {Object} b
 * @param {(rule: Object) => number|null} accessor
 * @returns {number}
 */
function compareBusinessRules(a, b, accessor) {
  const left = accessor(a);
  const right = accessor(b);

  const leftMissing = left === null || left === undefined;
  const rightMissing = right === null || right === undefined;
  if (leftMissing && rightMissing) return 0;
  if (leftMissing) return 1;
  if (rightMissing) return -1;
  if (right !== left) return right - left;

  const leftLift = finiteOrNull(a.lift);
  const rightLift = finiteOrNull(b.lift);
  if (leftLift !== null && rightLift !== null && rightLift !== leftLift) {
    return rightLift - leftLift;
  }

  const leftConfidence = finiteOrNull(a.confidence);
  const rightConfidence = finiteOrNull(b.confidence);
  if (
    leftConfidence !== null &&
    rightConfidence !== null &&
    rightConfidence !== leftConfidence
  ) {
    return rightConfidence - leftConfidence;
  }

  const leftKey = `${a.antecedent.join("|")}->${a.consequent.join("|")}`;
  const rightKey = `${b.antecedent.join("|")}->${b.consequent.join("|")}`;
  return leftKey.localeCompare(rightKey);
}

/**
 * Return a ranked copy of `rules` for one of the `BUSINESS_RANKINGS` views.
 *
 * @param {Object[]} rules enriched business rules
 * @param {string} [rankingId]
 * @returns {Object[]}
 */
function rankBusinessRules(rules, rankingId) {
  const ranking =
    BUSINESS_RANKINGS.find((entry) => entry.id === rankingId) ||
    BUSINESS_RANKINGS[0];
  return rules.slice().sort((a, b) => compareBusinessRules(a, b, ranking.value));
}

/**
 * Keep only the rules that beat the independent baseline (lift > 1), which is
 * the set worth putting in front of a business user.
 *
 * When the current thresholds happen to leave no rule above the baseline, the
 * full rule set is returned unchanged and `filtered` is set to `false`, so the UI
 * can say so rather than quietly showing an empty table.
 *
 * @param {Rule[]} rules
 * @returns {{rules: Rule[], filtered: boolean}}
 */
function selectBusinessRules(rules) {
  const aboveBaseline = rules.filter((rule) => {
    const lift = finiteOrNull(rule.lift);
    return lift !== null && lift > 1;
  });
  if (aboveBaseline.length > 0) return { rules: aboveBaseline, filtered: true };
  return { rules: rules.slice(), filtered: false };
}

/**
 * The rule with the largest reachable audience — the commercial recommendation.
 *
 * @param {Object[]} rules enriched business rules
 * @returns {Object|null}
 */
function pickRecommendation(rules) {
  return rankBusinessRules(rules, "a-without-b")[0] || null;
}

/**
 * The rule with the highest lift — the strongest statistical association.
 *
 * @param {Object[]} rules enriched business rules
 * @returns {Object|null}
 */
function pickStrongestAssociation(rules) {
  return rankBusinessRules(rules, "lift")[0] || null;
}

/**
 * A rule worth warning about: the highest lift among those with almost no
 * audience left to reach. Falls back to the strongest association when no rule
 * is that small, and reports which of the two happened.
 *
 * @param {Object[]} rules enriched business rules
 * @returns {{rule: Object|null, narrowed: boolean}}
 */
function pickMisleadingRule(rules) {
  const tinyReach = rules.filter((rule) => {
    const reach = finiteOrNull(rule.A_without_B);
    return reach !== null && reach <= MISLEADING_REACH_LIMIT;
  });
  if (tinyReach.length > 0) {
    return { rule: rankBusinessRules(tinyReach, "lift")[0], narrowed: true };
  }
  return { rule: pickStrongestAssociation(rules), narrowed: false };
}

// ---------------------------------------------------------------------------
// Formatting helpers
// ---------------------------------------------------------------------------

/**
 * Format a fraction as a percentage with two decimals.
 *
 * @param {number} fraction
 * @returns {string}
 */
function formatPercent(fraction) {
  return `${(fraction * 100).toFixed(2)}%`;
}

/**
 * Format a number with four significant decimals for the results table.
 *
 * @param {number} value
 * @returns {string}
 */
function formatMetric(value) {
  if (!Number.isFinite(value)) return "n/a";
  return value.toFixed(4);
}

/**
 * Format a basket count as a grouped integer, or `n/a` when the metric is
 * missing. A signed variant is used for `incremental_cooccurrence`, which is
 * negative when a pair co-occurs less often than independence predicts.
 *
 * @param {number|null} value
 * @param {boolean} [signed]
 * @returns {string}
 */
function formatCount(value, signed) {
  if (typeof value !== "number" || !Number.isFinite(value)) return "n/a";
  if (!signed) return Math.round(value).toLocaleString("en-US");
  const rounded = Math.round(value);
  return `${rounded > 0 ? "+" : ""}${rounded.toLocaleString("en-US")}`;
}

/**
 * Format a percentage for display, accepting `null` from a business metric that
 * could not be computed.
 *
 * @param {number|null} fraction
 * @returns {string}
 */
function formatRate(fraction) {
  if (typeof fraction !== "number" || !Number.isFinite(fraction)) return "n/a";
  return formatPercent(fraction);
}

/**
 * Render a list of stock codes as readable text.
 *
 * @param {string[]} stocks
 * @param {BasketIndex} index
 * @returns {string}
 */
function formatItemset(stocks, index) {
  if (!stocks || stocks.length === 0) return "(empty)";
  return stocks.map((stock) => describe(stock, index)).join(", ");
}

/**
 * Lookup a stock code's canonical description, falling back to the code. Filled
 * by `primeDescriptions()` once `init()` runs.
 */
const descriptionByStock = new Map();

/**
 * Render `STOCK — human readable description` for one item.
 *
 * @param {string} stock
 * @param {BasketIndex} index
 * @returns {string}
 */
function describe(stock, index) {
  const description = descriptionByStock.get(stock);
  return description ? `${stock} — ${description}` : stock;
}

// ---------------------------------------------------------------------------
// Rendering
// ---------------------------------------------------------------------------

/**
 * Render the dataset summary at the top of the page: total baskets, distinct
 * items, and the top five items by basket count.
 *
 * @param {BasketIndex} index
 * @param {HTMLElement|null} container
 * @returns {void}
 */
function renderDatasetSummary(index, container) {
  const target = container || document.getElementById("dataset-summary-body");
  if (!target) return;

  const counts = [...index.byStock.entries()]
    .map(([stock, posting]) => ({ stock, count: posting.size }))
    .sort((a, b) => b.count - a.count || a.stock.localeCompare(b.stock));
  const topFive = counts.slice(0, 5);

  const rows = topFive
    .map(
      (entry, i) =>
        `<tr><td class="num">${i + 1}</td><td>${escapeHtml(
          describe(entry.stock, index),
        )}</td><td class="num">${entry.count}</td></tr>`,
    )
    .join("");

  target.innerHTML = `
    <p class="summary-line">
      <strong>${index.n.toLocaleString("en-US")}</strong> baskets ·
      <strong>${index.byStock.size.toLocaleString("en-US")}</strong> distinct items
    </p>
    <details class="top-items">
      <summary>Top 5 items by basket count</summary>
      <table class="data-table">
        <thead><tr><th scope="col">#</th><th scope="col">Item</th><th scope="col">Baskets</th></tr></thead>
        <tbody>${rows}</tbody>
      </table>
    </details>
    <p class="provenance">${escapeHtml(data.dataset_provenance)}</p>
  `;
}

/**
 * Render the rule list into a table. Each row is clickable and shows the rule in
 * the detail panel.
 *
 * @param {Rule[]} rules
 * @param {BasketIndex} [index]
 * @param {HTMLElement|null} [container]
 * @returns {void}
 */
function renderResults(rules, index, container) {
  const target = container || document.getElementById("results");
  if (!target) return;
  const activeIndex = index || DATASET_INDEX;

  if (!rules || rules.length === 0) {
    target.innerHTML =
      '<p class="empty-state">No rules passed the current thresholds. ' +
      "Lower the minimum support or confidence and run again.</p>";
    return;
  }

  const n = (activeIndex && activeIndex.n) || N;

  // Each row is enriched on the way out: `enrichRule` fills in any missing count
  // or metric, then `enrichBusinessRule` attaches the commercial figures. Neither
  // call mutates the rule the miner produced.
  const body = rules
    .map((rule, rowIndex) => {
      const business = enrichBusinessRule(enrichRule(rule, activeIndex), n);
      return `
        <tr tabindex="0" data-rule-index="${rowIndex}">
          <td>${escapeHtml(formatItemset(business.antecedent, activeIndex))}</td>
          <td>${escapeHtml(formatItemset(business.consequent, activeIndex))}</td>
          <td class="num">${formatCount(business.countA)}</td>
          <td class="num">${formatCount(business.countAB)}</td>
          <td class="num">${formatCount(business.A_without_B)}</td>
          <td class="num">${formatRate(business.attach_rate)}</td>
          <td class="num">${formatRate(business.headroom)}</td>
          <td class="num">${formatRate(business.support)}</td>
          <td class="num">${formatMetric(business.lift)}</td>
        </tr>`;
    })
    .join("");

  // The business columns make this table wide, so it scrolls horizontally rather
  // than shrinking the type or dropping a column.
  target.innerHTML = `
    <p class="results-count">${rules.length} rule${rules.length === 1 ? "" : "s"}.</p>
    <div class="scroll-x" tabindex="0" role="group" aria-label="Rules table, scroll horizontally">
      <table class="data-table rules-table">
        <thead>
          <tr>
            <th scope="col">Antecedent (A)</th>
            <th scope="col">Consequent (B)</th>
            <th scope="col">A buyers</th>
            <th scope="col">A+B baskets</th>
            <th scope="col">A without B</th>
            <th scope="col">Attach rate (confidence)</th>
            <th scope="col">Headroom</th>
            <th scope="col">support</th>
            <th scope="col">lift</th>
          </tr>
        </thead>
        <tbody>${body}</tbody>
      </table>
    </div>`;

  target.querySelectorAll("tr[data-rule-index]").forEach((row) => {
    const activate = () => {
      const rule = rules[Number(row.dataset.ruleIndex)];
      renderRuleDetail(rule, activeIndex);
    };
    row.addEventListener("click", activate);
    row.addEventListener("keydown", (event) => {
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        activate();
      }
    });
  });
}

/**
 * Render the detail panel for one selected rule, including a "Reverse direction"
 * button that swaps A and B and re-renders the panel.
 *
 * Shows an inline note when `count(A)` or `count(B)` is zero, because confidence
 * and lift are then undefined.
 *
 * @param {Rule} rule
 * @param {BasketIndex} [index]
 * @param {HTMLElement|null} [container]
 * @returns {void}
 */
function renderRuleDetail(rule, index, container) {
  const target = container || document.getElementById("rule-detail");
  if (!target) return;
  const activeIndex = index || DATASET_INDEX;
  const enriched = enrichRule(rule, activeIndex);
  const reversed = reverseRule(enriched, activeIndex);
  const business = enrichBusinessRule(
    enriched,
    (activeIndex && activeIndex.n) || N,
  );

  const zeroDenominatorNotes = [];
  if (enriched.antecedentCount === 0) {
    zeroDenominatorNotes.push(
      "count(A) = 0, so confidence(A → B) is undefined (division by zero).",
    );
  }
  if (enriched.consequentCount === 0) {
    zeroDenominatorNotes.push(
      "count(B) = 0, so lift(A → B) is undefined (division by zero).",
    );
  }
  const noteHtml = zeroDenominatorNotes.length
    ? `<p class="warning">${zeroDenominatorNotes.map(escapeHtml).join(" ")}</p>`
    : "";

  target.innerHTML = `
    <dl class="rule-metrics">
      <dt>Antecedent (A)</dt><dd>${escapeHtml(formatItemset(enriched.antecedent, activeIndex))}</dd>
      <dt>Consequent (B)</dt><dd>${escapeHtml(formatItemset(enriched.consequent, activeIndex))}</dd>
      <dt>count(A∪B)</dt><dd class="num">${formatCount(enriched.jointCount)}</dd>
      <dt>count(A)</dt><dd class="num">${formatCount(enriched.antecedentCount)}</dd>
      <dt>count(B)</dt><dd class="num">${formatCount(enriched.consequentCount)}</dd>
      <dt>support</dt><dd class="num">${formatRate(enriched.support)}</dd>
      <dt>confidence</dt><dd class="num">${formatRate(enriched.confidence)}</dd>
      <dt>lift</dt><dd class="num">${formatMetric(enriched.lift)}</dd>
    </dl>
    <h3>Business reading</h3>
    <dl class="rule-metrics">
      <dt>A buyers</dt><dd class="num">${formatCount(business.countA)}</dd>
      <dt>A+B baskets</dt><dd class="num">${formatCount(business.countAB)}</dd>
      <dt>A without B</dt><dd class="num">${formatCount(business.A_without_B)}</dd>
      <dt>Attach rate</dt><dd class="num">${formatRate(business.attach_rate)}</dd>
      <dt>Headroom</dt><dd class="num">${formatRate(business.headroom)}</dd>
      <dt>Baseline for B</dt><dd class="num">${formatRate(business.baseline_B)}</dd>
      <dt>Expected A+B if independent</dt><dd class="num">${formatCount(business.expected_AB_if_independent)}</dd>
      <dt>Incremental co-occurrence</dt><dd class="num">${formatCount(business.incremental_cooccurrence, true)}</dd>
    </dl>
    <p class="hint">
      “A without B” counts the baskets that already contain A but not B. It is the
      currently reachable audience, not a forecast of incremental units.
    </p>
    <p class="comparison">
      Reverse direction (B → A): confidence
      <strong>${formatPercent(reversed.confidence)}</strong>, lift
      <strong>${formatMetric(reversed.lift)}</strong>.
      Confidence changes with direction; lift does not.
    </p>
    ${noteHtml}
    <button type="button" id="reverse-rule">Reverse direction (B → A)</button>
  `;

  const button = target.querySelector("#reverse-rule");
  button.addEventListener("click", () => {
    renderRuleDetail(reversed, activeIndex, target);
  });
}

/**
 * Render the optional worked-example readout panel.
 *
 * @param {ReturnType<typeof tinyWorkedExample>} example
 * @param {HTMLElement|null} [container]
 * @returns {void}
 */
function renderWorkedExample(example, container) {
  const target = container || document.getElementById("worked-example");
  if (!target) return;
  const rows = example.rules
    .map(
      (rule) => `
      <tr>
        <td>${rule.antecedent.join(", ")}</td>
        <td>${rule.consequent.join(", ")}</td>
        <td class="num">${rule.jointCount}</td>
        <td class="num">${rule.antecedentCount}</td>
        <td class="num">${rule.consequentCount}</td>
        <td class="num">${formatPercent(rule.support)}</td>
        <td class="num">${formatPercent(rule.confidence)}</td>
        <td class="num">${formatMetric(rule.lift)}</td>
      </tr>`,
    )
    .join("");
  target.innerHTML = `
    <p class="worked-example-intro">
      Five hand-built baskets, N = ${example.n}. The top row has lift exactly 1
      because <code>bread</code> appears in every basket; the second has lift &gt; 1;
      the third has lift &lt; 1.
    </p>
    <div class="scroll-x" tabindex="0" role="group" aria-label="Worked example table, scroll horizontally">
    <table class="data-table">
      <thead>
        <tr>
          <th scope="col">A</th><th scope="col">B</th>
          <th scope="col">count(A∪B)</th><th scope="col">count(A)</th><th scope="col">count(B)</th>
          <th scope="col">support</th><th scope="col">confidence</th><th scope="col">lift</th>
        </tr>
      </thead>
      <tbody>${rows}</tbody>
    </table>
    </div>`;
}

// ---------------------------------------------------------------------------
// Business rendering
//
// Each panel below answers a different question, and the point of showing them
// separately is that the answers disagree: the strongest association is rarely
// the biggest commercial prize. Nothing here decides anything on its own — the
// picks come from the transparent single-metric rankings above.
// ---------------------------------------------------------------------------

/**
 * Render one side of a rule (`A` or `B`) as a list of stock code + description.
 *
 * @param {string[]} stocks
 * @param {BasketIndex} index
 * @returns {string}
 */
function renderRuleSide(stocks, index) {
  if (!stocks || stocks.length === 0) return '<p class="empty-state">(empty)</p>';
  return `<ul class="item-list">${stocks
    .map(
      (stock) =>
        `<li><code>${escapeHtml(stock)}</code> <span class="muted">${escapeHtml(
          descriptionByStock.get(stock) || "",
        )}</span></li>`,
    )
    .join("")}</ul>`;
}

/**
 * Render the "Current settings" panel: the thresholds in force and how many
 * itemsets, rules and business rules came out of them.
 *
 * @param {Object} summary
 * @param {HTMLElement|null} [container]
 * @returns {void}
 */
function renderThresholdSummary(summary, container) {
  const target = container || document.getElementById("threshold-summary");
  if (!target) return;

  const baselineNote = summary.businessFiltered
    ? ""
    : ' — no rule beats the baseline here, so all generated rules are shown';

  target.innerHTML = `
    <dl class="summary-metrics">
      <dt>Minimum support</dt><dd class="num">${formatRate(summary.minSupport)}</dd>
      <dt>Minimum confidence</dt><dd class="num">${formatRate(summary.minConfidence)}</dd>
      <dt>Frequent itemsets</dt><dd class="num">${formatCount(summary.itemsetCount)}</dd>
      <dt>Generated rules</dt><dd class="num">${formatCount(summary.ruleCount)}</dd>
      <dt>Business rules (lift &gt; 1)</dt><dd class="num">${formatCount(summary.businessRuleCount)}${escapeHtml(baselineNote)}</dd>
      <dt>Table sorted by</dt><dd>${escapeHtml(summary.rankingLabel)}</dd>
    </dl>`;
}

/**
 * Render the "Recommended Cross-Sell Opportunity" card for the rule with the
 * largest reachable audience, together with a suggested test and the caveat that
 * makes the number honest.
 *
 * @param {Object|null} rule enriched business rule
 * @param {number} n
 * @param {BasketIndex} index
 * @param {HTMLElement|null} [container]
 * @returns {void}
 */
function renderRecommendation(rule, n, index, container) {
  const target = container || document.getElementById("recommendation");
  if (!target) return;

  if (!rule) {
    target.innerHTML =
      '<p class="empty-state">No rules passed the current thresholds, so there is no opportunity to recommend. Lower the minimum support or confidence and run again.</p>';
    return;
  }

  const reach = rule.A_without_B;
  const reachShare =
    typeof reach === "number" && typeof n === "number" && n > 0
      ? formatPercent(reach / n)
      : "an unknown share";
  const liftText = formatMetric(rule.lift);
  const baselineText = formatRate(rule.baseline_B);

  target.innerHTML = `
    <p class="lead">
      Ranked by <strong>A without B</strong> — the number of baskets already
      containing A but not B, i.e. the currently reachable audience.
    </p>
    <div class="two-col">
      <div>
        <h3>Antecedent (A)</h3>
        ${renderRuleSide(rule.antecedent, index)}
      </div>
      <div>
        <h3>Consequent (B)</h3>
        ${renderRuleSide(rule.consequent, index)}
      </div>
    </div>
    <dl class="summary-metrics">
      <dt>A buyers</dt><dd class="num">${formatCount(rule.countA)}</dd>
      <dt>A+B baskets</dt><dd class="num">${formatCount(rule.countAB)}</dd>
      <dt>A without B</dt><dd class="num">${formatCount(rule.A_without_B)}</dd>
      <dt>Attach rate</dt><dd class="num">${formatRate(rule.attach_rate)}</dd>
      <dt>Headroom</dt><dd class="num">${formatRate(rule.headroom)}</dd>
      <dt>Lift</dt><dd class="num">${liftText}</dd>
      <dt>Baseline for B</dt><dd class="num">${baselineText}</dd>
      <dt>Incremental co-occurrence</dt><dd class="num">${formatCount(rule.incremental_cooccurrence, true)}</dd>
    </dl>
    <p class="interpretation">
      Customers who buy A are <strong>${liftText}</strong> times more likely than
      baseline (<strong>${baselineText}</strong> of all baskets) to also buy B, but
      <strong>${formatCount(reach)}</strong> A-baskets currently do not contain B —
      ${escapeHtml(reachShare)} of all baskets. That makes B a candidate for a
      cross-sell test.
    </p>
    <h3>Suggested test</h3>
    <ul class="checklist">
      <li><strong>Recommendation slot:</strong> show the consequent product when the antecedent is added to the basket.</li>
      <li><strong>Bundle:</strong> test a bundle price for the A+B pair.</li>
      <li><strong>Second-item discount:</strong> test a small discount on B when A is already in the basket.</li>
    </ul>
    <p class="warning">
      “A without B” is the audience that exists today — it is not a forecast. It
      says nothing about how many of those baskets would change behaviour, so it
      must not be read as expected incremental units. Use the A/B test below to
      estimate the increment.
    </p>`;
}

/**
 * Render the "strongest association vs largest opportunity" panel, including the
 * educational warning about a high lift on a tiny audience.
 *
 * @param {Object|null} strongest
 * @param {Object|null} recommendation
 * @param {{rule: Object|null, narrowed: boolean}} misleading
 * @param {BasketIndex} index
 * @param {HTMLElement|null} [container]
 * @returns {void}
 */
function renderAssociationWarning(strongest, recommendation, misleading, index, container) {
  const target = container || document.getElementById("association-warning");
  if (!target) return;

  if (!strongest) {
    target.innerHTML =
      '<p class="empty-state">No rules passed the current thresholds, so there is nothing to compare.</p>';
    return;
  }

  const parts = [];

  parts.push(`
    <h3>Strongest association</h3>
    <p class="item-pair">
      <span>${escapeHtml(formatItemset(strongest.antecedent, index))}</span>
      <span class="arrow">→</span>
      <span>${escapeHtml(formatItemset(strongest.consequent, index))}</span>
    </p>
    <dl class="summary-metrics">
      <dt>Lift</dt><dd class="num">${formatMetric(strongest.lift)}</dd>
      <dt>Attach rate</dt><dd class="num">${formatRate(strongest.attach_rate)}</dd>
      <dt>A buyers</dt><dd class="num">${formatCount(strongest.countA)}</dd>
      <dt>A+B baskets</dt><dd class="num">${formatCount(strongest.countAB)}</dd>
      <dt>A without B</dt><dd class="num">${formatCount(strongest.A_without_B)}</dd>
    </dl>
    <p class="warning">
      Strong association, but limited remaining cross-sell headroom: only
      <strong>${formatCount(strongest.A_without_B)}</strong> baskets still contain A
      without B, so the reachable audience is
      <strong>${typeof strongest.A_without_B === "number" && N > 0 ? formatPercent(strongest.A_without_B / N) : "n/a"}</strong>
      of all baskets.
    </p>`);

  if (recommendation) {
    const sameRule =
      recommendation.antecedent.join("|") === strongest.antecedent.join("|") &&
      recommendation.consequent.join("|") === strongest.consequent.join("|");
    parts.push(`
      <h3>Largest opportunity</h3>
      <p class="item-pair">
        <span>${escapeHtml(formatItemset(recommendation.antecedent, index))}</span>
        <span class="arrow">→</span>
        <span>${escapeHtml(formatItemset(recommendation.consequent, index))}</span>
      </p>
      <dl class="summary-metrics">
        <dt>Lift</dt><dd class="num">${formatMetric(recommendation.lift)}</dd>
        <dt>Attach rate</dt><dd class="num">${formatRate(recommendation.attach_rate)}</dd>
        <dt>A buyers</dt><dd class="num">${formatCount(recommendation.countA)}</dd>
        <dt>A+B baskets</dt><dd class="num">${formatCount(recommendation.countAB)}</dd>
        <dt>A without B</dt><dd class="num">${formatCount(recommendation.A_without_B)}</dd>
      </dl>
      <p class="comparison">${
        sameRule
          ? "At these thresholds the strongest association and the largest opportunity happen to be the same rule."
          : `This is <strong>not</strong> the strongest association. The top lift is
             <strong>${formatMetric(strongest.lift)}</strong> on
             <strong>${formatCount(strongest.A_without_B)}</strong> reachable baskets,
             while the larger prize here is
             <strong>${formatCount(recommendation.A_without_B)}</strong> reachable
             baskets at lift <strong>${formatMetric(recommendation.lift)}</strong>.
             Ranking by strength and ranking by commercial reach answer different questions.`
      }</p>`);
  }

  if (misleading.rule) {
    const rule = misleading.rule;
    parts.push(`
      <h3>Why lift alone can mislead</h3>
      <p class="item-pair">
        <span>${escapeHtml(formatItemset(rule.antecedent, index))}</span>
        <span class="arrow">→</span>
        <span>${escapeHtml(formatItemset(rule.consequent, index))}</span>
      </p>
      <dl class="summary-metrics">
        <dt>Lift</dt><dd class="num">${formatMetric(rule.lift)}</dd>
        <dt>Attach rate</dt><dd class="num">${formatRate(rule.attach_rate)}</dd>
        <dt>A buyers</dt><dd class="num">${formatCount(rule.countA)}</dd>
        <dt>A+B baskets</dt><dd class="num">${formatCount(rule.countAB)}</dd>
        <dt>A without B</dt><dd class="num">${formatCount(rule.A_without_B)}</dd>
      </dl>
      <p class="interpretation">
        Very strong association, but little remaining audience is available for
        additional cross-sell: A+B already covers
        ${formatRate(rule.attach_rate)} of A buyers, leaving only
        <strong>${formatCount(rule.A_without_B)}</strong> basket(s) where B could
        still be attached.
        ${
          misleading.narrowed
            ? `This is the highest lift among the ${MISLEADING_REACH_LIMIT} rules with a reachable audience of ${MISLEADING_REACH_LIMIT} baskets or fewer.`
            : `No rule at these thresholds has a reachable audience of ${MISLEADING_REACH_LIMIT} baskets or fewer, so this is simply the highest-lift rule.`
        }
      </p>`);
  }

  target.innerHTML = parts.join("");
}

/**
 * Render the top three commercial opportunities, ranked by reachable audience.
 *
 * @param {Object[]} rules enriched business rules
 * @param {BasketIndex} index
 * @param {HTMLElement|null} [container]
 * @returns {void}
 */
function renderTopBusiness(rules, index, container) {
  const target = container || document.getElementById("top-business");
  if (!target) return;

  const top = rankBusinessRules(rules, "a-without-b").slice(0, 3);
  if (top.length === 0) {
    target.innerHTML =
      '<p class="empty-state">No rules passed the current thresholds.</p>';
    return;
  }

  const rows = top
    .map(
      (rule, position) => `
      <tr>
        <td class="num">${position + 1}</td>
        <td>${escapeHtml(formatItemset(rule.antecedent, index))}</td>
        <td>${escapeHtml(formatItemset(rule.consequent, index))}</td>
        <td class="num">${formatMetric(rule.lift)}</td>
        <td class="num">${formatRate(rule.attach_rate)}</td>
        <td class="num">${formatCount(rule.A_without_B)}</td>
      </tr>`,
    )
    .join("");

  target.innerHTML = `
    <p class="lead">The three rules with the most baskets already containing A but not B.</p>
    <div class="scroll-x" tabindex="0" role="group" aria-label="Top three opportunities, scroll horizontally">
    <table class="data-table">
      <thead>
        <tr>
          <th scope="col">#</th>
          <th scope="col">Antecedent (A)</th>
          <th scope="col">Consequent (B)</th>
          <th scope="col">lift</th>
          <th scope="col">Attach rate</th>
          <th scope="col">A without B</th>
        </tr>
      </thead>
      <tbody>${rows}</tbody>
    </table>
    </div>`;
}

/**
 * Escape text before inserting it into HTML.
 *
 * @param {string} text
 * @returns {string}
 */
function escapeHtml(text) {
  return String(text)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

// ---------------------------------------------------------------------------
// Test harness
// ---------------------------------------------------------------------------

/** Marker thrown by unimplemented `TODO(hw4)` stubs. */
const TODO_MARKER = "TODO(hw4)";

/**
 * Run the automated self-checks against the five-basket worked example and the
 * reference helpers. Unimplemented `TODO(hw4)` functions are reported as
 * `pending` rather than `fail`, so the harness is useful before and after the
 * assignment is implemented.
 *
 * @param {HTMLElement|null} [logElement] element that receives the text log
 * @returns {{passed: number, failed: number, pending: number, checks: Array<Object>}}
 */
function runTests(logElement) {
  const example = tinyWorkedExample();
  const checks = [];
  const basketObjects = example.baskets;

  const pass = (name, detail) => checks.push({ name, status: "PASS", detail });
  const fail = (name, detail) => checks.push({ name, status: "FAIL", detail });
  const pending = (name, detail) => checks.push({ name, status: "PENDING", detail });

  const close = (a, b, eps = 1e-9) => Math.abs(a - b) < eps;

  const check = (name, fn) => {
    try {
      const outcome = fn();
      if (outcome && outcome.pending) pending(name, outcome.pending);
      else pass(name, outcome === undefined ? "" : String(outcome));
    } catch (error) {
      if (String(error && error.message).includes(TODO_MARKER)) {
        pending(name, "TODO(hw4): not implemented yet.");
      } else {
        fail(name, String(error && error.message ? error.message : error));
      }
    }
  };

  // 1. Item counts on the fixture.
  check("fixture: item counts match the hand-computed values", () => {
    for (const [stock, expected] of Object.entries(example.itemCounts)) {
      const actual = countItem(basketObjects, stock);
      if (actual !== expected) {
        throw new Error(`countItem(${stock}) = ${actual}, expected ${expected}`);
      }
    }
    return "all five item counts correct";
  });

  // 2. lift == 1 case.
  check("fixture: bread -> milk has lift exactly 1", () => {
    const [breadMilk] = example.rules;
    const joint = countPair(basketObjects, "bread", "milk");
    const support = computeSupport(joint, example.n);
    const confidence = computeConfidence(joint, countItem(basketObjects, "bread"));
    const lift = computeLift(confidence, countItem(basketObjects, "milk"), example.n);
    if (!close(support.value, breadMilk.support)) throw new Error("support mismatch");
    if (!close(confidence.value, breadMilk.confidence)) throw new Error("confidence mismatch");
    if (!close(lift.value, breadMilk.lift)) throw new Error(`lift = ${lift.value}, expected 1`);
    return `support=${support.value.toFixed(4)} confidence=${confidence.value.toFixed(4)} lift=${lift.value.toFixed(4)}`;
  });

  // 3. lift > 1 case.
  check("fixture: milk -> jam has lift greater than 1", () => {
    const rule = example.rules[1];
    const joint = countPair(basketObjects, rule.antecedent[0], rule.consequent[0]);
    const confidence = computeConfidence(joint, countItem(basketObjects, rule.antecedent[0]));
    const lift = computeLift(confidence, countItem(basketObjects, rule.consequent[0]), example.n);
    if (!close(lift.value, rule.lift)) throw new Error(`lift = ${lift.value}, expected ${rule.lift}`);
    if (!(lift.value > 1)) throw new Error("expected lift > 1");
    return `lift=${lift.value.toFixed(4)} (> 1)`;
  });

  // 4. lift < 1 case.
  check("fixture: jam -> eggs has lift less than 1", () => {
    const rule = example.rules[2];
    const joint = countPair(basketObjects, rule.antecedent[0], rule.consequent[0]);
    const confidence = computeConfidence(joint, countItem(basketObjects, rule.antecedent[0]));
    const lift = computeLift(confidence, countItem(basketObjects, rule.consequent[0]), example.n);
    if (!close(lift.value, rule.lift)) throw new Error(`lift = ${lift.value}, expected ${rule.lift}`);
    if (!(lift.value < 1 && lift.value > 0)) throw new Error("expected 0 < lift < 1");
    return `lift=${lift.value.toFixed(4)} (0 < lift < 1)`;
  });

  // 5. Reversed confidence differs.
  check("fixture: reversed confidence differs (jam <-> eggs)", () => {
    const forwardJoint = countPair(basketObjects, "jam", "eggs");
    const forward = computeConfidence(forwardJoint, countItem(basketObjects, "jam"));
    const reversed = computeConfidence(forwardJoint, countItem(basketObjects, "eggs"));
    if (close(forward.value, reversed.value)) {
      throw new Error("confidence should differ between jam -> eggs and eggs -> jam");
    }
    if (!close(forward.value, 1 / 3) || !close(reversed.value, 1 / 2)) {
      throw new Error(`unexpected values: ${forward.value} vs ${reversed.value}`);
    }
    return `jam->eggs=${forward.value.toFixed(4)} vs eggs->jam=${reversed.value.toFixed(4)}`;
  });

  // 6. Duplicate items in a raw basket are counted once.
  check("duplicate items in a raw basket are counted once", () => {
    const deduped = dedupeBasket(["milk", "bread", "milk", "bread", "jam"]);
    if (deduped.length !== 3) throw new Error(`expected 3 unique items, got ${deduped.length}`);
    const duplicateBasket = [["milk", "bread", "milk", "bread"], ["milk"]];
    if (countItem(duplicateBasket, "milk") !== 2) {
      throw new Error("countItem must count each basket once, not each row");
    }
    if (countPair(duplicateBasket, "milk", "bread") !== 1) {
      throw new Error("countPair must count each basket once");
    }
    return "dedupeBasket removed repeats; counting is per basket";
  });

  // 7. Empty result set renders an empty-state note.
  check("empty rule set renders an empty-state note", () => {
    const scratch = document.createElement("div");
    renderResults([], DATASET_INDEX, scratch);
    if (!/no rules passed/i.test(scratch.textContent)) {
      throw new Error("expected an empty-state message");
    }
    return "empty state rendered";
  });

  // 8. Invalid thresholds are rejected.
  check("invalid thresholds are rejected", () => {
    if (validateThresholds(0.01, 0.3).ok !== true) throw new Error("valid thresholds rejected");
    if (validateThresholds(Number.NaN, 0.3).ok !== false) throw new Error("NaN support accepted");
    if (validateThresholds(0.01, 1.5).ok !== false) throw new Error("confidence > 1 accepted");
    if (validateThresholds(0, 0.3).ok !== false) throw new Error("zero support accepted");
    return "valid accepted, invalid rejected";
  });

  // 9. Zero-denominator guards.
  check("zero-denominator guards return undefined metrics", () => {
    if (computeConfidence(0, 0).defined !== false) throw new Error("count(A)=0 must be undefined");
    if (computeLift({ value: 0.5, defined: true }, 0, 5).defined !== false) {
      throw new Error("count(B)=0 must be undefined");
    }
    if (computeSupport(0, 0).defined !== false) throw new Error("N=0 must be undefined");
    return "zero denominators return defined=false";
  });

  // 10. Student function: frequent itemsets (pending until implemented).
  check("findFrequentItemsets reproduces the fixture's frequent itemsets", () => {
    const itemsets = findFrequentItemsets(basketObjects, 0.4);
    const pairs = itemsets.filter((set) => set.items.length === 2);
    if (pairs.length === 0) throw new Error("no frequent pairs found at support >= 0.4");
    return `${itemsets.length} itemsets`;
  });

  // 11. Student function: rules (pending until implemented).
  check("generateRules reproduces the fixture's rules", () => {
    const itemsets = findFrequentItemsets(basketObjects, 0.2);
    const rules = generateRules(itemsets, 0.5);
    if (rules.length === 0) throw new Error("no rules found at support >= 0.2, confidence >= 0.5");
    return `${rules.length} rules`;
  });

  const passed = checks.filter((c) => c.status === "PASS").length;
  const failed = checks.filter((c) => c.status === "FAIL").length;
  const pendingCount = checks.filter((c) => c.status === "PENDING").length;

  const lines = [
    `HW4 self-checks — pass ${passed}, fail ${failed}, pending ${pendingCount} (of ${checks.length})`,
    "",
    ...checks.map((c) => `${c.status.padEnd(7)} ${c.name}${c.detail ? ` — ${c.detail}` : ""}`),
  ];
  const text = lines.join("\n");

  const target = logElement || document.getElementById("testLog");
  if (target) target.textContent = text;

  return { passed, failed, pending: pendingCount, checks };
}

// ---------------------------------------------------------------------------
// Wiring
// ---------------------------------------------------------------------------

/**
 * Populate the stock -> description lookup used by the renderers from the
 * dictionary-encoded tables embedded in `window.HW4`.
 *
 * @returns {void}
 */
function primeDescriptions() {
  data.stocks.forEach((stock, i) => {
    if (!descriptionByStock.has(stock)) {
      descriptionByStock.set(stock, data.descriptions[i]);
    }
  });
}

/**
 * Read the two sliders and return the thresholds as fractions in `(0, 1]`.
 *
 * @returns {{minSupport: number, minConfidence: number}}
 */
function readThresholds() {
  const supportInput = document.getElementById("min-support");
  const confidenceInput = document.getElementById("min-confidence");
  return {
    minSupport: Number(supportInput.value) / 100,
    minConfidence: Number(confidenceInput.value) / 100,
  };
}

/** Update the `<output>` readouts next to the two sliders. */
function syncThresholdLabels() {
  const supportInput = document.getElementById("min-support");
  const confidenceInput = document.getElementById("min-confidence");
  const supportOutput = document.getElementById("min-support-value");
  const confidenceOutput = document.getElementById("min-confidence-value");
  if (supportOutput) supportOutput.textContent = `${Number(supportInput.value).toFixed(1)}%`;
  if (confidenceOutput) confidenceOutput.textContent = `${Number(confidenceInput.value).toFixed(0)}%`;
}

/**
 * Fill the ranking `<select>` from `BUSINESS_RANKINGS`, so the control and the
 * ranking logic can never drift apart. No ranking is hardcoded in the markup.
 *
 * @returns {void}
 */
function populateRankingControl() {
  const select = document.getElementById("rule-sort");
  if (!select) return;
  select.innerHTML = BUSINESS_RANKINGS.map(
    (ranking) =>
      `<option value="${escapeHtml(ranking.id)}">${escapeHtml(ranking.label)}</option>`,
  ).join("");
  select.value = DEFAULT_RANKING_ID;
}

/**
 * The ranking the user currently has selected, falling back to the default.
 *
 * @returns {string}
 */
function currentRankingId() {
  const select = document.getElementById("rule-sort");
  return select && select.value ? select.value : DEFAULT_RANKING_ID;
}

/**
 * Re-render the table and the settings panel from the cached run using the
 * currently selected ranking. This deliberately does not re-mine: ranking a rule
 * set is a sort, and re-mining on every dropdown change would make the control
 * feel broken at low support thresholds.
 *
 * @returns {void}
 */
function applyRanking() {
  if (!LAST_RESULT) return;
  const rankingId = currentRankingId();
  const ranking =
    BUSINESS_RANKINGS.find((entry) => entry.id === rankingId) ||
    BUSINESS_RANKINGS[0];
  const ranked = rankBusinessRules(LAST_RESULT.businessRules, rankingId);
  renderResults(ranked, DATASET_INDEX);
  renderThresholdSummary({
    ...LAST_RESULT,
    rankingId,
    rankingLabel: ranking.label,
  });
}

/**
 * Read the thresholds, run the student pipeline, and render the results.
 *
 * @returns {void}
 */
function runPipeline() {
  const status = document.getElementById("status");
  if (!DATASET_INDEX) {
    if (status) {
      status.textContent = "Dataset not ready yet — reload the page.";
    }
    return;
  }
  const { minSupport, minConfidence } = readThresholds();
  const validation = validateThresholds(minSupport, minConfidence);
  if (!validation.ok) {
    if (status) status.textContent = validation.errors.join(" ");
    return;
  }
  try {
    if (status) status.textContent = "Mining frequent itemsets…";
    const itemsets = findFrequentItemsets(TRANSACTIONS, minSupport);
    const rules = generateRules(itemsets, minConfidence);

    // The miner output is left exactly as produced. Presentation metrics are
    // attached on a copy, and the lift > 1 selection happens here rather than
    // inside `generateRules`.
    const n = DATASET_INDEX.n || N;
    const business = selectBusinessRules(rules);
    const enriched = business.rules.map((rule) =>
      enrichBusinessRule(enrichRule(rule, DATASET_INDEX), n),
    );

    LAST_RESULT = {
      n,
      minSupport,
      minConfidence,
      itemsetCount: itemsets.length,
      ruleCount: rules.length,
      businessRules: enriched,
      businessRuleCount: enriched.length,
      businessFiltered: business.filtered,
      recommendation: pickRecommendation(enriched),
      strongest: pickStrongestAssociation(enriched),
      misleading: pickMisleadingRule(enriched),
    };

    applyRanking();
    renderRecommendation(LAST_RESULT.recommendation, n, DATASET_INDEX);
    renderAssociationWarning(
      LAST_RESULT.strongest,
      LAST_RESULT.recommendation,
      LAST_RESULT.misleading,
      DATASET_INDEX,
    );
    renderTopBusiness(enriched, DATASET_INDEX);

    if (status) {
      status.textContent = `Done — ${rules.length} rule(s) at support \u2265 ${(minSupport * 100).toFixed(1)}% and confidence \u2265 ${(minConfidence * 100).toFixed(0)}%; ${enriched.length} of them beat the independent baseline (lift > 1).`;
    }
  } catch (error) {
    LAST_RESULT = null;
    const message = String(error && error.message ? error.message : error);
    const resultsEl = document.getElementById("results");
    if (resultsEl) {
      resultsEl.innerHTML =
        '<p class="empty-state">Run failed &mdash; the rule miner did not complete. ' +
        "Implement the <code>TODO(hw4)</code> functions, then press &ldquo;Run rules&rdquo;. " +
        `Error: ${escapeHtml(message)}</p>`;
    }
    ["threshold-summary", "recommendation", "association-warning", "top-business"].forEach(
      (id) => {
        const element = document.getElementById(id);
        if (element) {
          element.innerHTML = `<p class="empty-state">Run failed &mdash; ${escapeHtml(message)}</p>`;
        }
      },
    );
    if (status) status.textContent = message;
  }
}

/**
 * Wire the controls once the DOM is ready, then build the dataset index from the
 * baskets already decoded from `window.HW4` and render the summary.
 *
 * The data is already in memory (embedded by `transactions.js`), so there is no
 * fetch and no error path beyond the `window.HW4` guard at the top of this file.
 *
 * @returns {void}
 */
function init() {
  renderWorkedExample(tinyWorkedExample());
  syncThresholdLabels();

  const supportInput = document.getElementById("min-support");
  const confidenceInput = document.getElementById("min-confidence");
  if (supportInput) supportInput.addEventListener("input", syncThresholdLabels);
  if (confidenceInput) confidenceInput.addEventListener("input", syncThresholdLabels);

  const runButton = document.getElementById("run-rules");
  if (runButton) runButton.addEventListener("click", runPipeline);

  const testButton = document.getElementById("run-tests");
  if (testButton) testButton.addEventListener("click", () => runTests());

  populateRankingControl();
  const sortSelect = document.getElementById("rule-sort");
  if (sortSelect) sortSelect.addEventListener("change", applyRanking);

  const status = document.getElementById("status");
  primeDescriptions();
  N = TRANSACTIONS.length;
  DATASET_INDEX = buildIndex(TRANSACTIONS);
  try {
    renderDatasetSummary(DATASET_INDEX);
  } catch (error) {
    // `renderDatasetSummary` is scaffolding, but guard it defensively so that an
    // unimplemented TODO(hw4) stub can never take the whole page down at load
    // time. The harness's "Run tests" button stays usable regardless.
    const summary = document.getElementById("dataset-summary-body");
    if (summary) {
      summary.innerHTML =
        '<p class="empty-state">Implement the TODO(hw4) functions to activate ' +
        "the dataset summary.</p>";
    }
  }
  if (status) {
    status.textContent = `Dataset ready: ${N.toLocaleString("en-US")} baskets, ${data.N_ITEMS.toLocaleString("en-US")} distinct items. Set the thresholds and press “Run rules”.`;
  }
}

if (typeof document !== "undefined") {
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
}
