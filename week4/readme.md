# HW4 — Association Rules

**Course**: LLM4Rec, HSE University
**Instructor**: Seungmin Jin (sedzhin@hse.ru)
**Repository**: https://github.com/dryjins/RecSys-LLMs/tree/main/week4
**Task type**: Assignment (individual)
**Status**: Implementation complete — self-checks **11 passed / 0 failed / 0 pending**

This file keeps the original assignment framing (learning goals, dataset
provenance, submission rules, grading) and documents the delivered
implementation, the business interpretation, and the verification actually
performed.

---

## Goal

The UCI **Online Retail** transaction log is turned into association rules.

- One **`InvoiceNo` = one basket** (a transaction). Baskets are not split by
  customer, and there is no customer-level identity in this problem.
- **`StockCode` = item identity.** It is the only stable key; `Description` is a
  readable human label that changes in spelling and case across the source file,
  so it is never used for grouping.
- **`Description`** is displayed next to the code so a rule can be read by a
  merchandiser without looking it up.
- **Association Rule Mining (ARM)** finds items that co-occur far more often than
  independence would predict, and expresses them as `A → B`.

The three metrics:

| metric | question it answers |
|---|---|
| **support** | How often does this pattern occur at all? |
| **confidence** | Given A is in the basket, how often is B also there? |
| **lift** | How much more often than chance does B follow A? |

The assignment's business question (see §*Business interpretation*) is not only
*"what products are associated?"* but **"what B may be under-attached among
buyers of A, and which cross-sell should be tested?"** — that distinction drives
the whole analysis layer added on top of the miner.

---

## Run instructions and reproducibility

Open `week4/index.html` in a modern browser. **No build, no server, no install.**
Keep `week4/transactions.js` next to `index.html` — it holds the dataset; if it
is missing the page shows a "Failed to load `transactions.js`" message.

Reproducing the reported numbers:

1. Open the page. Confirm the dataset summary reads **17,080 baskets** and
   **3,653 distinct items**.
2. Press **Run tests** → expect `pass 11, fail 0, pending 0`.
3. Leave the sliders at their defaults (**1% support / 30% confidence**), press
   **Run rules**.
4. Confirm the status line: **1,219 frequent itemsets → 950 rules**, all 950 with
   lift > 1.
5. Read the four business panels from the top of the main column:
   recommended opportunity, strongest association, Top-3, then the sortable
   rules table.

To reproduce the threshold trade-off table below, move only the support slider
and re-run; each row is an independent full mining pass.

The page shows a **Controls** sidebar (minimum support / minimum confidence
sliders, **Run rules**, **Run tests**, **Reset**), a sortable **Rules** table
where each row opens in the detail panel, and a **Reverse direction (B → A)**
button that recomputes the opposite direction.

---

## Implementation

Implemented in `week4/script.js`. The seven `TODO(hw4)` stubs are complete and
the existing scaffolding (dataset decoding, `buildIndex` / `asIndex`,
`countItem` / `countPair`, `validateThresholds`, formatting, the test harness) is
left as-is.

**1. Basket construction — `dedupeBasket(rawItems)`**
A basket behaves as a *set*: a repeated `StockCode` inside the same invoice
counts once. First-appearance order is preserved so output is deterministic and
diffable.

**2. Inverted occurrence index — `buildIndex` / `asIndex`**
`StockCode → Set(basketId)`. Every count in the pipeline is then an intersection
size over posting lists rather than a rescan of all 17,080 baskets. This is the
single most important performance decision in the implementation: item counts are
O(1) and `countItemset` is O(size of the smallest posting list).

**3. Counting — `countItemset(basketsOrIndex, stocks)`**
Number of baskets containing *every* requested stock, accepting either raw
baskets or a prebuilt index. Returns `0` when any stock is unknown, so
downward-closure checks can never see a spurious count.

**4. Metrics — `computeSupport`, `computeConfidence`, `computeLift`**
Each returns `{ value, defined }` and guards its zero denominator:
`defined: false` rather than `Infinity` or `NaN`.

**5. Frequent itemsets — `findFrequentItemsets(transactions, minSupport)`**
Classic **level-wise Apriori**:

1. `L₁` = singletons whose count ≥ `minSupport · N`.
2. `Lₖ₊₁` = `Lₖ` joined with itself on the last `k − 1` items, then pruned by
   **downward closure**: a candidate is frequent only if *all* of its `k`
   subsets are present in `Lₖ`. Closure is checked directly against the level's
   own lookup, not by rescanning baskets.
3. Terminate when a level produces no frequent itemset.

Downward closure is the correctness property the assignment asks for: every
subset of a frequent itemset is frequent, so pruning on it is safe and is what
keeps the candidate space finite.

Two real bugs were found here **after** the fixture already passed, and both
were caught only on real data — see §*Debugging notes*.

**6. Rule generation — `generateRules(frequentItemsets, minConfidence)`**
For each frequent itemset of size ≥ 2, every proper non-empty antecedent /
consequent split is emitted in **both directions** (`A → B` and `B → A`), each
direction's confidence computed against its own antecedent count, and only rules
passing `minConfidence` are kept. The function returns **exactly the eight
documented fields** (`antecedent`, `consequent`, `jointCount`,
`antecedentCount`, `consequentCount`, `support`, `confidence`, `lift`) and
applies **no lift filter** — filtering on lift is a *business-analysis* step and
is done separately, which keeps the two concerns independent and auditable.

**7. Threshold handling and presentation**
The sliders feed `minSupport` / `minConfidence`. The **Rules** table contains
exactly the rules satisfying both. Downstream business analysis then retains
only rules with **lift > 1**, falling back to the unfiltered set if no rule
clears that bar, so the UI can never end up empty for a non-empty result set.

---

## Missing / duplicate handling

- **Duplicate `StockCode` inside the same basket counts once.** A basket is a
  transaction-level *set*, not a multiset; `dedupeBasket` enforces this. Two
  units of the same item in one invoice are one line-item identity.
- **A basket is a transaction, not a customer.** All metrics are computed over
  `N = 17,080` invoices. There is no customer-level rollup anywhere, so "baskets
  containing A" and "customers who bought A" are *not* interchangeable.
- **Imputation is not a relevant concept here.** Imputation addresses absent
  attribute values in a feature table. This dataset has no feature table and no
  per-basket attributes beyond item presence, so there is nothing to impute.
  The cleaning decisions that *were* applied (dropping cancellations,
  adjustments, non-positive quantity/price, blank descriptions, guest checkouts
  and non-product codes) are **row exclusions decided upstream**, not imputation,
  and are documented in §*Dataset provenance*.
- **Zero denominators** (`count(A) = 0` or `count(B) = 0`) are reported as
  *undefined* with an inline note in the detail panel — never as `Infinity` or
  `NaN`. With well-formed generated rules this cannot occur; it is implemented
  defensively and the UI path is exercised.

---

## Metrics

For an itemset `X`, let `count(X)` be the number of baskets containing every
item in `X`, and let `N` be the total number of baskets (`window.HW4.N_BASKETS`,
here **17,080**).

```
support(A → B)    = count(A ∪ B) / N
confidence(A → B) = count(A ∪ B) / count(A)
lift(A → B)       = confidence(A → B) / [ count(B) / N ]
```

- **support** measures how often the pattern occurs, not how strong the link is.
- **confidence** is a conditional probability and is **not symmetric**:
  `confidence(A → B)` and `confidence(B → A)` generally differ, because the
  denominators are different baskets. The **Reverse direction** button exists to
  demonstrate exactly this.
- **lift** compares observed co-occurrence against independence.
  `lift = 1` → independent, `lift > 1` → co-occurs more than chance,
  `lift < 1` → less than chance. Lift **is symmetric**:
  `lift(A → B) = lift(B → A)`, so the rule's *strength of association* does not
  depend on which item you write first — but its *actionability* does, via
  confidence and headroom.
- **lift > 1 means positive association, not causation.** See §*Suggested
  experiment*.

### Business metrics (added layer)

These are computed in a separate `enrichBusinessRule(rule, n)` helper and are
never written into the `generateRules` output:

| metric | definition | meaning |
|---|---|---|
| `countA` | `rule.antecedentCount` | baskets containing A |
| `countB` | `rule.consequentCount` | baskets containing B |
| `countAB` | `rule.jointCount` | baskets containing both |
| `A_without_B` | `countA − countAB` | A-baskets where B is **absent** |
| `attach_rate` | `rule.confidence` | share of A-baskets that already take B |
| `headroom` | `1 − attach_rate` | share of A-baskets that do **not** |
| `baseline_B` | `countB / N` | how common B is overall |
| `expected_AB_if_independent` | `countA · countB / N` | A+B baskets if independent |
| `incremental_cooccurrence` | `countAB − expected_AB_if_independent` | excess baskets over chance |

**`potential_attach_baskets = A_without_B` is NOT a predicted uplift.** It is the
size of today's audience that does not currently take B. It says how many
baskets *could* be exposed to a cross-sell; it does not predict how many will
change their behaviour. Converting reach into incremental sales is the job of
the experiment in §*Suggested experiment*, not of this table.

---

## Thresholds

UI defaults: **minimum support = 1%**, **minimum confidence = 30%**.

| min support | frequent itemsets | rules @ 30% conf | wall time |
|---|---|---|---|
| 2% | 294 | 96 | < 1 s |
| **1% (default)** | **1,219** | **950** | **~1–2 s** |
| 0.5% | 5,255 | 10,723 | ~7–9 s |
| 0.3% | 17,929 | 70,250 | ~15 s |
| 0.2% | 53,084 | 376,301 | ~34 s |
| 0.15% | 125,775 | 1,213,424 | ~76 s |
| 0.1% | *(not completed)* | *(not completed)* | > 305 s |

Confidence, holding support at 2%: 30% → 96 rules, 50% → 36 rules.

**Trade-off.** Lower thresholds admit more patterns but also more noise and
computationally more candidates — the growth above is combinatorial, because a
slightly more permissive threshold keeps every itemset of every larger size. At
0.1% support the rule set grows by more than an order of magnitude per step and
the page stops being interactive. Higher thresholds keep only the most common
patterns: fewer rules, but larger samples and less noise per rule.

Downward closure explains the direction: raising support shrinks `Lₖ` at every
level, and because each level is pruned against its predecessor, the shrinkage
compounds rather than staying proportional.

**1% / 30% is not claimed to be universally optimal.** It is the operating point
chosen here because it yields ~950 rules — enough to contain a commercially
meaningful top-3 with hundreds of reachable baskets, while keeping the smallest
antecedents above the noise floor. A different retail category, basket-size
distribution, or decision objective would justify a different point.

---

## Business interpretation

The teacher's question is not *"what products are associated?"* but:

> **What B may be under-attached among buyers of A, and which cross-sell should
> be tested?**

A high `lift` answers the first question. Only the combination of **lift** (is
the link real?) and **reach** — `countA`, and especially `A_without_B` (is
anyone left to sell to?) — answers the second.

That is why the page ranks by five explicit, named views rather than one
composite score. At the default settings all five disagree, which is the point:

| ranking | head rule | lift | reach (`A_without_B`) |
|---|---|---|---|
| A without B | 85099B → 22386 | 6.775 | 1,038 |
| lift | 22916 → 22917 | 85.062 | 10 |
| confidence | 22916 → 22917 | 85.062 | 10 |
| A+B baskets | 22386 → 85099B | 6.775 | 323 |
| incremental co-occurrence | 22697 → 22699 | 17.144 | 150 |

No single ranking is "correct"; each answers a different question, and the UI
labels which question is being asked. Changing the ranking re-sorts the
already-computed result — it does not re-mine.

---

## Strongest association vs commercial opportunity

### Strongest association (max lift)

```
22916  HERB MARKER THYME  ->  22917  HERB MARKER ROSEMARY
lift = 85.0617      attach rate = 94.62%
countA = 186        countAB = 176        A_without_B = 10
```

### Largest current opportunity (max `A_without_B`)

```
85099B  JUMBO BAG RED RETROSPOT  ->  22386  JUMBO BAG PINK POLKADOT
countA = 1584   countB = 869   countAB = 546   A_without_B = 1038
attach rate = 34.47%   headroom = 65.53%   lift = 6.775
expected_AB_if_independent = 80.59   incremental_cooccurrence = 465.41
```

**Maximum lift ≠ maximum commercial headroom.** The herb-marker rule has ~13×
the lift of the jumbo-bag rule and is almost certainly a *real* association — but
only **10** A-baskets do not already contain B, so there is essentially no
audience left to act on. The jumbo-bag rule has a far smaller lift and offers
**1,038** baskets that currently contain the red bag but not the pink one.

A rule is only a *misleading* one when the high metric is not backed by reach.
`22916 → 22917` is the extreme case: 94.62% attach and lift 85 look decisive, and
a naive "top rule = ship this" reading would be wrong. Under-attachment — not
association strength — is what selects the cross-sell worth testing.

**1,038 is a reachable audience, not 1,038 incremental sales.** Attaching B to all
1,038 would require every one of those customers to change behaviour, which no
observation supports.

### A note on the two kinds of misleading rule

The assignment suggests a misleading rule "for example" as *a very
high-confidence rule whose **lift is close to 1**, or a rule whose antecedent is
so frequent that the rule is trivially satisfied.* Both of those examples were
checked against the actual rule set and **neither exists at the chosen
thresholds**:

| shape | rules found |
|---|---|
| `1 < lift < 1.15` (high confidence, lift ≈ 1) | **0** |
| confidence ≥ 45% with lift ≤ 4 (B common anyway) | **0** |
| lowest lift anywhere in the 950-rule set | **2.6366** |
| highest confidence anywhere in the 950-rule set | **94.62%** |

The thresholds themselves already exclude vacuous rules: with support ≥ 1% and
confidence ≥ 30%, every surviving rule co-occurs at least 2.6× more than chance.
This is worth stating explicitly rather than manufacturing an example that the
data does not contain.

Two distinct failure modes remain, and the page surfaces the second one:

1. **Statistical vacuity** (`lift ≈ 1`) — *absent here by construction.* The
   closest available analogue is the low-lift end of the rules with a frequent
   antecedent: `82482 WOODEN PICTURE FRAME WHITE FINISH → 85123A WHITE HANGING
   HEART T-LIGHT HOLDER`, confidence 35.24%, **lift 3.07**, antecedent present in
   874 baskets (5.12%). `85123A` occurs in 1,959 baskets (11.47%) all by itself,
   which is exactly the starter's anchor point: a frequent item makes a
   *confidence* number easier to over-read, which is why lift is reported next
   to it.
2. **Commercial vacuity** — *present and surfaced by the UI.* `22916 → 22917` is
   the maximum-lift rule with 94.62% attach, yet only 10 A-baskets lack B. It is
   statistically excellent and commercially empty, and a naive "highest lift =
   ship this" reading is exactly wrong.

The misleading-rule panel reports case 2, because that is the mistake this
analysis exists to prevent.

---

## Suggested experiment

Association is not causation. `A → B` is a frequency statement about the observed
period only: both items may be bought together because of a seasonal event, a
promotion running at the same time, or a shared basket context (gift wrapping,
a set purchase), none of which promoting `A` would reproduce.

**Design.** Randomize at the basket or session level:

- **Treatment** — when A (`85099B`) is in the basket, surface B
  (`22386`) as a recommendation slot / second-item prompt, or test a bundle or a
  second-item discount.
- **Control** — the normal, unmodified experience.

**Primary KPI — incremental attach rate:**

```
P(B | A, treatment) − P(B | A, control)
```

**Secondary metrics:** units of B sold, revenue, average basket value, margin.

**Guardrails:** discount cost, conversion of A itself, cannibalization of other
B-adjacent items, and return rate.

**Decision rule.** Ship only if the attach-rate difference is positive, its
confidence interval excludes zero, and the margin gain clears the discount cost
without breaching a guardrail.

**Limitation and honesty requirement.** The exported dataset contains baskets
grouped by `InvoiceNo` with **no usable per-basket timestamp**. Temporal data is
**not** simulated anywhere in this submission. Validating stability requires a
later observation period or the A/B test above — data the starter does not
contain. A further limitation is that a rule is a statement about the observed
period only, and this log mixes wholesale with retail activity.

---

## Performance

At the default 1% support, mining takes roughly **1–2 seconds** in the
verification environment and produces **1,219** frequent itemsets and **950**
rules at 30% confidence. At 0.5% support the miner produces **5,255** itemsets
and roughly **10,700** rules in about 7–9 seconds.

Lower support causes combinatorial growth in the number of candidate itemsets,
and therefore in the rule count — see the sweep table in §*Thresholds*. The
practical ceiling of this implementation in this environment is around 0.15%
support (~76 s, 1.2 M rules); 0.10% support did not complete within 305 seconds.
The UI re-sorts and re-renders the cached result, so only the mining step pays
this cost.

---

## Verification

### Test progression

| stage | pass | fail | pending |
|---|---|---|---|
| Starter baseline (stubs untouched) | 2 | 0 | 9 |
| After metric + counting implementation | 9 | 0 | 2 |
| After Apriori | 10 | 0 | 1 |
| **Final** | **11** | **0** | **0** |

The provided `tinyWorkedExample` fixture's `lift = 1`, `lift > 1` and
`lift < 1` cases match their hand-computed values. **The provided tests were not
modified at any point.**

### Independent checks beyond the provided tests

- **Fixture brute force.** On the tiny fixture the rule output was compared
  against an independent enumeration of all antecedent/consequent splits:
  both return **38** rules at `minConfidence = 0.5`.
- **Real-data count cross-checks.** Itemset and pair counts from the inverted
  index were compared against a direct rescan of the raw baskets.
- **Downward-closure checks.** Every generated itemset was verified to have all
  of its subsets present in the previous level — the invariant that makes
  Apriori's pruning safe.
- **Rule metric cross-check.** Support, confidence and lift were recomputed from
  raw counts for sampled rules and matched the displayed values; the reported
  opportunity was hand-checked as `1584 × 869 / 17080 = 80.59`, then
  `546 − 80.59 = 465.41`.
- **Lift filter independence.** `generateRules` was asserted to keep rules with
  `lift < 1` (60 fixture rules: 6 below, 18 equal, 36 above) and the business
  layer's `lift > 1` subset was asserted to be exactly 36 of them — the two
  stages are verifiably distinct.
- **Business-layer purity.** `enrichBusinessRule` was checked not to mutate its
  input, and the business layer was scanned to confirm no stock code or product
  description is hardcoded in any decision path.
- **Browser UI verification.** The page was driven in Chrome (154) over the
  DevTools Protocol with real clicks and events — **27 / 27 checks passed**, with
  **0 console errors and 0 uncaught exceptions**. Covered: page load, Run rules,
  all panels rendered, the Sort-by control re-ordering the table, row click →
  detail panel, reverse direction (92.63% → 94.62%), slider input events, a
  threshold change propagating to every panel (950 → 36 rules, recommendation
  switching to `22384 → 20725`), restoration to defaults, and Run tests.
- **Layout / readability audit.** **15 / 15 checks passed**: panel order, table
  scrolling rather than shrinking, minimum text size, no clipped text, keyboard
  focusability of the scroll container, bound `<label>` for the sort control, and
  no horizontal overflow at a 375 px viewport.

---

## Manual verification

These are the checks I performed personally, in Chrome, on the local page:

- I opened the HW4 page locally in Chrome.
- I confirmed the default thresholds are 1% support / 30% confidence.
- I ran the rules and saw the recommendation `85099B → 22386`.
- I saw `A_without_B = 1,038`.
- I saw the strongest association `22916 → 22917`.
- I changed **Sort by** to **Lift** and observed the table re-sort.
- I ran **Run tests** and observed **11 passed / 0 failed / 0 pending**.
- I reviewed the Top-3 commercial opportunities.
- I reviewed the worked example.
- I reviewed the A/B validation section and the "Association is not causation"
  warning.

I did **not** manually recompute the metric formulas in the browser. Every
numeric cross-check listed in §*Verification* was performed separately, by
recomputing from raw counts outside the UI.

---

## Debugging notes

**A. Stale local starter.** The local `week4/` initially held a MovieLens
Two-Tower implementation (`app.js`, `graph.js`, `two-tower.js`, `data/`) while
the official upstream held Association Rules. Only `week4/` was synced from
upstream; the Two-Tower files were removed as part of the swap.

**B. Real-data Apriori bugs that the fixture did not catch.** The tiny fixture
passed while two genuine defects were still present, both surfacing only on the
real dataset:

- **False positives from an incorrect early exit** in candidate generation —
  a candidate was accepted without confirming all of its `k`-subsets were
  frequent, so non-frequent itemsets leaked into the output.
- **Wrong downward-closure subset check** — the subset test compared the wrong
  element positions rather than all `k` subsets of a `k+1`-itemset.
- **Stack overflow** from `push(...next)` on very large candidate arrays
  (argument-count limit), replaced by a loop.

The lesson recorded here: the tiny fixture alone was insufficient. Real-data
invariants — closure, itemset counts, and cross-checks against a direct rescan —
are what exposed the defects.

**C. The `generateRules` `N` trap.** `generateRules` receives no `N` argument,
so a natural-looking implementation reads the module-level `N`, which is the real
dataset's `N = 17,080`. On the fixture that silently corrupts support and lift
because the fixture's denominator is much smaller. The implementation instead
derives the effective `N` from the frequent-itemset count/support relationship
(`N ≈ count / support`), so it is correct for both the fixture and the real
dataset.

---

## Dataset provenance

- **Source**: UCI Machine Learning Repository, *Online Retail*, dataset id **352**.
  - Dataset page: `https://archive.ics.uci.edu/dataset/352/online+retail`
  - Raw download: `https://archive.ics.uci.edu/static/public/352/online+retail.zip`
  - Original workbook sha256:
    `43465a06f2ccf7c8b5bd2892bc7defb52f97487934fe93b16ae4c3936424676d`
- **Citation** (as required by the source): Daqing Chen, Sai Liang Sain, and Kun
  Guo, "Data mining for the online retail industry: A case study of RFM
  model-based customer segmentation using data mining", *Journal of Cases on
  Information Technology*, 2012.
- **Cleaning summary** (performed by `tools/build_week4_data.py`, a one-off
  generator that is **not** part of this assignment):
  - 541,909 raw rows → **384,911 kept rows**, **17,080 baskets**, **3,653 distinct items**.
  - Rows removed: cancellations (`InvoiceNo` starting with `C`) and adjustments
    (`A`), non-positive `Quantity`, non-positive `UnitPrice`, blank `Description`,
    guest checkouts (blank `CustomerID`), and non-product codes (postage,
    carriage, bank charges, manual entries, samples, discounts, gift vouchers,
    packing charges, internal adjustments).
  - `StockCode` is trimmed and upper-cased so case variants such as `84509c` and
    `84509C` collapse to one item; `Description` is upper-cased, whitespace is
    collapsed, trailing punctuation is dropped, and one canonical description is
    kept per stock code.
  - Baskets are grouped by `InvoiceNo`. Baskets with fewer than two distinct items
    are dropped because they cannot yield a rule. Baskets are **not** split by
    customer.
- The full provenance string is available as `window.HW4.dataset_provenance` (set
  by `week4/transactions.js`) and is displayed in the page's dataset summary.

---

## Learning goals

By the end of this assignment you should be able to:

- Turn a raw retail transaction log into **item counts** and **pairwise
  co-occurrence counts**, and explain why those counts are all that
  association-rule mining needs.
- Compute **support**, **confidence**, and **lift** for a rule, and explain what
  each metric does and does not measure.
- Implement **Apriori** (or an equivalent frequent-itemset miner) using the
  downward-closure property: every subset of a frequent itemset is frequent.
- Generate **candidate rules in both directions** (`A → B` and `B → A`) and show
  that confidence is *not* symmetric while lift *is*.
- Filter a rule list with **support / confidence thresholds**, keep only rules with
  **lift > 1**, and describe how each filter changes the list of rules you keep.
- Distinguish a **genuinely useful rule** from a **misleading one** that only looks
  good because one item is very frequent.
- Reason about **association versus causation**, and propose a **validation plan**
  for a rule when the exported dataset carries no usable time information.

---

## Submission instructions

Submit the **modified `week4/` directory** containing:

| File | Role |
|---|---|
| `week4/transactions.js` | dictionary-encoded dataset embedded as a plain script (provided, **not modified**) |
| `week4/script.js` | implementation of the `TODO(hw4)` stubs + business-analysis layer |
| `week4/index.html` | page structure (extended with the business panels) |
| `week4/style.css` | styling (extended for the business panels) |
| `week4/readme.md` | this file |

Plus the course report (IEEE-aligned, per the homework guidelines) that answers
the analysis questions.

- **No Jupyter notebook** (`.ipynb`) is part of this deliverable.
- **No separate memo** is required; the analysis belongs in the report.
- **No Python** is part of this deliverable. `tools/build_week4_data.py` is the
  one-off dataset generator used by the instructor; it is not a student deliverable
  and must not be submitted.
- Do not commit generated artefacts, virtual environments, or log files.

---

## Grading criteria

Grading follows the course homework guidelines, **§8 — Rubric Criteria in Detail**
(course repository path: `docs/homework-guidelines/guidelines.md`). The two
criteria are **binary** (0 or 1) and combine into a per-assignment score of
**0, 1, or 2**:

- **c1 — Understanding.** Clear problem statement; all references valid;
  attribution accurate. A hallucinated citation, a broken reference URL, or a
  fabricated author/year is a **Gate 0** failure of c1.
- **c2 — AI Management.** The solution works (the page runs and the table matches
  the code output); the reasoning is accurate; and verification is cited (a
  hand-computed metric, a re-run, a source read beyond the abstract, or a
  cross-check of a number against the code).

Read §10 of the same guidelines for the **Gate 0** failure modes. The ones that
apply most directly here:

- a **hallucinated citation** (a paper that does not exist) — including the UCI
  source reference;
- **fabricated verification** — claiming you hand-computed or re-ran something you
  did not;
- a **broken solution** — the page does not run, or the numbers shown do not match
  the code's own output.

If you cannot verify a critical output, say so honestly in the report's AI-usage
section rather than claiming verification you did not perform. The *Verification*
and *Manual verification* sections above deliberately separate what was checked by
cross-check from what was checked by hand in the browser.

---

*Starter file generated 2026-09-29 from the HW4 work order; implementation and
verification sections completed for the delivered submission.*