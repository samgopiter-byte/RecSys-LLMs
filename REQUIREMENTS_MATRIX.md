> **CURRENT STATUS UPDATE — 2026-10-17:** the implementation and student self-checks have progressed beyond the historical pre-implementation audit below. The dated update at the end is authoritative for changed rows. Do not reuse the old 4/38 count or the statement that seven stubs remain unimplemented. See audit/HW6_PERFORMANCE_REVIEW_2026-10-17.md. Final compliance is still blocked.

# A06 / HW6: fresh requirements audit

Audit date: 2026-10-16 (Europe/Moscow). Branch: `a06-personalized-pagerank`, HEAD `00c032f` ("HW6: add graph-based Personalized PageRank starter (week6)"). `git diff HEAD upstream/main -- week6` is empty — the local starter is byte-identical to the instructor's upstream main. `git status --porcelain` shows only untracked workspace artifacts (sessions, tmp, course, audit, matrix files); no staged or tracked modification exists. This file supersedes the earlier A05 matrix as the active requirements record; the A05 audit detail remains in `audit/*` files referenced there.

Scope of this audit: read-only starter/requirements audit BEFORE any implementation. Only `REQUIREMENTS_MATRIX.md` and `VERIFICATION_LEDGER.md` are authored; no week6 file is edited, no package is installed, no commit/push, no report authored. Statuses: PASS = verified at the stated (usually documentation/static) scope; FAIL = concrete noncompliance; PENDING = missing runtime/personal/artifact evidence; BLOCKED = unresolved source conflict; NOT APPLICABLE = explicitly out of scope. Static code presence never implies runtime PASS or personal verification.

## A. Sources read and source gaps

| ID | Source | Authority / scope |
|---|---|---|
| U | Student prompt, 2026-10-16 (this session) | Read-only full starter audit; write only matrix/ledger/scratch; no package installs; no week6 edits; final 13-section audit output |
| G0 | Root `AGENTS.md` | Permanent workflow: requirements-first, no report before compliance, report structure, page cap, references, AI disclosure, personal verification, starter audit, 5-Loop, evidence, session, final gate |
| H6 | `week6/readme.md`, all 191 lines | Current assignment specification (primary): graph rules, reference fixture, Top-10 protocol, four-method comparison, alpha sweep, one-factor demo, cohort/Recall, popularity, ONE business question, deliverables, checklist |
| G | `course/homework_guidelines.pdf`, all 12 pages (text extracted with local pymupdf, scratch file deleted after use) | Binding course rules: 5-Loop structure, c1/c2 grading, session export, failure modes |
| F1-F4 | `tmp/feedback преподавателя/HW{1,2,3,4}_class_feedback.md`, complete files | Grading lessons: full-starter defect coverage (F1 §2, F2 §2, F3 §2), personal verification not agent-run (F2 §4, F3 §4, F4 §5), session native export (F4 §4), in-page self-check counts reported (F4 §5), references research (F4 §3), business axis (F4 §2) |
| I5 | `tmp/feedback преподавателя/HW5_individual_recall_clarification_2026-10-14.md` | Latest individual instructor reply (HW5 Recall semantics). It defines HW5 ranking/evaluation pools; no HW6 conflict arises from it — HW6 §5/§7 give their own candidate and exclusion rules |
| S | `course/sample_homework.pdf`, all 6 pages | Illustrative sample (different A01 topic); not current requirements or factual evidence |
| A2-A4 | `HW{2,3,4}_A0{2,3,4}_Answer.md` | Sample reports with student placeholders; not individual feedback; cannot validate claims |
| E | `most important.txt` | Empty (0 bytes); no requirements |
| C6 | Entire `week6/` starter: `graph.js` (1411 lines), `script.js` (3801 lines), `data.js`, `index.html`, `style.css`, `u.data`, `u.item`, `vendor/d3.min.js` | Current static implementation/scaffold evidence; not runtime results |
| P | `README.md`, `week1..week5` readmes, HW5 lecture PDF in `tmp/` | Historical framing; HW5 lecture is not an HW6 specification |
| Git | branch/HEAD/status/diff vs `upstream/main` | Provenance: starter unmodified, work preserved, nothing committed |

Gaps (remain visible, not waived):
- **No HW5 class feedback yet** — expected: grading follows cohort submission; AGENTS.md explicitly says this is not a missing prerequisite and not a reason to wait.
- **No HW6 lecture/slides found locally** (only the HW5 lecture PDF exists in `tmp/`).
- **No HW6 report draft exists** — no report-authoring precondition is met, and none is attempted.
- **No HW6 individual instructor reply** other than I5 (HW5-specific).
- **Submission directory/deadline/grading guide** are not fixed anywhere local; H6 §9 says they "will be announced by the instructor" and instructs confirming them before publishing.

## B. Consolidated requirements

Requirements-first; preserve week1–week5 and datasets; implement every H6 bullet literally; audit the entire starter for all major defects before/with implementation (F1–F3 lesson); leakage-free evaluation (graph from TRAIN rating>=4.0 only; alpha by validation only; test for final evaluation only); missing = unobserved, never zero; alpha = link-following probability throughout; TF.js core propagation with readiness/backend/tidy discipline; shared cohort and shared candidates for all four methods; validation exclusion = train-rated; test exclusion = train- or val-rated regardless of value; exactly ONE business question with an actionable deployment recommendation plus limitations and one online experiment; PPR scores never presented as ratings or purchase probabilities; report every exclusion/count actually observed; §2 Related Work + 3–7 validated numbered references (student's stricter safeguard); §6 AI disclosure with personal-verification and trusted-without-verification statements; ≤5 physical PDF pages including references; personal verification performed by the student (hand computation, browser reproduction, code/output cross-check, assignment-specific logic check); unmodified native session export; full final pre-submission gate with GATE-0/MINOR RISKS: NONE only when evidenced.

## C. Conflicts and resolutions

1. **Page count** (G internal five-vs-six disagreement): resolved by the stricter student rule — ≤5 total physical PDF pages including references (G0 §4).
2. **MF provenance tension inside H6**: H6 §6 says "reuse your previous implementation and selected model", while the starter declares "The MF reference is instructor-provided: students must not have to implement it. Report its selected checkpoint, not a hand-built model" (`script.js:35-36`), and ships a lazy-trained deterministic MF (`MF_SEED=20261013`, factors 16, epochs 30, lr 0.01, reg 0.05, batch 512; week5 used epochs 20/batch 64). Resolution by the HW5 precedent (starter JSDoc/harness is the operative contract): use the provided reference as-is, do not rewrite it, and state its exact hyperparameters, checkpoint epoch and validation RMSE in §3/§4, noting the difference from the week5 model. Report the tension; do not silently claim "the week5 model" was reused. No BLOCKED status: both readings converge on "MF is a fixed comparator, not tuned on test".
3. **Reference count**: F4 §3 allows <3 with genuine research; G0 §5 requires 3–7 with algorithm + dataset + official docs. Apply the stricter 3–7 safeguard.
4. **Deadline/submission channel**: H6 defers them to the instructor. Not assumable — PENDING row; confirm before submission.
5. **Report structure**: G's sample ordering differs from G0 §3 structure; G0 structure is mandatory (§2, §6, references never omitted — HW4 zero-score lesson).
6. **I5 (HW5 Recall clarification)**: applies to week5 semantics; H6 §5/§7.2 define a different, explicit candidate/exclusion protocol for HW6 (training-movie pool for both ranking and relevance; split-specific previously-rated exclusions). No contradiction to resolve; do not import HW5's metadata-wide ranking into H6's `recommendTopK` contract — H6 §5 governs.

## D. REQUIREMENTS_MATRIX

Report-section entries are required destinations, not claims that a report exists. Personal verification is PENDING until the student supplies actual evidence in chat; agent checks never fill this column.

| Requirement | Source | Code evidence | Report section | Personal verification | Status |
|---|---|---|---|---|---|
| R01 Read G0, current matrix, guidelines, all feedback, assignment before coding/report | U; G0 §1 | All sources in table A read this session | Audit record | N/A, audit workflow | PASS (available local sources; gaps listed) |
| R02 Show fresh matrix before implementation or final report | U; G0 §1 | This file created; presented in chat | Audit record | Student reviews matrix | PASS (documentation gate) |
| R03 Use a06 branch; inspect status/diffs; preserve work; no commit/push | U; G0 §1 | Branch/HEAD/status verified; week6 == upstream/main; nothing committed | Audit record | N/A | PASS (observed git scope) |
| R04 No package installs; environment-change requests answered explicitly | U | No install attempted; TF.js comes from CDN (`index.html:192`, v4.22.0), d3 vendored | §3 | N/A | PASS (no change needed; REQUIRES ENVIRONMENT CHANGE not triggered) |
| R05 Read the ENTIRE starter; cover all major defects, not the first one | U; G0 §8; F1 §2; F2 §2; F3 §2 | graph.js/script.js/data.js/index.html read end-to-end; genre parser already fixed (`fields.slice(6,24)`, 18 names — inherited fixed week5 code, so the HW2/HW3 defect is absent); no other defect proven yet; runtime paths unexercised | §3, §5 | Student reviews defect conclusions | PENDING (static read done; runtime/defect conclusion outstanding) |
| R06 Preserve datasets and prior weeks; no scaffold rewrite; do not modify week6 unless a defect demands a documented fix | U; H6 §2; G0 §8 | u.data/u.item SHA256 byte-identical to week5; data.js byte-identical to week5; no file modified | §3 | Student checks hashes/files | PENDING (byte-identity agent-verified; student confirmation outstanding) |
| R07 Report actual loaded counts, sparsity, positive train edges, zero-degree users/movies, split records | H6 §2, §3.1 | Split logic in data.js; independent Node recount: 943 users / 1682 movies / 100000 ratings; train 79619, val 9596, test 10785, excluded 0 | §4 | Student reproduces counts in browser | PENDING |
| R08 Missing ratings = unobserved, never zero; chronological per-user 80/10/10 split disclosed as not globally time-consistent | H6 §2 | data.js `splitByTimestamp` implements per-user floors/tie-break; matrix zeros used only as "no positive edge" (H6 §3.1 wording echoed in graph.js JSDoc) | §3 | Student checks one user's split boundaries | PENDING |
| R09 Graph from TRAIN rating>=4.0 only; bidirected user-movie edges; no user-user/movie-movie edges; explicit contiguous ID map; no padding node | H6 §3.1 | Stub `buildGraph` (graph.js:336-359) JSDoc restates contract; harness check 11 pins node count = observed users+movies | §3 | Student inspects graph node counts after build | PENDING |
| R10 Transition row-normalized; `p_next = alpha*P^T*p + (1-alpha)*v`; alpha = link-following everywhere; transpose does not reverse movement | H6 §3.2; G0 §7 checklist | Stubs `rowNormalize` (graph.js:365-392), `pageRank` (398-442); harness check 9 verifies orientation (hub mass > movie mass, symmetric tie) | §3 | Student reads one hand-computed update | PENDING |
| R11 Zero-degree policy: no divide-by-zero, no silent mass loss; `p_next = alpha*(Q^T*p + d*v) + (1-alpha)*v`; same rule for PR and PPR; counts reported | H6 §3.3 | `rowNormalize` returns `{Q, danglingMask}`; harness checks 3 and 8 (zero-degree fixture, danglingMask [false,true,false,false,true], mass 1 both paths) | §3, §4 | Student inspects check 8 output | PENDING |
| R12 Global PR: uniform v; PPR: one-hot v at user; init p=v; stop at L1<1e-6 or 500 iters; report iterations/residual/converged; cap != convergence; no forcing score=restart | H6 §3.4 | Stubs `pageRank`/`personalizedPageRank` JSDoc; `hw6ConvergenceVerdict` + convergence records exported to results.csv (iterations/residual/converged columns) | §3, §4 | Student reads a convergence record from the page | PENDING |
| R13 TF.js core propagation: batched tf.matMul/tidy, await tf.ready(), report backend, no per-edge loop replacement, no async inside tidy, no tensor accumulation across runs; English JSDoc | H6 §3.5 | TF.js 4.22.0 CDN; `hw6InitTf` awaits ready and records backend; harness summary prints backend; student stubs are the propagation entry points | §3, §4 | Student observes backend line in browser | PENDING |
| R14 Reference fixture values verified BEFORE MovieLens experiments: uniform [0.3725268513, 0.1958239118, 0.3941492369, 0.0375] C>A>B>D; restart-A [0.4522328999, 0.1921989825, 0.3555681176, 0] A>C>B>D | H6 §4.1 | Harness checks 1-2 compare student output to readme values at 1e-6; REFERENCE_GRAPH carries them; agent independently reproduced both fixed points by 5000-step power iteration (max abs diff 4.3e-11 = rounding) | §4 | Student runs harness checks 1-2 in browser and compares to readme table | PENDING (agent math verified; student/browser check outstanding) |
| R15 Additional tests reported honestly: rows sum 1, restarts nonnegative/sum 1, mass preserved, nonnegative scores, stationarity of one extra update, zero-degree mass fixture, direction/orientation, filtering — "do not claim checks you did not run" | H6 §4.2 | Harness checks 3-9 cover each named property (12 checks total) | §4 | Student quotes actual per-check PASS/FAIL/PENDING lines | PENDING |
| R16 Candidate pool = movies in TRAIN split; TEST excludes train+val rated (any value); VAL excludes train-rated; sort score desc, tie by movie id asc; up to 10; report <10 eligible and zero-score inclusion; PPR scores labeled walk probabilities only | H6 §5 | Stub `recommendTopK` (script.js:1120-1152) contract; `hw6ExclusionSet` (668-677) implements split-specific exclusions; harness check 10 verifies exclusion/sort/tie-break | §3, §4 | Student intersects a Top-10 with the user's seen list | PENDING |
| R17 Four methods under shared protocol: popularity (distinct train users rating>=4), global PR uniform, PPR seeded at user, MF as provided; no MF redefinition to resemble PPR; model/checkpoint stated | H6 §6; script.js:30-36 | Provided `popularityScores` (648), `mfScores` (1052), lazy deterministic MF (MF_SEED 20261013, epochs 30, lr 0.01, reg 0.05, batch 512, MF_INFO exposes factors/epochs/checkpoint/valRMSE) | §3, §4 | Student reads MF_INFO line from harness output | PENDING (contract resolved per conflict 2; usage not yet run) |
| R18 Alpha selection: sweep {0.50,0.70,0.85,0.95} on VALIDATION Recall@10 only; one shared alpha; ties -> smaller alpha; same alpha reused for global PR with disclosure; no test tuning; MF keeps validation-RMSE checkpoint rule; objectives differ (disclose) | H6 §6.1 | `ALPHA_GRID` (script.js:104); alpha-sweep runner (1787-1800) records val rows in results.csv (`split=val`); no test outcome path into selection | §3, §4 | Student verifies sweep table shows split=val and selected alpha | PENDING |
| R19 One-factor-at-a-time demo: two fixed cohort users; three settings (A/0.85 baseline, B/0.85 seed-only change, A/0.50 alpha-only change); never attribute a change after changing both | H6 §6.2 | Demo runner (script.js:1843-1908) uses cohort[0]=A, cohort[1]=B with exactly those three settings; renderer at 2238 | §4, §5 | Student reads the three rendered lists | PENDING |
| R20 Shared evaluation cohort for all four methods: positive train edge + MF-scorable + >=1 eligible relevant movie; exclusions disclosed with counts; warm-user scope stated | H6 §7.1 | `hw6SharedCohort` (694-711) and `hw6CohortExclusions` (722-759) implement the intersection and counts (excludedUnseen / excludedNoEligible / excludedNoPositiveEdge / unseenIds) | §4 | Student confirms cohort size matches across method rows in results.csv | PENDING |
| R21 Recall@10 = relevant eligible / all relevant eligible; relevant = split rating>=4 + in train pool + passes exclusion; macro-average; report cohort size, exclusions, unseen-ID count, users w/o eligible relevant; no PPR RMSE | H6 §7.2 | Stub `recallAtK` (1158-1185); RecallReport typedef carries all required counts; results.csv columns cohort_size/excluded_* | §4 | Student hand-computes one user's recall numerator/denominator | PENDING |
| R22 Popularity evidence: popularity from TRAIN>=4 only; every Top-10 shows per-movie popularity + mean; two-user common-movie counts per method; no platform-wide-impression or dislike claims | H6 §7.3 | `HW6_STATE.popularity` + `popularityScores()`; recommendations.csv has `training_popularity` column; results.csv has mean_popularity | §4 | Student spot-checks one movie's popularity against train data | PENDING |
| R23 Exactly ONE business question: keep MF / replace with PPR / use both — one strategy, supported by cohort Recall@10, both Top-10s with titles/genres, training histories, popularity, one-factor effects; name sections if "both"; state what offline cannot establish (intent, completion, retention, revenue); one online experiment with intervention, primary outcome, guardrail, user-level assignment | H6 §8 | Analysis is report-only; no code obligation; UI provides all required evidence inputs | §5 | Student explains the chosen strategy from actual lists | PENDING (no report authored) |
| R24 In-page self-checks reported with actual counts (12 checks; stub baseline vs final run) | F4 §5; H6 §4.2 | `runSelfChecks` (script.js:2868-3192): 12 checks; TODO(hw6) and interface issues become PENDING, never PASS; serializeVerification exports counts | §4 | Student personally reads the self-check log in browser | PENDING |
| R25 Deliverables: app + TF.js propagation, README with run instructions/versions, verification.json, results.csv, recommendations.csv, report.pdf, native session; disclose incomplete/failed/nonconverged work | H6 §9 | Exporters exist: serializeVerification/serializeResults/serializeRecommendations (3205/3243/3330); HTTP run instructions in readme §"Local run instructions"; README.md not yet written | All | Student checks each file exists and matches run | PENDING |
| R26 Preserve original dataset/previous assignment files; state any unavailable prior MF output | H6 §2, §9 | No week5 file touched; provided MF makes "unavailable prior MF output" moot but disclosure row remains | §3, §4 | Student diff check | PENDING |
| R27 Evidence integrity: cite dataset; verify paper titles/authors/years/URLs; tables match actual exports; measured vs proposed separated; no fabricated runtimes/results | H6 §10; G0 §10 | Report-only obligation; Harper & Konstan 2015 citation named in H6 §10 must be validated in primary record before use | §2, §4, References | Student verifies one citation in primary source | PENDING |
| R28 Report structure: title block, abstract 50-150w (4 sentences incl. one verified number), §1, §2 Related Work (researched, 1-3 alternatives), §3 method w/ real versions, §4 experiments w/ hardware/backend/deliverables, §5 5-Loop discussion with real failure, §6 AI disclosure, numbered references | U; G0 §3; HW4 zero-score lesson | No report exists; none may be authored until prerequisite rows PASS | Whole report | Student checks structure against G0 §3 | PENDING |
| R29 3-7 numbered references: algorithm source (PageRank/PPR), dataset provenance (Harper & Konstan 2015 + GroupLens), official TF.js docs actually used; every author/year/venue/DOI/URL validated against primary records; sections actually read | G0 §5; F4 §3 | Not started; do not inherit starter citations unvalidated | §2, References | Student reads one source section beyond abstract | PENDING |
| R30 §6 AI disclosure: tools/versions/models from inspectable records; how each used; "What I personally verified" (exact quantities/actions); "What I trusted without verification" (scope); native session filename + notes | G0 §6; HW4 c2 lesson | OpenCode (opencode/big-pickle) used for this audit; other tools must be listed from actual records when report is written | §6 | Student confirms actual tool list | PENDING |
| R31 Personal verification: student hand-computes one critical metric; performs one browser/result reproduction; does one code/output cross-check; engages one derivation/assumption | G0 §7; F2 §4; F3 §4; F4 §5 | VERIFICATION_LEDGER.md initialized with HW6 rows, all NO | §§4/6 | REQUIRED; not yet performed | PENDING |
| R32 Manual verification script presented in chat BEFORE final report; only student-confirmed YES rows may appear in §6 | G0 §7 | Script to be presented when implementation is ready; ledger exists with NO rows | §§4/6 | Student performs and reports each check | PENDING |
| R33 Native session: unmodified plaintext tool export via `opencode export`, raw prompts/tool calls/results, covers the actual coding work, parseable, consistent with §6; never edited/reconstructed | G0 §11; F4 §4; F1 §4 | Prior HW5 export precedent exists; HW6 coding session not yet created | §6, attached JSON | Student exports and confirms | PENDING |
| R34 Genuinely observed failure: real error/log excerpt, root cause, detection→change→confirmation, before/after, what worked/failed/surrounded, next step; distinguish observed from hypothetical | G0 §9; F3 §3 | No HW6 failure observed yet (nothing has been run) — must not reuse an HW5 failure | §5 | Student verifies the fix personally | PENDING |
| R35 Every reported number matches final code/browser output at same precision; retain config/code revision/backend/run identity; report negative results and limitations; no cherry-picking | G0 §10 | No results exist yet | §4, §5 | Student cross-checks report vs run | PENDING |
| R36 Submission directory, deadline, grading guide, AI-export requirement confirmed with instructor before publishing; exact naming `<id>_<assignment>_report.pdf` + session | H6 §9; G0 §12 | Not announced locally; no deadline assumable | Submission manifest | Student confirms course announcement | PENDING |
| R37 Report contains clickable student repo link + a06 branch/path + verified implementation commit; publication only of finished, verified artifacts on authorized branch | G0 §12 (publication authorization precedent was HW5-specific; a06 push needs authorization) | origin = samgopiter-byte/RecSys-LLMs; branch a06 exists locally; nothing published | §3, submission | Student inspects link after push | PENDING |
| R38 Final pre-submission audit printed: PASS counts, page count, references, §2/§6 presence, personal verification, session validation, code/tests, number consistency, named bullets, GATE-0/MINOR RISKS | G0 §12 | Cannot run before prerequisites; current risks open | Final gate | Student review | PENDING |

Mandatory PASS count at this pre-implementation checkpoint: **5/38** (R01-R04, R06 partially? No — R06 is PENDING. Count: R01, R02, R03, R04 = 4... plus R05? No, PENDING). Recount: PASS = R01, R02, R03, R04 → **4/38**, all workflow/documentation scope. No implementation, runtime, report, session or personal-verification row is PASS.

## E. Missing evidence for current HW6

- All seven `TODO(hw6)` stubs unimplemented (graph.js: buildGraph, rowNormalize, pageRank, personalizedPageRank; script.js: movieScores, recommendTopK, recallAtK) — page throws until written; harness will show PENDING-heavy baseline.
- No runtime evidence: no graph built, no backend observed, no self-check run, no alpha sweep, no cohort/Recall numbers, no Top-10 lists, no MF checkpoint observed.
- No README.md, verification.json, results.csv, recommendations.csv produced (exporters exist but nothing has been run).
- No failure/fix history yet; no report draft; no HW6 session export; no student personal verification (ledger all NO); no confirmed deadline/channel; no validated references; no HW6 lecture found.

## Pre-submission checkpoint

- Mandatory requirements: 4/38 PASS (workflow scope only).
- Report page count: unavailable (no PDF).
- References validation: PENDING (no reference list).
- §2/References and §6 present: PENDING (no draft).
- Personal verification: PENDING (ledger all NO).
- Native session validation: PENDING (no HW6 coding session).
- Code/tests: static audit done; 7 stubs throw; runtime tests not run; no proven defect closed out.
- Report/code number consistency: PENDING (no results).
- GATE-0 RISKS: OPEN — unimplemented required functions; any submission now would lack the entire deliverable set.
- MINOR RISKS: OPEN — report sections, references, disclosure, personal checks not started.

**SUBMISSION BLOCKED** as a status of readiness, not as a workflow stop: implementation work is the next authorized step, and no report may be authored until the prerequisite rows PASS.

## 2026-10-17 focused performance/runtime update (supersedes historical statements above)

Current tracked scope: week6/graph.js +382/-25 and script.js +116/-6; no staged change. This audit does not certify the complete starter or finish all 38 obligations. No final mandatory PASS count is claimed while report and experiment evidence remain incomplete.

| Requirement | Source | Code evidence | Report section | Personal verification | Status |
|---|---|---|---|---|---|
| R05 Entire starter and major defects | AGENTS §8; F1-F4 | Seven functions are implemented; runtime and provided-runner defects found; full final audit still needed | §3, §5 | Self-check inspection only confirmed | PENDING — no blanket starter PASS |
| R07 Dataset/graph statistics | HW6 §§2–3 | Current agent run: 943/1682/100000; split 79619/9596/10785; graph 2554, 45602 edges, zero-degree 1/253 | §4 | Full counts not separately confirmed in this chat | PENDING — runtime evidence exists; final browser/report evidence outstanding |
| R13 TF.js propagation and responsiveness | HW6 §3.5; student lag request | tf.matMul used; synchronous dataSync per iteration and full-cohort loops block event processing; tensor sample 0→0 | §3, §5 | Student reports lag; after-fix check outstanding | FAIL — blocking path requires repair; no tensor leak seen in sampled run |
| R18 Validation-selected alpha governs test | HW6 §6.1 | Controlled probe: test uses .85, returned best=.50, slider remains .85; best was injected fixture, not MovieLens result | §3–4 | Not yet checked personally | FAIL — test-before-selection flow and unapplied best |
| R20–R21 Shared cohort and Recall | HW6 §7 | val 846/test 861; popularity test JS .07602291343770759 agrees with independent Python; PPR/MF full comparison not run | §4 | Hand calculation pending | PENDING — partial agent evidence only |
| R24 Required 12 self-checks | HW6 §4; F4 §5 | Current-source Node CPU 12/0/0; student supplied WebGL 12/0/0 | §4 | YES for the cited pre-optimization run only | PASS — self-check count, not full assignment compliance |
| R25 Final exports and README | HW6 §9 | No final full-cohort run/exports validated; audit JSON is not a deliverable substitute | §4, artifacts | Not confirmed | PENDING |
| R31–R32 Manual protocol | AGENTS §7 | Ledger M4 now YES; manual script prepared in this chat; remaining rows NO | §4, §6 | Only M4 confirmed | PENDING — numerical and code-output checks still needed |
| R34 Real failure/fix evidence | AGENTS §9 | Lag reported and blocking/selection defects reproduced; fix not yet implemented | §5 | Before lag reported, after not confirmed | PENDING |
| R33/R35–R38 Final evidence gates | AGENTS §§10–12 | No final session/PDF/results certification attempted | §6, final gate | Pending | PENDING |
| P01 Responsive progress and cancellation | Student request 2026-10-17 | Existing PPR runs lack progress/cancel; add measured work counters and yielding | §5 | After repair | PENDING |
| P02 Semantics-preserving acceleration | Student request; HW6 §§3–7 | One-slot cache, R rebuild per user, repeated recommendation passes; optimization prompt prepared | §3, §5 | After repair | PENDING |

No new specification conflict is needed to explain F1: HW6 explicitly requires validation selection before final test evaluation. The previous MF provenance tension remains disclosed; HW5's individual answer is not a universal authority for HW6. This review does not change MF.

**SUBMISSION BLOCKED** — repair selection flow and responsiveness; complete final experiments, exports and required personal verification before report finalization. GATE-0 RISKS: OPEN. MINOR RISKS: OPEN.


## 2026-10-17 post-OpenCode checkpoint (supersedes previous performance checkpoint)

See audit/HW6_POST_OPENCODE_REVIEW_2026-10-17.md and its JSON evidence. No final authoring permitted. This is a focused regression checkpoint, not all 38 mandatory rows certified.

| Requirement | Source | Code evidence | Report section | Personal verification | Status |
|---|---|---|---|---|---|
| R13 No tensor accumulation | HW6 §3.5 | Mixed-stopping fixture: 5 -> 269 -> 533 -> 797 tensors | §3–5 | Agent only | FAIL |
| R18 Complete validation selection before test | HW6 §6.1 | Happy-path order repaired; failed alpha .70 still allows best .50 in injected fixture | §3–4 | Not performed | FAIL |
| R20–R21 Cohorts and Recall | HW6 §7 | Current CPU: val846/test861; popularity .07602291343770759, user1 2/19; full four-method run outstanding | §4 | Not performed on final code | PENDING |
| R24 Provided harness | HW6 §4 | Independent current-source Node CPU 12/0/0 | §4 | Previous WebGL confirmation predates changes | PASS — agent runtime scope only |
| R25 Complete consistent exports | HW6 §9 | Cancel retains unlabelled old CSV rows; Evaluate alone does not populate recommendationRows | §4/artifacts | Not performed | FAIL |
| R33 Native session | AGENTS §11 | alex_a06_session_performance_native.json byte-identical to fresh CLI export; 290 messages/307 tool records; covers current code hashes | §6 | Student says exported | PASS for this revision; repair requires new export |
| R31–R32 Post-change manual verification | AGENTS §7 | Student explicitly asks agent first and will verify later | §6 | NO for all six new checks | PENDING |
| P01 Safe responsive UI | Student request; HW6 §3.5 | Run self-checks remains enabled and disposes active tensors; injected interleaving crashes | §5 | Browser unavailable to agent | FAIL |
| P02 Warm run avoids redundant propagation | Student request | Identical cached prefill invokes kernel again | §4–5 | Not performed | FAIL |
| P03 Progress/timings | Student request | Chunk-boundary-only updates; MF cancel not wired; no retained final run timing | §4–5 | Not performed | PENDING |
| R28 Report structure | AGENTS §3; HW5 template | Root HW5 PDF physically 5 pages, required sections found; HW6 not authored | Whole report | Later audit candidate | PENDING |
| R29/R34–R38 Remaining final gates | AGENTS §§5/9–12 | Real failure evidence retained; fix/full final results/references/report still outstanding | Whole report | Pending | PENDING |

SUBMISSION BLOCKED. GATE-0 RISKS: OPEN. MINOR RISKS: OPEN. Do not claim the assignment solved based solely on 12 self-checks.


## 2026-10-18 repair verification (supersedes prior repaired-defect statuses)

| Requirement | Source | Code evidence | Report section | Personal verification | Status |
|---|---|---|---|---|---|
| R13 Mixed-column tensor lifecycle | HW6 §3.5 | Independent fixture 5->5->5->5 tensors | §3–5 | Agent only | PASS for reproduced leak; full-run lifecycle still to inspect |
| P02 Repeat prefill cache | Student request | Same users/alpha: no new kernel; reduced real PPR/PR check agrees | §4–5 | NO | PASS for tested reuse scope |
| P01 Conflicting actions | Student request | Disabled self-check and guarded handler preserve graph | §5 | NO | PASS for self-check conflict regression |
| R24 Harness | HW6 §4 | Current CPU12/0/0 | §4 | NO on this revision | PASS for agent harness |
| R18 Complete valid selection/evaluation | HW6 §6.1 | Failed alpha blocked; null Recall/incomplete method arrays still accepted by completion helpers | §3–4 | NO | FAIL |
| R25 Final exports | HW6 §9 | results uses snapshot; recommendations reads live rows under completed ID | §4/artifacts | NO | FAIL |
| P03 Progress accuracy | Student request | 100% at iteration1 for2 active users; elapsed summary added | §5 | NO | FAIL |
| R25 Runtime README | HW6 §9 | Local-run section lacks actual dependency versions, unchanged | Artifacts | NO | PENDING |
| R33 Native session | AGENTS §11 | Repair JSON byte-identical fresh export;481messages/502tool records | §6 | Export supplied by student | PASS for current revision |
| R20–21 Final full comparison | HW6 §7 | Cohorts846/861, popularity verified; no full WebGL final results | §4 | NO | PENDING |
| R28–32/R34–38 Report and final gates | AGENTS | HW5 template inspected; personal final checks/report not complete | Whole report | NO for new six checks | PENDING |

SUBMISSION BLOCKED. GATE-0 RISKS: OPEN. MINOR RISKS: OPEN. Current evidence in audit/HW6_REPAIR_REVIEW_2026-10-18.md; diagnostic results are not final experiment exports.


## 2026-10-18 after-restart checkpoint

| Requirement | Source | Code evidence | Report section | Personal verification | Status |
|---|---|---|---|---|---|
| R24 self-checks and numerical regressions | HW6 §4 | Current CPU12/0/0; popularity .07602291343770759; cohorts846/861 | §4 | NO for latest revision | PASS for agent test scope |
| R13/P01/P02 memory/cache/guards | HW6 §3.5; student | Previously failing fixtures remain fixed | §3–5 | NO | PASS for tested regressions |
| P03 progress accuracy | Student | During first iteration0% (0/2), formerly100% | §5 | NO | PASS for counter fixture; browser pending |
| R18 valid complete results | HW6 §6.1 | Full valid-shape arrays still accept null/empty/out-of-range Recall and wrong selected alpha | §3–4 | NO | FAIL |
| R25 immutable recommendation exports | HW6 §9 | Producer stores object with rows; exporter expects array, falls back to live rows; mutation reproduced | §4/artifacts | NO | FAIL |
| R25 README | HW6 §9 | Versions added; incorrect D3 relative path and file:// advice | Artifacts | NO | FAIL |
| R33 native export | AGENTS §11 | After-restart JSON byte-identical fresh export;547messages/564tool records | §6 | Export supplied | PASS for current revision |
| Final experiment/report/personal gates | AGENTS §§7/10–12 | Full WebGL run and final report outstanding; browser runtime unavailable | Whole report | Six checks still NO | PENDING |

Authoritative new regression: audit/HW6_FINAL_CONTRACT_REGRESSION_2026-10-18.json. No code edits by report assistant. SUBMISSION BLOCKED; GATE-0 RISKS: OPEN; MINOR RISKS: OPEN.


## 2026-10-19 verified repair checkpoint (current source; supersedes reproduced-defect FAIL rows above)

| Requirement | Source | Code evidence | Report section | Personal verification | Status |
|---|---|---|---|---|---|
| R13/P01/P02/P03 reproduced memory/cache/guard/progress defects | HW6 §3.5; student | Repeated fixture memory5->5, cache one call across repeat; prior guard/progress fixes retained | §3–5 | NO on latest revision | PASS for tested regressions; full browser still pending |
| R18 invalid numeric and incomplete outcomes rejected | HW6 §6.1 | HW6_CONTRACT_VERIFICATION_2026-10-19.json: valid accepted; seven invalid probes rejected | §3–4 | NO | PASS for reproduced validation defects; full selection run pending |
| R25 immutable recommendation export regression | HW6 §9 | Actual producer->snapshot->export100 synthetic rows, live mutation does not affect export | Artifacts/§4 | NO | PASS for snapshot regression; full real exports pending |
| R25 runtime README | HW6 §9 | tfjs4.22.0/d3v7.9.0, correct asset/HTTP/demo instructions | Artifacts | NO | PASS for corrected documentation |
| R24 current harness | HW6 §4 | Node CPU12/0/0 | §4 | NO current revision | PASS for agent harness |
| R33 native session | AGENTS §11 | Oct19JSON byte-identical native export725messages/757tools | §6 | Export supplied by student | PASS for current code coverage |
| R20–21 final four-method full cohort results | HW6 §7 | val846/test861 and popularity verified; complete WebGL run absent | §4 | NO | PENDING |
| P04 six browser checks/first-repeat durations | Student request | Browser tool fails at startup; no browser behavior claimed | §4–6 | NO | PENDING |
| R28–32/R34–38 report/citations/personal/submission | AGENTS §§3–12 | Failure/fix evidence exists; final-run, personal, references and artifact gates incomplete | Whole report | NO for six final browser checks | PENDING |

See audit/HW6_CHECKPOINT_2026-10-19.md. No final mandatory PASS count claimed. SUBMISSION BLOCKED until remaining prerequisites and final artifact audit pass.

Connected-path regression on2026-10-19: exit0,0assert failures; selected-alpha application, repeat-sweep reuse and cancel/repeat tensor cleanup verified on8-user subset with MF substitute. Full861-user four-method browser evidence remains PENDING.

## Fresh pre-report compliance checkpoint — 2026-10-19, after student verification

This checkpoint supersedes older claims that all MF-related concerns were resolved. No final report authored; no application source edited. Source hashes match the Oct19 tested revision. Native session hash remains fff5417ce28652c34530a4ba9b5cccb5335b40254ae70fd66c1664af1738cf10 and parses with 725 messages.

| Requirement | Source | Code evidence | Report section | Personal verification | Status |
|---|---|---|---|---|---|
| Full WebGL evaluation and self-checks | HW6 sections 4,6,7 | Student page evidence; tmp/verification.json 12/0/0; tmp/results.csv all four methods, alpha .50, val846/test861 | 4 | Guided runs confirmed in ledger | PASS for execution and displayed/export consistency |
| Numerical, browser and code/output personal checks | AGENTS section 7 | User1 hits2/relevant19; code hit/relevant.size; CSV PPR .11218853219486066 | 4,6 | Explicit calculation and code/page/CSV confirmations | PASS for required categories |
| Cancel, restart, slider isolation, repeated run | Student requested verification | Cancel state and restart complete; .70 demo slider leaves .50 results; successful run95.7s, repeat.2s | 4,5,6 | Confirmed guided actions; exact live progress counters not confirmed | PASS for observed behaviors; no claim all progress fields personally verified |
| Export lists, genres, popularity, seen exclusions | HW6 sections 5,7,9 | audit/HW6_FINAL_LIST_AUDIT_2026-10-19.json: 100 rows, no raw-data cross-check errors; overlap counts per method1/1/1/2 | 4,5 | Student exported files; PPR metric row personally compared | PASS for scoped automated cross-check |
| MF comparator health and provenance | HW6 section 6; starter provided-MF restriction; AGENTS full-starter audit | All20 exported MF scores equal float32 training mean3.5806026458740234. Current raw-L2 objective differs from submitted HW5 reg/batch objective. Root cause not yet measured | 3,4,5 | No healthy MF claim confirmed | BLOCKED pending targeted diagnosis and source resolution |
| References and related work | AGENTS sections 3,5 | Dataset README and TFJS versioned docs read; algorithm primary source validation incomplete | 2,References | Not personally source-verified | PENDING |
| PDF and disclosure | AGENTS sections 2–6 | HW5 five-page structure reviewed; no HW6 PDF generated | All | Report proofreading not yet possible | PENDING |
| Submission artifacts and repository link | HW6 section 9; student current instruction | Native JSON valid; exports present; code uncommitted on a06 branch; student confirms GitHub link required | Reproducibility | Student confirms PDF+JSON and link | PENDING publication/link and final artifact gate |

MF source interpretation correction: an instructor clarification explicitly about HW5 Recall does not grant blanket permission to apply every starter comment over the HW6 assignment. The earlier matrix's broad transfer was unsupported. Preserve its history but do not treat it as authoritative resolution of MF provenance. Assignment section6 has priority over starter commentary; any claimed reuse must be verified, and any forbidden necessary edit must be surfaced.

SUBMISSION BLOCKED. GATE-0 RISKS: OPEN. MINOR RISKS: OPEN. Do not infer that constant MF scores prove all candidate scores are equal or that raw L2 is the proven cause; see focused OpenCode diagnostic prompt.

## MF patch review — 2026-10-20

| Requirement | Source | Code evidence | Report section | Personal verification | Status |
|---|---|---|---|---|---|
| Normalize MF L2 by actual batch size | HW5 loss convention; approved focused repair | script.js:1149 uses reg / size; controlled coefficient checks include partial batch276 and pass; node syntax checks pass | 3,5 | New full run not yet performed | PASS for patch application; runtime effect PENDING |
| Preserve PPR/UI and previous export evidence | Student scope | graph.js/index.html/style.css SHA256 unchanged; existing three tmp exports unchanged; HTTP serves the repaired script | 3,4 | Earlier scoped checks retained as historical evidence | PASS for inspected files |
| Corrected final MF and shared-comparison evidence | HW6 sections6–7 | No full WebGL run or exports for new script hash45d79c3c2da045bc0615d264bc8f6b5651e8528dece28d1ecc04f604ee231467 yet | 4,5 | PENDING | PENDING |
| Native session covers repair | AGENTS section11 | Live native export has832messages and completed final turn; saved793-message export predates patch | 6 | Final export needed after work | PENDING saved final export |

Do not require a particular checkpoint epoch, RMSE improvement or test Recall as a manufactured PASS target. Check finite learning, score variation, validation-based selection/restoration, and report the actual measured outcome. The model still differs from saved HW5 run configuration; do not claim identical trained-model reuse. Prior full-run numbers and timings are historical, not final numbers for this patched revision. Report authoring/final submission remain gated by the outstanding requirements.

## Corrected export checkpoint — 2026-10-20

| Requirement | Source | Code evidence | Report section | Personal verification | Status |
|---|---|---|---|---|---|
| Corrected full evaluation | HW6 sections6–7 | WebGL;12/0/0; MF0.054668358285198206; alpha0.50;861 test users | 4 | Student full page and export delivery | PASS for final-run evidence |
| Recommendation export integrity | HW6 sections5,7,9 | audit/HW6_FINAL_EXPORT_AUDIT_2026-10-20.json:100rows,10lists,no errors; MF10 distinct scores/user | 4–5 | Export performed; detailed data cross-check automated | PASS for tested export scope |
| Native session coverage | AGENTS section11 | golankov_alexander_a06_session.json; direct native export and coverage metadata in audit/HW6_SESSION_EXPORT_CHECK_2026-10-20.json | 6 | Export performed by assistant with native CLI | PASS for parseability/native provenance and patch presence; final disclosure consistency pending |
| Final references, report and publication | AGENTS sections2–6,12 | Final PDF and publication audit outstanding | All | Earlier confirmed personal checks retained | PENDING |

SUBMISSION BLOCKED until report prerequisites and final artifact/publication checks pass. This checkpoint does not certify every assignment requirement.

## Pre-authoring checkpoint — 2026-10-20, final packaging request

This supersedes any interpretation that the one-line MF normalization repair alone proves literal reuse of the previous selected model. User explicitly authorizes commit/push of finished HW6 artifacts and will personally upload the PDF/JSON to the course service. No LMS upload by the assistant is required; no deadline is invented.

| Requirement | Source | Code evidence | Report section | Personal verification | Status |
|---|---|---|---|---|---|
| Final repaired graph/evaluation/export evidence | HW6 sections3–7,9 | Final script hash45d79c3c; WebGL12/0/0;100 exported recommendations cross-checked; run-1-1 | 3–5 | Confirmed browser run and earlier arithmetic/code check in ledger | PASS for inspected runtime and export scope |
| Reuse previous MF implementation and selected model | HW6 section6 | Previous branch uses clipped validation RMSE,20epochs/batch64; current comparator uses unclipped validation RMSE,30epochs/batch512 and seeded initialization. No persisted previous weights found in branch or current filename inventory. L2 repair alone does not resolve this. | 3–5 | New run verified, exact prior-model reuse not verified | BLOCKED pending resolution; disclosure alone is not literal reuse |
| Validated references | AGENTS section5 | audit/HW6_REFERENCES_VALIDATION_2026-10-20.md:4primary sources, relevant sections read | 2,References | Not claimed personally read | PASS for source validation; PDF citations pending |
| Normal HW6 commit/push authorization | User reply2026-10-20 | Explicit instruction to commit/push so report link contains the files | Reproducibility | User authorizes | PASS for authorization, publication still pending |
| Course-service upload | User reply2026-10-20 | User will upload the two files personally | Delivery | User owns upload | NOT APPLICABLE to assistant upload; no deadline assumed |
| PDF structure,page count,number consistency and final public links | AGENTS sections2–6,12 | Not yet authored/finalized; prerequisite conflict remains | All | Candidate proofreading later | PENDING |

SUBMISSION BLOCKED. The report may not assert GATE-0 RISKS:NONE or MINOR RISKS:NONE while the MF reuse requirement remains unresolved. No new full rerun or code mutation is authorized by this checkpoint alone; preserve current measured evidence.

## Explicit user decision to proceed — 2026-10-20

The user selected "Describe the deviation in the report" after the MF reuse difference was explained, then requested report creation. This explicitly overrides the stop-authoring workflow for that disclosed deviation; it does not establish instructor acceptance or erase the difference. The user also authorized normal HW6 commit/push and will upload PDF/JSON personally. Preserve the factual comparison and do not claim MINOR RISKS:NONE.

| Requirement | Source | Code evidence | Report section | Personal verification | Status |
|---|---|---|---|---|---|
| Author report with disclosed MF difference | User explicit decision2026-10-20 | Current run and exports retained; no new code/training | 3,5 | Current run confirmed | PASS for authorization; course interpretation remains disclosed |
| Runtime evidence | HW6 sections3–7 | final export audit and WebGL12/0/0 | 3–4 | Ledger YES rows only | PASS for recorded checks |
| Citations | AGENTS section5 | Four verified primary sources | 2,References | Not claimed as student reading | PASS source gate |
| PDF and final artifact checks | AGENTS section12 | To inspect actual PDF | All | Final proofreading not yet confirmed | PENDING |
