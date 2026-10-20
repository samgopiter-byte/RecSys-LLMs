# HW6 / A06 personal verification ledger

Created 2026-10-16. Student-confirmed results only. The manual verification script must be presented in chat before the final report. Expected values are not observations. Updated 2026-10-17: M4 is confirmed for the pre-optimization self-check run from the student message cited below. Other checks remain unconfirmed. A later code change requires affected checks to be repeated.

| Check | Action performed by student | Observed result | Evidence/screenshot if available | Confirmed by student: YES/NO |
|---|---|---|---|---|
| M1 Hand-computed one PageRank/PPR update or stationary value (e.g. the 4-node reference fixture at alpha=0.85) compared with page output | Not yet confirmed | Not recorded | Not supplied | NO |
| M2 Reference-fixture harness checks 1-2 in browser vs readme §4.1 table (uniform [0.3725268513, 0.1958239118, 0.3941492369, 0.0375]; restart-A [0.4522328999, 0.1921989825, 0.3555681176, 0]) | Not yet confirmed | Not recorded | Not supplied | NO |
| M3 Final browser run: tf backend line, graph build node counts (users+movies, no padding), PR/PPR iterations/residual/converged actually displayed | Not yet confirmed | Not recorded | Not supplied | NO |
| M4 In-page self-check log personally read (pre-optimization run; no stub-baseline claim) | Student ran/inspected self-checks and pasted all 12 lines in Chat HW LLM REC | 12 PASS / 0 FAIL / 0 PENDING; backend webgl; MF reference not trained | Student turn e31906b5-2a78-44d4-bf34-1fe524fffd7f, retrieved 2026-10-17; current source hashes in audit/HW6_AGENT_CHECKS_2026-10-17.json | YES |
| M5 Split/graph arithmetic hand-checked: loaded counts, positive train edges (rating>=4.0), zero-degree users/movies, one user's train/val/test boundaries and exclusion set | Not yet confirmed | Not recorded | Not supplied | NO |
| M6 Two-user demonstration read in page: three settings (A/0.85, B/0.85, A/0.50), Top-10 titles vs seen IDs, per-movie popularity, common-movie counts | Not yet confirmed | Not recorded | Not supplied | NO |
| M7 Alpha sweep personally inspected: four validation Recall@10 values, selected alpha, split=val rows only; results.csv cross-read against the page | Not yet confirmed | Not recorded | Not supplied | NO |
| M8 Recall/cohort cross-check: cohort size identical across the four method rows; one user's numerator/denominator hand-computed; exclusion counts match the page | Not yet confirmed | Not recorded | Not supplied | NO |
| M9 One reference/source verified in a primary record (e.g. Harper & Konstan 2015 bibliographic data, or TF.js API used) by reading the source section | Not yet confirmed | Not recorded | Not supplied | NO |
| M10 Actual lag / alpha-selection failure and subsequent fix | Student reports severe lag; agent reproduced blocking and selection-flow defects; student before/after fix confirmation still required | Before: lag reported; after: not available | audit/HW6_PERFORMANCE_REVIEW_2026-10-17.md; no personal fix verification yet | NO |
| M11 Report numbers vs the identified final run (audit candidate phase, after M1-M10 exist) | Not yet confirmed; no report draft exists | Not recorded | Not supplied | NO |

Only an explicit student reply in chat reporting the performed action and actual observed result can change a row to YES. Partial confirmation records its exact scope; confirming one check never confirms another. Unperformed rows stay NO and their content may appear only under agent-verified findings or "What I trusted without personal verification". Rerun-to-identity is not required for learned metrics; retain one identified final run and compare against it. M10/M11 are later-stage; do not invent a failure or create a report to fill them.


## Post-optimization six-check protocol, 2026-10-17

Student clarification: "Пока проверь ты; я выполню затем". Thus no post-optimization check below is a student-performed result yet. Earlier "все что ты просил я проверил" is historical broad confirmation, not confirmation of these new actions or of unspecified numerical observations.

| Check | Action performed by student | Observed result | Evidence/screenshot if available | Confirmed by student: YES/NO |
|---|---|---|---|---|
| P1 WebGL backend and graph build | To perform after repairs | Not recorded | None | NO |
| P2 Progress, scrolling and Cancel during sweep | To perform after repairs | Not recorded | None | NO |
| P3 Complete Evaluate and selected-alpha stability after demo-slider change | To perform after repairs | Not recorded | None | NO |
| P4 Repeat Evaluate: cached sweep and identical metrics | To perform after repairs | Not recorded | None | NO |
| P5 Export and inspect verification.json/results.csv/recommendations.csv | To perform after repairs; generate required recommendation lists before export | Not recorded; 846/861 denote cohort sizes, not CSV row counts | None | NO |
| P6 Record real browser cold/warm timings and responsiveness | To perform after repairs | Not recorded | None | NO |


## Student confirmation — 2026-10-19, current manual verification

Supersedes the unperformed status only for the specific checks below. Student said they performed Build graph -> Run tests and saw the values below. Backend was not provided; agent browser read attempt failed at runtime startup. Do not infer WebGL from the historical run or record all six checks as completed.

| Check | Action performed by student | Observed result | Evidence/screenshot if available | Confirmed by student: YES/NO |
|---|---|---|---|---|
| P1a Current graph build and displayed counts | Student clicked Build graph and inspected graph counts | 2554 nodes;45602 positive interactions | Direct student confirmation in chat2026-10-19 | YES |
| P1b Current self-check reproduction | Student clicked Run tests and read PASS/FAIL/PENDING | 12 PASS /0 FAIL /0 PENDING | Direct student confirmation in chat2026-10-19 | YES |
| P1c Current backend | Not yet supplied; agent could not read page | Not recorded | Browser tool startup failed | NO |

P2-P6 remain pending. These browser observations do not establish a hand calculation or code-output formula check.

## Additional browser evidence — 2026-10-19

Student explicitly reports clicking Evaluate and supplied screenshot codex-clipboard-db5deba3-882c-4abe-991a-1a042a9a90ea.png. This records only that action and the visible screenshot, not completion of all six checks.

| Check | Action performed by student | Observed result | Evidence/screenshot if available | Confirmed by student: YES/NO |
|---|---|---|---|---|
| P3a Evaluate attempt | Student clicked Evaluate Recall@10 — all four methods | Screenshot shows Evaluation incomplete — not cached; elapsed 10.1 s; backend webgl | User-provided screenshot listed above; user confirmed click in chat | YES |
| P1c backend specifically verified in DevTools | No specific DevTools inspection confirmed | Agent can read backend webgl in student's screenshot; this is page evidence, not confirmation of a separate student console check | Same screenshot | NO |

P3 completion and P6 completed-run timing remain unconfirmed. The 10.1 s value is NOT a successful complete-run benchmark. P2/P4/P5 remain pending.

## Student-guided full and repeated Evaluate — 2026-10-19

Student supplied page text after each explicit manual instruction. This confirms those browser actions only, not all six checks. Evidence files preserve submitted text. No clean cold-start claim is made.

| Check | Action performed by student | Observed result | Evidence/screenshot if available | Confirmed by student: YES/NO |
|---|---|---|---|---|
| P3b Full Evaluate after Build graph | Student performed guided Build graph / Evaluate and pasted the resulting page | Complete, WebGL, selected alpha 0.50, elapsed 95.7 s; test Recall popularity 0.0760, PageRank 0.0797, PPR 0.1122, MF 0.0211; cohort 861, exclusions 13/82 | audit/HW6_STUDENT_EVALUATE_COMPLETE_2026-10-19.txt | YES |
| P4a Repeated Evaluate preserves displayed results and runs quickly | Student repeated Evaluate without rebuilding/reloading as instructed, then pasted page text | Complete in 0.2 s; all four displayed test recalls, mean popularities, validation recalls, selected alpha and cohort/exclusions unchanged | audit/HW6_STUDENT_EVALUATE_REPEAT_2026-10-19.txt | YES |

Validation recalls at alpha 0.50/0.70/0.85/0.95: 0.1145/0.1048/0.0900/0.0738, cohort 846. Displayed test mean popularities: 272.00/257.82/244.21/96.76 (popularity/PageRank/PPR/MF). Page text does not explicitly label cache reuse or count kernel/sweep executions: fast identical repeat is observed, absence of recomputation is not proven by text alone. Responsiveness, cancellation, slider stability, export inspection and hand calculation remain unconfirmed. Timings are displayed run durations, not isolated algorithm speedup measurements.

## Student demo-alpha isolation check — 2026-10-19

| Check | Action performed by student | Observed result | Evidence/screenshot if available | Confirmed by student: YES/NO |
|---|---|---|---|---|
| P3c Demo slider does not rewrite completed comparison | Student followed instruction to change only the demo alpha and supplied full page text | Demo alpha changed from 0.85 to 0.70; primary PageRank/PPR alpha remains 0.50; Recall remains 0.0797/0.1122. All four comparison rows and validation sweep values unchanged | audit/HW6_STUDENT_ALPHA_SLIDER_2026-10-19.txt | YES |

This confirms slider isolation only; no cancellation, export inspection, hand calculation or code inspection is inferred.

## Student two-user demonstration — 2026-10-19

| Check | Action performed by student | Observed result | Evidence/screenshot if available | Confirmed by student: YES/NO |
|---|---|---|---|---|
| M6a Reproduce two-user demonstration | Student followed instruction to run two-user demo and supplied full page output | Complete: user1/0.85, user2/0.85, user1/0.50, ten movies each; page overlap counts 0/7/0. Primary comparison unchanged | audit/HW6_STUDENT_TWO_USER_DEMO_2026-10-19.txt | YES |

Scope is browser reproduction only. Independent student overlap counting, seen-item exclusion inspection, genre/source verification and manual numerical calculation are not yet confirmed. Agent read the three lists and checked their displayed intersections (0/7/0); this is not student hand verification.

## Student numerical calculation — 2026-10-19

| Check | Action performed by student | Observed result | Evidence/screenshot if available | Confirmed by student: YES/NO |
|---|---|---|---|---|
| M8a Individual Recall arithmetic | In response to the explicit instruction to calculate Hits / Relevant for user 1 (2 / 19), student returned their numerical result | Student: 0,105263157. Assistant checked that rounding to four decimal places gives the displayed 0.1053 | Direct student reply in chat; user 1 row in saved browser output | YES |

This confirms the student's calculation. Independent dataset derivation of the numerator/denominator and direct code inspection are not inferred. Student has not separately described their rounding/comparison step.

## Student code/calculation/output cross-check — 2026-10-19

| Check | Action performed by student | Observed result | Evidence/screenshot if available | Confirmed by student: YES/NO |
|---|---|---|---|---|
| M8b Direct code-to-calculation-to-browser comparison | Student opened week6/script.js, located and pasted the hit / relevant.size expression, then explicitly confirmed comparison with their calculation and the displayed user 1 Recall | 2 / 19 = 0.105263157…; rounded to 0.1053, matching the page. Student: compared formula with calculation and page value — matches | Student pasted actual source fragment and explicitly confirmed in chat; saved browser evidence contains user 1 relevant 19, hits 2, Recall 0.1053 | YES |

This completes the specific personal arithmetic and formula/output comparison. It does not independently verify dataset-derived membership of the 19 relevant movies or the two hits, and does not confirm the remaining cancel/export/source checks.

## Current student WebGL self-check reproduction — 2026-10-19

| Check | Action performed by student | Observed result | Evidence/screenshot if available | Confirmed by student: YES/NO |
|---|---|---|---|---|
| P1d Current Run tests after completed Evaluate and demo | Student followed instruction to click Run tests without refreshing and pasted the full resulting page | 12 PASS / 0 FAIL / 0 PENDING; backend webgl. Four-method comparison and alpha sweep remain displayed unchanged | audit/HW6_STUDENT_SELF_CHECKS_WEBGL_2026-10-19.txt | YES |

Page additionally reports MF reference factors 16, checkpoint epoch 1, validation RMSE 1.1830. These are observed runtime metadata, not a separate student verification of checkpoint selection or training correctness. Exports still require inspection. Self-check evidence does not by itself certify all real-data logic or all six browser checks.

## Student exports supplied — 2026-10-19

| Check | Action performed by student | Observed result | Evidence/screenshot if available | Confirmed by student: YES/NO |
|---|---|---|---|---|
| P5a Export three current browser artifacts | Student clicked the three export buttons and explicitly confirmed files saved in repository tmp | verification.json, results.csv, recommendations.csv are present | tmp/verification.json; tmp/results.csv; tmp/recommendations.csv; direct student confirmation | YES |
| P5b Personally inspect export contents against page | Not yet confirmed | Assistant inspected exports; student content inspection still pending | Same files | NO |

Assistant export inspection: JSON parseable, WebGL, 12 PASS / 0 FAIL / 0 PENDING. Results CSV has 10 data rows: four test metrics, four validation metrics and two convergence rows. Test cohort 861, validation cohort 846 (cohort sizes, not CSV row counts). Four test recalls round to 0.0760/0.0797/0.1122/0.0211 as on page; chosen PR/PPR alpha 0.50. Recommendations CSV contains 100 rows in 10 lists, each with ranks 1–10 and 10 distinct movies; per-list mean popularity recomputation matches. Both CSVs identify run-2-7, completed. This scope does not independently verify seen-item exclusions or source dataset correctness.

SHA256: verification.json DD0E98DEA4DB8D5432267AFE11EDE8CCB638844EE4E714972A159156629B1DE6; results.csv 60E553CE4CBA7AD2815708F1923039D23CC80025E37DC6A145BA4D01DD271AA3; recommendations.csv CB50C1D51208F5510F3CE4642798CAF0C1925EA8C481363C009739C2699D9FA3.

## Student CSV-to-browser check — 2026-10-19

| Check | Action performed by student | Observed result | Evidence/screenshot if available | Confirmed by student: YES/NO |
|---|---|---|---|---|
| P5b PPR results.csv row versus browser | Student opened results.csv, located the PPR test row, pasted its values and explicitly confirmed agreement with the browser | ppr,0.50,0.11218853219486066,861; browser Recall 0.1122 and cohort 861 match at displayed precision | Direct student reply: ppr,0.50,0.11218853219486066,861 совпало; tmp/results.csv | YES |

Scope: this specific PPR row and its alpha/Recall/cohort. Other CSV rows and recommendations columns were checked by the assistant, not independently confirmed by the student. Progress/cancel and subjective responsiveness remain pending.

## Student responsiveness and attempted cancellation — 2026-10-19

| Check | Action performed by student | Observed result | Evidence/screenshot if available | Confirmed by student: YES/NO |
|---|---|---|---|---|
| P2a Scrolling during attempted evaluation | Student explicitly reported that scrolling worked following the guided check | Scrolling worked; pasted page shows incomplete evaluation at 11.5 s, WebGL | Direct student statement; audit/HW6_STUDENT_CANCEL_ATTEMPT_2026-10-19.txt | YES |
| P2b Cancel during an active full graph evaluation | Attempt requested, but a cancelled state is not evidenced | Page says no graph yet, HW6_INTERFACE no graph is active, incomplete not cached; no cancelled status | Same evidence | NO |

Do not count this as successful cancellation or full-cohort sweep responsiveness. The cause of the absent graph is not established from page text. Previous successful exports are preserved and are not replaced by this incomplete attempt.

## Student successful cancellation — 2026-10-19

| Check | Action performed by student | Observed result | Evidence/screenshot if available | Confirmed by student: YES/NO |
|---|---|---|---|---|
| P2b Cancel an active evaluation after Build graph | Student followed the guided Evaluate -> Cancel steps and pasted the resulting page in chat | Evaluation cancelled; elapsed 10.4 s; backend webgl; graph 2554 nodes / 45602 positive edges remains. Method comparison and Retrieval evaluation show cancellation messages instead of final numbers | Full page text directly supplied by student in chat, 2026-10-19 | YES |

Supersedes the earlier unconfirmed cancellation attempt only. 10.4 s is elapsed run time at cancellation, NOT measured cancellation latency. Exact cancellation stage and progress counters are absent from this post-cancel text. The generic previous-completed-evidence-retained message does not prove preservation of a prior successful snapshot in this fresh tab. Successful restart after cancellation remains to be checked. Existing successful-run CSV exports must not be overwritten by cancelled-run exports.

## Student Evaluate after cancellation — 2026-10-19

| Check | Action performed by student | Observed result | Evidence/screenshot if available | Confirmed by student: YES/NO |
|---|---|---|---|---|
| P2c Evaluate can complete after cancellation | Student followed instruction to click Evaluate in the same tab after cancellation without rebuilding, and supplied page text | Evaluation run complete; WebGL; elapsed 0.1 s; selected alpha 0.50; four test recalls 0.0760/0.0797/0.1122/0.0211, cohort 861, exclusions 13/82; validation values unchanged | audit/HW6_STUDENT_RESTART_AFTER_CANCEL_2026-10-19.txt | YES |

This confirms the resulting completed UI state after the guided restart. A 0.1 s run is not evidence of a fresh full recomputation; text does not establish which cached intermediates were reused or the exact prior cancellation stage. Keep earlier run-2-7 exports as the identified report evidence. Active progress fields/counters remain not independently captured or specifically confirmed by the student.

## Student full WebGL run after MF L2 repair — 2026-10-20

| Check | Action performed by student | Observed result | Evidence/screenshot if available | Confirmed by student: YES/NO |
|---|---|---|---|---|
| Corrected MF full evaluation and self-checks | Student followed refresh / Build graph / Evaluate / Run tests instructions and returned full page text | WebGL; elapsed154.1s; alpha0.50; recalls popularity0.0760, PageRank0.0797, PPR0.1122, MF0.0547; cohort861, exclusions13/82; MF checkpoint6, valRMSE0.9878; tests12/0/0 | audit/HW6_STUDENT_MF_FIXED_WEBGL_2026-10-20.txt | YES |

This supersedes old MF0.0211 / checkpoint1 / valRMSE1.1830 as the candidate final result. Other methods and validation sweep remain unchanged at displayed precision. Final corrected CSV exports and direct examination of nonconstant MF recommendation scores are still pending. Earlier95.7s/0.2s timings belong to the pre-fix revision;154.1s is the observed corrected full-run duration, not a repeated benchmark.

## Corrected-run exports supplied — 2026-10-20

| Check | Action performed by student | Observed result | Evidence/screenshot if available | Confirmed by student: YES/NO |
|---|---|---|---|---|
| Export corrected run | Student confirms saving three new exports in Downloads after the guided export steps | verification.json, results.csv and recommendations.csv received; copies preserved byte-for-byte | tmp/HW6_final_2026-10-20/ | YES |

The assistant independently checked their contents: WebGL12/0/0; MF Recall0.054668358285198206;100 recommendation rows;10 distinct MF scores per exported user; no raw-data metadata/popularity/candidate/exclusion errors. These content checks are not additional student confirmations.
