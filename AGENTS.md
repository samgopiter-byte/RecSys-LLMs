# LLM4Rec: permanent homework rules

These rules apply to every LLM4Rec homework in this repository. They are the student's required workflow, consolidated from course guidelines, assignments, class feedback HW1-HW4, and the student's account of individual HW4 feedback. They do not certify the current implementation or any previous report.

## 1. REQUIREMENTS FIRST

Before implementation, fixes, experiments intended for submission, or report authoring:

1. Read this file and the current `REQUIREMENTS_MATRIX.md`.
2. Recursively discover and read all course/homework guidelines, the current assignment specification, README, lecture materials/slides, and applicable nested `AGENTS.md` files.
3. Recursively read `tmp/feedback/`, any feedback directories with different names (currently `tmp/feedback преподавателя/`), all earlier teacher/class feedback, and individual instructor replies supplied by the student. Report missing sources. An empty file is not feedback.
4. Inspect the actual branch, git status, unstaged and staged diffs. Preserve existing work. Never assume previous test results still apply to current files.
5. Create/update and SHOW a fresh requirements matrix before coding or writing a final report. Use exactly these columns: `Requirement | Source | Code evidence | Report section | Personal verification | Status`.
6. Give every named deliverable and every assignment bullet its own evidence obligation. Carry forward general lessons, but do not impose obsolete homework-specific algorithms on later homework.

Record the source file and section, actual code/output evidence, report destination, and student verification. Use `PASS`, `FAIL`, `PENDING`, `BLOCKED`, or `NOT APPLICABLE` with justification. Static inspection alone cannot establish runtime PASS. No assumption, previous claim, planned check, or agent action establishes personal verification.

### Source priority and conflicts

For instructor/course requirements, use this order:

1. Latest individual instructor reply.
2. Current assignment specification.
3. Latest class feedback.
4. General course guidelines.
5. Old templates and previous homework.

Follow explicit student workflow constraints, including the stricter page/reference/verification safeguards below. Identify source disagreements explicitly. Use the stricter page limit. If a conflict cannot be resolved by this priority, stop the affected work, show both requirements and concrete evidence, and ask the student; never silently choose or change a provided test to manufacture PASS. Sample answers are illustrative, not individual feedback or verified citation records.

## 2. NO REPORT BEFORE COMPLIANCE

Do not create a final report until every mandatory implementation, analysis, evidence, citation, and personal-verification prerequisite is PASS. Missing mandatory sources or unresolved conflicts block final authoring. Inspecting an existing draft is allowed; its assertions must be re-audited.

Keep two explicit gates to avoid circular checks: prerequisite compliance before report authoring, then PDF/artifact compliance after rendering. A PDF rendered for page-count and layout inspection is an audit candidate, not a final submission. Do not mark PDF existence/page count PASS before checking the actual artifact. Never finalize or package a report until ALL mandatory rows, including PDF and native-session rows, are PASS. No commit/push/submission without the student's authorization for that action.

## 3. REPORT STRUCTURE

Use IEEE-aligned numbered citations and this structure, without placeholders:

- Title block: concise title, student name, team or Individual, email, date, assignment/topic and week.
- Abstract: 50-150 words; four sentences covering problem, method, one verified numerical result, and takeaway.
- §1 Introduction: problem (2-4 sentences), motivation (2-3), concrete input/output example, 1-2 restrained contribution bullets.
- §2 Related Work: research actually consulted and cited; 1-3 alternatives with accurate rationale. Do not merely append a source list.
- §3 Method: approach, pipeline, tools/libraries with actual versions and official documentation, 2-3 design choices with rationale, hyperparameters, seeds/randomness, environment.
- §4 Experiments: dataset/provenance, evaluation protocol, hardware and actual backend, labeled results, baseline comparison, specific reproducible checks. Address every current assignment deliverable.
- §5 Discussion: substantive 5-Loop, actual failure and evidence, root cause, fix and before/after verification, what worked, what failed/surprised, concrete next improvement.
- §6 AI Usage Disclosure: every item required below, including explicit personal verification and trusted-without-verification statements.
- Numbered References: consistent with citations in §2 and throughout the report.

§2, References, and §6 may NEVER be omitted, including for a technically strong solution. Do not copy sample text or their personal verification claims. The student's explanation and reflection must represent their actual reasoning.

## 4. PAGE LIMIT

Use <=5 physical pages for the entire `report.pdf` as the student's safe hard cap, including references. This also satisfies the course's stricter <=5 main-text pages where sources inconsistently allow six or exclude references. If a current rule imposes fewer pages, use the stricter cap.

After PDF generation, reopen the exact candidate PDF and physically count pages with a PDF reader/tool; record actual count and filename. Inspect rendered pages for clipping, unreadable text, missing sections, and tables/citations. A source-file estimate is not a page-count check.

## 5. REFERENCES

Use 3-7 relevant, numbered sources as this repository's safeguard. Include algorithm sources, dataset provenance/file-format sources, and official documentation for APIs actually used. The newer HW4 class feedback allows fewer than three with genuine research; the student deliberately requires the stricter 3-7 here.

Validate every author, year, title, journal/conference, attribution, DOI when supplied, and URL against authoritative primary records. Read the relevant method/data/API section, not only an abstract. Verify URLs resolve to the cited material and APIs exist in the version used. Record verification source/date and scope. Never invent citations, APIs, numbers, access dates, or source-reading claims. Do not inherit citations from starter README or sample answers without validation; HW4 feedback explicitly warns of incorrect starter attribution.

## 6. AI DISCLOSURE

Always state:

- Tools, versions, and actual models/provider identifiers, based on inspectable records; mark genuinely unavailable details honestly.
- How each tool was used (research, code generation, debugging, testing, documentation, visualization).
- **What I personally verified**: exact quantities/actions, student method, result, and artifact.
- **What I trusted without verification**: explicit and honest scope. This does not waive mandatory verification of critical outputs.
- Exact attached native session filename and relevant session notes (for example a crash).

Agent-run checks must be attributed to the agent. Never turn them into first-person student claims. Disclose all tools actually used; do not infer model identity from marketing names or a sample template.

## 7. PERSONAL VERIFICATION

An agent-run test is NOT personal verification. Before submission the student must personally perform and confirm at least:

1. One critical metric/calculation, with inputs, formula, intermediate arithmetic and comparison to code output.
2. One browser/result reproduction with actual steps and observed result.
3. One code-output cross-check identifying the code/formula and matching displayed/logged value.

Also engage personally with at least one derivation and assumption from the 5-Loop. Record student-provided evidence or direct student confirmation of specific actions. Prepare instructions for the student where needed; leave rows PENDING until they perform them. Passive inspection while the agent computes statistics does not establish student numerical verification. Do not state anything was personally verified unless it actually was.

### Mandatory manual script and confirmation ledger

Before writing the final report, the report assistant MUST present a separate **MANUAL VERIFICATION SCRIPT** to the student IN CHAT. Each check needs concrete student actions, expected outputs or explicit comparison/invariants, and evidence to inspect/record. Use actual final-run values when available; distinguish expected values from observations. Do not require an unseeded rerun to produce identical learned metrics. Maximize meaningful checks the student can realistically perform, not trivial actions or inflated counts.

Mandatory workflow: **AI PREPARES CHECK -> STUDENT PERFORMS CHECK -> STUDENT REPORTS RESULT BACK IN CHAT -> AI RECORDS ONLY THE CONFIRMED RESULT -> FINAL REPORT MAY THEN STATE IT AS "What I personally verified"**.

Before finalizing §6 maintain `VERIFICATION_LEDGER.md` with exactly these columns: `Check | Action performed by student | Observed result | Evidence/screenshot if available | Confirmed by student: YES/NO`. Initialize unperformed/unconfirmed checks with NO and no invented observations. Only checks explicitly confirmed YES by the student may appear under **What I personally verified**. A screenshot is optional; direct specific student confirmation is required. Confirmation of one check never confirms other rows.

For every assignment target at least one hand-computed numerical check, one final browser reproduction, one direct code/output consistency check, one assignment-specific logic check, and one reference/source verification when practical. Include additional meaningful checks actually performed. Appropriate checks include independent data statistics, in-page PASS/FAIL inspection, recommendation/seen-item intersections, exclusions, source sections, real before/after failure evidence, and final experiment reruns.

Unconfirmed points must be attributed to automated/agent verification or placed under **What I trusted without personal verification** if actually trusted, not presented as personal checks. Honest disclosure does not make a missing mandatory personal check PASS. Revalidate affected checks after code/configuration changes. Report-number proofreading can occur on an audit candidate after the pre-authoring personal checks; finalization still requires its own evidence gate.

The student assigns coding/implementation to OpenCode and requirements/report authoring to this chat. Prepare OpenCode prompts for concrete assignment work; manage this manual script, ledger, requirements audit and report disclosure here. Do not ask OpenCode to certify the student's personal actions or write the final report.

### Current HW5 feedback availability

HW5 instructor feedback will be issued only after the cohort submits. Its absence is expected and is NOT a missing prerequisite, a reason to wait for grading, or a requirement for submission. Audit against the current HW5 assignment and accumulated earlier applicable feedback. An unresolved contradiction in those available sources remains a separate issue; do not confuse it with unavailable future feedback.

### HW5 Recall: individual instructor clarification received 2026-10-14

Read `tmp/feedback преподавателя/HW5_individual_recall_clarification_2026-10-14.md`. This pre-submission answer is available and has first priority; it is distinct from future cohort grading feedback.

Professor Jin clarified that there is NO conflict: `recommendTopK` ranks the global metadata catalog, excluding IDs absent from metadata and the user's train/validation-rated movies. Test-rated IDs remain eligible. Training-set movie IDs define ONLY the Recall evaluation pool, numerator/denominator eligibility, unseenIds exclusions and evaluable-user set. Metadata-only movies may occupy Top-K slots. Starter test #10 must return0 for rank11.

Follow the starter JSDoc/harness as the operative contract. Do NOT implement strict training-only reranking or describe it as a mandatory repair. Prior Codex advice/audits alleging a training-only requirement are superseded. Report the clarified protocol and exclusions; replace "unresolved conflict" text in the report. Preserve historical audit evidence but mark its superseded interpretation explicitly. Do not change correct code or retrain solely for this clarification.

## 8. STARTER AUDIT

From HW2 onward investigate the ENTIRE starter: data parsing, raw format and mappings, algorithms, evaluation, rendering, errors, tests, and all execution paths. Identify, explain, fix and verify all major defects, rather than stopping after the first. Preserve original datasets and scaffold as the current assignment requires. Do not rewrite a scaffold unnecessarily.

Record defect evidence, root cause, minimal repair, and regression verification. A passing toy fixture is insufficient evidence for real-data correctness. Do not manufacture a defect if none is found; document the audit honestly. If the assignment forbids modifying a file containing a necessary repair, or the required protocol conflicts with an immutable fixture, expose the conflict before editing.

Prevent train/validation/test leakage. Identify missing-data semantics, candidate universe, relevance definition, denominators, tie-breaking, baseline source, checkpoint selection and restoration explicitly. Never use test outcomes to tune or select a checkpoint. Audit ranking candidates as well as metric denominators.

## 9. DISCUSSION / 5-LOOP

Show BUILD (working baseline), ASK WHY (assumptions/alternatives), TEST (personal checks against theory/data/code), EXPLAIN (accurate student reasoning), and IMPROVE (one-variable iteration and concrete next step).

Include a real observed failure with an error/log/output excerpt where applicable, specific root cause, detection -> change -> confirmation, before/after evidence, what worked, what failed or surprised, and next improvement. Distinguish an observed defect from a hypothetical limitation. Do not reuse a failure from another HW as if it occurred in this one. Avoid generic filler, wrong mechanisms and unsupported reasons for rejecting alternatives.

## 10. EVIDENCE

Every reported number must literally match the final code/browser output at the same displayed precision. Record configuration, current code revision or file hash, dataset, backend, checkpoint, run and formatting. A subsequent code/configuration change invalidates affected evidence until rerun.

Report positive AND negative metrics, limitations, exclusions, weak ranking results and uncertainty. No cherry-picking, fabricated benchmarks, unjustified significance, discovery/causal/business claims, or unsupported performance claims. Distinguish measurements, assumptions and proposals. Retain inspectable evidence beyond a lone screenshot. Report provided in-page self-check counts and meaningful independent checks.

## 11. SESSION

Require the native, plaintext, UNMODIFIED export of the real coding agent used for the work, containing prompts, raw tool calls/results, file modifications and milestones. Use the tool's actual export facility (`opencode export <sessionID>` or equivalent) and the required submission filename.

Do not edit, reconstruct, filter, convert, summarize, encrypt, hand-write, or replace it with a chat/share link or an agent-compiled log. JSON wrapper shape alone does not determine validity. Verify parseability, actual coding-agent identity, raw records, session coverage, export provenance and consistency with §6. A hash helps preserve bytes but does not prove native provenance. Never modify earlier session artifacts. A previous export may not cover later work; require native evidence for subsequent coding sessions too, and resolve packaging requirements before submission.

## 12. FINAL PRE-SUBMISSION AUDIT

Before finalizing/submitting, print factual results for ALL of:

- Mandatory requirements PASS count: X/Y (all applicable mandatory rows; no PENDING/FAIL/BLOCKED).
- Report physical page count and exact file, <=5 (or stricter current cap).
- References validation: 3-7, algorithm + dataset + official docs, verified metadata and live sources.
- §2 Related Work and numbered References present.
- §6 AI Usage Disclosure present with all required fields.
- Personal verification present, with the three student-performed categories above.
- Native session validation: plaintext, parseable, unmodified tool export, raw calls/results and work coverage.
- Code/tests status: current full starter audit, page works, required self-checks, independent checks, no unresolved major defects.
- Report/code number consistency for ALL metrics, lists, exclusions and experiment settings.
- All named assignment bullets and analysis questions accounted for.
- `GATE-0 RISKS: NONE`.
- `MINOR RISKS: NONE`.

Only print NONE when supported by the completed audit. If ANY item is not PASS, explicitly print **SUBMISSION BLOCKED**, identify actual risks and missing evidence, and stop finalization/submission. Do not claim NONE while evidence is unavailable. Two minors give 0; excellent analysis does not cancel missing required sections.

Required deliverables include report PDF and native session JSON, plus the current assignment's code/data files. Follow `<student-id-or-name>_<assignment-id>_report.pdf` and matching `_session.json` naming where applicable; do not submit notebooks, separate memos, generators or other artifacts prohibited by the assignment. Verify deadline/channel from actual course information; do not assume a deadline. Late submissions are not accepted.

### HW5 repository link and authorized publication

The student explicitly authorized this assistant on 2026-10-13 to perform the later commit/push and necessary normal Git publication steps for the finished HW5 submission. This supersedes earlier no-commit/no-push instructions for the eventual verified publication step; do not ask for the same authorization again. It does not authorize publishing unfinished work, bypassing compliance, force-pushing, deleting history, changing remotes, or submitting through an unspecified LMS/channel.

Verified local origin URL: `https://github.com/samgopiter-byte/RecSys-LLMs.git`. Current target branch: `a05-matrix-factorization`, tracking `origin/a05-matrix-factorization`. Use the student's repository, not the instructor's `upstream` repository, for the submission/code link.

The report MUST contain a clickable student repository link and explicit HW5 branch/path: `https://github.com/samgopiter-byte/RecSys-LLMs/tree/a05-matrix-factorization/week5`. Include a verified implementation commit identifier/link for reproducibility when available. Do not invent a future hash or claim that an unpushed local state is available remotely.

Publish all relevant submission artifacts: final HW5 code/data, validated final report PDF, chosen unmodified native session export(s), and applicable requirements/verification documents. Stage files by explicit reviewed paths; do not use indiscriminate `git add .` in this mixed workspace. Exclude unrelated prior exports, career artifacts, temporary runners and generated diagnostics unless expressly needed for this submission. Preserve native export bytes and check content before publication.

Before publication check staged diff, syntax/tests, document compliance and artifact manifest. Use a normal commit and push to origin/a05-matrix-factorization; preserve remote changes. After push verify remote branch points to the intended commit and every report link/artifact is accessible to the grader. Print the final repository/branch/commit and artifact links. Publishing can use a verified implementation commit first, then report/artifact commit, so a PDF does not need to contain its own future commit hash. No new permission question is needed for these authorized steps; any platform-required technical approval is distinct from task authorization.

## LESSONS FROM FEEDBACK

Student-reported actual grades: **HW1 2/2; HW2 1/2; HW3 2/2; HW4 0/2**. Record these as individual results supplied by the student, separate from class statistics. Do not infer the exact HW2 deduction without individual evidence. The specific HW4 deduction is supplied explicitly below.

- **HW1:** audit trail is mandatory; investigate the full starter; diagnose, do not just patch. Verify against the library/version actually loaded. Missing or uninspectable required evidence can make strong work score 0.
- **HW2:** Related Work/references matter; hand-check at least one number; explicitly state **What I personally verified**. Check raw data field alignment and all genres, normalization, watched-item exclusion, bias and profile dilution; do not generalize from one input.
- **HW3:** implement every named deliverable literally; avoid leakage; make algorithm/data/missing-value choices explicit. Preserve dataset bytes. Personal verification cannot be agent-only. Two minor criteria already give 0.
- **HW4:** according to the student's individual instructor feedback, good analysis and verification still scored **0** because References/§2 were missing (c1 minor) AND §6 AI Usage Disclosure was missing (c2 minor). Never omit either section. Class feedback additionally requires an explicit actionable business axis for investigate/reject decisions, raw native session evidence and named independently verified quantities.

## Audited source inventory (2026-10-13)

- `course/homework_guidelines.pdf` (12 pages, binding rules; internal five/six-page disagreement).
- `course/sample_homework.pdf` (6 pages, illustrative sample, different A01 topic; not evidence for this student's work).
- `README.md`; `week1/prompt.md`; `week2/readme.md`; `week3/readme.md`; `week4/readme.md`; `week5/readme.md`.
- Entire `tmp/feedback преподавателя/`: `HW1_class_feedback.md`, `HW2_class_feedback.md`, `HW3_class_feedback.md`, `HW4_class_feedback.md`, `HW2_A02_Answer.md`, `HW3_A03_Answer.md`, `HW4_A04_Answer.md`, and empty `most important.txt`.
- The `_Answer.md` files identify themselves as samples with student placeholders. Do not represent them as personal instructor replies. The student's messages supply the individual HW4 grading explanation and stricter permanent safeguards.
- No exact `tmp/feedback/`, root instructions/matrix, current HW5 lecture/slides, local `docs/homework-guidelines/guidelines.md`, or HW5 report draft was discovered at initial audit. Missing material must remain visible in the matrix until provided or its applicability clarified.

Current HW5 is matrix factorization per `week5/readme.md`, overriding the old general course table's A05 PageRank topic. Do not copy old HW algorithms or sample section numbering into a current report.
