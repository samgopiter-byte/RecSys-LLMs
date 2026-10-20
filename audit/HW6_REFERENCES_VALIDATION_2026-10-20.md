# HW6 reference validation — 2026-10-20

Primary sources read and bibliographic metadata verified by the report assistant. This is not student personal source verification. No unverified DOI is included.

| Source | Primary URL | Verified scope | Status |
|---|---|---|---|
| Taher H. Haveliwala, Topic-Sensitive PageRank, WWW 2002 | https://snap.stanford.edu/class/cs224w-readings/Haveliwala02Topicsenitive.pdf | Title/author/WWW2002 date on p1; section2 fixed-point iteration and nonuniform personalization, pp2–3. Paper alpha denotes restart; HW6 alpha denotes link-following. Do not copy the opposite convention. Topic-sensitive web search is related work, not a claim the movie method reproduces the paper. | PASS |
| F. Maxwell Harper and Joseph A. Konstan, The MovieLens Datasets: History and Context, ACM TiiS5(4), article19,2015; DOI10.1145/2827872 | https://files.grouplens.org/papers/harper-tiis2015.pdf | Author manuscript title/authors; metadata confirmed at https://www.grouplens.org/blog/movielens-datasets-context-and-history/ . Read section3/TableII and sections3.2–3.3:100K counts, sampling, rating times not viewing times, limitations. Manuscript has placeholder venue header; final venue metadata comes from GroupLens publication record. | PASS |
| GroupLens, MovieLens 100K Dataset | https://grouplens.org/datasets/movielens/100k/ | Official dataset release/download provenance. Landing page rounds user/movie counts; use actual loaded counts and paper TableII for exact numbers. | PASS |
| TensorFlow.js, API reference4.22.0 | https://js.tensorflow.org/api/4.22.0/ | Read tf.matMul transpose flags, tf.tidy prohibition on Promise/async, variables needing explicit disposal and tf.dispose. Exact version matches index.html. Undated documentation; accessed2026-10-20, do not invent publication year/authors. | PASS |

Four sources cover algorithm, dataset and actual API. Each relevant-work summary should stay short and describe the adaptation honestly. Do not attribute a MovieLens online benefit to Haveliwala's web-search results. Student did not claim reading these sources. Full citation formatting and links in the PDF still require artifact validation.
