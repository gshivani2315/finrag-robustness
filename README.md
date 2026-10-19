# FinTrust-RAG - Track A (Retrieval, Evaluation, Analysis) on TAT-QA

**Status.** Retrieval, retrieval metrics, experiments and tests are done. EM/F1 exist but have only passed
unit tests: no real model predictions have been evaluated. The perturbation experiments (noise, redundancy,
conflict), end-to-end pipeline wiring, degradation curves and heatmaps depend on Track B and are not done.

## 1. Setup (Windows PowerShell)
```powershell
cd <path-to>\project
.\.venv\Scripts\Activate.ps1                      # use the environment that actually has the packages
python -c "import numpy, rank_bm25, torch, transformers; print('environment OK')"
pip install -r requirements.txt                    # only if the line above fails (downloads PyTorch)
python -m src.data_loader --download               # train/dev/test JSON into data\
python -m pytest -q                                # expected: 34 passed
```
Always run commands from the `project` folder. The first Contriever run downloads `facebook/contriever`
from Hugging Face (internet needed).

## 2. What is in the repository
| Path | Purpose |
|---|---|
| `src/data_loader.py` | download/load TAT-QA, one record per question, gold evidence mapping |
| `src/passage_builder.py` | paragraph and table-row passages, stable IDs, optional table-aware search text |
| `src/retriever.py` | BM25, Contriever encoder (cached), BM25+Contriever reranker, `reserve_row_slots` |
| `src/evaluate.py` | retrieval metrics only: Recall@k, nDCG@k (strict and condensed) |
| `src/answer_metrics.py` | EM and token F1 (simplified, not the official TAT-QA scorer) |
| `src/pipeline.py` | `retrieve()` interface and the batch command (JSONL output) |
| `scripts/*.py` | analysis scripts, see section 6 |
| `tests/` | 34 tests |
| `data/`, `outputs/` | downloaded data, experiment outputs (both git-ignored) |

## 3. TAT-QA facts this project relies on (checked on the real files)
| split | docs | questions | answers / evidence |
|---|---|---|---|
| train | 2201 | 13215 | yes |
| dev | 278 | 1668 | yes |
| test | 278 | 1669 | **none** (can be retrieved against, cannot be scored) |

* `rel_paragraphs` lists paragraph `order` values and labels **paragraphs only**. Table rows are never
  labelled, and table-only questions have no evidence label.
* Dev: 389 questions have complete paragraph gold (text-only), 512 partial (text+table), 767 none.
  The 901 labelled questions are the only ones retrieval can be scored on.
* No table or question overlaps between train, dev and test. Each corpus is built from one split only.
* Dev corpus = 3582 passages (paragraphs + table rows).

## 4. Passages, IDs, and the final retrieval configuration
Passage IDs: paragraph `<table_uid>::p<order>`, table row `<table_uid>::r<row_index>`.

**Original baseline:** BM25 over paragraphs and linearised table rows; optional Contriever rerank of BM25's
top 100 (dot product).

**Final (table-aware) configuration:** each paragraph is indexed together with its own document's table
header and first 20 row labels (`search_text`). Only the index changes; results always show the original
text. Of the top 5, 4 slots go to paragraphs and 1 to a table row (`reserve_paragraph_slots=4`), because
otherwise paragraphs crowd table rows out of the top 5 and no label can show that loss.
```python
from src.pipeline import retrieve
retrieve(question, top_k=5, method="bm25", paragraph_context="table_labels", reserve_paragraph_slots=4)
```
Defaults (`retrieve(question)`) reproduce the original baseline. `doc_id=` restricts the search to one document.
Return value: `{question, method, split, doc_id, top_k, paragraph_context, reserve_paragraph_slots, results}`;
each result has `passage_id, text, rank, source_type, doc_id, bm25_score, bm25_rank`, plus `contriever_score`
(method `bm25+contriever`) and `rank_before_slots` (when slots are on).

## 5. Reproduce the experiments
Every command writes to its own folder, so nothing is overwritten.
```powershell
# original baseline: 200 labelled questions, seed 42 (BM25 and BM25+Contriever)
python -m src.pipeline --split dev --n 200 --seed 42 --only-evaluable --methods bm25 bm25+contriever --store-k 100 --candidate-n 100 --out outputs\dev200_k100
# same questions, search restricted to the question's own document
python -m src.pipeline --split dev --n 200 --seed 42 --only-evaluable --methods bm25 bm25+contriever --store-k 100 --doc-scope --out outputs\dev200_docscope
# table-aware BM25 (add --reserve-paragraph-slots 4 for the slot rule)
python -m src.pipeline --split dev --n 200 --seed 42 --only-evaluable --methods bm25 --store-k 100 --paragraph-context table_labels --out outputs\exp4_ctx_eval
# held-out sample: dev questions from documents NOT in the 200 (used to choose the fusion weight and slot split)
python -m src.pipeline --split dev --n 0 --only-evaluable --exclude-docs-from outputs\dev200_k100\retrieval_bm25.jsonl --methods bm25 bm25+contriever --store-k 100 --candidate-n 100 --out outputs\tune
# fresh-question confirmation: run on documents outside the held-out sample, then compare on the 386 never-used questions
python -m src.pipeline --split dev --n 0 --only-evaluable --exclude-docs-from outputs\tune\retrieval_bm25.jsonl --methods bm25 bm25+contriever --store-k 100 --candidate-n 100 --out outputs\final_orig
python -m src.pipeline --split dev --n 0 --only-evaluable --exclude-docs-from outputs\tune\retrieval_bm25.jsonl --methods bm25 --store-k 100 --paragraph-context table_labels --out outputs\final_ctx
python scripts\final_confirmation.py
```
Sampling is deterministic: questions are sorted by ID, then `random.Random(seed)` samples.
`--reserve-paragraph-slots 4` in the batch command has been tested on a small run only; the 4+1 results in
section 7 come from `scripts\slot_analysis.py` and `scripts\final_confirmation.py`, which rearrange saved rankings.

## 6. Metrics, output files and scripts
* **Recall@5** = gold passages in the top 5 / gold passages. **nDCG@10** = binary-relevance nDCG.
  Questions without gold are excluded ("not evaluable"), never scored 0.
* **Strict:** table rows count as non-relevant (a lower bound, because rows are unlabelled).
  **Condensed:** table rows are removed from the ranking before scoring (the primary view).
  The condensed view depends on how many passages are stored, so compare runs stored at the same depth (100).
* Confidence intervals: paired document-level bootstrap (questions of one document are correlated), 10,000
  resamples (5,000 in `plot_results.py`).
* EM/F1 (`src/answer_metrics.py`) score generated answers, not retrieval. Track A produces no answers.

Run folders contain `retrieval_bm25.jsonl`, `retrieval_bm25_contriever.jsonl` (when that method ran) and
`metrics_summary.json`. One JSONL row per question: `question_id, split, doc_id, question, gold_answer,
gold_answer_text, answer_type, answer_from, gold_evidence_ids, evidence_status, retriever, doc_scope,
reserve_paragraph_slots, retrieved[], retrieval_metrics`. `retrieved[]` items hold `passage_id, text, rank,
source_type, bm25_score, bm25_rank` (+ `contriever_score`, `rank_before_slots`). `retrieval_metrics` is
`null` without gold, else `recall@5_strict, ndcg@10_strict, recall@5_condensed, ndcg@10_condensed`.

| Script (run from `project`) | What it does |
|---|---|
| `scripts\verify_outputs.py --dir outputs\dev200` | recompute metrics independently, structure checks, paired CIs |
| `scripts\failure_analysis.py` | questions both retrievers missed, examples, rule-based flags |
| `scripts\pool_and_docscope.py` | BM25 top-100 pool coverage, oracle ceiling, document-restricted results |
| `scripts\pool_depth.py` | pool coverage at 100/200/300 (needs `outputs\exp2_bm25_k300`) |
| `scripts\compare_runs.py --cand <file> --name <label>` | candidate vs BM25 and baseline; `--vs-bm25-only` for BM25 variants |
| `scripts\fusion_analysis.py` | BM25+Contriever score blending from saved scores |
| `scripts\ctx_checks.py`, `scripts\slot_analysis.py` | table-aware checks: per-group effect, table-row share, answer-string proxy, slot split |
| `scripts\final_confirmation.py` | the single fresh-question comparison |
| `scripts\plot_results.py` | tables and figures into `outputs\figures` |

## 7. Experiment record (condensed Recall@5 / nDCG@10 unless noted)
| Folder | Questions | Result | Status |
|---|---|---|---|
| `dev200_k100` | 200 | BM25 0.545 / 0.487; BM25+Contriever 0.515 / 0.441 | baseline, exploratory |
| `dev200_docscope` | 200 | nDCG@10 BM25 0.847, Contriever 0.802 (random order 0.632) | exploratory |
| `exp1_cosine` | 200 | cosine 0.480 / 0.401 (worse than dot) | exploratory |
| `exp2_bm25_k300` | 200 | questions with a gold paragraph in BM25's top 100/200/300: 0.760 / 0.795 / 0.805 | exploratory |
| `tune` | 315 held-out | BM25 0.657 / 0.591; Contriever 0.613 / 0.528 | tuning sample |
| `exp3_fusion` | 200 | blend (w=0.6) 0.563 / 0.496; vs BM25 +0.009 nDCG@10, CI includes 0 | exploratory |
| `exp4_ctx_eval`, `exp4_ctx_tune` | 200; 315 | table-aware BM25 0.698 / 0.625; 0.740 / 0.640 | exploratory + replication |
| `exp6_ctx_*`, `exp6_fusion` | 200; 315 | blend on table-aware text 0.715 / 0.625 = no gain over table-aware BM25 | exploratory |
| `final_orig`, `final_ctx` | 386 fresh | below | **confirmation** |

**Fresh-question confirmation (386 questions, 132 documents, never used for tuning or comparison):**
original BM25 0.448 / 0.405; original BM25+Contriever 0.452 / 0.385; table-aware BM25 with 4+1 slots 0.636 / 0.551.
Proposed minus original BM25: strict Recall@5 +0.190 (CI +0.127 to +0.256), condensed nDCG@10 +0.146 (+0.104 to +0.191).
Proposed minus original BM25+Contriever: strict Recall@5 +0.181 (+0.107 to +0.255), condensed nDCG@10 +0.167 (+0.116 to +0.216).
Answer-string proxy in the top 5: 0.726 (proposed), 0.743 (BM25), 0.586 (BM25+Contriever).

## 8. Known limitations
* Only paragraphs are labelled; table rows are not, and the test split has no labels. All experiments use dev.
* Contriever, cosine, a larger pool and rank fusion gave no demonstrated gain over BM25 alone.
* Gains come from table-aware indexing and are concentrated in text+table questions. Text-only questions show
  a small decline in two of three samples that is not statistically established.
* The answer-string proxy is crude (span questions only); arithmetic questions are unmeasured. The result
  concerns retrieval of the labelled gold paragraphs, not end-to-end answer quality.
* Every paragraph in a document shares the same appended table text, so paragraphs within a document are
  harder to tell apart.
* **EM/F1 have only passed unit tests.** No real predictions have been scored.
* Not done (needs Track B): evidence tagging, synthesis, verification wiring; noise, redundancy and conflict
  runs; degradation curves and heatmaps; query-type x noise tables.
