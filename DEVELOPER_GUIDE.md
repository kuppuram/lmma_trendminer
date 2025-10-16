# lmma_trendminer – Developer Guide

This guide is for developers who want to explore, extend, and run **lmma_trendminer** locally or in CI. It covers setup, training (NER + intent), ingestion, configs, running the pipeline end‑to‑end, adding an LLM summarizer, and troubleshooting.

---

## 1) What the repo does

`lmma_trendminer` is a **30‑second trend mining** pipeline over large text corpora (e.g., Amazon reviews).

1. Parse the **natural language query** → metadata (date range, rating/score, category keywords).  
2. **Vector‑only retrieval** from a vector DB (OpenSearch) based on time & optional filters.  
3. **UMAP** → **HDBSCAN** to discover clusters quickly (auto cluster count + noise).  
4. Select one **representative document** per cluster (centroid‑closest).  
5. Produce a human‑readable **trend summary**, optionally with an LLM for crisp titles.

> This is **not RAG**: we cluster across many docs to surface themes vs. retrieving a few passages.

---

## 2) Repository layout

```
lmma_trendminer/
  providers/
    registry.py                 # central access to trained models (intent & NER)
    custom_ner.py               # NER wrapper (HF pipeline + aggregation + normalization)
    custom_intent.py            # Intent wrapper (HF pipeline)
    vector_store.py             # OpenSearch adapter
  pipeline/
    retrieve.py                 # vector-only search, return ids + embeddings
    reduce.py                   # UMAP dimensionality reduction
    cluster.py                  # HDBSCAN clustering
    label.py                    # TF-IDF labels for clusters
    metadata_from_entities.py   # NER+regex → {start_date,end_date,rating,...}
  config.py                     # profile/field mapping (text/vector/date/score)
  runner.py                     # orchestrates analyze_query_and_summarize
tools/
  ingest/
    configs/opensearch-ingest-data.yaml   # template for ingestion field mapping
    index_data.py                         # optional ingestion script
  train/
    datasets/ner_data.py                  # seed examples (Python dicts)
    make_ner_jsonl_from_py.py             # converts ner_data.py → JSONL
    train_ner_cli.py                      # NER training CLI (Transformers)
    intent_cli.py                         # Intent training CLI
data/
  ner_train.jsonl  ner_eval.jsonl         # token-labeled data (post-conversion)
models/
  ner-extractor/                          # trained NER (tokenizer + model)
  intent-classifier/                      # trained intent classifier
test2.py  test3.py                        # runnable E2E samples
```

---

## 3) Prerequisites & installation

- Python **3.10+** (3.11 OK)  
- OpenSearch **2.x** running locally (`localhost:9200` by default)  
- (Optional) GPU/CUDA for faster training

```bash
# create & activate venv
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

# install package + extras
pip install -U pip
pip install -e ".[ingest,train,llm]"

# CPU-only torch (if needed)
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

**Helpful one-liners** (Windows PowerShell via `tasks.ps1`):
```powershell
pwsh -File .\tasks.ps1 install           # editable install + extras
pwsh -File .\tasks.ps1 ner-jsonl         # build data/ner_*.jsonl from ner_data.py
pwsh -File .\tasks.ps1 train-ner-dslim   # recommended base for tiny data
pwsh -File .\tasks.ps1 train-intent
pwsh -File .\tasks.ps1 run-demo          # run test3.py
```

**Makefile** (macOS/Linux):
```bash
make install
make ner-jsonl
make train-ner-dslim
make train-intent
make run-demo
```

---

## 4) Data formats

### 4.1 NER training JSONL (used by `ner_cli.py`)

Each line has aligned tokens & labels:

```json
{"tokens":["reviews","from","jan","to","march","2014"],
 "labels":["O","O","B-MONTH","I-MONTH","I-MONTH","B-YEAR"]}
```

If you prefer authoring Python rows like:
```python
{"text":"reviews from jan to march 2014",
 "labels":"O O B-MONTH I-MONTH I-MONTH B-YEAR"}
```
convert using:
```bash
python -m tools.train.make_ner_jsonl_from_py tools.train.datasets.ner_data \
  --train-out ./data/ner_train.jsonl \
  --eval-out  ./data/ner_eval.jsonl
```

### 4.2 Intent CSV

Simple `text,label` (e.g., `trend_analysis`, `simple_search`, `greeting`).  
See `data/intent_data.csv` in the repo.

---

## 5) Training models

### 5.1 NER (MONTH/YEAR/SCORE)

Use a NER‑tuned base for tiny datasets:

```bash
python -m tools.train.ner_cli \
  --train-file ./data/ner_train.jsonl \
  --eval-file  ./data/ner_eval.jsonl \
  --base-model dslim/bert-base-NER \
  --output-dir ./models/ner-extractor \
  --epochs 12 --batch-size 8
```

Or train from a plain encoder:
```bash
python -m tools.train.ner_cli \
  --train-file ./data/ner_train.jsonl \
  --eval-file  ./data/ner_eval.jsonl \
  --base-model distilbert-base-uncased \
  --output-dir ./models/ner-extractor \
  --epochs 12 --batch-size 8
```

**Tips for tiny data**
- Add **80–120** examples covering month ranges (`Nov to Dec 2011`, `Apr-May 2011`), single month+year, rating phrases (`>=4 score`, `high 5 score`, `score 3`). Vary casing/hyphens. Keep labels **consistent**.
- Validate JSONL alignment before training (tokens == labels in each row).

### 5.2 Intent

```bash
python -m tools.train.intent_cli ./data/intent_data.csv \
  --output-dir ./models/intent-classifier \
  --epochs 5 --batch-size 16 --val-ratio 0.3
```

With very small CSVs, increase `--val-ratio` so each class appears in eval, or omit eval for a quick POC.

---

## 6) Configuring the app

### 6.1 Point to your trained models

```python
from lmma_trendminer.providers.registry import configure_models, model_paths
configure_models(intent_path=r"./models/intent-classifier",
                 ner_path=r"./models/ner-extractor")
print("[DEBUG MODELS]", model_paths())
```

### 6.2 Index field mapping

Ensure your OpenSearch index has (minimum):
- `review_embedding` (dense vector used in retrieval)
- `text` (full text)
- `timestamp` (`YYYY-MM-DD`)
- `score` (integer 1–5 or keyword `"1".."5"`)

Map/rename in:
- `tools/ingest/configs/opensearch-ingest-data.yaml` (ingestion)
- `config.py` profile (runtime). The runner adapts rating filters:
  - numeric → `range` query (`gte/lte`)
  - keyword → `terms` query across enumerated values

---

## 7) Running end‑to‑end

```python
from opensearchpy import OpenSearch
from lmma_trendminer.providers.vector_store import OpenSearchStore
from lmma_trendminer.providers.registry import configure_models
from lmma_trendminer.runner import analyze_query_and_summarize

configure_models(intent_path=r"./models/intent-classifier",
                 ner_path=r"./models/ner-extractor")

client = OpenSearch(hosts=[{"host":"localhost","port":9200}])
store  = OpenSearchStore(client, index="amazon-food-reviews", vector_field="review_embedding")

query = "top trends in Foods for Nov to Dec 2011 which are high 5 score"

summary, details = analyze_query_and_summarize(
    query, store,
    min_reviews=1000,
    text_field="text",
    # umap_cfg={"n_components": 20, "n_neighbors": 30, "min_dist": 0.0, "metric": "cosine"},
    # hdbscan_cfg={"min_cluster_size": 10, "metric": "euclidean"},
    # summarizer=groq_summarizer,     # optional LLM step
)
print("=== SUMMARY ===\n", summary)
print("=== META ===\n", details.get("meta"))
```

Expected:
- `META` includes `start_date`, `end_date`, `rating` (from NER or **regex fallback**).
- A concise trend list with cluster sizes and TF‑IDF hints (or LLM‑generated titles).

---

## 8) LLM summarizer (optional)

Wire in Groq/OpenAI for nicer titles:

```python
from lmma_trendminer.summarizers.groq_sum import groq_summarizer
summary, details = analyze_query_and_summarize(
    query, store, text_field="text", summarizer=groq_summarizer
)
```
Set `GROQ_API_KEY` (or your chosen provider’s env var) before running.

---

## 9) Troubleshooting

- **`META {}`**  
  Ensure you use `build_metadata_from_entities(...)` and the **regex fallback** exists in `metadata_from_entities.py`. Add debug:
  ```python
  from lmma_trendminer.providers.registry import get_ner
  ents = get_ner().extract(query)
  print("[DEBUG NER OUTPUT]", ents)
  print("[DEBUG META]", meta)
  ```

- **Score filter missing in OS query**  
  `meta["rating"]` must be present (e.g., `{"gte":4,"lte":5}`). The runner maps it to `range`/`terms` based on the score field type.

- **Transformers kwarg errors (e.g., evaluation_strategy)**  
  Upgrade:  
  `pip install -U "transformers>=4.44,<5" "accelerate>=0.34" "datasets>=2.19" "seqeval>=1.2.2"`

- **Custom NER not used**  
  Confirm you call `providers.registry.get_ner()` (not raw HF pipeline). Add debug prints in `registry.get_ner()` and `custom_ner.__init__()` to log the model path.

- **Windows symlink warning (HF cache)**  
  Cosmetic; or enable Developer Mode / run VS Code as admin.

---

## 10) Quality tips

- Keep the **regex fallback**; it guarantees production‑grade robustness for date/score phrasing.
- Expand NER training data with **real queries**; aim for 80–120 examples covering month ranges and score expressions.
- Train **12–20 epochs** for tiny sets; DistilBERT is cheap on CPU.
- For intent with tiny CSVs, include **15–30 examples per class** to stabilize decisions.

---

## 11) CI hooks (suggested)

- Add a smoke test that checks:
  - `get_ner().extract("trends for May 2011 with score 5")` returns non‑empty **or**
  - `build_metadata_from_entities(...)` falls back and yields a valid date range.
- Optionally validate OpenSearch connectivity and basic mapping.

---

## 12) License & contributing

- License: **MIT** (see `LICENSE`).
- PRs welcome! Good first issues:
  - Add more NER examples (month ranges / numeric filters).
  - Improve cluster labeling and summaries.
  - Add configs for other vector stores.

---

## Appendix A – Field mapping reference

These are the field names the runner expects from the **active profile** in `config.py`:

- **Vector**: name of the dense vector field (e.g., `review_embedding`).
- **Text**: one or more fields that hold the full text (e.g., `text`).
- **Timestamp**: a date field used for range filters (e.g., `timestamp`).
- **Score**: numeric (int 1–5) *or* keyword (`"1".."5"`).

The query builder adapts:
- Numeric score → `{"range": {"score": {"gte": lo, "lte": hi}}}`
- Keyword score → `{"terms": {"score": ["4","5"]}}`

Keep ingestion (`tools/ingest/configs/opensearch-ingest-data.yaml`) and runtime (`config.py`) in sync.

---

## Appendix B – Notes on training from different bases

- `dslim/bert-base-NER` is a NER‑tuned base. Even though your head is new (MONTH/YEAR/SCORE), the encoder tends to learn boundaries faster on tiny data.  
- If you have a previously working local model dir, you can **continue training** from that as `--base-model`.
