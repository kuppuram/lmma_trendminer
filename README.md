# lmma_trendminer

> **30-second trend mining** over large text corpora (e.g., Amazon reviews).  
> Parse a natural-language query → fetch **vectors only** → **UMAP** → **HDBSCAN** → pick a few representative docs → output clean trend bullets (optional LLM titles).

This is **not** RAG. We cluster many documents to surface themes, not fetch a handful of passages to answer a single question.

---

## Key features

- **Vector-only retrieval** (OpenSearch): tiny payloads, fast.
- **UMAP → HDBSCAN**: automatic cluster count, outlier filtering.
- **Smart labeling**: TF-IDF hints + optional LLM summarizer (Groq/OpenAI).
- **Robust query parsing**: custom NER (**MONTH/YEAR/SCORE**) + **regex fallback** so dates/scores are extracted even with tiny NER data.
- **Configurable field mapping**: easily map your index fields (text, timestamp, score, vector).

---

## Quickstart

> Requires Python 3.10+ and a running OpenSearch 2.x (default: `localhost:9200`).

```bash
# 1) clone
git clone https://github.com/<you>/lmma_trendminer.git
cd lmma_trendminer

# 2) create venv
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# 3) install
pip install -U pip
pip install -e ".[ingest,train,llm]"
# If you want CPU-only torch:
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### Index data (example)

Provide an OpenSearch index with at least these fields:

- `review_embedding` (dense vector)
- `text` (full text)
- `timestamp` (`YYYY-MM-DD`)
- `score` (int 1–5 or keyword `"1".."5"`)

See `tools/ingest/configs/opensearch-ingest-data.yaml` as a template.

### Run a sample query

```python
# test3.py
from opensearchpy import OpenSearch
from lmma_trendminer.providers.vector_store import OpenSearchStore
from lmma_trendminer.providers.registry import configure_models
from lmma_trendminer.runner import analyze_query_and_summarize

configure_models(intent_path=r"./models/intent-classifier",
                 ner_path=r"./models/ner-extractor")

client = OpenSearch(hosts=[{"host":"localhost","port":9200}])
store = OpenSearchStore(client, index="amazon-food-reviews", vector_field="review_embedding")

query = "top trends in Foods for Nov to Dec 2011 which are high 5 score"

summary, details = analyze_query_and_summarize(
    query,
    store,
    min_reviews=1000,
    text_field="text",
)

print("=== SUMMARY ===\\n", summary)
print("=== META ===\\n", details.get("meta"))
```

```bash
python test3.py
```

You should see:

- **META** like `{'start_date':'2011-11-01','end_date':'2011-12-31','rating':{'gte':5,'lte':5}}`
- Trend bullets with **(n=cluster_size)** (or richer titles if you enable an LLM summarizer).

---

## How it works

1. **Parse query → metadata**  
   - `providers/custom_ner.py` wraps your trained NER with **aggregation** and **label normalization**.  
   - `pipeline/metadata_from_entities.py` converts entities to `{start_date,end_date,rating}`.  
   - If NER misses, a **regex fallback** extracts dates/scores (e.g., “Nov to Dec 2011”, “>=4 score”, “high 5 score”).

2. **Vector-only retrieval**  
   - `providers/vector_store.py` issues a search that returns **only IDs + vectors** (and filters by date/score).  
   - No heavy text transfer.

3. **UMAP → HDBSCAN**  
   - `pipeline/reduce.py` (UMAP) → `pipeline/cluster.py` (HDBSCAN).  
   - HDBSCAN picks the number of clusters automatically, marks noise.

4. **Label & summarize**  
   - `pipeline/label.py`: TF-IDF keywords per cluster.  
   - Representative example retrieval via `mget`.  
   - Optional LLM titles (e.g., Groq) → polished trend bullets.

---

## Model training (optional but supported)

### NER (MONTH/YEAR/SCORE)

**Data format** (JSONL, aligned tokens/labels):

```json
{"tokens":["reviews","from","jan","to","march","2014"],
 "labels":["O","O","B-MONTH","I-MONTH","I-MONTH","B-YEAR"]}
```

You can keep seed data as Python rows and convert:

```bash
python -m tools.train.make_ner_jsonl_from_py tools.train.datasets.ner_data \
  --train-out ./data/ner_train.jsonl \
  --eval-out  ./data/ner_eval.jsonl
```

**Train** (base can be `distilbert-base-uncased` or `dslim/bert-base-NER` for faster convergence on tiny data):

```bash
python -m tools.train.ner_cli \
  --train-file ./data/ner_train.jsonl \
  --eval-file  ./data/ner_eval.jsonl \
  --base-model dslim/bert-base-NER \
  --output-dir ./models/ner-extractor \
  --epochs 12 --batch-size 8
```

### Intent (trend_analysis / simple_search / greeting)

CSV with `text,label`. Train:

```bash
python -m tools.train.intent_cli ./data/intent_data.csv \
  --output-dir ./models/intent-classifier \
  --epochs 5 --batch-size 16 --val-ratio 0.3
```

> With very small datasets, increase `--val-ratio` or skip eval.

---

## Configuration

- **Model paths**: set once at startup

```python
from lmma_trendminer.providers.registry import configure_models
configure_models(intent_path="./models/intent-classifier",
                 ner_path="./models/ner-extractor")
```

- **Index field mapping**: see `config.py` (profile) and `tools/ingest/configs/opensearch-ingest-data.yaml`.  
  The runner adapts the score filter to numeric **range** or keyword **terms** automatically.

---

## Optional: LLM summarizer

Add a Groq/OpenAI summarizer for prettier trend titles:

```python
from lmma_trendminer.summarizers.groq_sum import groq_summarizer
summary, details = analyze_query_and_summarize(
    query, store, text_field="text", summarizer=groq_summarizer
)
```

Set `GROQ_API_KEY` in your environment.

---

## Troubleshooting

- **`META {}`**  
  Ensure you’re calling `build_metadata_from_entities(...)` and that `metadata_from_entities.py` includes the **regex fallback**. Add debug prints:

  ```python
  ents = get_ner().extract(query)
  print("[DEBUG NER OUTPUT]", ents)
  print("[DEBUG META]", meta)
  ```

- **Score filter not applied**  
  Confirm `meta["rating"] = {"gte": X, "lte": Y}` is present. The runner maps this to OS `range`/`terms` depending on your score field type.

- **Transformers kwarg errors**  
  Upgrade:  
  `pip install -U "transformers>=4.44,<5" "accelerate>=0.34" "datasets>=2.19" "seqeval>=1.2.2"`

- **Custom NER not used**  
  Make sure code calls `providers.registry.get_ner()` (not a raw HF `pipeline`). Add debug in `registry.get_ner()` to print model path.

- **Windows symlink warning (HF cache)**  
  Safe to ignore; or enable Windows Developer Mode / run as admin.

---

## Why a regex fallback?

Tiny, custom NER datasets can miss spans—users also write dates in many shapes (`Nov–Dec ’11`, `>=4 score`). The fallback ensures you **always** extract dates/scores and keep latency low. As you add NER data/epochs, reliance on the fallback naturally decreases.

---

## License

MIT (see `LICENSE`).

---

## Acknowledgements

- UMAP (McInnes et al.), HDBSCAN (Campello et al.)
- Hugging Face Transformers & Datasets
- OpenSearch Python client

---

## Contributing

PRs welcome!  
Good first issues:
- Add more NER examples (month ranges, numeric filters).
- Improve cluster labeling and summaries.
- Add configs for other vector stores.
