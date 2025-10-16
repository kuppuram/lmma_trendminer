# test3.py
from opensearchpy import OpenSearch
from lmma_trendminer.providers.vector_store import OpenSearchStore
from lmma_trendminer.providers.registry import configure_models
from lmma_trendminer.runner import analyze_query_and_summarize

configure_models(intent_path=r".\models\intent-classifier", ner_path=r".\models\ner-extractor")

client = OpenSearch(hosts=[{"host":"localhost","port":9200}])
store = OpenSearchStore(client, index="amazon-food-reviews", vector_field="review_embedding")

query = "top trends in Foods for Nov to Dec 2011 which are high 5 score"

summary, details = analyze_query_and_summarize(
    query,
    store,
    min_reviews=1000,
    text_field="text",
)

print("=== SUMMARY ===")
print(summary)
print("=== META ===")
print(details.get("meta"))
