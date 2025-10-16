# # test2.py
# test2.py
import os
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
from dotenv import load_dotenv; load_dotenv()

from opensearchpy import OpenSearch
from lmma_trendminer.providers.vector_store import OpenSearchStore
from lmma_trendminer.runner import analyze_query_and_summarize
from lmma_trendminer.providers.registry import configure_models
from lmma_trendminer.summarizers.groq_sum import groq_summarizer   # <-- add

# configure_models(intent_path="./models_old/intent-classifier", ner_path="./models_old/ner-extractor")
configure_models(intent_path="./models/intent-classifier", ner_path="./models/ner-extractor")

client = OpenSearch(hosts=[{"host":"localhost","port":9200}])
# store index still passed here (transport + default index), but runner will use fields from profile
store = OpenSearchStore(client, index="amazon-food-reviews", vector_field="review_embedding")

# query = "top trends in Foods for May 2011 which got score greater than 3"

# query = "top trends in Foods for Apr to May 2011 which are high 5 score"

# query = "top trends in Foods for Apr to May 2011 which got score greater than 3"

# query = "top trends in Foods for Nov to Dec 2011 which got score greater than 3"

query = "top trends in Foods for Nov to Dec 2011 which are high 5 score"

summary, details = analyze_query_and_summarize(
    query,
    store,
    profile="amazon_food_reviews",     # <-- tell runner which profile block to use
    min_reviews=1000,
    umap_cfg={"n_components": 20, "n_neighbors": 30, "min_dist": 0.0, "metric": "cosine"},
    hdbscan_cfg={"min_cluster_size": 10, "metric": "euclidean"},
    summarizer=groq_summarizer,      # optional
)

print("\n=== SUMMARY ===")
print(summary)
print("\n=== DETAILS META ===")
print(details.get("meta"))
print("\n=== FIELDS USED ===")
print(details.get("fields"))


# import os
# os.environ["TRANSFORMERS_OFFLINE"] = "1"
# os.environ["HF_HUB_OFFLINE"] = "1"
# # os.environ["GROQ_API_KEY"] = "..."   # set in your env

# from opensearchpy import OpenSearch
# from lmma_trendminer.providers.vector_store import OpenSearchStore
# from lmma_trendminer.runner import analyze_query_and_summarize
# from lmma_trendminer.providers.registry import configure_models
# from lmma_trendminer.summarizers.groq_sum import groq_summarizer   # <-- add

# configure_models(intent_path="./models/intent-classifier", ner_path="./models/ner-extractor")

# client = OpenSearch(hosts=[{"host":"localhost","port":9200}])
# store = OpenSearchStore(client, index="amazon-food-reviews", vector_field="review_embedding")

# # query = "top trends in Foods for Apr to May 2011 which are high 5 score"
# query = "top trends in Foods for May 2011 which got score greater than 3 score"

# summary, details = analyze_query_and_summarize(
#     query,
#     store,
#     min_reviews=1000,
#     text_field="text",
#     umap_cfg={"n_components": 20, "n_neighbors": 30, "min_dist": 0.0, "metric": "cosine"},
#     hdbscan_cfg={"min_cluster_size": 10, "metric": "euclidean"},
#     summarizer=groq_summarizer,   # <-- pass it here
# )

# print("\n=== SUMMARY ===")
# print(summary)
# print("\n=== META ===")
# print(details.get("meta"))
