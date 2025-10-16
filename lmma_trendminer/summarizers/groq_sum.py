# # lmma_trendminer/summarizers/groq_sum.py
# from typing import List
# import os
# from groq import Groq
# from dotenv import load_dotenv

# load_dotenv()

# def groq_summarizer(rep_texts: List[str]) -> str:
#     """
#     Takes one representative text per cluster and returns a concise summary.
#     Keep it pure: input -> output string.
#     """
#     if not rep_texts:
#         return "No representative texts to summarize."

#     prompt_context = "\n\n".join(
#         [f"Trend Example {i+1}: \"{t}\"" for i, t in enumerate(rep_texts)]
#     )
#     prompt = f"""
# You are a senior product analyst. Identify and summarize distinct trends from the
# representative customer reviews below. For each trend, produce a short title and
# one-sentence explanation. Keep it bullet-pointed and tight.

# {prompt_context}

# Return only the bullet list.
# """.strip()

#     client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
#     chat = client.chat.completions.create(
#         model="llama-3.1-8b-instant",
#         messages=[{"role":"user","content":prompt}],
#         temperature=0.2,
#         max_tokens=400,
#     )
#     return chat.choices[0].message.content.strip()

# lmma_trendminer/summarizers/groq_sum.py
from typing import List
import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()


def groq_summarizer(rep_texts: List[str]) -> str:
    if not rep_texts:
        return "No representative texts to summarize."
    # prompt_context = "\n\n".join([f"[Cluster {i+1}] {t}" for i, t in enumerate(rep_texts)])
    prompt_context = "\n".join([f"Trend Example {i+1}: \"{text}\"" for i, text in enumerate(rep_texts)])

    prompt = f"""
You are summarizing CLUSTERS of customer reviews. Each item below is the most
representative review of a DISTINCT trend (one example per cluster).

TASK:
- Produce ONE bullet per cluster, in the SAME ORDER.
- Format EXACTLY as: "• **<Short, specific trend title>** — <one-sentence concrete description>"
- Use nouns/phrases found in the example (brands, product types, complaints).
- Avoid generic phrases like “value for money,” “quality,” “customer service,” unless they are explicit.

INPUT EXAMPLES (one per cluster):
{prompt_context}

OUTPUT:
- A bullet list with exactly {len(rep_texts)} bullets, one per cluster, in order.
- No preamble or conclusion.
""".strip()

    # prompt = f"""
    # You are a senior product analyst AI. Your task is to identify and summarize the top trends from recent customer feedback.
    # I have already clustered the feedback and am providing you with the single most representative review from each distinct trend.

    # Analyze these representative reviews and generate a concise summary report. For each trend, give it a short title and a one-sentence description.

    # {prompt_context}

    # Provide your output as a simple list of trends.
    # """
    client = Groq(api_key=os.environ["GROQ_API_KEY"])
    chat = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=500,
    )
    return chat.choices[0].message.content.strip()
