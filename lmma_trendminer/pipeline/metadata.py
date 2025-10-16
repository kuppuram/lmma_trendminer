# # lmma_trendminer/pipeline/metadata.py
# from typing import Dict, List, Any
# from lmma_trendminer.utils.date_ranges import parse_date_range
# from lmma_trendminer.providers.intent import IntentProvider
# from lmma_trendminer.providers.ner import NERProvider

# # keep your label set in one place (or pass it in from caller)
# DEFAULT_INTENT_LABELS: List[str] = [
#     "product_quality_issue","packaging_issue","delivery_delay",
#     "price_complaint","warranty_service","feature_request",
#     "positive_feedback","spam_or_irrelevant"
# ]

# def build_metadata_from_query(
#     query: str,
#     intent_provider: IntentProvider,
#     ner_provider: NERProvider,
#     *,
#     labels: List[str] = DEFAULT_INTENT_LABELS,
# ) -> Dict[str, Any]:
#     intent_res = intent_provider.predict(query, labels)
#     entities = ner_provider.extract(query)
#     start_date, end_date = parse_date_range(query)

#     return {
#         "intent": intent_res["intent"],
#         "intent_scores": intent_res["scores"],
#         "entities": entities,
#         "start_date": start_date,   # "YYYY-MM-DD" or None
#         "end_date": end_date,       # "YYYY-MM-DD" or None
#     }

# lmma_trendminer/pipeline/metadata.py  (replace helpers + build function)

from typing import Dict, List, Any, Optional, Tuple
import re, calendar
from datetime import datetime

MONTHS = {
    "jan":1,"january":1,"feb":2,"february":2,"mar":3,"march":3,"apr":4,"april":4,
    "may":5,"jun":6,"june":6,"jul":7,"july":7,"aug":8,"august":8,"sep":9,"sept":9,"september":9,
    "oct":10,"october":10,"nov":11,"november":11,"dec":12,"december":12
}

def _span_texts(entities: List[Dict[str, Any]]) -> List[Tuple[str, str]]:
    # collapse B-xxx + I-xxx runs into (label, text)
    spans = []
    prev_lab = None; buf = []
    for e in entities:
        lab = e.get("entity_group","")
        word = (e.get("word") or "").strip()
        if not lab or not word: 
            continue
        if lab.startswith("B-") or (prev_lab and lab != prev_lab and not lab.startswith("I-")):
            if buf and prev_lab:
                spans.append((prev_lab.replace("B-",""), " ".join(buf)))
            buf = [word]; prev_lab = lab
        elif lab.startswith("I-") and prev_lab and lab.replace("I-","") == prev_lab.replace("B-",""):
            buf.append(word)
        else:
            # new unrelated tag
            if buf and prev_lab:
                spans.append((prev_lab.replace("B-",""), " ".join(buf)))
            buf = [word]; prev_lab = lab
    if buf and prev_lab:
        spans.append((prev_lab.replace("B-",""), " ".join(buf)))
    return spans

def _extract_years(text: str) -> List[int]:
    return [int(y) for y in re.findall(r"\b(19|20)\d{2}\b", text)]

def _first_last_month(spans: List[Tuple[str,str]], raw_text: str) -> Tuple[Optional[str], Optional[str]]:
    # Prefer MONTH spans; if none, fall back to DATE spans that include a month word
    month_candidates = [s for t,s in spans if t=="MONTH"]
    date_candidates  = [s for t,s in spans if t=="DATE"]
    parts = month_candidates or date_candidates
    if not parts:
        # last fallback: scan raw text for months
        parts = re.findall(r"\b(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:t|tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\b", raw_text, flags=re.I)
        parts = [p] if parts else []

    if not parts:
        return None, None

    # detect months in order of appearance
    tokens = re.findall(r"\w+", raw_text.lower())
    month_seq = [tok for tok in tokens if tok in MONTHS]
    if not month_seq and parts:
        # try from spans
        month_seq = [p.split()[0].lower() for p in parts if p.split()[0].lower() in MONTHS]

    if not month_seq:
        return None, None

    m1 = MONTHS[month_seq[0]]
    m2 = MONTHS[month_seq[-1]]

    # pick years near/inside the text
    years = _extract_years(raw_text)
    if not years:
        return None, None
    y1 = years[0]; y2 = years[-1]
    # if only one year, use for both
    if len(years) == 1:
        y2 = y1
        # if month order implies wrap (e.g., Nov to Feb) keep same year unless explicitly two years are present

    start = datetime(y1, m1, 1)
    end   = datetime(y2, m2, calendar.monthrange(y2, m2)[1])
    if start > end:
        start, end = end, start
    return start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")

def _extract_rating(spans: List[Tuple[str,str]], raw_text: str) -> Optional[Dict[str,int]]:
    # From SCORE spans, pick the numbers and detect "or above/below" in raw text
    scores = []
    for t, s in spans:
        if t == "SCORE":
            scores += [int(x) for x in re.findall(r"\b[0-5]\b", s)]
    if not scores:
        # quick raw fallback
        m = re.search(r"\b([0-5])\s*(?:star|score)\b", raw_text.lower())
        if m:
            scores = [int(m.group(1))]
    if not scores:
        return None

    lo = min(scores); hi = max(scores)
    t = raw_text.lower()
    if re.search(r"\b(or\s+above|at\s+least|>=)\b", t):
        return {"rating": {"gte": hi}}
    if re.search(r"\b(or\s+below|<=)\b", t):
        return {"rating": {"lte": lo}}
    if len(scores) == 2 and re.search(r"\b[-to]\b", t):
        return {"rating": {"gte": lo, "lte": hi}}
    # default equality to max mention
    return {"rating": {"gte": hi, "lte": hi}}

def build_metadata_from_query(query: str, intent_provider, ner_provider) -> dict:
    
    intent_res = intent_provider.predict(query)     # provider handles its label space
    ents   = ner_provider.extract(query),
    spans = _span_texts(ents)

    start_date, end_date = _first_last_month(spans, query)
    rating = _extract_rating(spans, query)

    return {
        "intent": intent_res["intent"],
        "intent_scores": intent_res["scores"],
        "entities": ents,               # raw spans from your model
        "start_date": start_date,       # from MONTH/DATE logic
        "end_date": end_date,
        "rating": rating,               # {'rating': {...}} or None
    }
