# from typing import List, Dict, Any, Optional, Tuple
# import re, calendar
# from datetime import datetime

# MONTHS = {
#     "jan":1,"january":1,"feb":2,"february":2,"mar":3,"march":3,"apr":4,"april":4,
#     "may":5,"jun":6,"june":6,"jul":7,"july":7,"aug":8,"august":8,"sep":9,"sept":9,"september":9,
#     "oct":10,"october":10,"nov":11,"november":11,"dec":12,"december":12
# }

# def _strip_bio(label: str) -> str:
#     return label[2:] if label.startswith(("B-","I-")) else label

# def _month_num(token: str) -> Optional[int]:
#     t = token.lower().strip()
#     return MONTHS.get(t) or MONTHS.get(t[:3])

# def _month_bounds(y: int, m: int) -> Tuple[str, str]:
#     start = f"{y:04d}-{m:02d}-01"
#     end   = f"{y:04d}-{m:02d}-{calendar.monthrange(y, m)[1]:02d}"
#     return start, end

# def _explode_month_span(text: str) -> List[int]:
#     """
#     Handles single-entity spans like 'apr to may' / 'apr-may' / 'apr–may'.
#     Returns a list of month numbers found in order.
#     """
#     parts = re.split(r"\b(?:to|through|–|-)\b", text, flags=re.I)
#     out: List[int] = []
#     for p in parts:
#         p = p.strip()
#         # keep only the first word fragment per side (e.g., 'apr', 'may')
#         head = p.split()[0] if p else ""
#         m = _month_num(head)
#         if m:
#             out.append(m)
#     return out

# def _score_from_text(raw: str) -> list[int]:
#     return [int(s) for s in re.findall(r"\b([0-5])\b", raw)]

# def _score_range_from_query(raw: str, candidate_scores: list[int]) -> dict | None:
#     """
#     Interpret comparators in the raw query. Returns {'rating': {'gte': x, 'lte': y}} or similar.
#     Examples:
#       "greater than 3" -> gte 4
#       ">= 4" or "at least 4" -> gte 4
#       "less than 4" or "< 4"  -> lte 3
#       "between 3 and 5" or "3-5" -> gte 3, lte 5
#     Falls back to equality if no comparator words found.
#     """
#     t = raw.lower()

#     # between A and B (or A-B)
#     m = re.search(r"\bbetween\s*([0-5])\s*(?:and|to|-)\s*([0-5])\b", t)
#     if not m:
#         m = re.search(r"\b([0-5])\s*[-to]\s*([0-5])\b", t)
#     if m:
#         a, b = int(m.group(1)), int(m.group(2))
#         lo, hi = min(a, b), max(a, b)
#         return {"rating": {"gte": lo, "lte": hi}}

#     # greater than N / above N / > N
#     m = re.search(r"\b(greater\s+than|above|>\s*)([0-5])\b", t)
#     if m:
#         n = int(m.group(2))
#         return {"rating": {"gte": min(5, n + 1)}}

#     # at least N / >= N
#     m = re.search(r"\b(at\s+least|>=)\s*([0-5])\b", t)
#     if m:
#         n = int(m.group(2))
#         return {"rating": {"gte": n}}

#     # less than N / below N / < N
#     m = re.search(r"\b(less\s+than|below|<\s*)([0-5])\b", t)
#     if m:
#         n = int(m.group(2))
#         return {"rating": {"lte": max(0, n - 1)}}

#     # at most N / <= N
#     m = re.search(r"\b(at\s+most|<=)\s*([0-5])\b", t)
#     if m:
#         n = int(m.group(2))
#         return {"rating": {"lte": n}}

#     # no comparators; if we saw a score, default to equality of the *max* mentioned
#     if candidate_scores:
#         hi = max(candidate_scores)
#         # return {"rating": {"gte": hi, "lte": hi}}
#         return {"gte": hi, "lte": hi}

#     return None

# def build_metadata_from_entities(query_text: str, entities: List[Dict[str, Any]]) -> Dict[str, Any]:
#     meta: Dict[str, Any] = {}

#     # 1) normalize
#     norm = []
#     for e in entities:
#         g = _strip_bio(e.get("entity_group","") or "")
#         w = (e.get("word") or "").strip()
#         if g and w:
#             norm.append({"group": g.upper(), "word": w})

#     months: List[int] = []
#     years:  List[int] = []
#     scores: List[int] = []

#     # 2) collect months/years/scores in mention order
#     for e in norm:
#         if e["group"] == "MONTH":
#             text = e["word"]
#             # NEW: handle span inside one token
#             ms = _explode_month_span(text)
#             if ms:
#                 months.extend(ms)
#             else:
#                 m = _month_num(text)
#                 if m:
#                     months.append(m)
#         elif e["group"] in ("YEAR","DATE"):
#             full = re.findall(r"\b(19\d{2}|20\d{2})\b", e["word"])
#             if full:
#                 years += [int(full[0])]
#         elif e["group"] == "SCORE":
#             m = re.search(r"\b([0-5])\b", e["word"])
#             if m:
#                 scores.append(int(m.group(1)))

#     if scores:
#         meta["score"] = max(scores)  # or last/first if you prefer
#         score_range = _score_range_from_query(query_text, scores)
#         if score_range:
#             meta["score_range"] = score_range  # note: key 'rating' holds dict {'gte':..,'lte':..} etc.
    
#     # 3) build date range
#     if months and years:
#         y1 = years[0]
#         y2 = years[-1] if len(years) > 1 else y1
#         m1 = months[0]
#         m2 = months[-1]
#         s = datetime(y1, m1, 1)
#         e = datetime(y2, m2, calendar.monthrange(y2, m2)[1])
#         if s > e:
#             s, e = e, s
#         meta["start_date"] = s.strftime("%Y-%m-%d")
#         meta["end_date"]   = e.strftime("%Y-%m-%d")
#         return meta

#     if len(months) == 1 and len(years) == 1:
#         meta["start_date"], meta["end_date"] = _month_bounds(years[0], months[0])
#         return meta

#     if not months and len(years) == 1:
#         y = years[0]
#         meta["start_date"], meta["end_date"] = f"{y}-01-01", f"{y}-12-31"
#         return meta
    
#     return meta

# lmma_trendminer/pipeline/metadata_from_entities.py
from __future__ import annotations
import re, calendar
from typing import List, Dict, Any, Optional

# --- helpers for months ---
from calendar import month_abbr, month_name
_MONTHS = {m.lower(): i for i, m in enumerate(month_name) if m}
_MONTHS.update({m.lower(): i for i, m in enumerate(month_abbr) if m})
def _mm(x: str) -> Optional[int]:
    return _MONTHS.get(x.strip(".").lower())

def _month_last_day(y: int, m: int) -> str:
    return f"{y:04d}-{m:02d}-{calendar.monthrange(y, m)[1]:02d}"

# --- regex fallback if NER misses ---
def _fallback_parse(query: str) -> Dict[str, Any] | None:
    q = query.lower()

    # e.g. "nov to dec 2011", "apr to may 2011"
    m = re.search(r"\b([a-z]{3,9})\s+to\s+([a-z]{3,9})\s+(\d{4})\b", q)
    if m:
        m1, m2, y = m.groups()
        M1, M2 = _mm(m1), _mm(m2)
        if M1 and M2:
            y = int(y)
            start = f"{y:04d}-{M1:02d}-01"
            end   = _month_last_day(y, M2)
            rating = None
            # "high 5 score" or "score 5" or "5 score"
            if re.search(r"\b(high\s*)?5\s*score\b|\bscore\s*5\b", q):
                rating = {"gte": 5, "lte": 5}
            elif re.search(r"\bscore\s*([0-9])\b", q):
                s = int(re.search(r"\bscore\s*([0-9])\b", q).group(1))
                rating = {"gte": s, "lte": s}
            elif re.search(r"(>=|=>|above|greater than)\s*([0-9])", q):
                s = int(re.search(r"([0-9])", q).group(1))
                rating = {"gte": s}
            return {"start_date": start, "end_date": end, "rating": rating}

    # e.g. "may 2011"
    m = re.search(r"\b([a-z]{3,9})\s+(\d{4})\b", q)
    if m:
        mon, y = m.groups()
        M = _mm(mon)
        if M:
            y = int(y)
            start = f"{y:04d}-{M:02d}-01"
            end   = _month_last_day(y, M)
            return {"start_date": start, "end_date": end}

    return None

def build_metadata_from_entities(query: str, entities: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    entities: aggregated output from HF pipeline with entity_group normalized by custom_ner.py
    Expected groups: MONTH, YEAR, SCORE (we normalize 'B-'/'I-' or 'LABEL_*' inside custom_ner)
    """
    meta: Dict[str, Any] = {}
    months: List[str] = []
    year: Optional[int] = None
    rating_val: Optional[int] = None

    # 1) Use NER if available
    for e in entities:
        grp = (e.get("entity_group") or "").upper()
        word = (e.get("word") or "").strip()
        if grp == "MONTH":
            months.extend([w for w in word.split() if _mm(w)])  # allow "apr to may" span
        elif grp == "YEAR":
            try:
                year = int(re.sub(r"\D", "", word))
            except Exception:
                pass
        elif grp == "SCORE":
            try:
                rating_val = int(re.sub(r"\D", "", word))
            except Exception:
                pass

    # (a) month range + year
    if len(months) >= 2 and year:
        m1, m2 = _mm(months[0]), _mm(months[1])
        if m1 and m2:
            meta["start_date"] = f"{year:04d}-{m1:02d}-01"
            meta["end_date"]   = _month_last_day(year, m2)

    # (b) single month + year
    if "start_date" not in meta and len(months) == 1 and year:
        m1 = _mm(months[0])
        if m1:
            meta["start_date"] = f"{year:04d}-{m1:02d}-01"
            meta["end_date"]   = _month_last_day(year, m1)

    # (c) score → rating bounds
    if rating_val is not None:
        meta["rating"] = {"gte": rating_val, "lte": rating_val}

    # 2) Fallback if NER incomplete
    if "start_date" not in meta or "end_date" not in meta:
        fb = _fallback_parse(query)
        if fb:
            meta.update({k: v for k, v in fb.items() if v is not None})

    return meta
