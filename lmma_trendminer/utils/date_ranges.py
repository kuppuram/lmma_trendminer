import re
from datetime import datetime, timedelta
import calendar
import dateparser

# Default to Asia/Kolkata per your environment
DEFAULT_TZ = "Asia/Kolkata"

ISO = "%Y-%m-%d"

def _to_date(s: str, now=None):
    return dateparser.parse(
        s,
        settings={
            "TIMEZONE": DEFAULT_TZ,
            "RETURN_AS_TIMEZONE_AWARE": False,
            "PREFER_DAY_OF_MONTH": "first",
            "RELATIVE_BASE": now or datetime.now(),
        },
    )

def _month_bounds(dt: datetime):
    start = dt.replace(day=1)
    last_day = calendar.monthrange(dt.year, dt.month)[1]
    end = dt.replace(day=last_day)
    return start, end

def _year_bounds(dt: datetime):
    return dt.replace(month=1, day=1), dt.replace(month=12, day=31)

def parse_date_range(text: str, now: datetime | None = None) -> tuple[str | None, str | None]:
    """
    Returns (start_date_iso, end_date_iso) or (None, None) if nothing found.
    """
    now = now or datetime.now()

    # 1) Explicit ISO range like 2011-04-01 to 2011-04-30 / through / –
    m = re.search(r"(\d{4}-\d{2}-\d{2})\s*(?:to|through|–|-)\s*(\d{4}-\d{2}-\d{2})", text, re.I)
    if m:
        s = datetime.fromisoformat(m.group(1))
        e = datetime.fromisoformat(m.group(2))
        if s > e:
            s, e = e, s
        return s.strftime(ISO), e.strftime(ISO)

    # 2) "from X to Y" / "between X and Y"
    m = re.search(r"(?:from|between)\s+(.+?)\s+(?:to|and)\s+(.+)", text, re.I)
    if m:
        s = _to_date(m.group(1), now=now)
        e = _to_date(m.group(2), now=now)
        if s and e:
            if s > e: s, e = e, s
            return s.strftime(ISO), e.strftime(ISO)

    t = text.lower().strip()

    # 3) Relative phrases
    if "last 7 days" in t or "past 7 days" in t or "previous 7 days" in t:
        s, e = now - timedelta(days=7), now
        return s.strftime(ISO), e.strftime(ISO)

    if "last week" in t:
        # interpret as previous 7 days (simple & fast)
        s, e = now - timedelta(days=7), now
        return s.strftime(ISO), e.strftime(ISO)

    if "last month" in t:
        y, mth = (now.year - 1, 12) if now.month == 1 else (now.year, now.month - 1)
        s = datetime(y, mth, 1)
        e = datetime(y, mth, calendar.monthrange(y, mth)[1])
        return s.strftime(ISO), e.strftime(ISO)

    if "yesterday" in t:
        y = now - timedelta(days=1)
        return y.strftime(ISO), y.strftime(ISO)

    if "today" in t:
        return now.strftime(ISO), now.strftime(ISO)

    # 4) Single month like "April 2011" or "Apr 2011" or "April"
    m = _to_date(text, now=now)
    if m:
        # If the text mentions a month name, treat as month span
        if re.search(r"\b(jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec|january|february|march|april|may|june|july|august|september|october|november|december)\b", t):
            s, e = _month_bounds(m)
            return s.strftime(ISO), e.strftime(ISO)
        # If the text looks like a bare year
        if re.search(r"\b\d{4}\b", t) and not re.search(r"\d{4}-\d{2}-\d{2}", t):
            s, e = _year_bounds(m)
            return s.strftime(ISO), e.strftime(ISO)

    # 5) Fallback: try to extract any date(s) present and make a sensible range
    # Find all date-like mentions
    dates = []
    for chunk in re.findall(r"(\d{4}-\d{2}-\d{2}|\d{1,2}\s+\w+\s+\d{4}|\w+\s+\d{4})", text):
        d = _to_date(chunk, now=now)
        if d:
            dates.append(d)
    if len(dates) == 1:
        d = dates[0]
        return d.strftime(ISO), d.strftime(ISO)
    if len(dates) >= 2:
        s, e = min(dates), max(dates)
        return s.strftime(ISO), e.strftime(ISO)

    return None, None
