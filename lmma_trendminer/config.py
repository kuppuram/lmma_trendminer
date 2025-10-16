# # lmma_trendminer/config.py
# from dataclasses import dataclass, field
# from typing import List, Dict, Optional, Any
# import os, yaml

# @dataclass
# class ScoreCfg:
#     name: str
#     type: str = "keyword"   # "numeric" or "keyword"
#     min: int = 0
#     max: int = 5

# @dataclass
# class FieldsCfg:
#     id: Optional[str] = None
#     vector: str = "embedding"
#     text: List[str] = field(default_factory=lambda: ["text"])
#     timestamp: str = "timestamp"
#     score: ScoreCfg = field(default_factory=lambda: ScoreCfg(name="score"))
#     category: Optional[str] = None
#     brand: Optional[str] = None
#     product: Optional[str] = None

# @dataclass
# class ProfileCfg:
#     index: str
#     fields: FieldsCfg
#     query_term_map: Dict[str, List[str]] = field(default_factory=dict)
#     synonyms: Dict[str, List[str]] = field(default_factory=dict)

# @dataclass
# class AppCfg:
#     version: int
#     profiles: Dict[str, ProfileCfg]

# def _as_score(d: Dict[str, Any]) -> ScoreCfg:
#     return ScoreCfg(**d)

# def _as_fields(d: Dict[str, Any]) -> FieldsCfg:
#     d = dict(d)
#     d["score"] = _as_score(d.get("score", {"name": "score"}))
#     return FieldsCfg(**d)

# def _as_profile(d: Dict[str, Any]) -> ProfileCfg:
#     d = dict(d)
#     d["fields"] = _as_fields(d["fields"])
#     return ProfileCfg(**d)

# def load_config(path: str = None) -> AppCfg:
#     path = path or os.getenv("LMMA_TRENDMINER_CONFIG", "lmma_trendminer/config.yaml")
#     with open(path, "r", encoding="utf-8") as f:
#         raw = yaml.safe_load(f)
#     profs = {k: _as_profile(v) for k, v in raw["profiles"].items()}
#     return AppCfg(version=int(raw["version"]), profiles=profs)

# lmma_trendminer/config.py

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
import os, yaml

@dataclass
class ScoreCfg:
    name: str
    type: str = "keyword"     # "numeric" or "keyword"
    min: int = 0
    max: int = 5
    cast: str = "int"         # NEW: "int" or "string" (for keyword fields stored as strings)

@dataclass
class FieldsCfg:
    id: Optional[str] = None
    vector: str = "embedding"
    text: List[str] = field(default_factory=lambda: ["text"])
    timestamp: str = "timestamp"
    score: ScoreCfg = field(default_factory=lambda: ScoreCfg(name="score"))
    category: Optional[str] = None
    brand: Optional[str] = None
    product: Optional[str] = None

@dataclass
class ProfileCfg:
    index: str
    fields: FieldsCfg
    query_term_map: Dict[str, List[str]] = field(default_factory=dict)
    synonyms: Dict[str, List[str]] = field(default_factory=dict)

@dataclass
class AppCfg:
    version: int
    profiles: Dict[str, ProfileCfg]

def _as_score(d: Dict[str, Any]) -> ScoreCfg:
    # Keep only known keys to avoid future surprises
    allowed = {"name", "type", "min", "max", "cast"}
    return ScoreCfg(**{k: v for k, v in d.items() if k in allowed})

def _as_fields(d: Dict[str, Any]) -> FieldsCfg:
    d = dict(d)
    d["score"] = _as_score(d.get("score", {"name": "score"}))
    return FieldsCfg(**d)

def _as_profile(d: Dict[str, Any]) -> ProfileCfg:
    d = dict(d)
    d["fields"] = _as_fields(d["fields"])
    return ProfileCfg(**d)

def load_config(path: str = None) -> AppCfg:
    path = path or os.getenv("LMMA_TRENDMINER_CONFIG", "lmma_trendminer/config.yaml")
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    profs = {k: _as_profile(v) for k, v in raw["profiles"].items()}
    return AppCfg(version=int(raw["version"]), profiles=profs)
