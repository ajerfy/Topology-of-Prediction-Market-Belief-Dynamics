from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import pandas as pd


CRYPTO_KEYWORDS = [
    "bitcoin",
    "btc",
    "ethereum",
    "eth",
    "microstrategy",
    "mstr",
    "etf",
    "reserve",
    "stablecoin",
    "crypto",
]

EXCLUDE_PATTERNS = [
    re.compile(r"\binsider trading\b", re.I),
    re.compile(r"\baccused\b", re.I),
    re.compile(r"\bsatoshi move\b", re.I),
]


@dataclass(frozen=True)
class MarketClassification:
    market_family: str
    asset: str | None
    threshold: float | None
    direction: str | None
    target_date: str | None
    is_core: bool
    is_satellite: bool
    selection_reason: str | None
    exclusion_reason: str | None


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).lower()


def market_text(row: pd.Series) -> str:
    tags = row.get("tags")
    if isinstance(tags, (list, tuple)):
        tags_text = " ".join(str(tag) for tag in tags)
    else:
        tags_text = str(tags or "")
    return " ".join(
        clean_text(row.get(field))
        for field in ["question", "slug", "description", "category"]
    ) + " " + tags_text.lower()


def is_crypto_candidate(row: pd.Series) -> bool:
    text = market_text(row)
    return any(re.search(rf"\b{re.escape(keyword)}\b", text) for keyword in CRYPTO_KEYWORDS)


def extract_threshold(text: str) -> float | None:
    patterns = [
        r"\$([0-9]+(?:,[0-9]{3})*(?:\.[0-9]+)?)\s*(k|m|b)?",
        r"\b([0-9]+(?:,[0-9]{3})*(?:\.[0-9]+)?)\s*(k|m|b)\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.I)
        if not match:
            continue
        raw = match.group(1).replace(",", "")
        multiplier = (match.group(2) or "").lower()
        value = float(raw)
        if multiplier == "k":
            value *= 1_000
        elif multiplier == "m":
            value *= 1_000_000
        elif multiplier == "b":
            value *= 1_000_000_000
        return value
    return None


def infer_direction(text: str) -> str | None:
    if re.search(r"\b(dip|below|under|less than)\b", text):
        return "below"
    if re.search(r"\b(reach|hit|above|all time high|ath|greater than)\b", text):
        return "above"
    return None


def classify_market(row: pd.Series) -> MarketClassification:
    text = market_text(row)
    question = clean_text(row.get("question"))
    target_date = row.get("end_date")

    if not is_crypto_candidate(row):
        return MarketClassification("excluded", None, None, None, target_date, False, False, None, "not_crypto_candidate")

    for pattern in EXCLUDE_PATTERNS:
        if pattern.search(text):
            return MarketClassification("excluded", None, None, None, target_date, False, False, None, "crypto_news_or_identity_false_positive")

    threshold = extract_threshold(question)
    direction = infer_direction(question)

    if re.search(r"\b(bitcoin|btc)\b", text) and threshold is not None and direction is not None:
        return MarketClassification("btc_price", "BTC", threshold, direction, target_date, True, False, "btc_price_threshold", None)

    if re.search(r"\b(ethereum|eth)\b", text) and threshold is not None and direction is not None:
        return MarketClassification("eth_price", "ETH", threshold, direction, target_date, True, False, "eth_price_threshold", None)

    if re.search(r"\b(microstrategy|mstr)\b", text):
        return MarketClassification("microstrategy", "MSTR", threshold, direction, target_date, False, True, "microstrategy_crypto_exposure", None)

    if re.search(r"\betf\b", text):
        asset = "BTC" if re.search(r"\b(bitcoin|btc)\b", text) else "ETH" if re.search(r"\b(ethereum|eth)\b", text) else None
        return MarketClassification("crypto_etf", asset, threshold, direction, target_date, False, True, "crypto_etf_policy", None)

    if re.search(r"\b(reserve|stablecoin|regulation|bill|senate|congress|sec|cftc|policy)\b", text):
        return MarketClassification("crypto_policy", None, threshold, direction, target_date, False, True, "crypto_policy", None)

    return MarketClassification("excluded", None, threshold, direction, target_date, False, False, None, "generic_crypto_not_core_or_satellite")


def classify_markets(markets: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in markets.iterrows():
        classification = classify_market(row)
        enriched = row.to_dict()
        enriched.update(classification.__dict__)
        rows.append(enriched)
    return pd.DataFrame(rows)
