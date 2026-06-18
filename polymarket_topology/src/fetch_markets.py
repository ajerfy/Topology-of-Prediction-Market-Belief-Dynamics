from __future__ import annotations

import argparse
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from utils import (
    GAMMA_BASE_URL,
    ensure_dirs,
    parse_jsonish,
    project_root,
    request_json,
    safe_float,
    setup_logging,
    str_to_bool,
    write_json,
)


RAW_FIELDS = [
    "id",
    "conditionId",
    "question",
    "title",
    "category",
    "outcomes",
    "outcomePrices",
    "clobTokenIds",
    "startDate",
    "endDate",
    "closedTime",
    "active",
    "closed",
    "archived",
    "volume",
    "volumeNum",
    "liquidity",
    "liquidityNum",
    "umaResolutionStatus",
    "umaResolutionStatuses",
    "resolvedBy",
    "resolutionSource",
    "events",
    "tags",
    "slug",
    "description",
    "enableOrderBook",
    "volumeClob",
]


def market_event_id(market: dict[str, Any]) -> str | None:
    events = market.get("events") or []
    if isinstance(events, list) and events:
        return str(events[0].get("id")) if events[0].get("id") is not None else None
    return None


def market_tags(market: dict[str, Any]) -> list[str]:
    tags: list[str] = []
    for source in [market.get("tags") or []]:
        if isinstance(source, list):
            for tag in source:
                if isinstance(tag, dict):
                    label = tag.get("label") or tag.get("slug")
                    if label:
                        tags.append(str(label))
                elif tag:
                    tags.append(str(tag))
    events = market.get("events") or []
    if isinstance(events, list):
        for event in events:
            for tag in event.get("tags") or []:
                if isinstance(tag, dict):
                    label = tag.get("label") or tag.get("slug")
                    if label:
                        tags.append(str(label))
    return sorted(set(tags))


def market_category(market: dict[str, Any]) -> str | None:
    if market.get("category"):
        return str(market.get("category"))
    events = market.get("events") or []
    if isinstance(events, list):
        for event in events:
            if isinstance(event, dict) and event.get("category"):
                return str(event.get("category"))
    return None


def normalize_market(market: dict[str, Any]) -> dict[str, Any]:
    outcomes = parse_jsonish(market.get("outcomes"), [])
    prices = parse_jsonish(market.get("outcomePrices"), [])
    token_ids = parse_jsonish(market.get("clobTokenIds"), [])

    yes_idx = None
    for idx, outcome in enumerate(outcomes):
        if str(outcome).strip().lower() == "yes":
            yes_idx = idx
            break

    yes_token_id = None
    yes_price = None
    if yes_idx is not None:
        if yes_idx < len(token_ids):
            yes_token_id = str(token_ids[yes_idx])
        if yes_idx < len(prices):
            yes_price = safe_float(prices[yes_idx])

    resolved_outcome = None
    if prices and outcomes and len(prices) == len(outcomes):
        numeric_prices = [safe_float(price) for price in prices]
        if any(price is not None for price in numeric_prices):
            max_idx = max(
                range(len(numeric_prices)),
                key=lambda idx: numeric_prices[idx] if numeric_prices[idx] is not None else -1,
            )
            if numeric_prices[max_idx] is not None and numeric_prices[max_idx] >= 0.99:
                resolved_outcome = str(outcomes[max_idx])

    return {
        "market_id": str(market.get("id")),
        "condition_id": market.get("conditionId"),
        "event_id": market_event_id(market),
        "question": market.get("question") or market.get("title"),
        "category": market_category(market),
        "tags": market_tags(market),
        "outcomes": outcomes,
        "outcome_prices": [safe_float(price) for price in prices],
        "clob_token_ids": [str(token_id) for token_id in token_ids],
        "yes_token_id": yes_token_id,
        "yes_price_current": yes_price,
        "start_date": market.get("startDate"),
        "end_date": market.get("endDate"),
        "close_date": market.get("closedTime"),
        "active": bool(market.get("active")) if market.get("active") is not None else None,
        "closed": bool(market.get("closed")) if market.get("closed") is not None else None,
        "archived": bool(market.get("archived")) if market.get("archived") is not None else None,
        "resolved_outcome": resolved_outcome,
        "volume": safe_float(market.get("volumeNum") or market.get("volume")),
        "liquidity": safe_float(market.get("liquidityNum") or market.get("liquidity")),
        "volume_clob": safe_float(market.get("volumeClob")),
        "uma_resolution_status": market.get("umaResolutionStatus"),
        "uma_resolution_statuses": parse_jsonish(market.get("umaResolutionStatuses"), []),
        "resolved_by": market.get("resolvedBy"),
        "resolution_source": market.get("resolutionSource"),
        "enable_order_book": market.get("enableOrderBook"),
        "slug": market.get("slug"),
        "description": market.get("description"),
        "is_binary": len(outcomes) == 2 and {str(o).lower() for o in outcomes} == {"yes", "no"},
        "has_yes_token": yes_token_id is not None,
    }


def fetch_markets(
    *,
    limit: int,
    closed: bool,
    page_size: int,
    order: str,
    ascending: bool,
    category: str | None,
) -> list[dict[str, Any]]:
    session = requests.Session()
    markets: list[dict[str, Any]] = []
    offset = 0
    while len(markets) < limit:
        batch_limit = min(page_size, limit - len(markets))
        params: dict[str, Any] = {
            "limit": batch_limit,
            "offset": offset,
            "closed": str(closed).lower(),
            "order": order,
            "ascending": str(ascending).lower(),
        }
        if category:
            params["category"] = category
        payload = request_json(session, f"{GAMMA_BASE_URL}/markets", params=params)
        if not payload:
            logging.info("No payload at offset=%s; stopping", offset)
            break
        if not isinstance(payload, list):
            logging.warning("Unexpected markets payload type: %s", type(payload).__name__)
            break
        markets.extend(payload)
        logging.info("Fetched %s/%s markets", len(markets), limit)
        if len(payload) < batch_limit:
            break
        offset += len(payload)
    return markets[:limit]


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch and clean Polymarket Gamma market metadata.")
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--closed", type=str_to_bool, default=True)
    parser.add_argument("--page-size", type=int, default=100)
    parser.add_argument("--order", default="volumeNum")
    parser.add_argument("--ascending", type=str_to_bool, default=False)
    parser.add_argument("--category", default=None)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    setup_logging(args.log_level)
    root = project_root()
    raw_dir = root / "data" / "raw"
    processed_dir = root / "data" / "processed"
    ensure_dirs([raw_dir, processed_dir])

    markets = fetch_markets(
        limit=args.limit,
        closed=args.closed,
        page_size=args.page_size,
        order=args.order,
        ascending=args.ascending,
        category=args.category,
    )
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    raw_path = raw_dir / f"gamma_markets_{stamp}.json"
    write_json(raw_path, markets)
    write_json(raw_dir / "gamma_markets_latest.json", markets)

    rows = [normalize_market(market) for market in markets]
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df[df["is_binary"] & df["has_yes_token"]].copy()
        df = df.sort_values(["volume_clob", "volume"], ascending=False, na_position="last")
    output_path = processed_dir / "markets.parquet"
    df.to_parquet(output_path, index=False)

    logging.info("Saved raw metadata to %s", raw_path)
    logging.info("Saved %s binary markets to %s", len(df), output_path)


if __name__ == "__main__":
    main()
