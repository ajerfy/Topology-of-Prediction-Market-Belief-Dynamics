from __future__ import annotations

import argparse
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from utils import (
    DATA_API_BASE_URL,
    append_jsonl,
    ensure_dirs,
    project_root,
    request_json,
    setup_logging,
)

DEFAULT_CRYPTO_KEYWORDS = [
    "bitcoin",
    "btc",
    "ethereum",
    " eth ",
    "solana",
    " sol ",
    "xrp",
    "doge",
    "crypto",
    "cryptocurrency",
    "coinbase",
    "binance",
    "tether",
    "usdt",
    "microstrategy",
    "mstr",
    "stablecoin",
    "etf",
]


def token_ids(market: pd.Series) -> tuple[str | None, str | None]:
    ids = market.get("clob_token_ids")
    if isinstance(ids, list) and len(ids) >= 2:
        return str(ids[0]), str(ids[1])
    return str(market.get("yes_token_id")), None


def trade_yes_price(trade: dict[str, Any], yes_token_id: str | None, no_token_id: str | None) -> float | None:
    price = trade.get("price")
    if price is None:
        return None
    try:
        price_float = float(price)
    except (TypeError, ValueError):
        return None

    asset = str(trade.get("asset")) if trade.get("asset") is not None else None
    outcome = str(trade.get("outcome") or "").strip().lower()

    if asset and yes_token_id and asset == yes_token_id:
        return price_float
    if asset and no_token_id and asset == no_token_id:
        return 1.0 - price_float
    if outcome == "yes":
        return price_float
    if outcome == "no":
        return 1.0 - price_float
    return None


def normalize_trades(trades: list[dict[str, Any]], market: pd.Series) -> list[dict[str, Any]]:
    yes_token_id, no_token_id = token_ids(market)
    rows: list[dict[str, Any]] = []
    for trade in trades:
        timestamp = trade.get("timestamp")
        yes_price = trade_yes_price(trade, yes_token_id, no_token_id)
        if timestamp is None or yes_price is None:
            continue
        rows.append(
            {
                "timestamp": pd.to_datetime(timestamp, unit="s", utc=True),
                "market_id": str(market["market_id"]),
                "token_id": str(yes_token_id) if yes_token_id else None,
                "yes_price": yes_price,
                "category": market.get("category"),
                "event_id": str(market.get("event_id")) if pd.notna(market.get("event_id")) else None,
                "resolved_outcome": market.get("resolved_outcome"),
                "market_family": market.get("market_family"),
                "asset": market.get("asset"),
                "is_core": bool(market.get("is_core")) if pd.notna(market.get("is_core")) else False,
                "is_satellite": bool(market.get("is_satellite")) if pd.notna(market.get("is_satellite")) else False,
                "source": "data_api_trades",
                "trade_asset": trade.get("asset"),
                "trade_outcome": trade.get("outcome"),
                "trade_side": trade.get("side"),
                "trade_size": trade.get("size"),
                "transaction_hash": trade.get("transactionHash"),
            }
        )
    return rows


def market_text(market: pd.Series) -> str:
    tags = market.get("tags")
    if isinstance(tags, (list, tuple)):
        tags_text = " ".join(str(tag) for tag in tags)
    elif tags is None:
        tags_text = ""
    else:
        tags_text = str(tags)
    fields = [
        market.get("question"),
        market.get("category"),
        market.get("slug"),
        market.get("description"),
        tags_text,
    ]
    return f" {' '.join(str(field or '') for field in fields).lower()} "


def filter_markets_by_keywords(markets: pd.DataFrame, keywords: list[str]) -> pd.DataFrame:
    if not keywords:
        return markets
    patterns = []
    for keyword in keywords:
        term = keyword.lower().strip()
        if not term:
            continue
        if term.isalpha() and len(term) <= 4:
            patterns.append(re.compile(rf"\b{re.escape(term)}\b"))
        else:
            patterns.append(re.compile(re.escape(term)))
    mask = markets.apply(
        lambda row: any(pattern.search(market_text(row)) for pattern in patterns),
        axis=1,
    )
    return markets[mask].copy()


def fetch_trades_for_market(
    session: requests.Session,
    condition_id: str,
    *,
    page_size: int,
    max_trades: int,
) -> list[dict[str, Any]]:
    trades: list[dict[str, Any]] = []
    offset = 0
    while len(trades) < max_trades:
        limit = min(page_size, max_trades - len(trades))
        params = {"market": condition_id, "limit": limit, "offset": offset}
        payload = request_json(session, f"{DATA_API_BASE_URL}/trades", params=params)
        if not payload:
            break
        if not isinstance(payload, list):
            logging.warning("Unexpected trade payload type for %s: %s", condition_id, type(payload).__name__)
            break
        trades.extend(payload)
        if len(payload) < limit:
            break
        offset += len(payload)
    return trades


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch Polymarket Data API trades and reconstruct YES-token probabilities."
    )
    parser.add_argument("--input", default="data/processed/market_universe.parquet")
    parser.add_argument("--output", default="data/processed/prices_long.parquet")
    parser.add_argument("--limit", type=int, default=None, help="Maximum markets to scan.")
    parser.add_argument("--page-size", type=int, default=500)
    parser.add_argument("--max-trades-per-market", type=int, default=50000)
    parser.add_argument("--min-volume-clob", type=float, default=0.0)
    parser.add_argument(
        "--keywords",
        default=None,
        help="Comma-separated market keyword filter. Use 'crypto' for a built-in crypto keyword set.",
    )
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    setup_logging(args.log_level)
    root = project_root()
    input_path = Path(args.input)
    output_path = Path(args.output)
    if not input_path.is_absolute():
        input_path = root / input_path
    if not output_path.is_absolute():
        output_path = root / output_path

    raw_dir = root / "data" / "raw"
    ensure_dirs([raw_dir, output_path.parent])

    markets = pd.read_parquet(input_path)
    if {"is_core", "is_satellite"}.issubset(markets.columns):
        markets = markets[markets["is_core"].fillna(False) | markets["is_satellite"].fillna(False)].copy()
    markets = markets[markets["condition_id"].notna() & markets["yes_token_id"].notna()].copy()
    if "volume_clob" in markets.columns and args.min_volume_clob > 0:
        markets = markets[markets["volume_clob"].fillna(0) >= args.min_volume_clob]
    if args.keywords:
        if args.keywords.strip().lower() == "crypto":
            keywords = DEFAULT_CRYPTO_KEYWORDS
        else:
            keywords = [keyword.strip() for keyword in args.keywords.split(",") if keyword.strip()]
        before = len(markets)
        markets = filter_markets_by_keywords(markets, keywords)
        logging.info("Keyword filter kept %s/%s markets for keywords=%s", len(markets), before, keywords)
    if args.limit:
        markets = markets.head(args.limit)

    session = requests.Session()
    rows: list[dict[str, Any]] = []
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    raw_path = raw_dir / f"data_api_trades_{stamp}.jsonl"

    for idx, (_, market) in enumerate(markets.iterrows(), start=1):
        condition_id = str(market["condition_id"])
        trades = fetch_trades_for_market(
            session,
            condition_id,
            page_size=args.page_size,
            max_trades=args.max_trades_per_market,
        )
        append_jsonl(
            raw_path,
            [
                {
                    "market_id": str(market["market_id"]),
                    "condition_id": condition_id,
                    "payload": trades,
                }
            ],
        )
        market_rows = normalize_trades(trades, market)
        rows.extend(market_rows)
        logging.info(
            "Fetched trades %s/%s market_id=%s trades=%s usable_points=%s",
            idx,
            len(markets),
            market["market_id"],
            len(trades),
            len(market_rows),
        )

    prices = pd.DataFrame(rows)
    required = [
        "timestamp",
        "market_id",
        "token_id",
        "yes_price",
        "category",
        "event_id",
        "resolved_outcome",
        "market_family",
        "asset",
        "is_core",
        "is_satellite",
    ]
    for column in required:
        if column not in prices.columns:
            prices[column] = pd.Series(dtype="object")
    if not prices.empty:
        prices = prices.sort_values(["market_id", "timestamp"])
        prices = prices[(prices["yes_price"] >= 0) & (prices["yes_price"] <= 1)].copy()
        dedupe_cols = ["transaction_hash", "market_id", "timestamp", "trade_asset", "trade_size", "yes_price"]
        prices = prices.drop_duplicates(subset=dedupe_cols, keep="last")
    prices.to_parquet(output_path, index=False)
    logging.info(
        "Saved %s trade-derived rows across %s markets to %s",
        len(prices),
        prices["market_id"].nunique() if not prices.empty else 0,
        output_path,
    )
    logging.info("Saved raw trade payloads to %s", raw_path)


if __name__ == "__main__":
    main()
