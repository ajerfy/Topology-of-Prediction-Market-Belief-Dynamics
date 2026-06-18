from __future__ import annotations

import argparse
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from utils import (
    CLOB_BASE_URL,
    append_jsonl,
    ensure_dirs,
    project_root,
    request_json,
    setup_logging,
)


def normalize_history(payload: Any, market: pd.Series) -> list[dict[str, Any]]:
    history = payload.get("history") if isinstance(payload, dict) else payload
    if not history or not isinstance(history, list):
        return []

    rows: list[dict[str, Any]] = []
    for point in history:
        if not isinstance(point, dict):
            continue
        timestamp = point.get("t") or point.get("timestamp")
        price = point.get("p") or point.get("price")
        if timestamp is None or price is None:
            continue
        rows.append(
            {
                "timestamp": pd.to_datetime(timestamp, unit="s", utc=True),
                "market_id": str(market["market_id"]),
                "token_id": str(market["yes_token_id"]),
                "yes_price": float(price),
                "category": market.get("category"),
                "event_id": str(market.get("event_id")) if pd.notna(market.get("event_id")) else None,
                "resolved_outcome": market.get("resolved_outcome"),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch Polymarket CLOB YES-token price history.")
    parser.add_argument("--input", default="data/processed/markets.parquet")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--interval", default="max")
    parser.add_argument("--fidelity", type=int, default=60, help="CLOB sampling fidelity in minutes.")
    parser.add_argument("--min-volume-clob", type=float, default=0.0)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    setup_logging(args.log_level)
    root = project_root()
    input_path = Path(args.input)
    if not input_path.is_absolute():
        input_path = root / input_path

    raw_dir = root / "data" / "raw"
    processed_dir = root / "data" / "processed"
    ensure_dirs([raw_dir, processed_dir])

    markets = pd.read_parquet(input_path)
    markets = markets[markets["yes_token_id"].notna()].copy()
    if "volume_clob" in markets.columns and args.min_volume_clob > 0:
        markets = markets[markets["volume_clob"].fillna(0) >= args.min_volume_clob]
    if args.limit:
        markets = markets.head(args.limit)

    session = requests.Session()
    all_rows: list[dict[str, Any]] = []
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    raw_path = raw_dir / f"clob_price_history_{stamp}.jsonl"

    for idx, (_, market) in enumerate(markets.iterrows(), start=1):
        token_id = str(market["yes_token_id"])
        params = {"market": token_id, "interval": args.interval, "fidelity": args.fidelity}
        payload = request_json(session, f"{CLOB_BASE_URL}/prices-history", params=params)
        raw_record = {
            "market_id": str(market["market_id"]),
            "token_id": token_id,
            "params": params,
            "payload": payload,
        }
        append_jsonl(raw_path, [raw_record])

        rows = normalize_history(payload, market) if payload else []
        all_rows.extend(rows)
        logging.info(
            "Fetched history %s/%s market_id=%s points=%s",
            idx,
            len(markets),
            market["market_id"],
            len(rows),
        )

    prices = pd.DataFrame(
        all_rows,
        columns=[
            "timestamp",
            "market_id",
            "token_id",
            "yes_price",
            "category",
            "event_id",
            "resolved_outcome",
        ],
    )
    if not prices.empty:
        prices = prices.sort_values(["market_id", "timestamp"])
    output_path = processed_dir / "prices_long.parquet"
    prices.to_parquet(output_path, index=False)
    logging.info("Saved raw histories to %s", raw_path)
    logging.info(
        "Saved %s rows across %s markets to %s",
        len(prices),
        prices["market_id"].nunique() if not prices.empty else 0,
        output_path,
    )


if __name__ == "__main__":
    main()
