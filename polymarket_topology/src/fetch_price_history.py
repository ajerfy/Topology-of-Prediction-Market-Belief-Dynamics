from __future__ import annotations

import argparse
import logging
from datetime import datetime, timedelta, timezone
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
                "market_family": market.get("market_family"),
                "asset": market.get("asset"),
                "is_core": bool(market.get("is_core")) if pd.notna(market.get("is_core")) else False,
                "is_satellite": bool(market.get("is_satellite")) if pd.notna(market.get("is_satellite")) else False,
                "source": "clob_prices_history",
            }
        )
    return rows


def active_bounds(market: pd.Series) -> tuple[pd.Timestamp | None, pd.Timestamp | None]:
    start = pd.to_datetime(market.get("start_date"), utc=True, errors="coerce")
    close = pd.to_datetime(market.get("close_date"), utc=True, errors="coerce")
    end = pd.to_datetime(market.get("end_date"), utc=True, errors="coerce")
    stop = close if pd.notna(close) else end
    if pd.isna(start) or pd.isna(stop) or stop <= start:
        return None, None
    return start, stop


def history_windows(start: pd.Timestamp, end: pd.Timestamp, chunk_days: int) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    windows = []
    cursor = start
    delta = timedelta(days=chunk_days)
    while cursor < end:
        stop = min(cursor + delta, end)
        windows.append((cursor, stop))
        cursor = stop
    return windows


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch Polymarket CLOB YES-token price history in bounded time windows.")
    parser.add_argument("--input", default="data/processed/market_universe.parquet")
    parser.add_argument("--output", default="data/processed/clob_price_history_long.parquet")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--interval", default=None)
    parser.add_argument("--fidelity", type=int, default=60, help="CLOB sampling fidelity in minutes.")
    parser.add_argument("--chunk-days", type=int, default=14, help="Maximum days per prices-history request.")
    parser.add_argument("--min-volume-clob", type=float, default=0.0)
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
        start, end = active_bounds(market)
        if start is None or end is None:
            logging.info("Skipping history %s/%s market_id=%s missing active bounds", idx, len(markets), market["market_id"])
            continue
        market_rows: list[dict[str, Any]] = []
        windows = history_windows(start, end, args.chunk_days)
        for window_start, window_end in windows:
            params = {
                "market": token_id,
                "startTs": int(window_start.timestamp()),
                "endTs": int(window_end.timestamp()),
                "fidelity": args.fidelity,
            }
            if args.interval:
                params["interval"] = args.interval
            payload = request_json(session, f"{CLOB_BASE_URL}/prices-history", params=params)
            raw_record = {
                "market_id": str(market["market_id"]),
                "token_id": token_id,
                "params": params,
                "payload": payload,
            }
            append_jsonl(raw_path, [raw_record])
            rows = normalize_history(payload, market) if payload else []
            market_rows.extend(rows)
        all_rows.extend(market_rows)
        logging.info(
            "Fetched history %s/%s market_id=%s windows=%s points=%s",
            idx,
            len(markets),
            market["market_id"],
            len(windows),
            len(market_rows),
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
            "market_family",
            "asset",
            "is_core",
            "is_satellite",
            "source",
        ],
    )
    if not prices.empty:
        prices = prices.sort_values(["market_id", "timestamp"])
        prices = prices[(prices["yes_price"] >= 0) & (prices["yes_price"] <= 1)].copy()
        prices = prices.drop_duplicates(subset=["market_id", "timestamp", "token_id", "yes_price"], keep="last")
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
