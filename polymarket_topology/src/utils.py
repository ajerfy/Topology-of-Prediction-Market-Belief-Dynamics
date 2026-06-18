from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any, Iterable

import requests

GAMMA_BASE_URL = "https://gamma-api.polymarket.com"
CLOB_BASE_URL = "https://clob.polymarket.com"


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def str_to_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    lowered = value.lower()
    if lowered in {"true", "1", "yes", "y"}:
        return True
    if lowered in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected boolean, got {value!r}")


def ensure_dirs(paths: Iterable[Path]) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def parse_jsonish(value: Any, default: Any = None) -> Any:
    if value is None:
        return default
    if isinstance(value, (list, dict)):
        return value
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return default
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return default
    return default


def safe_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def append_jsonl(path: Path, records: Iterable[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def request_json(
    session: requests.Session,
    url: str,
    params: dict[str, Any] | None = None,
    *,
    timeout: int = 30,
    retries: int = 4,
    backoff: float = 1.5,
    rate_limit_seconds: float = 0.25,
) -> Any | None:
    for attempt in range(retries + 1):
        try:
            response = session.get(url, params=params, timeout=timeout)
            if response.status_code == 429 or 500 <= response.status_code < 600:
                raise requests.HTTPError(
                    f"retryable status {response.status_code}", response=response
                )
            response.raise_for_status()
            if not response.text.strip():
                return None
            return response.json()
        except (requests.RequestException, ValueError) as exc:
            if attempt >= retries:
                logging.warning("Giving up on %s params=%s error=%s", url, params, exc)
                return None
            sleep_for = backoff**attempt + rate_limit_seconds
            logging.debug(
                "Retrying %s params=%s attempt=%s sleep=%.2fs error=%s",
                url,
                params,
                attempt + 1,
                sleep_for,
                exc,
            )
            time.sleep(sleep_for)
    return None
