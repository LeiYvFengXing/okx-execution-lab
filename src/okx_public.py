"""Minimal OKX public REST client.

Only **public** endpoints are used here, so **no API key is required**.
Everything in this repository runs on public market data alone.

Set ``OKX_HOST`` to override the base URL (useful behind a restricted network):

    OKX_HOST=https://www.okx.com python run_all.py
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request

DEFAULT_HOST = "https://www.okx.com"


def base_url() -> str:
    """Return the REST base URL (honours the ``OKX_HOST`` environment variable)."""
    return os.environ.get("OKX_HOST", DEFAULT_HOST).rstrip("/")


def request(
    method: str,
    path: str,
    params: dict | None = None,
    retries: int = 3,
    timeout: int = 15,
) -> list:
    """Call a public OKX endpoint and return the ``data`` array.

    Raises ``RuntimeError`` when the exchange reports a non-zero ``code``.
    """
    url = base_url() + path
    if params:
        url += "?" + urllib.parse.urlencode(params, doseq=True)

    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, method=method)
            req.add_header("User-Agent", "okx-execution-lab/1.0")
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
            if payload.get("code") not in ("0", 0):
                raise RuntimeError(f"OKX error {payload.get('code')}: {payload.get('msg')}")
            return payload.get("data", [])
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, RuntimeError) as exc:
            last_err = exc
            if attempt < retries - 1:
                time.sleep(0.6 * (attempt + 1))
    raise RuntimeError(f"request failed after {retries} attempts: {url} ({last_err})")


# --------------------------------------------------------------------------- #
# Public market data
# --------------------------------------------------------------------------- #

def fetch_book(inst_id: str, sz: int = 400) -> dict:
    """Return a level-2 order-book snapshot: ``{'asks': [...], 'bids': [...], 'ts': int}``.

    Each level is ``[price, size, deprecated, order_count]``.
    ``asks`` are sorted low -> high, ``bids`` high -> low.

    Note: OKX caps ``sz`` at 400 for this endpoint, so a full-book depth
    measurement is only ever *partial* for large notionals.
    """
    data = request("GET", "/api/v5/market/books", {"instId": inst_id, "sz": min(sz, 400)})
    if not data:
        raise RuntimeError(f"empty order book for {inst_id}")
    row = data[0]
    return {
        "instId": inst_id,
        "ts": int(row["ts"]),
        "asks": [[float(p), float(s)] for p, s, _, _ in row["asks"]],
        "bids": [[float(p), float(s)] for p, s, _, _ in row["bids"]],
    }


def fetch_klines(inst_id: str, bar: str = "1H", limit: int = 300) -> list:
    """Return raw candles ``[ts, o, h, l, c, vol, ...]`` (newest first)."""
    return request("GET", "/api/v5/market/candles", {"instId": inst_id, "bar": bar, "limit": limit})


def mid_price(book: dict) -> float:
    """Mid price of an order-book snapshot."""
    return (book["asks"][0][0] + book["bids"][0][0]) / 2.0


def spread_bps(book: dict) -> float:
    """Quoted bid-ask spread in basis points (1 bp = 0.01%)."""
    best_ask, best_bid = book["asks"][0][0], book["bids"][0][0]
    return (best_ask - best_bid) / ((best_ask + best_bid) / 2.0) * 1e4
