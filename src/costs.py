"""Order-book cost decomposition.

Answers a single question: **when I trade, who actually takes my money?**

Three layers are measured separately:

1. **Half-spread** - paid to whoever was resting on the other side.
2. **Market impact** - the cost of walking up the book with your own size.
3. **Fees** - paid to the exchange.

The headline result on BTC/ETH: layer 3 dominates layers 1-2 by two orders
of magnitude, i.e. *the market is nearly free and the exchange is not*.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from . import okx_public as api
from .common import FEE_MAKER, FEE_TAKER, save_csv, savefig, style

DEFAULT_COINS = [
    "BTC-USDT", "ETH-USDT", "SOL-USDT",
    "DOGE-USDT", "XRP-USDT", "ADA-USDT", "PEPE-USDT",
]
DEFAULT_NOTIONALS = [100.0, 1_000.0, 10_000.0]


def walk_book(levels: list[list[float]], notional: float, side: str) -> tuple[float, float]:
    """Walk the book until ``notional`` USDT is filled.

    ``side='buy'`` consumes asks, ``side='sell'`` consumes bids.

    Returns ``(average_price, filled_notional)``. If the snapshot is exhausted
    before the order is filled, the partial fill is still reported with its own
    average price - the caller can detect this via ``filled_notional``.
    """
    filled_value = 0.0
    filled_size = 0.0
    for price, size in levels:
        level_value = price * size
        take_value = min(level_value, notional - filled_value)
        take_size = take_value / price
        filled_value += take_value
        filled_size += take_size
        if filled_value >= notional - 1e-9:
            break
    if filled_size == 0:
        return float("nan"), 0.0
    return filled_value / filled_size, filled_value


def analyze(
    inst: str,
    notionals: list[float] | None = None,
    fee_taker: float = FEE_TAKER,
    fee_maker: float = FEE_MAKER,
    sz: int = 400,
) -> dict:
    """Measure spread, depth and round-trip cost for one instrument."""
    notionals = notionals or DEFAULT_NOTIONALS
    book = api.fetch_book(inst, sz=sz)
    mid = api.mid_price(book)
    best_bid, best_ask = book["bids"][0][0], book["asks"][0][0]

    depth_usdt = sum(p * s for p, s in book["bids"]) + sum(p * s for p, s in book["asks"])
    half_spread_bps = (best_ask - best_bid) / mid / 2.0 * 1e4

    rows = []
    for notional in notionals:
        buy_px, bought = walk_book(book["asks"], notional, "buy")
        sell_px, sold = walk_book(book["bids"], notional, "sell")
        # Cost of a round trip relative to mid, in bps, excluding fees.
        buy_slip = (buy_px - mid) / mid * 1e4 if bought else float("nan")
        sell_slip = (mid - sell_px) / mid * 1e4 if sold else float("nan")
        rows.append({
            "notional_usdt": notional,
            "buy_avg": buy_px,
            "sell_avg": sell_px,
            "buy_slip_bps": buy_slip,
            "sell_slip_bps": sell_slip,
            "round_trip_exec_bps": buy_slip + sell_slip,
        })
    impact = pd.DataFrame(rows)

    # Decomposition at the smallest notional tested (impact-free reference).
    ref = impact.iloc[0]
    half_spread_rt = half_spread_bps * 2
    impact_rt = max(ref["round_trip_exec_bps"] - half_spread_rt, 0.0)
    fee_taker_rt = fee_taker * 2 * 1e4
    fee_maker_rt = fee_maker * 2 * 1e4

    return {
        "instId": inst,
        "mid": mid,
        "spread_bps": (best_ask - best_bid) / mid * 1e4,
        "half_spread_bps": half_spread_bps,
        "depth_usdt": depth_usdt,
        "impact": impact,
        "half_spread_rt_bps": half_spread_rt,
        "impact_rt_bps": impact_rt,
        "fee_taker_rt_bps": fee_taker_rt,
        "fee_maker_rt_bps": fee_maker_rt,
        "taker_total_bps": half_spread_rt + impact_rt + fee_taker_rt,
        "maker_total_bps": half_spread_rt + impact_rt + fee_maker_rt,
        "fee_share": fee_taker_rt / (half_spread_rt + impact_rt + fee_taker_rt),
    }


def summarize(results: list[dict], notional: float = 1_000.0) -> pd.DataFrame:
    """Build the headline comparison table."""
    rows = []
    for r in results:
        ref = r["impact"][r["impact"]["notional_usdt"] == notional]
        exec_bps = float(ref["round_trip_exec_bps"].iloc[0]) if len(ref) else float("nan")
        rows.append({
            "instId": r["instId"],
            "spread_bps": r["spread_bps"],
            "depth_usdt": r["depth_usdt"],
            "exec_cost_bps": exec_bps,
            "half_spread_rt_bps": r["half_spread_rt_bps"],
            "impact_rt_bps": r["impact_rt_bps"],
            "fee_taker_rt_bps": r["fee_taker_rt_bps"],
            "taker_total_bps": r["taker_total_bps"],
            "maker_total_bps": r["maker_total_bps"],
            "fee_share_pct": r["fee_share"] * 100,
        })
    return pd.DataFrame(rows)


def plot(results: list[dict], notional: float = 1_000.0) -> Path:
    """Liquidity map + BTC cumulative depth curve."""
    style()
    summary = summarize(results, notional=notional)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))

    ax = axes[0]
    ax.scatter(summary["spread_bps"], summary["depth_usdt"], s=70, zorder=3)
    for _, row in summary.iterrows():
        ax.annotate(row["instId"].split("-")[0], (row["spread_bps"], row["depth_usdt"]),
                    textcoords="offset points", xytext=(6, 5), fontsize=8)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Quoted spread (bps, lower is better)")
    ax.set_ylabel(f"Book depth, top 400 levels (USDT)")
    ax.set_title("Liquidity map: top-left = cheap AND deep")

    ax = axes[1]
    btc = next((r for r in results if r["instId"] == "BTC-USDT"), results[0])
    book = api.fetch_book(btc["instId"], sz=400)
    mid = api.mid_price(book)
    for side, levels, colour in (("bids", book["bids"], "tab:green"), ("asks", book["asks"], "tab:red")):
        xs, ys, cum = [], [], 0.0
        for price, size in levels:
            cum += price * size
            xs.append((price - mid) / mid * 1e4)
            ys.append(cum)
        ax.plot(xs, ys, color=colour, label=side)
        ax.fill_between(xs, ys, alpha=0.15, color=colour)
    ax.axvline(0, color="black", lw=0.8, ls="--")
    ax.set_yscale("log")
    ax.set_xlim(-60, 60)
    ax.set_xlabel("Distance from mid (bps)")
    ax.set_ylabel("Cumulative size (USDT, log)")
    ax.set_title(f"{btc['instId']} depth curve - flatter = more impact per dollar")
    ax.legend(loc="upper left")

    fig.suptitle("Order-book cost decomposition", fontsize=12)
    return savefig(fig, "costs.png")


def run(coins: list[str] | None = None, notional: float = 1_000.0, sz: int = 400) -> pd.DataFrame:
    """Fetch snapshots for ``coins`` and return the summary table."""
    coins = coins or DEFAULT_COINS
    results = []
    for inst in coins:
        try:
            results.append(analyze(inst, sz=sz))
            print(f"  [ok] {inst}")
        except Exception as exc:  # noqa: BLE001 - network is best-effort
            print(f"  [skip] {inst}: {exc}")
    if not results:
        raise RuntimeError("no order books could be fetched (check network / OKX_HOST)")

    summary = summarize(results, notional=notional)
    save_csv(summary, "costs.csv")
    plot(results, notional=notional)

    print()
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    print()
    print("=" * 78)
    print(f"  Fee share of round-trip cost at ${notional:,.0f} (taker):")
    for _, row in summary.iterrows():
        print(f"      {row['instId']:<12} {row['fee_share_pct']:5.1f}%  "
              f"(half-spread {row['half_spread_rt_bps']:.2f} + impact "
              f"{row['impact_rt_bps']:.2f} + fee {row['fee_taker_rt_bps']:.2f} bps)")
    print()
    print("  Reading: the market itself is nearly free; the exchange is not.")
    print("  Lever available immediately: post maker orders -> 0.10% becomes 0.08%.")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    run()
