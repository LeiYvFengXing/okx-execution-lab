#!/usr/bin/env python3
"""Run every study in okx-execution-lab.

    python run_all.py                 # offline: uses the bundled BTC 1H sample
    python run_all.py --with-orderbook  # also fetch live order books (network)

Nothing here needs an API key. The order-book study is the only part that
touches the network, and it degrades gracefully if the exchange is unreachable.
"""

from __future__ import annotations

import argparse
import sys
import time


def banner(title: str) -> None:
    print()
    print("#" * 78)
    print(f"#  {title}")
    print("#" * 78)
    print()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--with-orderbook", action="store_true",
                    help="also run the live order-book cost study (needs network)")
    ap.add_argument("--csv", default=None, help="path to an OHLC CSV (default: bundled sample)")
    ap.add_argument("--bar", default="1H", choices=["1H", "1D"],
                    help="bar size of the CSV - controls the annualisation factor")
    ap.add_argument("--capital", type=float, default=1_000.0, help="starting capital for the grid")
    ap.add_argument("--fee", type=float, default=None, help="override the maker fee (e.g. 0.0008)")
    ap.add_argument("--skip", nargs="*", default=[],
                    choices=["costs", "maker", "grid"], help="studies to skip")
    args = ap.parse_args()

    from src import grid, maker_fill
    from src.common import FEE_MAKER

    fee = args.fee if args.fee is not None else FEE_MAKER
    t0 = time.time()

    if args.with_orderbook and "costs" not in args.skip:
        from src import costs
        banner("STUDY 1/3 - order-book cost decomposition (live snapshots)")
        try:
            costs.run()
        except Exception as exc:  # noqa: BLE001
            print(f"  [skipped] order-book study failed: {exc}")
            print("  (this study needs network access; use OKX_HOST=... behind a firewall)")

    if "maker" not in args.skip:
        banner("STUDY 2/3 - maker vs taker: fill rate and adverse selection")
        maker_fill.run(csv_path=args.csv, fee_maker=fee)

    if "grid" not in args.skip:
        banner("STUDY 3/3 - grid trading vs a same-exposure control")
        grid.run(csv_path=args.csv, bar=args.bar, fee=fee, capital=args.capital)

    print()
    print(f"done in {time.time() - t0:.1f}s - figures in images/, tables in reports/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
