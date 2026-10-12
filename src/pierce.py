"""穿透 vs 触碰 —— 挂单成交率的真实代价

问题出在哪：
    okx-execution-lab 的 maker_fill 用 1H K 线的 low 判断"挂单是否成交"。
    但 `low <= limit` 只说明【价格碰到过你的挂单价】，
    不说明【你在队列里排到了】。

    真实情况有两种，1H 数据把它们混为一谈：
      (a) 价格下探到 100.00 然后立刻弹走 —— 你的买单在 100.00 排队，
          前面还压着 500 个同价单子。你没成交。
      (b) 价格砸到 99.90 并在那里待了一段 —— 卖单不断打进来，
          这时的成交才真轮到你。

    把 (a) 也算成成交，会【系统性高估挂单成交率】。

方法：
    用 15m K 线看 1H 挂单的 bar 内路径。
    对每个挂单机会 t：
        P      = close_t * (1 - delta)              挂单价
        pierce = (P - min(low[t+1 .. t+H])) / P     最深穿透多少（bps）
    然后对不同的"穿透门槛" k 重新判成交：
        只有 pierce >= k 才算这一单成交了。
    观察：成交率和挂单优势如何随 k 变化。

    k = 0 就是旧的"触碰即成交"规则，作为基准。
"""
import os
import sys

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt                                   # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (                                              # noqa: E402
    DATA_DIR, IMAGE_DIR, REPORT_DIR, FEE_TAKER, FEE_MAKER, HALF_SPREAD,
    load_csv, style, savefig, save_csv,
)

SAMPLE_15M = os.path.join(DATA_DIR, "BTC-USDT_15m.csv")
FIG = os.path.join(IMAGE_DIR, "pierce.png")

DELTAS = [0.0002, 0.0005, 0.001, 0.002, 0.005, 0.01]
HORIZONS = [4, 12, 24, 48]
# 穿透门槛（bps）：挂单价被穿多深才算真成交
KS = [0.0, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0]
BARS_PER_HOUR = 4          # 15m


def load_15m(path=None):
    """读 15m CSV（默认用自带样本 data/BTC-USDT_15m.csv）。"""
    return load_csv(path or SAMPLE_15M)


def scan(df, deltas=DELTAS, horizons=HORIZONS, ks=KS, stride=BARS_PER_HOUR):
    """返回长表：每行 = (delta, horizon, k) 的成交率与收益。"""
    close = df["close"].to_numpy(float)
    low = df["low"].to_numpy(float)
    n = len(close)

    # 决策点：每 stride 根取一个（15m → 1H）
    idx = np.arange(0, n, stride, dtype=np.int64)

    rows = []
    for delta in deltas:
        for H in horizons:
            win = H * BARS_PER_HOUR
            fwd_min = pd.Series(low).rolling(win).min().shift(-win).to_numpy()
            close_end = pd.Series(close).shift(-win).to_numpy()

            c0 = close[idx]
            P = c0 * (1.0 - delta)
            fm = fwd_min[idx]
            ce = close_end[idx]

            ok = np.isfinite(fm) & np.isfinite(ce)
            c0, P, fm, ce = c0[ok], P[ok], fm[ok], ce[ok]

            touched = fm <= P
            with np.errstate(invalid="ignore", divide="ignore"):
                pierce_bps = np.where(touched, (P - fm) / P * 1e4, np.nan)

            # 挂单：成交在 P，结束时按 close_end 卖出（maker 双边费率）
            maker_ret = (ce * (1 - FEE_MAKER)) / P - 1.0
            # 吃单：立刻在 close*(1+半个点差) 买入，同样时点卖出（taker 双边费率）
            taker_ret = (ce * (1 - FEE_TAKER)) / (c0 * (1 + HALF_SPREAD)) - 1.0
            taker_mean = float(np.mean(taker_ret))

            for k in ks:
                filled = touched & (pierce_bps >= k)
                # 按【机会】平均：没成交的那次记 0，不是丢掉
                maker_all = np.where(filled, maker_ret, 0.0)
                rows.append({
                    "delta_pct": delta * 100,
                    "horizon_h": H,
                    "k_bps": k,
                    "n_opp": len(c0),
                    "fill_rate": float(filled.mean()),
                    "taker_mean_pct": taker_mean * 100,
                    "maker_mean_pct": float(maker_all.mean()) * 100,
                    "maker_only_pct": float(np.mean(maker_ret[filled])) * 100 if filled.any() else np.nan,
                    "edge_bps": (float(maker_all.mean()) - taker_mean) * 1e4,
                    "mean_pierce_bps": float(np.nanmean(pierce_bps[touched])) if touched.any() else np.nan,
                })
    return pd.DataFrame(rows)


def adverse_by_pierce(df, deltas=(0.0005,), horizons=(24,),
                      edges=(0.0, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 1e9)):
    """成交后继续跌的比例，按穿透深度分桶 —— 检验"越容易成交越是坏消息"。

    ⚠️ 最后一个桶必须是开区间（1e9）—— 否则最深的那些样本会被悄悄丢掉，
       而它们恰恰是"成交了但接到的是真下跌"的主体。
    """
    close = df["close"].to_numpy(float)
    low = df["low"].to_numpy(float)
    n = len(close)
    idx = np.arange(0, n, BARS_PER_HOUR, dtype=np.int64)

    rows = []
    for delta in deltas:
        for H in horizons:
            win = H * BARS_PER_HOUR
            fwd_min = pd.Series(low).rolling(win).min().shift(-win).to_numpy()
            close_end = pd.Series(close).shift(-win).to_numpy()
            c0 = close[idx]
            P = c0 * (1.0 - delta)
            fm, ce = fwd_min[idx], close_end[idx]
            ok = np.isfinite(fm) & np.isfinite(ce)
            c0, P, fm, ce = c0[ok], P[ok], fm[ok], ce[ok]
            touched = fm <= P
            with np.errstate(invalid="ignore", divide="ignore"):
                pierce = np.where(touched, (P - fm) / P * 1e4, np.nan)
            # 从挂单价 P 到期末 close 的收益（成交后你拿到什么）
            ret = ce / P - 1.0

            lo = 0.0
            for hi in edges[1:]:
                m = touched & (pierce >= lo) & (pierce < hi)
                if m.sum() < 30:
                    lo = hi
                    continue
                rows.append({
                    "delta_pct": delta * 100, "horizon_h": H,
                    "pierce_lo_bps": lo, "pierce_hi_bps": hi,
                    "n": int(m.sum()),
                    "mean_ret_pct": float(ret[m].mean()) * 100,
                    "pct_down_pct": float((ret[m] < 0).mean()) * 100,
                })
                lo = hi
    return pd.DataFrame(rows)


def plot(tbl, adv, focus_delta=0.05, focus_H=24):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    fig.suptitle("Pierce vs touch - does a strict fill rule change the maker edge?", fontsize=13)

    ax = axes[0]
    sub = tbl[(tbl["delta_pct"] == focus_delta) & (tbl["horizon_h"] == focus_H)]
    if not sub.empty:
        ax.plot(sub["k_bps"], sub["fill_rate"] * 100, "o-", color="#2E86AB")
    ax.set_xlabel("Required penetration (bps)")
    ax.set_ylabel("Fill rate (%)")
    ax.set_title(f"(1) Stricter rule = fewer fills\nlimit {focus_delta}% below close, max wait {focus_H}h")
    ax.grid(alpha=.3)

    ax = axes[1]
    for H in HORIZONS:
        sub = tbl[(tbl["delta_pct"] == focus_delta) & (tbl["horizon_h"] == H)]
        if sub.empty:
            continue
        ax.plot(sub["k_bps"], sub["edge_bps"], "o-", label=f"max wait {H}h")
    ax.axhline(0, color="k", lw=1)
    ax.set_xlabel("Required penetration (bps)")
    ax.set_ylabel("Maker edge vs taker (bps)")
    ax.set_title(f"(2) Does the edge survive?\nlimit {focus_delta}% below close")
    ax.legend(fontsize=8)
    ax.grid(alpha=.3)

    ax = axes[2]
    # 注意：adv 表的 delta_pct 已经是百分数（0.05 = 下移 0.05%），
    # 而 focus_delta 用的是小数（0.05）—— 两者在这里恰好同值，但语义不同，别改错。
    sub = adv[adv["delta_pct"] == focus_delta]
    if not sub.empty:
        labels = [f"{a:.0f}-{b:.0f}" if b < 1e8 else f"{a:.0f}+"
                  for a, b in zip(sub["pierce_lo_bps"], sub["pierce_hi_bps"])]
        x = np.arange(len(sub))
        ax.bar(x, sub["mean_ret_pct"], color="#E4572E")
        ax.axhline(0, color="k", lw=1)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, fontsize=8)
    ax.set_xlabel("Penetration bucket (bps)")
    ax.set_ylabel("Mean return from limit to exit (%)")
    ax.set_title("(3) How deep it pierced vs what you got")
    ax.grid(alpha=.3, axis="y")

    fig.tight_layout(rect=(0, 0, 1, 0.90))
    return savefig(fig, "pierce.png")


def main(csv_path=None):
    style()
    print("=" * 100)
    df = load_15m(csv_path)
    t0, t1 = df["dt"].min(), df["dt"].max()
    print(f"  Pierce vs touch · BTC-USDT 15m · {len(df):,} bars · {t0} ~ {t1} "
          f"({(t1 - t0).days} days ~ {(t1 - t0).days / 365:.1f} years)")
    print("=" * 100)

    tbl = scan(df)
    adv = adverse_by_pierce(df)

    print("\n  【表 1】能不能成交，以及成交后拿到什么（挂单价下移 0.05%，最长等 24h）")
    print("  " + "-" * 84)
    s = tbl[(tbl["delta_pct"] == 0.05) & (tbl["horizon_h"] == 24)]
    if not s.empty:
        print(f"  吃单基准：同样这些机会，立刻吃单买入的平均收益 = "
              f"{float(s['taker_mean_pct'].iloc[0]):+.3f}%")
    print()
    print(f"  {'穿透门槛(bps)':>13} {'成交率':>9} {'成交单均值':>12} "
          f"{'吃单均值':>10} {'成交单 − 吃单':>14}")
    for _, r in s.iterrows():
        print(f"  {r['k_bps']:>13.1f} {r['fill_rate']*100:>8.1f}% "
              f"{r['maker_only_pct']:>11.3f}% {r['taker_mean_pct']:>9.3f}% "
              f"{r['maker_only_pct'] - r['taker_mean_pct']:>13.3f}%")
    print("\n  ↑ 最后一列是【条件在成交上】的比较：负 = 你能抢到的那些单，还不如直接吃单。")

    print("\n  【表 2】挂单优势 vs 吃单（bps）—— 负值 = 还不如直接吃单")
    print("  " + "-" * 100)
    piv = tbl.pivot_table(index="k_bps", columns=["delta_pct", "horizon_h"], values="edge_bps")
    hdr = "  " + f"{'门槛':>6}" + "".join(f"{d:.2f}%/{h}h".rjust(13) for d in sorted(tbl['delta_pct'].unique()) for h in [24])
    print(hdr)
    for k in KS:
        line = f"  {k:>6.1f}"
        for d in sorted(tbl["delta_pct"].unique()):
            # pivot 的 index 是 k_bps，columns 是 (delta_pct, horizon_h) 的元组
            if k in piv.index and (d, 24) in piv.columns:
                v = piv.loc[k, (d, 24)]
                line += f"{v:>13.2f}"
            else:
                line += f"{'—':>13}"
        print(line)

    print("\n  【表 3】穿透越深，成交后的收益越差？（挂单价下移 0.05%，终点 24h）")
    print("  " + "-" * 84)
    print(f"  {'穿透区间(bps)':>16} {'样本数':>10} {'占已成交':>10} "
          f"{'从挂单价到期末':>16} {'亏损概率':>10}")
    sub3 = adv[adv["horizon_h"] == 24]
    tot = int(sub3["n"].sum()) or 1
    for _, r in sub3.iterrows():
        hi = float(r["pierce_hi_bps"])
        rng = f"{r['pierce_lo_bps']:.0f} ~ {hi:.0f}" if hi < 1e8 else f"{r['pierce_lo_bps']:.0f} 以上"
        print(f"  {rng:>16} {int(r['n']):>10,} {r['n']/tot*100:>9.1f}% "
              f"{r['mean_ret_pct']:>15.3f}% {r['pct_down_pct']:>9.1f}%")
    print(f"\n  已成交样本合计 {tot:,} 根（表 1 里门槛 0 的成交次数）")

    fig = plot(tbl, adv)
    print(f"\n  图 → {fig}")

    out = save_csv(tbl, "pierce.csv")
    out2 = save_csv(adv, "pierce_adverse.csv")
    print(f"  表 → {out}\n     → {out2}")


if __name__ == "__main__":
    import argparse
    _ap = argparse.ArgumentParser(description="Pierce vs touch: the real cost of a limit order")
    _ap.add_argument("--csv", default=None, help="15m OHLC csv (default: bundled BTC-USDT sample)")
    main(csv_path=_ap.parse_args().csv)
