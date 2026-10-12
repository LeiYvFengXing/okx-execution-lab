"""诊断：表 1（scan）与表 3（adverse_by_pierce）对同一批成交样本的均值为什么对不上。

表 1 说 成交单均值 = +0.033%
表 3 的加权平均  ≈ -0.191%（8 个桶加权）
两者必须相等（同一个集合、同一个终点、只差手续费）。

不勾稽清楚，任何结论都是错的。
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "examples"))
from db import load_candles                                        # noqa: E402

FEE_MAKER = 0.0008
DELTA = 0.0005
H = 24
BARS_PER_HOUR = 4

lib = load_candles()
df = lib[(lib["instId"] == "BTC-USDT") & (lib["bar"] == "15m")].sort_values("ts").reset_index(drop=True)
close = df["close"].to_numpy(float)
low = df["low"].to_numpy(float)
n = len(close)
win = H * BARS_PER_HOUR
idx = np.arange(0, n, BARS_PER_HOUR, dtype=np.int64)

fwd_min = pd.Series(low).rolling(win).min().shift(-win).to_numpy()
close_end = pd.Series(close).shift(-win).to_numpy()

c0 = close[idx]
P = c0 * (1.0 - DELTA)
fm = fwd_min[idx]
ce = close_end[idx]

ok = np.isfinite(fm) & np.isfinite(ce)
print(f"决策点 {len(idx):,} → 有效 {ok.sum():,}（丢掉 {len(idx)-ok.sum():,} 个：末尾没有完整前视窗口）")
c0, P, fm, ce = c0[ok], P[ok], fm[ok], ce[ok]

touched = fm <= P
print(f"触碰（naive 成交）{touched.sum():,}  成交率 {touched.mean()*100:.1f}%")

ret = ce / P - 1.0
maker_ret = (ce * (1 - FEE_MAKER)) / P - 1.0

print()
print("【A】全样本（含未成交）")
print(f"  ret         均值 {ret.mean()*100:+.4f}%")
print(f"  maker_ret   均值 {maker_ret.mean()*100:+.4f}%   ← scan 表 1 的 maker_mean_pct")

print()
print("【B】只在成交样本上")
print(f"  mean(ret[touched])        = {ret[touched].mean()*100:+.4f}%   ← 表 3 的口径")
print(f"  mean(maker_ret[touched])  = {maker_ret[touched].mean()*100:+.4f}%   ← scan 表 1 的 maker_only_pct")
print()
print(f"  两者应当相差 ≈ {FEE_MAKER*100:.2f}%：")
print(f"  {ret[touched].mean()*100:+.4f}% - {FEE_MAKER*100:.2f}% = "
      f"{ret[touched].mean()*100 - FEE_MAKER*100:+.4f}%")

print()
print("【C】按 penet 分桶求和，检查是否等于 B")
pierce = np.where(touched, (P - fm) / P * 1e4, np.nan)
edges = [0.0, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 1e9]
tot = 0.0
cnt = 0
for lo, hi in zip(edges[:-1], edges[1:]):
    m = touched & (pierce >= lo) & (pierce < hi)
    if m.sum() == 0:
        continue
    tot += ret[m].sum()
    cnt += int(m.sum())
    print(f"  [{lo:>4.1f},{hi:>6.1f})  n={m.sum():>7,}  mean(ret)={ret[m].mean()*100:+.4f}%")
print(f"  合计 n={cnt:,}  加权均值 = {tot/cnt*100:+.4f}%")

print()
print("【D】为什么会有差异？—— 看未成交样本长什么样")
print(f"  未成交 {int((~touched).sum()):,} 个：mean(ret)={ret[~touched].mean()*100:+.4f}%")
print(f"  已成交 {int(touched.sum()):,} 个：mean(ret)={ret[touched].mean()*100:+.4f}%")
print()
print(f"  全样本加权 = {(ret[touched].sum()+ret[~touched].sum())/len(ret)*100:+.4f}%")
