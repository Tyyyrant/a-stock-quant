#!/usr/bin/env python3
"""
用新浪日K(不复权) 补齐：2024+ 历史 + 今日。合并保留已有数据，只补缺。

背景：
  - mootdx 返回空；东财 push2his 限流封IP(~53个请求)；腾讯 fqkline 现被 WAF 挑战封。
  - 新浪 quotes.sina.cn getKLineData 经 curl 稳定、不复权价格与缓存逐位一致、
    耐受高频(20连发无失败，~0.4s/只)、datalen=800 覆盖 2023-06 起。
  - 新浪字段: [day, open, high, low, close, volume(股)]，无 amount、无换手率。
    → volume 是「股」，÷100 得「手」(与缓存一致)；
    → amount ≈ volume(股) × close(仅成交额排序用，误差<1%)；
    → 换手率(turnover)留空，后续由东财 fetch_klines_em 单独补。

合并策略(保留已有数据，只补缺)：
  - 读已有 parquet(date/open/close/high/low/volume/amount[/turnover])；
  - 拉新浪全历史(约800日=2023-06起)；
  - 按 date 去重，已有优先(保留精确 amount)，新增日补 amount≈vol_股*close；
  - 输出 8 列: date/open/close/high/low/volume/amount/turnover。

用法:
  python3 scripts/fetch_klines_sina.py [--limit N] [--all] [--interval S]
"""
import os, sys, json, subprocess, time
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
STOCK_DIR = ROOT / "data" / "stocks"
MIN_DATE = "2024-01-01"   # 视为已补齐的最早日期
OK_MAX = "2026-09-29"     # 视为已含最新交易日

MAIN_10CM = ('600', '601', '603', '605', '000', '001', '002', '003')
COLS = ["date", "open", "close", "high", "low", "volume", "amount", "turnover"]


def is_10cm(code):
    return code.startswith(MAIN_10CM)


def fetch_sina(code):
    prefix = "sh" if code.startswith("6") else "sz"
    url = (f"https://quotes.sina.cn/cn/api/json_v2.php/CN_MarketDataService.getKLineData"
           f"?symbol={prefix}{code}&scale=240&ma=no&datalen=800")
    try:
        out = subprocess.run(["curl", "-s", "--max-time", "12",
                              "-H", "User-Agent: Mozilla/5.0", url],
                             capture_output=True, text=True, timeout=15).stdout
        return json.loads(out)
    except Exception:
        return []


def parse(days):
    rows = []
    for r in days:
        try:
            d, o, h, l, c = r["day"], float(r["open"]), float(r["high"]), \
                            float(r["low"]), float(r["close"])
            vol_gu = float(r["volume"])          # 股
            vol_shou = round(vol_gu / 100, 2)     # 手，与缓存一致
            rows.append((d, o, c, h, l, vol_shou, round(vol_gu * c, 2), None))
        except (KeyError, ValueError):
            continue
    return rows


def main():
    args = sys.argv[1:]
    limit = None
    only_10cm = True
    interval = 0.3
    for a in args:
        if a.startswith("--limit="):
            limit = int(a.split("=", 1)[1])
        elif a == "--all":
            only_10cm = False
        elif a.startswith("--interval="):
            interval = float(a.split("=", 1)[1])

    files = sorted(f for f in os.listdir(STOCK_DIR)
                   if f.endswith(".parquet") and not f.startswith("INDEX_"))
    if only_10cm:
        files = [f for f in files if is_10cm(f[:-8])]
    if limit:
        files = files[:limit]

    ok = skip = err = consec_empty = 0
    print(f"新浪补齐 {len(files)} 只（{'10cm' if only_10cm else '全市场'}，节流 {interval}s）...",
          flush=True)
    t0 = time.time()
    for i, fname in enumerate(files):
        code = fname[:-8]
        path = STOCK_DIR / fname

        # 断点续跑
        try:
            old = pd.read_parquet(path, columns=["date"])
            if len(old) and str(old["date"].min()) <= MIN_DATE \
                    and str(old["date"].max()) >= OK_MAX:
                skip += 1
                continue
        except Exception:
            pass

        day = fetch_sina(code)
        rows = parse(day) if isinstance(day, list) else []
        if not rows:
            err += 1
            consec_empty += 1
            if consec_empty >= 20:   # 疑似被限流，退避等解除
                wait = 90
                print(f"  ⚠ 连续 {consec_empty} 个空响应，疑似限流，等待 {wait}s ...", flush=True)
                time.sleep(wait)
                consec_empty = 0
            else:
                time.sleep(interval)
            continue
        consec_empty = 0

        tdf = pd.DataFrame(rows, columns=COLS)
        try:
            existing = pd.read_parquet(path)
            if "turnover" not in existing.columns:
                existing["turnover"] = None
            existing = existing[COLS]
            merged = pd.concat([existing, tdf], ignore_index=True)
            merged = merged.drop_duplicates(subset="date", keep="first")  # 已有优先(精确amount)
        except Exception:
            merged = tdf
        merged = merged.sort_values("date").reset_index(drop=True)
        merged.to_parquet(path, index=False)
        ok += 1
        time.sleep(interval)

        if (i + 1) % 200 == 0:
            el = time.time() - t0
            print(f"  {i+1}/{len(files)} 补{ok} 跳{skip} 错{err} | {el:.0f}s 剩余~{el/(i+1)*(len(files)-i-1):.0f}s",
                  flush=True)

    el = time.time() - t0
    print(f"\n完成: 补齐 {ok} 只, 跳过(已最新) {skip} 只, 失败 {err} 只 | 耗时 {el:.0f}s", flush=True)


if __name__ == "__main__":
    main()
