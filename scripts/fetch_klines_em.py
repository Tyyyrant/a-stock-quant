#!/usr/bin/env python3
"""
用东财 push2his(不复权 fqt=0) 全量补齐日K：2024+ 历史 + 今日 + 换手率。

背景与教训：
  - mootdx(通达信) 源当前返回空(服务器/沙箱不可用)。
  - Python requests 被东财 TLS 指纹拦(RemoteDisconnected)，curl 正常。
  - 东财 fqt=0 价格与现有 mootdx 缓存逐位一致(已验证)。
  - ⚠ 东财有 IP 限流：sleep 0.12s 会在 ~53 个请求后被封。必须 1.5s 节流(同 em_get)。

动作：对 data/stocks/*.parquet(默认仅 10cm 主板) 逐个 curl 东财日K，
      覆盖写入 date/open/close/high/low/volume/amount/turnover 八列。
      东财返回空时保留原缓存不覆盖；可断点续跑(--skip-ok 默认开)。

用法:
  python3 scripts/fetch_klines_em.py [--limit N] [--force] [--all] [--interval S]
    --limit N     只处理前 N 只（试跑）
    --force       已补齐的也重下
    --all         处理全部股票(默认仅 10cm 主板)
    --interval S  节流秒数(默认 1.5)
"""
import os, sys, json, subprocess, time
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
STOCK_DIR = ROOT / "data" / "stocks"
BEG = "20240101"
MIN_DATE = "2024-01-03"   # 视为已补齐的最早日期
OK_MAX = "2026-09-29"     # 视为已含最新交易日的最晚日期

MAIN_10CM = ('600', '601', '603', '605', '000', '001', '002', '003')


def is_10cm(code):
    return code.startswith(MAIN_10CM)


def fetch_em(code, market):
    secid = f"{market}.{code}"
    url = (f"https://push2his.eastmoney.com/api/qt/stock/kline/get"
           f"?secid={secid}&fields1=f1,f2,f3"
           f"&fields2=f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61"
           f"&klt=101&fqt=0&beg={BEG}&end=20500101")
    try:
        out = subprocess.run(
            ["curl", "-s", "--max-time", "20",
             "-H", "User-Agent: Mozilla/5.0",
             "-H", "Referer: https://quote.eastmoney.com/", url],
            capture_output=True, text=True, timeout=25).stdout
        d = json.loads(out)
        return (d.get("data") or {}).get("klines") or []
    except Exception:
        return []


def parse(klines):
    rows = []
    for x in klines:
        p = x.split(",")
        if len(p) < 11:
            continue
        try:
            hs = float(p[10]) if p[10] not in ("", "-") else None
            rows.append((p[0], float(p[1]), float(p[2]), float(p[3]), float(p[4]),
                         float(p[5]), float(p[6]), hs))
        except ValueError:
            continue
    return rows


def main():
    args = sys.argv[1:]
    limit = None
    force = False
    only_10cm = True
    interval = 1.5
    for a in args:
        if a.startswith("--limit="):
            limit = int(a.split("=", 1)[1])
        elif a == "--force":
            force = True
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
    print(f"全量补齐 {len(files)} 只（{'10cm' if only_10cm else '全市场'}，节流 {interval}s）...")
    t0 = time.time()
    for i, fname in enumerate(files):
        code = fname[:-8]
        market = 1 if code.startswith("6") else 0
        path = STOCK_DIR / fname

        # 断点续跑：已补齐的直接跳过
        if not force:
            try:
                old = pd.read_parquet(path, columns=["date"])
                if len(old) and str(old["date"].min()) <= MIN_DATE \
                        and str(old["date"].max()) >= OK_MAX:
                    skip += 1
                    continue
            except Exception:
                pass

        k = fetch_em(code, market)
        rows = parse(k)
        if not rows:
            err += 1
            consec_empty += 1
            # 疑似被封：退避等待，等限流解除
            if consec_empty >= 20:
                wait = 60
                print(f"  ⚠ 连续 {consec_empty} 个空响应，疑似被封，等待 {wait}s ...")
                time.sleep(wait)
                consec_empty = 0
            time.sleep(interval)
            continue
        consec_empty = 0
        df = pd.DataFrame(rows, columns=[
            "date", "open", "close", "high", "low", "volume", "amount", "turnover"])
        df.to_parquet(path, index=False)
        ok += 1
        time.sleep(interval)

        if (i + 1) % 100 == 0:
            el = time.time() - t0
            rate = (i + 1) / el
            print(f"  {i+1}/{len(files)} 补{ok} 跳{skip} 错{err} | "
                  f"{el:.0f}s 剩余~{(len(files)-i-1)*interval:.0f}s | 速率{rate:.2f}/s")

    el = time.time() - t0
    print(f"\n完成: 补齐 {ok} 只, 跳过(已最新) {skip} 只, 失败 {err} 只 | 耗时 {el:.0f}s")


if __name__ == "__main__":
    main()
