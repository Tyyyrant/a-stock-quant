#!/usr/bin/env python3
"""
并发增量更新 K 线到最新交易日：新浪日K(datalen=800, 全量拉取后按 date 合并去重，
已有优先保留精确 amount)。只处理 max(date) < TARGET 的股票，已最新的跳过。

用法: python3 scripts/fetch_today_kline.py [YYYY-MM-DD] [--all]
      缺省只更新 10cm 主板；--all 更新全市场(含创业板/科创板)。
"""
import os, sys, json, subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
STOCK_DIR = ROOT / 'data' / 'stocks'
MAIN_10CM = ('600', '601', '603', '605', '000', '001', '002', '003')
COLS = ['date', 'open', 'close', 'high', 'low', 'volume', 'amount', 'turnover']
WORKERS = 6


def is_10cm(code):
    return code.startswith(MAIN_10CM)


def fetch(code):
    prefix = 'sh' if code.startswith('6') else 'sz'
    url = (f'https://quotes.sina.cn/cn/api/json_v2.php/CN_MarketDataService.getKLineData'
           f'?symbol={prefix}{code}&scale=240&ma=no&datalen=800')
    out = subprocess.run(['curl', '-s', '--max-time', '12',
                          '-H', 'User-Agent: Mozilla/5.0', url],
                         capture_output=True, text=True, timeout=15).stdout
    try:
        return json.loads(out)
    except Exception:
        return []


def update(fname):
    code = fname[:-8]
    path = STOCK_DIR / fname
    try:
        days = fetch(code)
    except Exception:
        return ('err', code)
    if not isinstance(days, list) or not days:
        return ('empty', code)
    rows = []
    for r in days:
        try:
            d, o, h, l, c = r['day'], float(r['open']), float(r['high']), \
                            float(r['low']), float(r['close'])
            v = float(r['volume']) / 100          # 股 -> 手
            rows.append((d, o, c, h, l, round(v, 2), round(v * c, 2), None))
        except (KeyError, ValueError):
            continue
    if not rows:
        return ('empty', code)
    tdf = pd.DataFrame(rows, columns=COLS)
    try:
        old = pd.read_parquet(path)
        if 'turnover' not in old.columns:
            old['turnover'] = None
        old = old[COLS]
        merged = pd.concat([old, tdf], ignore_index=True)
        merged = merged.drop_duplicates(subset='date', keep='first')  # 已有优先
    except Exception:
        merged = tdf
    merged = merged.sort_values('date').reset_index(drop=True)
    merged.to_parquet(path, index=False)
    mx = str(merged['date'].max())
    return ('stale', code, mx) if mx < TARGET else ('ok', code)


def main():
    args = sys.argv[1:]
    only_10cm = '--all' not in args
    global TARGET
    TARGET = next((a for a in args if a.startswith('20')), '2026-09-30')
    if TARGET.startswith('--'):
        TARGET = '2026-09-30'

    files = [f for f in os.listdir(STOCK_DIR)
             if f.endswith('.parquet') and not f.startswith('INDEX_')]
    if only_10cm:
        files = [f for f in files if is_10cm(f[:-8])]

    todo = []
    for f in files:
        try:
            if str(pd.read_parquet(STOCK_DIR / f, columns=['date'])['date'].max()) >= TARGET:
                continue
        except Exception:
            pass
        todo.append(f)
    print(f'并发更新 K 线 → {TARGET} ｜ 待更新 {len(todo)}/{len(files)} 只', flush=True)

    ok = err = stale = 0
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = [ex.submit(update, f) for f in todo]
        for i, fu in enumerate(as_completed(futs), 1):
            r = fu.result()
            if r[0] == 'ok':
                ok += 1
            elif r[0] == 'stale':
                stale += 1
            else:
                err += 1
            if i % 300 == 0:
                print(f'  {i}/{len(todo)} 已最新{ok} 停牌/无今日{stale} 失败{err}', flush=True)
    print(f'完成：已更新到{TARGET} {ok} 只 ｜ 仍滞后(停牌等) {stale} 只 ｜ 失败 {err} 只', flush=True)


if __name__ == '__main__':
    main()
