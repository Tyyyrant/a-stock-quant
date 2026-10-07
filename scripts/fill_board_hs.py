#!/usr/bin/env python3
"""
用同花顺 getharden(零鉴权, 可回溯2024+) 补 board_dataset.csv 的 换手率 + 涨停原因/题材。

字段来源：
  - 换手率(d2_hs/d3_hs)：getharden 的 huanshou(%)。覆盖 2024+ 全部交易日。
  - 涨停原因(d2_reason/d3_reason)：getharden 的 reason(题材归因)，龙头战法辨识度核心。
  - 涨停时间/炸板次数：getharden 无 → 本脚本不碰，由 fill_board_zt.py(东财涨停池, 仅3周) 补。

动作：读 data/board_dataset.csv → 取所有 d2/d3 日期 → 逐日抓 getharden
      → 缓存 data/ths_harden_cache.json {date: {code: {hs, reason}}}
      → 回填 d2_hs/d3_hs + 新增 d2_reason/d3_reason。

用法: python3 scripts/fill_board_hs.py [--refresh] [--interval S]
      --refresh 强制重抓缓存；--interval 节流(默认 0.15s)
"""
import sys, os, json, subprocess, time
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CSV = ROOT / 'data' / 'board_dataset.csv'
CACHE = ROOT / 'data' / 'ths_harden_cache.json'
UA = 'Mozilla/5.0'


def fetch_day(date, interval):
    url = (f'http://zx.10jqka.com.cn/event/api/getharden/date/{date}/'
           f'orderby/date/orderway/desc/charset/GBK/')
    try:
        out = subprocess.run(['curl', '-s', '--max-time', '10',
                              '-H', f'User-Agent: {UA}',
                              '-H', 'Referer: https://www.10jqka.com.cn/', url],
                             capture_output=True, text=True, timeout=12).stdout
        j = json.loads(out)
        if j.get('errocode', 0) != 0:
            return {}
        m = {}
        for it in (j.get('data') or []):
            code = it.get('code', '')
            if not code:
                continue
            hs = it.get('huanshou')
            try:
                hs = round(float(hs), 2) if hs not in (None, '', '-') else None
            except (ValueError, TypeError):
                hs = None
            m[code] = {'hs': hs, 'reason': (it.get('reason') or '').strip()}
        return m
    except Exception:
        return {}


def main():
    args = sys.argv[1:]
    refresh = '--refresh' in args
    interval = 0.15
    for a in args:
        if a.startswith('--interval='):
            interval = float(a.split('=', 1)[1])

    df = pd.read_csv(CSV, converters={'code': lambda x: str(x).zfill(6)})
    dates = sorted(set(df['d2'].dropna()) | set(df['d3'].dropna()))
    print(f'{len(dates)} 个交易日待抓 getharden ...', flush=True)

    cache = {}
    if not refresh and CACHE.exists():
        cache = json.load(open(CACHE))
        print(f'  已有缓存 {len(cache)} 日', flush=True)

    t0 = time.time()
    new = 0
    for i, d in enumerate(dates):
        if d in cache:
            continue
        m = fetch_day(d, interval)
        cache[d] = m
        new += 1
        time.sleep(interval)
        if (i + 1) % 100 == 0:
            print(f'  {i+1}/{len(dates)} 新增{new} | {time.time()-t0:.0f}s', flush=True)

    json.dump(cache, open(CACHE, 'w'), ensure_ascii=False)
    hit = sum(1 for d in cache if cache[d])
    print(f'抓取完成：缓存 {len(cache)} 日，其中 {hit} 日有涨停数据 | 耗时 {time.time()-t0:.0f}s', flush=True)

    # 回填
    def fill(col_date, col_hs, col_reason):
        for i, row in df.iterrows():
            d = row[col_date]
            if pd.isna(d):
                continue
            m = cache.get(str(d), {})
            it = m.get(str(row['code']))
            if it:
                if it.get('hs') is not None:
                    df.at[i, col_hs] = it['hs']
                if it.get('reason'):
                    df.at[i, col_reason] = it['reason']

    df['d2_reason'] = ''
    df['d3_reason'] = ''
    fill('d2', 'd2_hs', 'd2_reason')
    fill('d3', 'd3_hs', 'd3_reason')

    df.to_csv(CSV, index=False, encoding='utf-8-sig')
    n2 = df['d2_hs'].astype(str).ne('').sum()
    n3 = df['d3_hs'].astype(str).ne('').sum()
    r2 = df['d2_reason'].astype(str).ne('').sum()
    print(f'\n回填完成 → {CSV}')
    print(f'  d2_hs 已填 {n2}/{len(df)} ｜ d3_hs 已填 {n3}/{len(df)}')
    print(f'  涨停原因已填 {r2}/{len(df)} 行(至少2板日)')


if __name__ == '__main__':
    main()
