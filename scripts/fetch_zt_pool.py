#!/usr/bin/env python3
"""
从东方财富「涨停池」抓取当日涨停明细（独立数据源，非连板网）。
字段：code/name/连板数/首次封板时间/最后封板时间/封单资金/炸板次数/换手率/成交额
输出 data/zt_pool.json（供 compute_emotion.py 合并）
用法: python3 scripts/fetch_zt_pool.py [YYYY-MM-DD]   # 缺省取 K 线最新交易日
"""
import sys, json
from pathlib import Path
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent
POOL_PATH = ROOT / 'data' / 'zt_pool.json'
UT = '7eea3edcaed734bea9cbfc24409ed989'
URL = 'https://push2ex.eastmoney.com/getTopicZTPool'


def latest_trade_date():
    """K 线缓存里最大的日期作为目标交易日"""
    mx = None
    for f in (ROOT / 'data' / 'stocks').iterdir():
        if not f.name.endswith('.parquet') or f.name.startswith('INDEX_'):
            continue
        try:
            d = str(pd.read_parquet(f, columns=['date'])['date'].max())
        except Exception:
            continue
        if mx is None or d > mx:
            mx = d
    return mx


def fmt_time(v):
    """92500 -> '09:25:00'"""
    if v in (None, '', '-', 0):
        return None
    s = str(int(v)).zfill(6)
    return f'{s[:2]}:{s[2:4]}:{s[4:6]}'


def main():
    date = sys.argv[1] if len(sys.argv) > 1 else latest_trade_date()
    ymd = date.replace('-', '')

    S = requests.Session()
    S.headers.update({'User-Agent': 'Mozilla/5.0', 'Referer': 'https://quote.eastmoney.com/'})
    r = S.get(URL, params={'ut': UT, 'dpt': 'wz.ztzt', 'Pageindex': 0, 'pagesize': 500,
                           'sort': 'fbt:asc', 'date': ymd}, timeout=20)
    r.raise_for_status()
    pool = (r.json().get('data') or {}).get('pool') or []
    if not pool:
        print(f'涨停池 {date}: 空（非交易日或接口无数据）')
        return

    stocks = []
    for it in pool:
        stocks.append({
            'code': it.get('c'),
            'name': it.get('n'),
            'hybk': it.get('hybk'),
            'lbc': it.get('lbc'),
            'fbt': fmt_time(it.get('fbt')),
            'lbt': fmt_time(it.get('lbt')),
            'fund': it.get('fund'),
            'zbc': it.get('zbc'),
            'hs': it.get('hs'),
            'amount': it.get('amount'),
        })
    POOL_PATH.write_text(json.dumps({'date': date, 'count': len(stocks), 'stocks': stocks},
                                    ensure_ascii=False, indent=1), encoding='utf-8')
    print(f'涨停池 {date}: {len(stocks)} 只 → {POOL_PATH}')


if __name__ == '__main__':
    main()
