#!/usr/bin/env python3
"""
从东方财富「涨停池」回溯抓取历史涨停明细（逐日），供 compute_emotion.py 做历史辨识度对比。
字段：code/name/行业/连板数/首次封板时间/最后封板时间/封单资金/炸板次数/换手率/成交额
输出 data/zt_pool_history.json（按日期 key）
用法: python3 scripts/fetch_zt_pool_history.py [YYYY-MM-DD]   # 起始日，缺省 2026-09-01

注意：东财 getTopicZTPool 接口只回溯约 3 周，更早的日期会返回空（count=0）。
     涨停/连板/炸板仍可由 K 线推导，但封单/首封时间/换手/题材这些字段无法补。
"""
import sys, json, time
from pathlib import Path
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent
HIST_PATH = ROOT / 'data' / 'zt_pool_history.json'
UT = '7eea3edcaed734bea9cbfc24409ed989'
URL = 'https://push2ex.eastmoney.com/getTopicZTPool'


def trading_days():
    """从 K 线缓存取全体交易日期（升序，去重）"""
    dates = set()
    for f in (ROOT / 'data' / 'stocks').iterdir():
        if not f.name.endswith('.parquet') or f.name.startswith('INDEX_'):
            continue
        try:
            for d in pd.read_parquet(f, columns=['date'])['date'].astype(str):
                dates.add(d)
        except Exception:
            continue
    return sorted(dates)


def fmt_time(v):
    if v in (None, '', '-', 0):
        return None
    s = str(int(v)).zfill(6)
    return f'{s[:2]}:{s[2:4]}:{s[4:6]}'


def fetch_pool(session, ymd):
    r = session.get(URL, params={'ut': UT, 'dpt': 'wz.ztzt', 'Pageindex': 0,
                                 'pagesize': 500, 'sort': 'fbt:asc', 'date': ymd},
                    timeout=20)
    r.raise_for_status()
    pool = (r.json().get('data') or {}).get('pool') or []
    return [{
        'code': it.get('c'), 'name': it.get('n'), 'hybk': it.get('hybk'),
        'lbc': it.get('lbc'), 'fbt': fmt_time(it.get('fbt')), 'lbt': fmt_time(it.get('lbt')),
        'fund': it.get('fund'), 'zbc': it.get('zbc'), 'hs': it.get('hs'),
        'amount': it.get('amount'),
    } for it in pool]


def main():
    start = sys.argv[1] if len(sys.argv) > 1 else '2026-09-09'
    days = [d for d in trading_days() if d >= start]
    if not days:
        print(f'K 线里没有 >= {start} 的交易日'); return

    S = requests.Session()
    S.headers.update({'User-Agent': 'Mozilla/5.0', 'Referer': 'https://quote.eastmoney.com/'})

    # 合并已有历史，避免覆盖更早(接口已不可回溯)的旧数据
    hist = {}
    if HIST_PATH.exists():
        try:
            hist = json.loads(HIST_PATH.read_text(encoding='utf-8'))
            print(f'已有历史 {len(hist)} 日，合并更新 {start} 之后', flush=True)
        except Exception:
            hist = {}
    missed = []
    for i, d in enumerate(days):
        ymd = d.replace('-', '')
        try:
            stocks = fetch_pool(S, ymd)
        except Exception as e:
            print(f'{d} 抓取失败: {e}'); missed.append(d); continue
        if not stocks:
            missed.append(d)
            print(f'{d} 接口返回空（回溯窗口外）')
        else:
            hist[d] = {'count': len(stocks), 'stocks': stocks}
            print(f'{d} {len(stocks):>3} 只')
        time.sleep(0.3)

    HIST_PATH.write_text(json.dumps(hist, ensure_ascii=False, indent=1), encoding='utf-8')
    print(f'\n完成：{len(hist)} 个交易日 → {HIST_PATH}')
    if missed:
        print(f'未取到（接口窗口外/失败）：{missed}')


if __name__ == '__main__':
    main()
