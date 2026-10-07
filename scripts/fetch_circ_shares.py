#!/usr/bin/env python3
"""
批量抓取涨停池 10cm 票的流通股本/流通市值（东财），供 fund_score 归一化封单。
流通股本 = 流通市值 / 最新价；历史日流通市值 = 流通股本 × 当日收盘价。
输出 data/zt_pool_circ.json: {code: {name, circ_shares, circ_market_cap, price, date}}
用法: python3 scripts/fetch_circ_shares.py
"""
import json, time
from pathlib import Path
import requests

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / 'data' / 'zt_pool_circ.json'
UT = '7eea3edcaed734bea9cbfc24409ed989'
MAIN = ('600', '601', '603', '605', '000', '001', '002', '003')


def load_codes():
    zt = json.load(open(ROOT / 'data' / 'zt_pool_history.json'))
    codes = set()
    for d, s in zt.items():
        for x in s['stocks']:
            if x['code'].startswith(MAIN):
                codes.add(x['code'])
    return sorted(codes)


def secid(code):
    return ('1.' if code.startswith(('6', '9')) else '0.') + code


def fetch_batch(S, codes):
    secids = ','.join(secid(c) for c in codes)
    r = S.get('https://push2.eastmoney.com/api/qt/ulist.np/get',
              params={'ut': UT, 'fltt': 2, 'invt': 2,
                      'fields': 'f2,f12,f14,f21', 'secids': secids}, timeout=20)
    r.raise_for_status()
    return (r.json().get('data') or {}).get('diff') or []


def fetch_one(S, code):
    r = S.get('https://push2.eastmoney.com/api/qt/stock/get',
              params={'ut': UT, 'fltt': 2, 'invt': 2,
                      'fields': 'f2,f12,f14,f21', 'secid': secid(code)}, timeout=20)
    r.raise_for_status()
    return r.json().get('data')


def parse(it):
    try:
        code = str(it['f12'])
        price = float(it['f2'])
        cmc = float(it['f21'])
    except (TypeError, ValueError, KeyError):
        return None
    if not code or not price or not cmc:
        return None
    return code, it.get('f14'), price, cmc


def main():
    codes = load_codes()
    print(f'待取 {len(codes)} 只(10cm)')
    S = requests.Session()
    S.headers.update({'User-Agent': 'Mozilla/5.0', 'Referer': 'https://quote.eastmoney.com/'})

    out = {}
    if OUT.exists():
        old = json.load(open(OUT))
        out = {k: v for k, v in old.items() if not k.startswith('_')}
    todo = [c for c in codes if c not in out]
    print(f'已缓存 {len(out)} 只，剩余 {len(todo)} 只')

    B = 30
    for i in range(0, len(todo), B):
        batch = todo[i:i + B]
        got = {}
        # 批量，失败重试+退避
        for attempt in range(3):
            try:
                for it in fetch_batch(S, batch):
                    p = parse(it)
                    if p:
                        got[p[0]] = p
                break
            except Exception as e:
                if attempt == 2:
                    print(f'  批量 {i}-{i+len(batch)} 连续失败: {e}')
                time.sleep(2 + attempt * 3)
        # 批量没拿全的，逐个补
        for c in batch:
            if c in got:
                continue
            try:
                it = fetch_one(S, c)
                p = parse(it) if it else None
                if p:
                    got[p[0]] = p
                time.sleep(0.15)
            except Exception:
                pass
        for code, name, price, cmc in got.values():
            out[code] = {'name': name, 'price': round(price, 3),
                         'circ_market_cap': round(cmc, 0), 'circ_shares': round(cmc / price, 0)}
        # 增量落盘，防丢
        OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding='utf-8')
        print(f'  {i + len(batch)}/{len(todo)} 累计 {len(out)} 只')
        time.sleep(1.0)

    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding='utf-8')
    print(f'完成：{len(out)} 只 → {OUT}')


if __name__ == '__main__':
    main()
