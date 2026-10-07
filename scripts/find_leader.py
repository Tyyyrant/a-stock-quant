#!/usr/bin/env python3
"""
龙头识别 + 板块时序（验证版）
- 涨停池(封板时间/封单/炸板/换手/行业) + F10 核心题材(ssbk 所属板块)
- 辨识度评分排序连板股(≥2板)：高度/封板时间/封单/换手/板块效应/炸板
- 板块时序：近 N 日各行业涨停家数，标「点火(0→N)」与「退潮(N→0)」
用法: python3 scripts/find_leader.py [YYYY-MM-DD]
"""
import os, sys, json, time
from pathlib import Path
from collections import Counter, defaultdict
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent
STOCK_DIR = ROOT / 'data' / 'stocks'
UT = '7eea3edcaed734bea9cbfc24409ed989'
MAIN_10CM = ('600', '601', '603', '605', '000', '001', '002', '003')


def is_10cm(c):
    return c.startswith(MAIN_10CM)


def l_up(pc):
    return round(pc * 1.1, 2)


def fetch_zt_pool(date):
    """涨停池 -> {code: {...}}"""
    ymd = date.replace('-', '')
    try:
        r = requests.get('https://push2ex.eastmoney.com/getTopicZTPool',
                         params={'ut': UT, 'dpt': 'wz.ztzt', 'Pageindex': 0, 'pagesize': 500,
                                 'sort': 'fbt:asc', 'date': ymd},
                         headers={'User-Agent': 'Mozilla/5.0', 'Referer': 'https://quote.eastmoney.com/'},
                         timeout=20)
        r.raise_for_status()
        pool = (r.json().get('data') or {}).get('pool') or []
    except Exception:
        pool = []
    out = {}
    for it in pool:
        out[it.get('c')] = {
            'name': it.get('n'), 'hybk': it.get('hybk'), 'lbc': it.get('lbc'),
            'fbt': it.get('fbt'), 'lbt': it.get('lbt'), 'fund': it.get('fund'),
            'zbc': it.get('zbc'), 'hs': it.get('hs'), 'ltsz': it.get('ltsz'),
        }
    return out


def fetch_ssbk(code):
    """F10 所属板块 -> list[str]"""
    mkt = 'SZ' if code[0] in ('0', '3') else 'SH'
    try:
        r = requests.get(f'https://emweb.securities.eastmoney.com/PC_HSF10/CoreConception/PageAjax?code={mkt}{code}',
                         headers={'User-Agent': 'Mozilla/5.0'}, timeout=12)
        d = r.json()
        return [x.get('BOARD_NAME') for x in (d.get('ssbk') or []) if x and x.get('BOARD_NAME')]
    except Exception:
        return []


# ---- 评分 ----
def seal_score(fbt):
    """首封时间：越早越强，尾盘板最弱（次日晋级核心信号）"""
    if not fbt:
        return 6
    h = fbt // 10000
    sec = h * 3600 + (fbt // 100 % 100) * 60
    if sec <= 34200: return 20   # 竞价/秒板(09:30前)
    if sec <= 36000: return 16   # 10:00 前
    if sec <= 41400: return 11   # 11:30 前
    if sec <= 46800: return 6    # 13:00 前
    if sec <= 52200: return 3    # 14:30 前
    return 0                      # 尾盘板(14:30后)最弱


def fund_score(fund):
    if not fund: return 3
    if fund >= 2e8: return 12
    if fund >= 1e8: return 10
    if fund >= 5e7: return 7
    if fund >= 2e7: return 4
    return 2


def hs_score(hs, ptype):
    if hs is None: return 4
    if ptype == '一字板':
        return 8   # 一字板换手天然低=封死抢不到, 不是弱
    if 10 <= hs <= 30: return 10   # 换手/T字/回封的健康区间(分歧转一致)
    if 5 <= hs < 10 or 30 < hs <= 50: return 6
    return 3


PTYPE_BONUS = {'一字板': 6, 'T字板': 3, '回封板': 0, '换手板': -2}


def score(board, fbt, fund, hs, ptype, sect_cnt):
    parts = {
        '高度': min((board - 1) * 15, 60),
        '时间': seal_score(fbt),
        '封单': fund_score(fund),
        '换手': hs_score(hs, ptype),
        '板块': min(sect_cnt, 8) * 2,
        '板型': PTYPE_BONUS.get(ptype, 0),
    }
    return sum(parts.values()), parts


def fbt_str(fbt):
    if not fbt:
        return '—'
    s = str(fbt).zfill(6)
    return f"{s[:2]}:{s[2:4]}"


def ptype(open_at_limit, low_at_limit, zbc):
    if open_at_limit:
        if zbc is not None:
            return '一字板' if zbc == 0 else 'T字板'
        return '一字板' if low_at_limit else 'T字板'
    return '回封板' if (zbc is not None and zbc > 0) else '换手板'


def main():
    target = sys.argv[1] if len(sys.argv) > 1 else None

    # 加载 10cm K 线
    close_map, open_map, low_map = {}, {}, {}
    for f in os.listdir(STOCK_DIR):
        if not f.endswith('.parquet') or f.startswith('INDEX_'):
            continue
        code = f[:-8]
        if not is_10cm(code):
            continue
        df = pd.read_parquet(STOCK_DIR / f)
        close_map[code] = dict(zip(df['date'], df['close']))
        open_map[code] = dict(zip(df['date'], df['open']))
        low_map[code] = dict(zip(df['date'], df['low']))
    all_dates = sorted(set().union(*[set(m.keys()) for m in close_map.values()]))
    if target is None or target not in all_dates:
        target = all_dates[-1]
    idx = all_dates.index(target)

    # 近 16 日涨停集合（用于连板计数）
    up_by_date = {}
    for di in range(max(0, idx - 15), idx + 1):
        d = all_dates[di]
        prev = all_dates[di - 1] if di > 0 else None
        up = set()
        for code, cm in close_map.items():
            if d not in cm or prev is None or prev not in cm:
                continue
            if abs(cm[d] - l_up(cm[prev])) < 0.005:
                up.add(code)
        up_by_date[d] = up

    prev_date = all_dates[idx - 1]
    t_up = up_by_date[target]

    # 涨停池
    pool = fetch_zt_pool(target)
    # 所属行业 + 板块效应（10cm 内）
    hybk_of = {}
    for code in t_up:
        p = pool.get(code)
        if p and p.get('hybk'):
            hybk_of[code] = p['hybk']
    sect_cnt = Counter(hybk_of.values())

    # 连板数 + 板型
    rows = []
    for code in t_up:
        n = 1
        j = idx - 1
        while j >= 0 and code in up_by_date.get(all_dates[j], set()):
            n += 1
            j -= 1
        lp = l_up(close_map[code][prev_date])
        oal = abs(open_map[code].get(target, 0) - lp) < 0.005
        lal = abs(low_map[code].get(target, 0) - lp) < 0.005
        p = pool.get(code) or {}
        rows.append({
            'code': code, 'name': p.get('name') or '', 'board': n,
            'ptype': ptype(oal, lal, p.get('zbc')),
            'fbt': p.get('fbt'), 'fund': p.get('fund'), 'zbc': p.get('zbc'),
            'hs': p.get('hs'), 'hybk': p.get('hybk'),
            'sect': hybk_of.get(code),
        })

    no_pool = [r for r in rows if r['board'] >= 2 and not r['name']]
    cand = [r for r in rows if r['board'] >= 2 and r['name']]
    for r in cand:
        r['score'], r['parts'] = score(r['board'], r['fbt'], r['fund'], r['hs'],
                                       r['ptype'], sect_cnt.get(r['sect'], 0))
    cand.sort(key=lambda x: -x['score'])

    # 题材标签（仅龙头候选）
    for r in cand:
        r['ssbk'] = fetch_ssbk(r['code'])
        time.sleep(0.2)

    # 板块时序（近 12 日）
    N = 12
    hist_dates = all_dates[max(0, idx - N):idx + 1]
    seq = defaultdict(dict)
    for d in hist_dates:
        pd_ = fetch_zt_pool(d)
        for c, p in pd_.items():
            if not is_10cm(c):
                continue
            hb = p.get('hybk')
            if hb:
                seq[hb][d] = seq[hb].get(d, 0) + 1

    # 输出报告
    # 未入涨停池 = 未封死(尾盘封单未吃完), 不是数据缺失 —— 直接排除
    excl = f" · 未封死排除{len(no_pool)}只[{','.join(r['code'] for r in no_pool)}]" if no_pool else ""
    print('=' * 72)
    print(f"龙头识别 · {target}  (10cm 涨停 {len(t_up)} 只 / 连板≥2 {len(cand)} 只{excl})")
    print('=' * 72)
    print(f"{'#':<2}{'股票':<10}{'板':<4}{'板型':<5}{'封板':<7}{'封单':<8}{'换手':<7}{'炸':<3}{'行业':<8}{'分':<5}")
    for i, r in enumerate(cand, 1):
        fund = f"{r['fund']/1e8:.2f}亿" if r['fund'] else '—'
        hs = f"{r['hs']:.1f}%" if r['hs'] else '—'
        print(f"{i:<2}{r['name']:<10}{r['board']}板{'':<2}{r['ptype']:<5}{fbt_str(r['fbt']):<7}"
              f"{fund:<8}{hs:<7}{r['zbc'] or 0:<3}{(r['hybk'] or '—')[:6]:<8}{r['score']:<5}")
    print()
    print('【龙头候选 · 题材标签(ssbk 前6)】')
    for i, r in enumerate(cand[:8], 1):
        tags = ' / '.join(r['ssbk'][:6]) if r['ssbk'] else '(无)'
        print(f"  {i}. {r['name']}({r['code']}) {r['board']}板: {tags}")

    print()
    print('【主线板块 · 涨停家数时序】(近13日 · ↑点火=冷(≤1)→热(≥3) · ↓退潮=热(≥3)→0)')
    print('   ' + ' '.join(d[5:] for d in hist_dates))
    hot = sorted(seq.items(), key=lambda kv: -(kv[1].get(target, 0)))
    for hb, m in hot[:18]:
        series = [m.get(d, 0) for d in hist_dates]
        if series[-1] < 2:
            continue
        cells = []
        for k, v in enumerate(series):
            cell = f"{v:>2}"
            if v >= 3 and (k == 0 or series[k - 1] <= 1):
                cell += '↑'
            elif v == 0 and k > 0 and series[k - 1] >= 3:
                cell += '↓'
            cells.append(cell)
        print(f"  {hb[:8]:<8} " + ' '.join(cells))


if __name__ == '__main__':
    main()
