#!/usr/bin/env python3
"""
8月回测 —— 用 K 线推导涨停池（东财接口只回溯3周，8月抓不到）。

涨停/连板由 K 线推导（验证过：与东财涨停池误判率 0.5%），但封单资金/首封时间/换手/题材补不了，
所以选股退化为「成交额排序」，达标门槛退化为「非一字次日高开≥1%」。
方法本体不变：满仓一只、2板非一字打板、晋级躺、断板走。

用法: python3 scripts/backtest_aug_kline.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from compute_emotion import is_10cm, load_names, ROOT
import pandas as pd

START = '2026-08-03'   # 8月第一个交易日
END_CAND = '2026-08-31'  # 只从8月选新候选；之后只用于判定8月仓位的断板出场


def load_all():
    close, open_, amt = {}, {}, {}
    for f in (ROOT / 'data' / 'stocks').iterdir():
        if not f.name.endswith('.parquet') or f.name.startswith('INDEX_'):
            continue
        code = f.name[:-8]
        if not is_10cm(code):
            continue
        df = pd.read_parquet(f, columns=['date', 'open', 'close', 'amount'])
        close[code] = dict(zip(df['date'], df['close']))
        open_[code] = dict(zip(df['date'], df['open']))
        amt[code] = dict(zip(df['date'], df['amount']))
    return close, open_, amt


def build_lbc(close):
    """逐日连板数: {date: {code: 连板数}}"""
    lbc = {}
    for code, cmap in close.items():
        ds = sorted(cmap)
        run = 0
        for i in range(1, len(ds)):
            d, prev = ds[i], ds[i - 1]
            pc, c = cmap[prev], cmap[d]
            if pc and c and abs(c - round(pc * 1.1, 2)) < 0.005:
                run += 1
                lbc.setdefault(d, {})[code] = run
            else:
                run = 0
    return lbc


def main():
    names = load_names()
    close, open_, amt = load_all()
    lbc = build_lbc(close)
    tdays = sorted(set().union(*(set(m) for m in close.values())))
    tdays = [d for d in tdays if d >= '2026-07-30']
    ti = {d: i for i, d in enumerate(tdays)}
    dates = [d for d in tdays if d >= START]

    def select_cands(d):
        if d > END_CAND:
            return []
        idx = ti[d]
        prev = tdays[idx - 1] if idx > 0 else None
        if prev is None:
            return []
        out = []
        for code, b in lbc.get(d, {}).items():
            if b != 2 or code not in close:
                continue
            pc = close[code].get(prev)
            o = open_[code].get(d)
            if pc is None or o is None:
                continue
            if abs(o - round(pc * 1.1, 2)) < 0.005:  # 开盘涨停(一字/T字)剔除，留可买到的换手/回封
                continue
            out.append({'code': code, 'name': names.get(code, code), 'amt': amt[code].get(d, 0)})
        out.sort(key=lambda x: -(x['amt'] or 0))  # 成交额大的排前（人气/辨识度近似）
        return out

    capital = 1.0
    holding = None
    pending = []
    trades = []

    for d in dates:
        idx = ti[d]
        prev_d = tdays[idx - 1] if idx > 0 else None
        # ① 空仓则次日高开达标打板
        if holding is None and pending and prev_d:
            hit = None
            for cand in pending:
                o = open_.get(cand['code'], {}).get(d)
                cl = close.get(cand['code'], {}).get(prev_d)
                if not o or not cl:
                    continue
                opct = (o / cl - 1) * 100
                if opct >= 1.0:  # 非一字统一门槛：高开≥1%
                    hit = (cand, opct)
                    break
            if hit:
                cand, opct = hit
                holding = {'code': cand['code'], 'name': cand['name'],
                           'entry_open': open_[cand['code']][d], 'entry': d, 'opct': opct}
        # ② 断板卖出
        if holding:
            c = holding['code']
            if c not in lbc.get(d, {}):  # 当日不再涨停 = 断板
                cl = close[c].get(d)
                if cl:
                    r = (cl / holding['entry_open'] - 1) * 100
                    capital *= (1 + r / 100)
                    trades.append((holding, d, r))
                    holding = None
        # ③ 空仓选明日候选
        pending = select_cands(d) if holding is None else []

    print(f"{'买入日':<9}{'票':<9}{'高开':>7}{'连板轨迹':<30}{'卖出日':<9}{'收益%':>8}")
    print('-' * 76)
    for h, d, r in trades:
        path = []
        for dd in dates:
            if dd < h['entry']:
                continue
            if h['code'] in lbc.get(dd, {}):
                path.append(f"{lbc[dd][h['code']]}板")
            else:
                break
        print(f"{h['entry'][5:]:<9}{h['name']:<9}{h['opct']:>+6.1f}%  {'→'.join(path):<30}{d[5:]:<9}{r:>+8.1f}%")
    print('-' * 76)
    n = len(trades)
    wins = sum(1 for h, d, r in trades if r > 0)
    print(f"共 {n} 笔(已平) ｜ 赚 {wins} 亏 {n - wins} ｜ 满仓复利终值 {capital:.4f}（{(capital-1)*100:+.1f}%）")
    if holding:
        cl = close[holding['code']].get(dates[-1])
        if cl:
            ur = (cl / holding['entry_open'] - 1) * 100
            print(f"未平仓: {holding['name']} {holding['entry'][5:]}买入 浮盈 {ur:+.1f}%")


if __name__ == '__main__':
    main()
