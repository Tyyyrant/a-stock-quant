#!/usr/bin/env python3
"""
构建「2板/3板」数据集 —— 找出所有走到 3 板及以上的 10cm 主板股票，
记录其 2板日、3板日 的开盘/收盘/换手率/涨停时间/炸板次数 + 后续走到第几板还是断板。

字段口径（务必认清，别高估）：
  - 开盘/收盘/连板数/板型(一字/T字/换手)/后续走到第几板：K 线精确推导。
  - 换手率(hs)：读 parquet 的 turnover 列（由 fetch_klines_em.py 从东财补齐）。
  - 涨停时间(首封 fbt)：日线无、涨停池只回溯3周 → 留空，需 Tushare limit_list_d。
  - 炸板次数(zbc)：日线无 → 留空；日线最多给「盘中是否开板」近似(=非一字板)。

用法:
  python3 scripts/build_board_dataset.py [--start 2024-01-01]
  输出: data/board_dataset.csv (utf-8-sig, Excel 直接打开)
"""
import sys, os
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
STOCK_DIR = ROOT / "data" / "stocks"

MAIN_10CM = ('600', '601', '603', '605', '000', '001', '002', '003')

# 指数/股票代码冲突：all_stocks.json 里这些号只留了指数(market=1)，股票名缺失
_NAME_FALLBACK = {
    '000016': '深康佳A', '000010': '美丽生态', '000004': '国华网安',
    '000078': '海王生物', '000903': '云内动力',
}


def is_10cm(code: str) -> bool:
    return code.startswith(MAIN_10CM)


def load_names():
    from compute_emotion import load_names as _ln
    names = _ln()
    names.update(_NAME_FALLBACK)
    return names


def load_ohlc():
    """读 10cm 主板 K 线 → {code: {date: (open, close, high, low, turnover)}}"""
    m = {}
    for f in os.listdir(STOCK_DIR):
        if not f.endswith('.parquet') or f.startswith('INDEX_'):
            continue
        code = f[:-8]
        if not is_10cm(code):
            continue
        df = pd.read_parquet(STOCK_DIR / f)
        hs_map = dict(zip(df['date'], df['turnover'])) if 'turnover' in df.columns else {}
        m[code] = {r.date: (r.open, r.close, r.high, r.low, hs_map.get(r.date))
                   for r in df.itertuples()}
    return m


def detect_runs(ohlc):
    """逐股识别连续涨停段(连板 run)，返回 [(code, [d1,d2,d3,...])]，仅保留 >=3 板的 run"""
    out = []
    for code, m in ohlc.items():
        ds = sorted(m)
        run = []
        prev_close = None
        for d in ds:
            o, c, h, l, _ = m[d]
            is_up = (prev_close is not None and c
                     and abs(c - round(prev_close * 1.1, 2)) < 0.005)
            if is_up:
                run.append(d)
            else:
                if len(run) >= 3:
                    out.append((code, list(run)))
                run = []
            prev_close = c
        if len(run) >= 3:
            out.append((code, list(run)))
    return out


def ptype_of(open_, low, lim):
    """板型(日线近似)：一字/T字/换手。回封与换手日线分不清，统一归「换手」"""
    oal = abs(open_ - lim) < 0.005
    lal = abs(low - lim) < 0.005
    if oal and lal:
        return '一字板'
    if oal and not lal:
        return 'T字板'
    return '换手板'


def main():
    start = '2024-01-01'
    for a in sys.argv[1:]:
        if a.startswith('--start='):
            start = a.split('=', 1)[1]

    names = load_names()
    print("读取 10cm 主板 K 线 ...")
    ohlc = load_ohlc()
    print(f"  载入 {len(ohlc)} 只")

    print("识别连板 run（>=3板） ...")
    runs = detect_runs(ohlc)
    runs.sort(key=lambda x: x[1][2])  # 按 3板日 排序
    print(f"  共 {len(runs)} 段 3板+ 连板")

    rows = []
    for code, run in runs:
        d1, d2, d3 = run[0], run[1], run[2]
        peak = len(run)
        if d3 < start:
            continue
        ds = sorted(ohlc[code])
        next_after_peak = None
        for i, d in enumerate(ds):
            if d == run[-1] and i + 1 < len(ds):
                next_after_peak = ds[i + 1]
                break
        outcome = ('断板' if peak == 3 else f'走到{peak}板') if next_after_peak else '进行中'

        def day_info(d, prev_close):
            o, c, h, l, hs = ohlc[code][d]
            lim = round(prev_close * 1.1, 2)
            return {'open': round(o, 2), 'close': round(c, 2),
                    'ptype': ptype_of(o, l, lim),
                    'hs': round(hs, 2) if hs is not None and hs == hs else ''}

        i2 = day_info(d2, ohlc[code][d1][1])
        i3 = day_info(d3, ohlc[code][d2][1])
        rows.append({
            'code': code, 'name': names.get(code, ''),
            'd2': d2, 'd2_open': i2['open'], 'd2_close': i2['close'], 'd2_hs': i2['hs'],
            'd2_ptype': i2['ptype'],
            'd3': d3, 'd3_open': i3['open'], 'd3_close': i3['close'], 'd3_hs': i3['hs'],
            'd3_ptype': i3['ptype'],
            'peak_board': peak, 'outcome': outcome,
            # 涨停时间/炸板次数：日线无，留空待 Tushare
            'd2_zt_time': '', 'd3_zt_time': '', 'd2_zha_cnt': '', 'd3_zha_cnt': '',
        })

    cols = ['code', 'name',
            'd2', 'd2_open', 'd2_close', 'd2_hs', 'd2_zt_time', 'd2_zha_cnt', 'd2_ptype',
            'd3', 'd3_open', 'd3_close', 'd3_hs', 'd3_zt_time', 'd3_zha_cnt', 'd3_ptype',
            'peak_board', 'outcome']
    df = pd.DataFrame(rows)[cols]
    # code 补零成 6 位字符串，避免 000/001/002/003 深市股丢前导零(int64 会吃掉)
    df['code'] = df['code'].astype(str).str.zfill(6)
    out_path = ROOT / 'data' / 'board_dataset.csv'
    df.to_csv(out_path, index=False, encoding='utf-8-sig')

    n = len(df)
    n_duan = (df['outcome'] == '断板').sum()
    n_up = df['outcome'].str.startswith('走到').sum()
    n_ing = (df['outcome'] == '进行中').sum()
    n_hs = df['d2_hs'].astype(str).ne('').sum()
    print(f"\n写入 {out_path} ｜ {n} 行")
    print(f"  3板后断板 {n_duan} ｜ 走到4板+ {n_up} ｜ 进行中 {n_ing}")
    print(f"  日期范围: {df['d3'].min()} ~ {df['d3'].max()}")
    print(f"  换手率已填 {n_hs}/{n} 行")
    if n_up + n_duan:
        print(f"  3板后晋级率(走到4板+): {n_up/(n_up+n_duan)*100:.1f}%")
    print("  ⚠ 涨停时间/炸板次数：日线无法推导，需 Tushare limit_list_d 补")


if __name__ == '__main__':
    main()
