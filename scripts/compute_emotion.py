#!/usr/bin/env python3
"""
情绪周期计算引擎（参考连板网 lianban.net 框架）
- 只看 10cm 主板：600/601/603/605/000/001/002/003
- 从 K 线计算涨停/跌停/炸板/连板梯队/晋级率/情绪温度/情绪阶段
- 输出 output/emotion/emotion_data.js（供 emotion_board.html 渲染）
用法: python3 scripts/compute_emotion.py [目标日期]
"""
import os, sys, json
from pathlib import Path
from collections import Counter, defaultdict
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
STOCK_DIR = ROOT / 'data' / 'stocks'
OUT_DIR = ROOT / 'output' / 'emotion'
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 10cm 主板前缀
MAIN_10CM = ('600', '601', '603', '605', '000', '001', '002', '003')


def is_10cm(code: str) -> bool:
    return code.startswith(MAIN_10CM)


def l_up(pc): return round(pc * 1.1, 2)
def l_down(pc): return round(pc * 0.9, 2)


def load_names():
    names = {}
    try:
        data = json.load(open(ROOT / 'data' / 'all_stocks.json'))
        for it in data:
            if isinstance(it, dict) and it.get('code') and it.get('name'):
                nm = str(it['name']).replace('\x00', '').strip()
                if nm:
                    names[it['code']] = nm
    except Exception:
        pass
    return names


def load_zt_pool():
    """读取东方财富涨停池缓存，返回 {'date': str|None, 'map': {code: {...}}}"""
    p = ROOT / 'data' / 'zt_pool.json'
    if not p.exists():
        return {'date': None, 'map': {}}
    try:
        d = json.load(open(p))
    except Exception:
        return {'date': None, 'map': {}}
    return {'date': d.get('date'), 'map': {s['code']: s for s in d.get('stocks', [])}}


def emotion_temp(limit_up, limit_down, break_cnt, ladder, max_board):
    """情绪温度 0-100（自研近似连板网）"""
    lian = sum(ladder.values())  # 连板家数(>=2板)
    seal = limit_up / (limit_up + break_cnt) * 100 if (limit_up + break_cnt) else 0
    breadth = min(limit_up / 90.0, 1.0) * 35        # 涨停广度 0-35
    continuity = min(lian / 18.0, 1.0) * 25         # 连板强度 0-25
    height = min((max_board - 1) / 5.0, 1.0) * 15   # 高度空间 0-15
    quality = max(min((seal - 35) / 55.0, 1.0), 0.0) * 15  # 封板质量 0-15
    negative = min(limit_down / 20.0, 1.0) * 30     # 跌停惩罚 0-30
    t = breadth + continuity + height + quality - negative
    return max(0, min(100, round(t)))


def classify(temp, prev_temp, prev_phase):
    """情绪阶段状态机：温度水平 + 趋势"""
    if temp <= 20:
        return '冰点'
    delta = temp - prev_temp
    if temp >= 75:
        return '高潮' if delta >= 0 else '降温'
    if delta > 5:
        return '修复' if temp < 40 else '升温'
    if delta < -5:
        # 退潮持续下跌到底 → 冰点；否则退潮/降温
        if prev_phase == '退潮' and temp < 45:
            return '冰点'
        return '退潮' if temp < 50 else '降温'
    return prev_phase  # 小幅震荡，延续


def main():
    target = sys.argv[1] if len(sys.argv) > 1 else None
    names = load_names()

    # 加载 10cm 主板 K 线
    close_map, high_map, open_map, low_map = {}, {}, {}, {}
    for f in os.listdir(STOCK_DIR):
        if not f.endswith('.parquet') or f.startswith('INDEX_'):
            continue
        code = f[:-8]
        if not is_10cm(code):
            continue
        df = pd.read_parquet(STOCK_DIR / f)
        close_map[code] = dict(zip(df['date'], df['close']))
        high_map[code] = dict(zip(df['date'], df['high']))
        open_map[code] = dict(zip(df['date'], df['open']))
        low_map[code] = dict(zip(df['date'], df['low']))

    all_dates = sorted(set().union(*[set(m.keys()) for m in close_map.values()]))
    if target and target in all_dates:
        all_dates = [d for d in all_dates if d <= target]

    up_by_date = {}   # date -> set of 涨停 codes
    board_map = {}    # date -> {code: 连板数}
    prev_phase = '修复'
    history = []

    for i, d in enumerate(all_dates):
        prev = all_dates[i - 1] if i > 0 else None
        up, down, zha = set(), set(), set()
        for code, cm in close_map.items():
            if d not in cm or prev is None or prev not in cm:
                continue
            c = cm[d]; pc = cm[prev]
            lu = l_up(pc); ld = l_down(pc)
            h = high_map[code].get(d, c)
            if abs(c - lu) < 0.005:
                up.add(code)
            elif h >= lu - 0.005:
                zha.add(code)
            if abs(c - ld) < 0.005:
                down.add(code)

        # 连板梯队 + 最高板
        ladder = Counter()
        for code in up:
            n = 1
            j = i - 1
            while j >= 0 and code in up_by_date.get(all_dates[j], set()):
                n += 1; j -= 1
            ladder[n] += 1
        up_by_date[d] = up

        # 涨跌家数
        up_cnt = down_cnt = 0
        for code, cm in close_map.items():
            if d not in cm or prev is None or prev not in cm:
                continue
            if cm[d] > cm[prev]: up_cnt += 1
            elif cm[d] < cm[prev]: down_cnt += 1

        # 昨日涨停今表现（溢价）
        yest_up = up_by_date.get(prev, set()) if prev else set()
        perfs = []
        for code in yest_up:
            if code in close_map and d in close_map[code] and prev in close_map[code]:
                p0 = close_map[code][prev]
                if p0 > 0:
                    perfs.append((close_map[code][d] / p0 - 1) * 100)
        yest_lu_perf = round(sum(perfs) / len(perfs), 2) if perfs else 0.0

        # 晋级率
        prev_ladder = history[-1]['ladder'] if history else {}
        advance = {}
        for n in range(1, 5):
            yc = prev_ladder.get(str(n), 0)
            tc = ladder.get(n + 1, 0)
            advance[f"{n}to{n+1}"] = round(tc / yc * 100, 1) if yc else None

        # 情绪温度 + 阶段
        lian_total = sum(v for k, v in ladder.items() if k >= 2)
        max_board = max(ladder.keys()) if ladder else 0
        temp = emotion_temp(len(up), len(down), len(zha), {str(k): v for k, v in ladder.items()}, max_board)
        prev_temp = history[-1]['temp'] if history else temp
        phase = classify(temp, prev_temp, prev_phase)
        prev_phase = phase

        # 涨停个股（全部，含首板，按高度降序）
        limit_up_stocks = []
        for code in up:
            n = 1
            j = i - 1
            while j >= 0 and code in up_by_date.get(all_dates[j], set()):
                n += 1; j -= 1
            lp = l_up(close_map[code][prev])
            open_at_limit = abs(open_map[code].get(d, 0) - lp) < 0.005
            low_at_limit = abs(low_map[code].get(d, 0) - lp) < 0.005
            limit_up_stocks.append({
                'code': code, 'name': names.get(code, ''),
                'board': n, 'close': round(close_map[code][d], 2),
                'open_at_limit': open_at_limit, 'low_at_limit': low_at_limit,
            })
        limit_up_stocks.sort(key=lambda x: (-x['board'], x['code']))
        leaders = [s for s in limit_up_stocks if s['board'] >= 2]
        board_map[d] = {s['code']: s['board'] for s in limit_up_stocks}

        seal_rate = round(len(up) / (len(up) + len(zha)) * 100, 1) if (len(up) + len(zha)) else 0.0
        history.append({
            'date': d,
            'limit_up': len(up), 'limit_down': len(down), 'break_cnt': len(zha),
            'seal_rate': seal_rate, 'max_board': max_board,
            'ladder': {str(k): ladder[k] for k in sorted(ladder)},
            'lian_total': lian_total,
            'up_cnt': up_cnt, 'down_cnt': down_cnt,
            'yest_lu_perf': yest_lu_perf, 'advance': advance,
            'temp': temp, 'phase': phase,
            'leaders': leaders,
            'limit_up_stocks': limit_up_stocks,
        })

    # 合并东方财富涨停池（封板时间/封单额/炸板/换手），仅最新一日
    zt = load_zt_pool()
    if history and zt['date'] == history[-1]['date']:
        for s in history[-1]['limit_up_stocks']:
            m = zt['map'].get(s['code'])
            if m:
                s.update({k: m[k] for k in ('fbt', 'lbt', 'fund', 'zbc', 'hs', 'hybk') if k in m})

    # 板型判定（一字/T字/回封/换手）
    if history:
        for s in history[-1]['limit_up_stocks']:
            oal = s.get('open_at_limit', False)
            lal = s.get('low_at_limit', False)
            zbc = s.get('zbc')
            if oal:
                if zbc is not None:
                    s['ptype'] = '一字板' if zbc == 0 else 'T字板'
                else:
                    s['ptype'] = '一字板' if lal else 'T字板'
            else:
                s['ptype'] = '回封板' if (zbc is not None and zbc > 0) else '换手板'

    # 行业分布聚合（涨停家数按行业降序）
    if history:
        sector_cnt = Counter()
        for s in history[-1]['limit_up_stocks']:
            hb = s.get('hybk')
            if hb:
                sector_cnt[hb] += 1
        history[-1]['sectors'] = [{'name': k, 'count': v} for k, v in sector_cnt.most_common()]

    # 连板晋级（昨日涨停股今日表现）
    if len(all_dates) >= 2:
        today = all_dates[-1]
        prev = all_dates[-2]
        yest_board = board_map.get(prev, {})
        today_up = up_by_date.get(today, set())
        today_board = board_map.get(today, {})
        groups = defaultdict(lambda: {'jin': [], 'duan': []})
        for code, nb in yest_board.items():
            if code in today_up:
                groups[nb]['jin'].append({'code': code, 'name': names.get(code, ''), 'today': today_board.get(code, nb + 1)})
            else:
                chg = None
                if code in close_map and today in close_map[code] and prev in close_map[code]:
                    p0 = close_map[code][prev]
                    if p0 > 0:
                        chg = round((close_map[code][today] / p0 - 1) * 100, 2)
                groups[nb]['duan'].append({'code': code, 'name': names.get(code, ''), 'chg': chg})
        promotion = []
        for nb in sorted(groups.keys(), reverse=True):
            g = groups[nb]
            g['jin'].sort(key=lambda x: -x['today'])
            g['duan'].sort(key=lambda x: (x['chg'] is None, -(x['chg'] or 0)))
            promotion.append({'board': nb, 'jin': g['jin'], 'duan': g['duan']})
        history[-1]['promotion'] = promotion

    # 输出 emotion_data.js
    payload = {
        'generated_at': history[-1]['date'] if history else '',
        'universe': '10cm主板(600/601/603/605/000/001/002/003)',
        'total_stocks': len(close_map),
        'latest': history[-1] if history else {},
        'history': [{k: v for k, v in h.items() if k != 'limit_up_stocks'} for h in history[-120:]],  # 最近120个交易日(剔除涨停明细瘦身)
    }
    js = 'window.EMOTION_DATA = ' + json.dumps(payload, ensure_ascii=False) + ';\n'
    (OUT_DIR / 'emotion_data.js').write_text(js, encoding='utf-8')

    # 控制台摘要（最近15天）
    print(f"10cm主板 {len(close_map)} 只 | 输出 {OUT_DIR/'emotion_data.js'}")
    print(f"{'日期':<11}{'涨停':>4}{'跌停':>4}{'炸板':>4}{'封板率':>7}{'最高板':>6}{'温度':>5}  阶段")
    for h in history[-15:]:
        print(f"{h['date']:<11}{h['limit_up']:>4}{h['limit_down']:>4}{h['break_cnt']:>4}"
              f"{h['seal_rate']:>6.1f}%{h['max_board']:>5}板{h['temp']:>5}°  {h['phase']}")


if __name__ == '__main__':
    main()
