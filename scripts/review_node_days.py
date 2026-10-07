#!/usr/bin/env python3
"""
龙头断板日(节点日)复盘：用逐日涨停池 + K线，输出「候选池 → 次日竞价/开盘预期 → 达标判定」。

逻辑（龙头战法）：
  1. 辨识度六维只负责圈出候选池（排名），不预测谁晋级——中晶排第一就是第一，没问题。
  2. 真正决定打谁的是次日竞价/开盘是否落在「它自己该有的预期」里（1775样本统计拟合，见 expectation 注释）：
     一字板 → 须继续一字(≥9.5%)，开板(2~7%)即弃；
     T字板 → 一字开(59%)或高开≥4%(56%)，微高开0-2%(21.7%)弃；
     换手板 → 一字开(52.4%)或低开反包(43.2%)，中高开2-6%(30-32%)弃。
  3. 决策铁律：只打「竞价达标 + 开盘强势」里辨识度最高的；第一若低于预期 → 降级看第二；全不达标 → 空仓。

用法: python3 scripts/review_node_days.py [起始日]   # 缺省 2026-09-04(涨停池最早)
供 compute_emotion.py 调用 build_node_review() 嵌入盘面 HTML。
"""
import sys, json
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compute_emotion import (leader_score, hy_topic, is_10cm, l_up, load_names, ROOT)

START = sys.argv[1] if len(sys.argv) > 1 else '2026-09-04'


def load_kline():
    close, open_ = {}, {}
    for f in (ROOT / 'data' / 'stocks').iterdir():
        if not f.name.endswith('.parquet') or f.name.startswith('INDEX_'):
            continue
        code = f.name[:-8]
        if not is_10cm(code):
            continue
        df = pd.read_parquet(f)
        close[code] = dict(zip(df['date'], df['close']))
        open_[code] = dict(zip(df['date'], df['open']))
    return close, open_


def ptype_of(code, d, prev, s, close, open_):
    """一字/T字/回封/换手，与 compute_emotion 同口径"""
    pc = close[code].get(prev)
    if pc is None:
        return '换手板'
    lp = l_up(pc)
    oal = abs(open_[code].get(d, 0) - lp) < 0.005
    zbc = s.get('zbc')
    if oal:
        return '一字板' if (zbc == 0) else 'T字板'
    return '回封板' if (zbc is not None and zbc > 0) else '换手板'


def fbt_sec(fbt):
    """首封时间 → 当日秒数，用于区分秒板/早封/尾盘"""
    if not fbt:
        return None
    try:
        hh, mm, ss = map(int, str(fbt).split(':'))
        return hh * 3600 + mm * 60 + ss
    except Exception:
        return None


def expectation(ptype, fbt, zbc):
    """次日开盘预期（1775样本统计拟合 board_dataset.csv：2板型 × 3板开盘 → 晋级4板+率）
    规律：开盘不是越强越好，而是「要么够强(一字/超高开)、要么够弱(低开反包)」，卡中间2~5%最危险。
    (fbt 封板时间暂未入模——历史涨停时间缺失；近3周涨停池数据齐了可再拟合)
    → (竞价预期, 开盘预期)
    """
    if zbc is not None and zbc >= 3:
        return '平开/低开正常', '看弱转强'
    if ptype == '一字板':
        # 一字开60.8% / 8-9.5%仅35.7% / 6-8%仅28.1% / 2-7%开板仅7.7~27% → 非一字即弃
        return '须继续一字(≥9.5%)', '开板(2~7%)即弃'
    if ptype == 'T字板':
        # 一字开59% / 高开4-6%56% / 8-9.5%55% / 微高开0-2%仅21.7% → 避开微高开
        return '一字开或高开≥4%', '微高开0-2%弃'
    if ptype == '换手板':
        # 一字开52.4% / 低开反包43.2% / 中间2-6%仅30-32% → 两极分化，中间最尴尬
        return '一字开或低开反包', '中高开2-6%弃'
    # 回封板(炸<3)
    return '高开1~3%', '回封不炸'


def judge_open(ptype, fbt, zbc, opct):
    """次日开盘达标判定（口径对齐 expectation 的统计拟合）：达标/超预期/低于/弱转强"""
    if opct is None:
        return '-'
    if zbc is not None and zbc >= 3:
        return '弱转强'
    if ptype == '一字板':
        return '达标' if opct >= 9.5 else '低于'        # 非一字即弃(开板2~7%仅7.7~27%)
    if ptype == 'T字板':
        if opct >= 9.5:
            return '超预期'                              # 一字开59%(黄金组合,非诱多)
        return '达标' if opct >= 4 else '低于'           # 0-2%微高开21.7%死亡区
    if ptype == '换手板':
        if opct >= 9.5:
            return '超预期'                              # 一字开52.4%
        if opct < 0:
            return '弱转强'                              # 低开反包43.2%
        return '低于'                                    # 0-9.5%中间尴尬区(30-42%)
    # 回封板(炸<3)
    return '达标' if opct >= 1 else '低于'


def build_node_review(start='2026-09-04'):
    """返回结构化复盘：[{date, dragon, groups:[{label, rows:[{...}], decision:{...}}]}]"""
    zt = json.load(open(ROOT / 'data' / 'zt_pool_history.json'))
    names = load_names()
    close, open_ = load_kline()
    dates = sorted(d for d in zt if d >= start)
    pool = {d: {s['code']: s for s in zt[d]['stocks']} for d in dates}

    review = []
    for i, d in enumerate(dates):
        prev = dates[i - 1] if i > 0 else None
        if prev is None or prev not in pool:
            continue
        pprev = pool[prev]
        pmax = max((s['lbc'] for c, s in pprev.items() if is_10cm(c) and c in close), default=0)
        if pmax < 2:
            continue
        top = [c for c, s in pprev.items() if s['lbc'] == pmax and is_10cm(c) and c in close]
        cur = pool[d]
        if any(c in cur for c in top):
            continue  # 龙头今日续板，非节点

        dragon = '、'.join(f"{names.get(c, pprev[c]['name'])}({c}){pprev[c]['lbc']}板[{pprev[c].get('hybk')}]" for c in top)
        nxt = dates[i + 1] if i + 1 < len(dates) else None

        def _group(cands, label):
            rows = []
            for c, s in cands:
                if not (is_10cm(c) and c in close):
                    continue
                ptype = ptype_of(c, d, prev, s, close, open_)
                hb = s.get('hybk')
                followers = sum(1 for t in cur if t != c and hb and hy_topic(cur[t].get('hybk')) == hy_topic(hb))
                score, _ = leader_score(s['lbc'], s.get('fbt'), s.get('fund'), s.get('hs'), ptype, followers, s.get('amount'))
                ex_open, ex_intra = expectation(ptype, s.get('fbt'), s.get('zbc'))
                opct = res = None
                if nxt and nxt in pool:
                    o = open_[c].get(nxt)
                    cl = close[c].get(d)
                    if o and cl:
                        opct = (o / cl - 1) * 100
                    ns = pool[nxt].get(c)
                    if ns and ns['lbc'] == s['lbc'] + 1:
                        res = f"晋级{ns['lbc']}板"
                    elif ns:
                        res = "涨停(连板异常)"
                    else:
                        res = "断板"
                rows.append({
                    'score': score, 'name': names.get(c, s['name']), 'code': c,
                    'board': s['lbc'], 'ptype': ptype, 'zbc': s.get('zbc'), 'hs': s.get('hs'),
                    'fund': s.get('fund'), 'exp_open': ex_open, 'exp_intra': ex_intra,
                    'opct': opct, 'verdict': judge_open(ptype, s.get('fbt'), s.get('zbc'), opct), 'result': res,
                })
            rows.sort(key=lambda x: -x['score'])
            play = [r for r in rows if r['verdict'] in ('达标', '超预期')]
            avoid = [r['name'] for r in rows if r['verdict'] == '低于']
            weak = [{'n': r['name'], 'opct': r['opct']} for r in rows if r['verdict'] == '弱转强']
            decision = {
                'play': [{'n': r['name'], 'opct': r['opct']} for r in play],
                'pick': {'n': play[0]['name'], 'c': play[0]['code'], 'opct': play[0]['opct']} if play else None,
                'avoid': avoid,
                'weak': weak,
            }
            return {'label': label, 'rows': rows, 'decision': decision}

        groups = [
            _group([(c, cur[c]) for c in cur if cur[c]['lbc'] == 3], '候选 · 2进3'),
            _group([(c, cur[c]) for c in cur if cur[c]['lbc'] == 2], '候选 · 1进2'),
        ]
        review.append({'date': d, 'dragon': dragon, 'groups': groups})
    return review


def _fmt_weak(weak):
    return ', '.join(f"{w['n']}{w['opct']:+.1f}%" for w in weak)


def main():
    review = build_node_review(START)
    for nd in review:
        print(f"\n{'=' * 92}")
        print(f"【节点日 {nd['date']}】龙头断板: {nd['dragon']}")
        for g in nd['groups']:
            print(f"  {g['label']}:")
            for r in g['rows']:
                op = f"{r['opct']:+.1f}%" if r['opct'] is not None else " - "
                print(f"    #{r['score']:>3} {r['name']}({r['code']}) {r['board']}板 {r['ptype']} "
                      f"炸{r['zbc']} 换手{r['hs']:.1f}% | 竞价预期:{r['exp_open']} 开盘:{r['exp_intra']} "
                      f"| 次日{op}[{r['verdict']}] → {r['result']}")
            dec = g['decision']
            if dec['pick']:
                play = ', '.join(f"{p['n']}{p['opct']:+.1f}%" for p in dec['play'])
                msg = f"    → 决策: 达标可打 {play} ｜ 打辨识度最高 {dec['pick']['n']}({dec['pick']['c']}){dec['pick']['opct']:+.1f}%"
                if dec['avoid']:
                    msg += f" ｜ 避 {', '.join(dec['avoid'])}"
                if dec['weak']:
                    msg += f" ｜ 看开盘 {_fmt_weak(dec['weak'])}"
                print(msg)
            else:
                parts = []
                if dec['avoid']:
                    parts.append(f"避 {', '.join(dec['avoid'])}")
                if dec['weak']:
                    parts.append(f"看开盘弱转强 {_fmt_weak(dec['weak'])}")
                print(f"    → 决策: 无达标 ｜ {' ｜ '.join(parts) if parts else '全弱转强'} → 空仓/看开盘弱转强")


if __name__ == '__main__':
    main()
