#!/usr/bin/env python3
"""
第一位(高切低吃低位) 2进3 接力回测 —— 满仓一只、躺到断板，全程前视，不马后炮。

规则（只用当日收盘已知信息选票）：
  候选 = 当日封死涨停的 2 板票（10cm 主板），明日冲 3 板。
  过滤1 可买到：剔除一字板（开盘即涨停+炸板0 = 封死买不进），留 T字/回封/换手板。
  过滤2 不接烂板：换手率 ≤ 20%；炸板次数 ≤ 2。
  排序 = leader_score 六维辨识度。

买卖（前视）：
  打 = 次日竞价/开盘达标（换手秒板高开≥3%、早封≥1%、回封≥1%、T字3~6%不顶一字）
        → 开盘价买入；不达标/诱多/弱转强 → 空仓不动。
  躺 = 买入后只要次日继续涨停(晋级)就持有，一路躺；
  走 = 断板(不涨停)当日收盘卖出。
  满仓一只：持仓期间不开新仓，空仓才找下一个。

用法: python3 scripts/backtest_weiyi.py
"""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from compute_emotion import leader_score, hy_topic, is_10cm, load_names, ROOT
from review_node_days import load_kline, ptype_of, judge_open

START = '2026-09-04'


def main():
    zt = json.load(open(ROOT / 'data' / 'zt_pool_history.json'))
    names = load_names()
    close, open_ = load_kline()
    dates = sorted(d for d in zt if d >= START)
    pool = {d: {s['code']: s for s in zt[d]['stocks']} for d in dates}
    tdays = sorted(set().union(*(set(m) for m in close.values())))
    tdays = [d for d in tdays if d >= '2026-09-01']
    ti = {d: i for i, d in enumerate(tdays)}

    def select_cands(d):
        """当日收盘：可买到的 2 板候选，按辨识度降序"""
        idx = ti.get(d)
        prev = tdays[idx - 1] if idx and idx > 0 else None
        if prev is None:
            return []
        cur = pool.get(d, {})
        out = []
        for c, s in cur.items():
            if s['lbc'] != 2 or not is_10cm(c) or c not in close:
                continue
            ptype = ptype_of(c, d, prev, s, close, open_)
            if ptype == '一字板':
                continue
            hs = s.get('hs')
            if hs is not None and hs > 20:
                continue
            zbc = s.get('zbc')
            if zbc is not None and zbc > 2:
                continue
            hb = s.get('hybk')
            followers = sum(1 for t in cur if t != c and hb
                            and hy_topic(cur[t].get('hybk')) == hy_topic(hb))
            score, _ = leader_score(2, s.get('fbt'), s.get('fund'), hs, ptype, followers, s.get('amount'))
            out.append({'code': c, 'name': names.get(c, s['name']), 'ptype': ptype,
                        'fbt': s.get('fbt'), 'zbc': zbc, 'hs': hs, 'score': score})
        out.sort(key=lambda x: -x['score'])
        return out

    capital = 1.0
    holding = None          # {'code','name','entry_open','entry','board'}
    pending = []            # 昨日收盘选的候选，供今日竞价达标
    trades = []

    for d in dates:
        idx = ti.get(d)
        prev_d = tdays[idx - 1] if idx and idx > 0 else None
        # ① 开盘：空仓则竞价达标打板
        if holding is None and pending and prev_d:
            hit = None
            for cand in pending:
                o = open_.get(cand['code'], {}).get(d)
                cl = close.get(cand['code'], {}).get(prev_d)
                if not o or not cl:
                    continue
                opct = (o / cl - 1) * 100
                v = judge_open(cand['ptype'], cand['fbt'], cand['zbc'], opct)
                if v in ('达标', '超预期'):
                    hit = (cand, opct, v)
                    break                       # pending 已按辨识度降序，首个达标即打
            if hit:
                cand, opct, v = hit
                holding = {'code': cand['code'], 'name': cand['name'],
                           'entry_open': open_[cand['code']][d], 'entry': d,
                           'board': 2, 'opct': opct, 'verdict': v}
        # ② 收盘：持仓断板则卖出
        if holding:
            if holding['code'] not in pool.get(d, {}):
                cl = close[holding['code']].get(d)
                if cl:
                    r = (cl / holding['entry_open'] - 1) * 100
                    capital *= (1 + r / 100)
                    trades.append((holding, d, cl, r))
                    holding = None
        # ③ 收盘：空仓选明日候选，满仓则不选
        pending = select_cands(d) if holding is None else []

    # 输出
    print(f"{'买入日':<9}{'票':<9}{'买入高开':>7}{'连板轨迹':<34}{'卖出日':<9}{'收益%':>8}")
    print('-' * 80)
    for h, d, cl, r in trades:
        # 连板轨迹
        path = []
        for dd in dates:
            if dd < h['entry']:
                continue
            if h['code'] in pool[dd]:
                path.append(f"{pool[dd][h['code']]['lbc']}板")
            else:
                break
        traj = '→'.join(path)
        mark = ' ▶' if abs(r) > 0 else ''
        print(f"{h['entry'][5:]:<9}{h['name']:<9}{h['opct']:>+7.1f}%  {traj:<34}{d[5:]:<9}{r:>+8.1f}%")
    print('-' * 80)
    n = len(trades)
    wins = sum(1 for h, d, cl, r in trades if r > 0)
    print(f"共 {n} 笔(已平) ｜ 赚 {wins} 亏 {n - wins} ｜ 满仓复利终值 {capital:.4f}（{(capital-1)*100:+.1f}%）")
    if holding:
        cl = close[holding['code']].get(dates[-1])
        if cl:
            ur = (cl / holding['entry_open'] - 1) * 100
            print(f"未平仓: {holding['name']} 于 {holding['entry'][5:]} 买入(高开{holding['opct']:+.1f}%)，"
                  f"最新浮盈 {ur:+.1f}%（含入终值则 {(capital*(1+ur/100)-1)*100:+.1f}%）")


if __name__ == '__main__':
    main()
