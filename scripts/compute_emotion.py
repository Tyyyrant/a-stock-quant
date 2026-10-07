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
    # 深市股票 market=0，沪市股票 market=1；但沪市指数也用 000xxx(与深市股票同号)，
    # 如 000993 闽东电力(股) vs 全指信息(指)，需按交易所前缀过滤，否则指数名会覆盖股票名
    SZ = ('000', '001', '002', '003', '300', '301')
    SH = ('600', '601', '603', '605', '688', '689')
    try:
        data = json.load(open(ROOT / 'data' / 'all_stocks.json'))
        for it in data:
            if not isinstance(it, dict) or not it.get('code') or not it.get('name'):
                continue
            code = str(it['code'])
            nm = str(it['name']).replace('\x00', '').replace(' ', '').strip()
            if not nm:
                continue
            mkt = it.get('market')
            if code.startswith(SZ):
                if mkt != 0:
                    continue
            elif code.startswith(SH):
                if mkt != 1:
                    continue
            else:
                continue  # 395xxx 等板块指数/非股票 code
            names[code] = nm
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


def classify(limit_up, limit_down, max_board, lian, node, prev_phase):
    """情绪阶段(5阶段, 对齐连板网)：结构驱动——高度/广度/晋级 + 龙头断板节点
    冰点→启动→发酵→高潮→退潮；退潮需明确启动信号(高度破4板+广度回升)才转"""
    # 冰点：亏钱效应压倒(跌停>涨停) 或 高度被压到2板以下
    if limit_down > limit_up or max_board <= 2:
        return '冰点'
    # 退潮硬信号：龙头断板日
    if node:
        return '退潮'
    # 高潮：批量涨停潮(>=80) 或 连板密集(>=15)
    if limit_up >= 80 or lian >= 15:
        return '高潮'
    # 退潮/冰点后的延续：需高度破4板且广度回升(涨停>=55)才算启动，否则仍退潮
    if prev_phase in ('退潮', '冰点'):
        if max_board >= 4 and limit_up >= 55:
            return '启动'
        return '退潮'
    # 高潮后：涨停骤降(<65)即退潮
    if prev_phase == '高潮':
        return '退潮' if limit_up < 65 else '高潮'
    # 发酵：高度打开(5板+) 或 4板+广度维持
    if max_board >= 5 or (max_board >= 4 and limit_up >= 55):
        return '发酵'
    # 启动：有3板+高度
    return '启动'


# ---- 龙头辨识度评分 ----
def seal_score(fbt):
    """首封时间：越早越强，尾盘板最弱（次日晋级核心信号）"""
    if not fbt:
        return 6
    if isinstance(fbt, str):
        try:
            hh, mm, ss = map(int, fbt.split(':'))
            sec = hh * 3600 + mm * 60 + ss
        except Exception:
            return 6
    else:
        h = fbt // 10000
        sec = h * 3600 + (fbt // 100 % 100) * 60
    if sec <= 34200: return 20   # 竞价/秒板(09:30前)
    if sec <= 36000: return 16   # 10:00 前
    if sec <= 41400: return 11   # 11:30 前
    if sec <= 46800: return 6    # 13:00 前
    if sec <= 52200: return 3    # 14:30 前
    return 0                      # 尾盘板(14:30后)最弱


def fund_score(fund, hs=None, amount=None):
    """封单归一化：封单/流通市值(=成交额/换手率)，消除市值污染(大票天然封单大)。"""
    if not fund or not hs or not amount:
        return 3
    circ = amount / (hs / 100)   # 自由流通市值 ≈ 成交额/换手率
    if circ <= 0:
        return 3
    ratio = fund / circ          # 封单占自由流通市值比例
    if ratio >= 0.08: return 14  # >=8% 一字锁死
    if ratio >= 0.04: return 12
    if ratio >= 0.02: return 10
    if ratio >= 0.01: return 8
    if ratio >= 0.005: return 6
    if ratio >= 0.002: return 4
    return 2


def hs_score(hs, ptype):
    """换手=分歧温度计: 越低=获利盘越锁仓=分歧越小=越强(该弱不弱就是强)。一字封死天然最低。"""
    if hs is None: return 4
    if ptype == '一字板':
        return 10   # 一字封死=锁仓极致=最强
    if hs <= 3: return 8    # 缩量锁仓(分歧极小)
    if hs <= 8: return 6    # 惜售(分歧小)
    if hs <= 15: return 4   # 温和换手
    if hs <= 30: return 3   # 放量分歧(炸板高发)
    return 2                 # 爆量(筹码松动)


PTYPE_BONUS = {'一字板': 6, 'T字板': 3, '回封板': 0, '换手板': -2}


# 东财3级行业 -> 题材大类：只归并"同题材"的产业链(传媒/房地产/汽车/化工…)
# 电子(元件/半导体/消费电子/光学光电/军工电子)与机械(通用/专用设备)各细分题材独立，不强行合并
HY_TOPIC = {
    # 文化传媒链：出版/电视广播/影视/游戏/广告 常同题材联动
    '出版': '传媒', '电视广播': '传媒', '影视院线': '传媒', '游戏Ⅱ': '传媒',
    '广告营销': '传媒', '数字媒体': '传媒', '文化用品': '传媒',
    # 房地产：开发/服务/经营 同产业链
    '房地产服': '房地产', '房地产开': '房地产', '房地产营': '房地产', '园区开发': '房地产',
    # 纺织服装
    '服装家纺': '纺织服装', '纺织制造': '纺织服装', '饰品': '纺织服装',
    # 汽车：整车/零部/服务/商用车 同产业链
    '汽车整车': '汽车', '汽车零部': '汽车', '汽车服务': '汽车', '商用车': '汽车', '摩托车': '汽车',
    # 家电
    '其他家电': '家电', '照明设备': '家电', '白色家电': '家电', '黑色家电': '家电', '小家电': '家电',
    # 医药
    '化学制药': '医药', '医疗器械': '医药', '中药': '医药', '生物制品': '医药', '医疗服务': '医药',
    # 化工
    '化学制品': '化工', '化学原料': '化工', '化学纤维': '化工', '塑料制品': '化工', '橡胶制品': '化工',
    # 有色
    '工业金属': '有色', '金属新材': '有色', '小金属': '有色', '贵金属': '有色', '能源金属': '有色',
    # 建材
    '水泥': '建材', '玻璃玻纤': '建材',
    # 建筑装饰
    '装修装饰': '建筑装饰', '专业工程': '建筑装饰', '工程咨询': '建筑装饰', '房屋建设': '建筑装饰',
    # 食品饮料
    '食品加工': '食品饮料', '饮料制造': '食品饮料',
    # 商贸零售
    '一般零售': '商贸零售', '专业连锁': '商贸零售',
    # 农林牧渔
    '种植业': '农林牧渔', '养殖业': '农林牧渔', '农产品加': '农林牧渔', '饲料': '农林牧渔', '渔业': '农林牧渔',
}


def hy_topic(hb):
    """3级行业 -> 题材大类(未归并的保持自身，即电子/机械各细分题材独立)"""
    return HY_TOPIC.get(hb, hb)


def leader_score(board, fbt, fund, hs, ptype, followers, amount=None):
    parts = {
        '高度': min((board - 1) * 15, 60),
        '时间': seal_score(fbt),
        '封单': fund_score(fund, hs, amount),
        '换手': hs_score(hs, ptype),
        '带动': min(followers, 8) * 2,   # 一板二板小弟跟风家数(吃独食=0)
        '板型': PTYPE_BONUS.get(ptype, 0),
    }
    return sum(parts.values()), parts


def main():
    target = sys.argv[1] if len(sys.argv) > 1 else None
    names = load_names()

    # 节点复盘/次日竞价开盘预期（在 review_node_days.py，延迟导入避免循环依赖）
    try:
        from review_node_days import expectation, build_node_review
    except Exception:
        expectation, build_node_review = None, None

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
    prev_phase = '冰点'
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

        # 龙头断板判定(提前, 供阶段分类)：昨日最高标今日全部未涨停
        node = False
        node_dragon = []
        if prev and board_map.get(prev):
            prev_max = max(board_map[prev].values())
            if prev_max >= 2:
                top_prev = [c for c, b in board_map[prev].items() if b == prev_max]
                if not any(c in up for c in top_prev):
                    node = True
                    node_dragon = [{'code': c, 'name': names.get(c, '')} for c in top_prev]

        # 情绪温度 + 阶段(结构驱动)
        lian_total = sum(v for k, v in ladder.items() if k >= 2)
        max_board = max(ladder.keys()) if ladder else 0
        temp = emotion_temp(len(up), len(down), len(zha), {str(k): v for k, v in ladder.items()}, max_board)
        prev_temp = history[-1]['temp'] if history else temp
        phase = classify(len(up), len(down), max_board, lian_total, node, prev_phase)
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

        # 节点日候选(高切低/弱转强) —— node/node_dragon 已在上方判定
        new_leader_cand = []
        new_leader_cand_1to2 = []  # 断板当天1进2(第二位打法: 昨日首板→今日2板弱转强)
        if node:
            # 新龙候选：昨日2板→今日3板(2进3)，逆势=当日跌多涨少
            for c in up:
                if board_map[prev].get(c) == 2 and board_map[d].get(c) == 3:
                    new_leader_cand.append({
                        'code': c, 'name': names.get(c, ''),
                        'nishi': down_cnt > up_cnt,
                    })
            # 第二位打法：断板当天找1进2(昨日首板→今日2板)，弱转强启动
            for c in up:
                if board_map[prev].get(c) == 1 and board_map[d].get(c) == 2:
                    s = next((x for x in limit_up_stocks if x['code'] == c), None)
                    new_leader_cand_1to2.append({
                        'code': c, 'name': names.get(c, ''),
                        'yizi': (s['open_at_limit'] if s else False),  # 今日开盘即涨停(一字/T字)
                    })

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
            'node': node, 'node_dragon': node_dragon, 'new_leader_cand': new_leader_cand,
            'new_leader_cand_1to2': new_leader_cand_1to2,
            'leaders': leaders,
            'limit_up_stocks': limit_up_stocks,
        })

    # 合并东方财富涨停池（封板时间/封单额/炸板/换手），仅最新一日
    zt = load_zt_pool()
    if history and zt['date'] == history[-1]['date']:
        for s in history[-1]['limit_up_stocks']:
            m = zt['map'].get(s['code'])
            if m:
                s.update({k: m[k] for k in ('fbt', 'lbt', 'fund', 'zbc', 'hs', 'hybk', 'amount') if k in m})

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

    # 龙头识别：连板股(≥2板)辨识度评分排序
    # 总龙=辨识度第一(一字/T字封死优先)；跟风=换手/回封且与一字龙同题材(买不进总龙→做同题材可买到=蹭)
    # 带动性=同题材(大类)的跟风票家数——不限板数(高标也能带动)、排除自己；吃独食(0)要扣分。名字维度不用(只看题材)
    if history:
        last = history[-1]
        all_up = last['limit_up_stocks']
        lian = [s for s in all_up if s['board'] >= 2]
        if lian:
            for s in lian:
                hb = s.get('hybk')
                followers = sum(1 for t in all_up
                                if t['code'] != s['code']
                                and hb and hy_topic(t.get('hybk')) == hy_topic(hb))
                s['score'], s['parts'] = leader_score(
                    s['board'], s.get('fbt'), s.get('fund'), s.get('hs'),
                    s.get('ptype', '换手板'), followers, s.get('amount'))
                s['sealed'] = s.get('fbt') is not None  # 入涨停池=封死
                s['followers'] = followers
            lian.sort(key=lambda x: -x['score'])
            sealed = [s for s in lian if s.get('ptype') in ('一字板', 'T字板')]
            for i, s in enumerate(lian):
                if i == 0:
                    s['role'] = '总龙'
                elif s.get('ptype') in ('一字板', 'T字板'):
                    s['role'] = '支线龙'  # 其他封死龙(各自题材, 非主线)
                elif s.get('ptype') in ('换手板', '回封板') and s.get('hybk') \
                        and any(hy_topic(y.get('hybk')) == hy_topic(s.get('hybk')) for y in sealed):
                    s['role'] = '跟风'   # 蹭一字龙(封死买不进→资金做同题材可买到)
                else:
                    s['role'] = '中军'
        history[-1]['leader_rank'] = lian

    # 明日推荐：节点日(龙头断板)逆势晋级的候选里，挑辨识度第一(一字/T字封死+低换手+炸板少)，附预期
    # 纪律：符合预期或超预期才做，否则不做(龙空龙——无龙空仓)
    tomorrow_reco = {'node_day': False, 'dragon': [], 'stocks': [], 'note': ''}
    if history:
        last = history[-1]
        if last.get('node'):
            cand = set()
            for x in last.get('new_leader_cand', []):
                cand.add(x['code'])
            for x in last.get('new_leader_cand_1to2', []):
                cand.add(x['code'])
            picks = [s for s in last.get('leader_rank', [])
                     if s['code'] in cand
                     and s.get('ptype') in ('一字板', 'T字板')
                     and (s.get('zbc') or 0) <= 1]
            picks.sort(key=lambda s: -s['score'])
            def _expect(s):
                nb = s['board'] + 1
                if s.get('ptype') == '一字板':
                    return f'继续一字或高开秒板封{nb}板'
                return f'弱转强：高开秒板封{nb}板'
            dragon = '、'.join(x['name'] for x in last['node_dragon'])
            tomorrow_reco = {
                'node_day': True,
                'dragon': last['node_dragon'],
                'note': f'节点日（{dragon}断板）→ 退潮期。明日只看晋级确认：',
                'stocks': [{
                    'name': s['name'], 'code': s['code'], 'board': s['board'],
                    'ptype': s.get('ptype', '换手板'), 'hybk': s.get('hybk', ''),
                    'fbt': s.get('fbt'), 'lbt': s.get('lbt'),
                    'fund': s.get('fund'), 'hs': s.get('hs'), 'zbc': s.get('zbc'),
                    'score': s['score'], 'role': s.get('role', ''),
                    'next_board': s['board'] + 1, 'expect': _expect(s),
                } for s in picks],
            }

    # 今晚决策：最高标(总龙) + 2进3低位接力 + 1进2弱转强，各附次日竞价/开盘预期
    # 铁律：节点之外干最高标，节点日切低位找2进3/1进2辨识度逆势；只打「竞价达标+开盘强势」里辨识度最高的
    night_reco = {'date': '', 'phase': '', 'node': False, 'max_board': 0, 'longtou': None, 'groups': []}
    if history:
        last = history[-1]
        phase = last.get('phase', '')
        node = last.get('node', False)
        all_up = last['limit_up_stocks']
        lian = last.get('leader_rank', [])

        # 给 1板 涨停打分(1进2候选)：首板高度=0，只看时间/封单/换手/板型/带动
        for s in all_up:
            if s['board'] == 1:
                hb = s.get('hybk')
                followers = sum(1 for t in all_up if t['code'] != s['code']
                                and hb and hy_topic(t.get('hybk')) == hy_topic(hb))
                s['score'], _ = leader_score(1, s.get('fbt'), s.get('fund'), s.get('hs'),
                                             s.get('ptype', '换手板'), followers, s.get('amount'))
                s['sealed'] = s.get('fbt') is not None

        def _exp(s):
            return expectation(s.get('ptype', '换手板'), s.get('fbt'), s.get('zbc')) if expectation else ('', '')

        def _stock(s, next_board):
            eo, ei = _exp(s)
            return {'name': s['name'], 'code': s['code'], 'board': s['board'],
                    'ptype': s.get('ptype', '换手板'), 'hybk': s.get('hybk', ''),
                    'fbt': s.get('fbt'), 'lbt': s.get('lbt'),
                    'fund': s.get('fund'), 'hs': s.get('hs'), 'zbc': s.get('zbc'),
                    'score': s.get('score'), 'role': s.get('role', ''),
                    'next_board': next_board, 'exp_open': eo, 'exp_intra': ei}

        # 1) 最高标(总龙)=辨识度第一：节点之外干、节点日不碰；按高度给持有/兑现建议
        longtou = None
        if lian:
            s = lian[0]
            b = s['board']
            eo, ei = _exp(s)
            if node:
                advice = '龙头已断板 → 不碰最高标，切低位找新龙'
            elif b >= 6:
                advice = f'{b}板高位 → 高潮兑现：不追，只卖不买'
            elif b >= 4:
                advice = f'{b}板中位 → 持有不加仓，注意兑现'
            else:
                advice = f'{b}板低位 → 干最高标：明日竞价/开盘达标则持有/加仓'
            longtou = {'name': s['name'], 'code': s['code'], 'board': b,
                       'ptype': s.get('ptype', '换手板'), 'hybk': s.get('hybk', ''),
                       'fbt': s.get('fbt'), 'lbt': s.get('lbt'),
                       'fund': s.get('fund'), 'hs': s.get('hs'), 'zbc': s.get('zbc'),
                       'score': s['score'], 'role': s.get('role', ''),
                       'exp_open': eo, 'exp_intra': ei, 'advice': advice}

        def _pick(board, hs_cap=None):
            out = [s for s in all_up if s['board'] == board and s.get('sealed')]
            if hs_cap is not None:
                out = [s for s in out if (s.get('hs') or 99) <= hs_cap]
            out.sort(key=lambda s: (
                0 if s.get('ptype') in ('一字板', 'T字板') else
                (1 if s.get('ptype') == '回封板' else 2),
                -(s.get('score') or 0)))
            return out[:5]

        risky = phase in ('退潮', '冰点')
        groups = [
            {'label': '2进3 · 低位接力', 'next_board': 3,
             'note': f'{phase}期 → ' + ('退潮/冰点接力胜率低，只做最强回封，其余空仓' if risky else '可做，只看封死型辨识度'),
             'stocks': [_stock(s, 3) for s in _pick(2, hs_cap=20)]},
            {'label': '1进2 · 弱转强', 'next_board': 2,
             'note': f'{phase}期 → ' + ('退潮期只做最强回封' if risky else '封死型(一字/T字)优先'),
             'stocks': [_stock(s, 2) for s in _pick(1)]},
        ]
        night_reco = {'date': last['date'], 'phase': phase, 'node': node,
                      'max_board': last.get('max_board', 0),
                      'longtou': longtou, 'groups': groups}

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

    # 连板天梯(3日演变)：最近3个交易日各板位个股(最新在左)，供连板晋级卡横着看3日演变/接力/断板
    # boards=当日封住涨停(N板)；duan=昨日N板今日未涨停的晋级失败票，挂在N+1板位(晋级失败的位置)，附涨跌幅
    ladder_3d = []
    win = all_dates[-3:]
    for di, d in enumerate(win):
        bm = board_map.get(d, {})
        by_board = {}
        for code, nb in bm.items():
            by_board.setdefault(nb, []).append({'code': code, 'name': names.get(code, '')})
        for nb in by_board:
            by_board[nb].sort(key=lambda x: x['code'])
        duan_by_target = {}
        if di > 0:
            prev_d = win[di - 1]
            pbm = board_map.get(prev_d, {})
            today_up = up_by_date.get(d, set())
            for code, nb in pbm.items():
                if code not in today_up:
                    chg = None
                    if code in close_map and d in close_map[code] and prev_d in close_map[code]:
                        p0 = close_map[code][prev_d]
                        if p0 > 0:
                            chg = round((close_map[code][d] / p0 - 1) * 100, 2)
                    duan_by_target.setdefault(nb + 1, []).append({
                        'code': code, 'name': names.get(code, ''), 'chg': chg, 'from': nb,
                    })
            for nb in duan_by_target:
                duan_by_target[nb].sort(key=lambda x: (x['chg'] is None, -(x['chg'] or 0)))
        ladder_3d.append({
            'date': d,
            'boards': {str(k): by_board[k] for k in sorted(by_board, reverse=True)},
            'duan': {str(k): duan_by_target[k] for k in sorted(duan_by_target, reverse=True)},
        })

    # 节点日复盘（候选池 + 次日竞价/开盘预期 + 达标判定），嵌入盘面底部
    node_review = []
    if build_node_review:
        try:
            node_review = build_node_review()
        except Exception as e:
            print(f"[node_review] 跳过: {e}")

    # 输出 emotion_data.js
    payload = {
        'generated_at': history[-1]['date'] if history else '',
        'universe': '10cm主板(600/601/603/605/000/001/002/003)',
        'total_stocks': len(close_map),
        'latest': history[-1] if history else {},
        'tomorrow_reco': tomorrow_reco,
        'night_reco': night_reco,
        'ladder_3d': ladder_3d,
        'node_review': node_review,
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

    print()
    print("【节点 · 龙头断板日 → 2进3(第一位/高切低) 与 1进2(第二位/弱转强)】")
    for h in history[-15:]:
        if h['node']:
            dragons = '、'.join(f"{x['name']}({x['code']})" for x in h['node_dragon'])
            cand = '、'.join(f"{x['name']}({x['code']}){'[逆势]' if x['nishi'] else ''}" for x in h['new_leader_cand']) or '(无2进3)'
            cand12 = '、'.join(f"{x['name']}({x['code']}){'[一字]' if x['yizi'] else ''}" for x in h['new_leader_cand_1to2']) or '(无1进2)'
            print(f"  {h['date']} 断板: {dragons}")
            print(f"        2进3(高切低): {cand}")
            print(f"        1进2(弱转强): {cand12}")

    # 今晚决策摘要
    print()
    print("【今晚决策 · 做谁】")
    lt = night_reco.get('longtou')
    if lt:
        print(f"  最高标(总龙): {lt['name']}({lt['code']}) {lt['board']}板 {lt['ptype']} | 竞价:{lt['exp_open']} 开盘:{lt['exp_intra']} | {lt['advice']}")
    else:
        print(f"  最高标(总龙): 无连板股 → 空仓")
    for g in night_reco.get('groups', []):
        names_ = '、'.join(f"{s['name']}({s['board']}板{'·'+s['ptype']})" for s in g['stocks']) or '(无)'
        print(f"  {g['label']}: {names_}")


if __name__ == '__main__':
    main()
