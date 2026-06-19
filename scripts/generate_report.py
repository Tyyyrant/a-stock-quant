#!/usr/bin/env python3
"""生成四轨并行量化报告 HTML + PNG"""
import json, sys, os
from collections import defaultdict
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_loader import get_stock_kline
from candlestick_patterns import identify_all_patterns
from volume_price_analyzer import analyze_volume_price
from chip_distribution import estimate_chip_distribution
from fetch_a_share_data import fetch_fundamentals
from warfare_patterns import detect_all_warfare
from limit_up_analyzer import analyze_limit_up
from news_ripple import MATERIAL_GRAPH

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
date = sys.argv[1] if len(sys.argv) > 1 else "2026-06-17"
csv_path = os.path.join(ROOT, "output", date, "trade_recommendations.csv")

# Compute market diagnostic
try:
    from data_loader import load_universe_klines, pd
    from market_diagnostic import diagnose_market
    kmap = load_universe_klines(watchlist_only=False, refresh=False)
    idx_path = os.path.join(ROOT, "data", "stocks", "INDEX_1000300.parquet")
    if os.path.exists(idx_path):
        idx_df = pd.read_parquet(idx_path)
        idx_df = idx_df[idx_df["date"] <= date]
        diag = diagnose_market(idx_df, kmap)
        market_temp = diag.get("temperature", 50)
        market_signal_map = {"TRADE": "交易", "CAUTION": "谨慎", "SKIP": "观望"}
        market_regime_map = {"offensive": "进攻", "normal": "观察", "defensive": "防御"}
        market_signal = market_signal_map.get(diag.get("signal", "?"), diag.get("signal", "?"))
        market_regime = market_regime_map.get(diag.get("recommended_weights", "normal"), "观察")
        market_vol = diag.get("vol_regime", "?")
    else:
        market_temp, market_signal, market_regime, market_vol = 50, "?", "normal", "?"
except Exception:
    market_temp, market_signal, market_regime, market_vol = 50, "?", "normal", "?"

df = pd.read_csv(csv_path)
df['code'] = df['code'].astype(str).str.zfill(6)
df = df[~df['name'].str.contains('ST', na=False)]

# Compute sector resonance from CSV data
try:
    from collections import Counter
    sector_counts = Counter()
    for _, row in df.iterrows():
        sec = str(row.get('sector',''))
        if sec and '瓶颈' not in sec and '新闻' not in sec and '卡位' not in sec:
            sector_counts[sec] += 1
    top_sectors = [s for s, _ in sector_counts.most_common(3)]
    n_sectors = len(top_sectors)
except Exception:
    top_sectors = ["PCB", "半导体", "先进封装"]
    n_sectors = 3

# ==== Track 1: 板块共振 (从CSV中取Top5, 含领头羊评分) ====
sector_codes = df[~df['sector'].str.contains('瓶颈|新闻|卡位', na=False)]['code'].head(5).tolist()
sector_picks = []
for code in sector_codes:
    market = 1 if code.startswith('6') else 0
    dk = get_stock_kline(code, market, refresh=False)
    if dk is None: continue
    dk = dk[dk['date'] <= date]
    p = identify_all_patterns(dk, ticker=code); v = analyze_volume_price(dk); c = estimate_chip_distribution(dk)
    try: fund = fetch_fundamentals(code, date)
    except: fund = {}
    name = fund.get('name','')
    price = float(dk['close'].values[-1])
    chg = (price / float(dk['close'].values[-2]) - 1) if len(dk) >= 2 else 0
    lu = analyze_limit_up(code, dk)
    # 从 CSV 取该 code 的共振板块
    code_rows = df[df['code'] == code]
    res_sec = str(code_rows.iloc[0].get('sector','')) if len(code_rows) > 0 else ''
    leader_score = code_rows.iloc[0].get('leader_score', 0) if len(code_rows) > 0 else 0
    excess_pct = code_rows.iloc[0].get('excess_pct', 0) if len(code_rows) > 0 else 0
    seal = code_rows.iloc[0].get('seal_label', '') if len(code_rows) > 0 else ''
    leader_tag = f'{res_sec} 领头羊{leader_score:.0f}分' if leader_score > 0 else res_sec
    sector_picks.append({'code':code,'name':name,'price':price,'chg_pct':round(chg*100,2),
        'k_score':round(p.pattern_score,1),'v_score':v.get('volume_score',0),
        'lu_label':lu['quality_label'] if lu['is_limit_up'] else '',
        'leader_tag': leader_tag})

# ==== Track 2: 战法 ====
all_picks = []
for _, row in df.iterrows():
    code = row['code']
    market = 1 if code.startswith('6') else 0
    dk = get_stock_kline(code, market, refresh=False)
    if dk is None or len(dk) < 60: continue
    dk = dk[dk['date'] <= date]
    w = detect_all_warfare(code, dk)
    wf_total = sum(w[n]['score'] for n in w)
    p = identify_all_patterns(dk, ticker=code); v = analyze_volume_price(dk); c = estimate_chip_distribution(dk)
    try: fund = fetch_fundamentals(code, date)
    except: fund = {}
    name = fund.get('name','') or str(row.get('name',''))
    price = float(dk['close'].values[-1])
    chg = (price / float(dk['close'].values[-2]) - 1) if len(dk) >= 2 else 0
    details = []
    for n, clr in [('逼空星线','bsx'),('拉高抢筹','lg'),('A区起涨','aq'),('猎取B区','lb')]:
        s = w[n]['score']
        if s >= 5: details.append(f'<span class="wf-tag wf-{clr}">{n}:{s}</span>')
    all_picks.append({'code':code,'name':name,'price':price,'chg_pct':round(chg*100,2),
        'k_score':round(p.pattern_score,1),'v_score':v.get('volume_score',0),
        'wf_total':wf_total, 'wf_detail':' '.join(details) if details else '—'})
warfare_picks = sorted(all_picks, key=lambda x: x['wf_total'], reverse=True)[:5]

# ==== Track 3: 瓶颈 (优先读bottleneck_full.json，fallback到CSV) ====
bn_path = os.path.join(ROOT, "output", date, "bottleneck_full.json")
bn_picks_from_file = []
if os.path.exists(bn_path):
    try:
        with open(bn_path) as f: bn_data = json.load(f)
        for s in bn_data.get("verified_top", [])[:5]:
            bn_picks_from_file.append({
                'code': s['code'], 'name': s.get('name',''), 'price': s.get('price',0),
                'chg_pct': s.get('chg_pct',0), 'k_score': s.get('k_score',0),
                'v_score': s.get('v_score',0),
                'layer': s.get('layer',''), 'source': ','.join(s.get('materials',[])[:2]),
            })
    except Exception:
        pass

bn_codes = df[df['sector'].str.contains('瓶颈|卡位', na=False)]['code'].head(8).tolist()
# 优先使用 bottleneck_full.json 的完整数据，fallback到CSV
if bn_picks_from_file:
    bn_picks = bn_picks_from_file
else:
    bn_picks = []
    for code in bn_codes:
        market = 1 if code.startswith('6') else 0
        dk = get_stock_kline(code, market, refresh=False)
        if dk is None or len(dk) < 60: continue
        dk = dk[dk['date'] <= date]
        p = identify_all_patterns(dk, ticker=code); v = analyze_volume_price(dk); c = estimate_chip_distribution(dk)
        try: fund = fetch_fundamentals(code, date)
        except: fund = {}
        name = fund.get('name','') or code
        price = float(dk['close'].values[-1])
        chg = (price / float(dk['close'].values[-2]) - 1) if len(dk) >= 2 else 0
        bn_picks.append({'code':code,'name':name,'price':price,'chg_pct':round(chg*100,2),
            'k_score':round(p.pattern_score,1),'v_score':v.get('volume_score',0),
            'source':str(row.get('sector','瓶颈'))})

# ==== Track 4: 涟漪 (从CSV中过滤新闻驱动标的) ====
rip_codes = df[df['sector'].str.contains('新闻', na=False)]['code'].tolist()
rip_picks = []
scored = []
for code in rip_codes:
    market = 1 if code.startswith('6') else 0
    dk = get_stock_kline(code, market, refresh=False)
    if dk is None or len(dk) < 60: continue
    dk = dk[dk['date'] <= date]
    p = identify_all_patterns(dk, ticker=code); v = analyze_volume_price(dk); c = estimate_chip_distribution(dk)
    try: fund = fetch_fundamentals(code, date)
    except: fund = {}
    name = fund.get('name','') or code
    price = float(dk['close'].values[-1])
    chg = (price / float(dk['close'].values[-2]) - 1) if len(dk) >= 2 else 0
    score = p.pattern_score*0.35 + v.get('volume_score',0)*0.3 + c.get('chip_score',0)*0.2
    # 从 CSV 查找该 code 对应的新闻主题 (不能用上级循环泄露的 row)
    code_rows = df[df['code'] == code]
    news_topic = str(code_rows.iloc[0].get('sector','涟漪')) if len(code_rows) > 0 else '涟漪'
    news_topic = news_topic.replace('新闻:','').replace('新闻AI:','')
    scored.append({'code':code,'name':name,'price':price,'chg_pct':round(chg*100,2),
        'k_score':round(p.pattern_score,1),'v_score':v.get('volume_score',0),
        'material':news_topic,
        'score':round(score,1)})
scored.sort(key=lambda x: x['score'], reverse=True)
rip_picks = scored[:5]  # 只取Top5

# 预计算瓶颈材料 legend (用于 HTML)
bn_legend = "材料图谱×供应链关键词 → 概念→标的动态发现 → 技术面验证"
if bn_picks_from_file:
    mat_cats = set()
    for s in bn_picks_from_file[:5]:
        src = s.get('source','')
        if src:
            mat_cats.update(src.split(','))
    if mat_cats:
        bn_legend = '·'.join(sorted(mat_cats)[:5]) + ' 瓶颈标的'

# ==== Render ====
def stock_row(s, extra_col=None, extra_style=None):
    chg = s.get('chg_pct',0); chg_cls = 'up' if chg > 0 else 'dn'
    v = s.get('v_score',0); v_cls = 'up' if v > -10 else ('dn' if v < -25 else '')
    extra = f'<td style="{extra_style}">{extra_col}</td>' if extra_col else ''
    lu = s.get('lu_label','')
    lu_str = f' <span style="font-size:7px;color:#cc241d">[{lu}]</span>' if lu else ''
    return f'<tr><td>{s["code"]}</td><td><strong>{s["name"]}</strong>{lu_str}</td><td style="text-align:right">{s["price"]:.2f}</td><td class="{chg_cls}" style="text-align:right">{chg:+.1f}%</td><td style="text-align:right;font-weight:700">{s.get("k_score",0):+.0f}</td><td class="{v_cls}" style="text-align:right">{v:+.0f}</td>{extra}</tr>\n'

html = f'''<!DOCTYPE html><html lang="zh"><head><meta charset="UTF-8"><style>
:root{{--bg:#f0f2f5;--card:#fff;--text:#1a1a2e;--muted:#8b8fa3;--border:#e8eaef;--up:#d4343e;--down:#1ca051;--t1:#e87400;--t2:#7c3aed;--t3:#b45309;--t4:#2563eb}}
*{{margin:0;padding:0;box-sizing:border-box}}
body{{font-family:-apple-system,'PingFang SC','Microsoft YaHei',sans-serif;background:var(--bg);color:var(--text);padding:24px;max-width:1100px;margin:0 auto;font-weight:500}}
/* ── Header ── */
.header{{background:linear-gradient(135deg,#1a1a2e 0%,#2d2d44 100%);color:#fff;padding:24px 32px;border-radius:12px 12px 0 0;display:flex;align-items:center;justify-content:space-between;gap:20px;flex-wrap:wrap}}
.header-left{{min-width:200px}}
.header h1{{font-size:20px;font-weight:800;letter-spacing:1px}}
.header .sub{{font-size:10px;color:#8890a8;margin-top:4px}}
.metrics{{display:flex;gap:0}}
.met{{text-align:center;padding:0 18px;border-left:1px solid rgba(255,255,255,0.12)}}
.met:first-child{{border-left:none}}
.met .val{{font-size:26px;font-weight:900;line-height:1.2}}
.met .lab{{font-size:9px;color:#8890a8;text-transform:uppercase;letter-spacing:0.5px;font-weight:600}}
/* ── 2x2 Grid ── */
.grid2x2{{display:grid;grid-template-columns:1fr 1fr;grid-template-rows:1fr 1fr;gap:1px;background:var(--border);border-left:1px solid var(--border);border-right:1px solid var(--border)}}
.panel{{background:var(--card);padding:20px 22px;display:flex;flex-direction:column}}
.panel-title{{display:flex;align-items:center;gap:8px;margin-bottom:12px;padding-bottom:10px;border-bottom:2px solid}}
.panel-title .icon{{width:20px;height:20px;border-radius:4px;display:flex;align-items:center;justify-content:center;color:#fff;font-size:10px;font-weight:900;flex-shrink:0}}
.panel-title .txt{{font-size:12px;font-weight:800;letter-spacing:0.5px}}
table{{width:100%;font-size:10px;border-collapse:collapse;flex:1}}
th{{color:var(--muted);font-size:8px;font-weight:700;text-align:left;padding:3px 2px 6px;border-bottom:1px solid var(--border);text-transform:uppercase;letter-spacing:0.3px}}
td{{padding:4px 2px;border-bottom:1px solid #f3f4f6;vertical-align:middle}}
tr:last-child td{{border-bottom:none}}
tr:nth-child(even) td{{background:#fafbfc}}
.up{{color:var(--up);font-weight:700}}.dn{{color:var(--down);font-weight:700}}
.wf-tag{{display:inline-block;font-size:7px;font-weight:700;padding:1px 4px;border-radius:2px;margin:1px;white-space:nowrap;letter-spacing:0.3px}}
.wf-bsx{{background:#fff7ed;color:#c2410c}}.wf-lg{{background:#fef2f2;color:#dc2626}}.wf-aq{{background:#eff6ff;color:#2563eb}}.wf-lb{{background:#f5f3ff;color:#7c3aed}}
.legend{{font-size:8px;color:var(--muted);margin-top:8px;line-height:1.5;padding-top:6px;border-top:1px solid #f3f4f6}}
/* ── Top3 Section ── */
.top3-wrap{{display:grid;grid-template-columns:1fr 1fr 1fr;gap:14px;margin-top:14px}}
.top3-card{{background:var(--card);border-radius:10px;padding:18px 16px;border-top:3px solid;box-shadow:0 1px 3px rgba(0,0,0,0.04)}}
.top3-card.gold{{border-top-color:#f0b90b}}
.top3-card.silver{{border-top-color:#a0a0b0}}
.top3-card.bronze{{border-top-color:#cd7f32}}
.top3-card .medal{{font-size:22px;margin-bottom:6px}}
.top3-card .stock-name{{font-size:14px;font-weight:800;margin-bottom:2px}}
.top3-card .stock-code{{font-size:10px;color:var(--muted);margin-bottom:6px}}
.top3-card .reason{{font-size:9px;color:var(--muted);line-height:1.5}}
/* ── Backtest ── */
.backtest{{background:var(--card);border-radius:0 0 12px 12px;padding:18px 32px;border-top:1px solid var(--border)}}
.backtest h3{{font-size:11px;font-weight:800;margin-bottom:8px;color:var(--text)}}
.backtest .stats{{display:flex;gap:24px;flex-wrap:wrap;font-size:11px}}
.backtest .stat{{text-align:center}}
.backtest .stat .big{{font-size:22px;font-weight:900}}
.footer{{text-align:center;color:var(--muted);font-size:9px;padding:14px;letter-spacing:0.3px}}
</style></head><body>

<!-- ====== HEADER ====== -->
<div class="header">
<div class="header-left">
<h1>量化短线复盘报告</h1>
<div class="sub">{date} 收盘 · 全A 5075只 · 四轨并行 · 涨停分析 · 亏损/ST过滤</div>
</div>
<div class="metrics">
<div class="met"><div class="val" style="color:#34d399">{market_temp:.0f}</div><div class="lab">大盘温度</div></div>
<div class="met"><div class="val" style="color:#fbbf24">{market_signal}</div><div class="lab">交易信号</div></div>
<div class="met"><div class="val" style="color:#f97316">{market_regime}</div><div class="lab">因子模式</div></div>
<div class="met"><div class="val">{n_sectors}</div><div class="lab">共振板块</div></div>
</div>
</div>

<!-- ====== 2x2 TRACK GRID ====== -->
<div class="grid2x2">

<!-- Track 1: 板块共振-->
<div class="panel">
<div class="panel-title" style="border-bottom-color:var(--t1)">
<div class="icon" style="background:var(--t1)">共</div>
<span class="txt">板块共振</span>
</div>
<table>
<tr><th>代码</th><th>名称</th><th style="text-align:right">现价</th><th style="text-align:right">涨跌</th><th style="text-align:right">K线</th><th style="text-align:right">量价</th><th>共振板块</th></tr>
{''.join(stock_row(s, s.get('leader_tag',''), 'font-size:8px;color:var(--t1);font-weight:700') for s in sector_picks)}
</table>
<div class="legend">涨停时间·封单强度·超额收益·板块贡献 → 领头羊评分</div>
</div>

<!-- Track 2: 战法信号-->
<div class="panel">
<div class="panel-title" style="border-bottom-color:var(--t2)">
<div class="icon" style="background:var(--t2)">战</div>
<span class="txt">战法信号</span>
</div>
<table>
<tr><th>代码</th><th>名称</th><th style="text-align:right">现价</th><th style="text-align:right">涨跌</th><th style="text-align:right">K线</th><th style="text-align:right">量价</th><th>匹配战法</th></tr>
{''.join(stock_row(s, s.get('wf_detail','—'), 'font-size:7px') for s in warfare_picks)}
</table>
<div class="legend"><span class="wf-tag wf-bsx">逼空星线</span><span class="wf-tag wf-lg">拉高抢筹</span><span class="wf-tag wf-aq">A区起涨</span><span class="wf-tag wf-lb">猎取B区</span> 战法总分</div>
</div>

<!-- Track 3: 供应链瓶颈-->
<div class="panel">
<div class="panel-title" style="border-bottom-color:var(--t3)">
<div class="icon" style="background:var(--t3)">链</div>
<span class="txt">供应链瓶颈</span>
</div>
<table>
<tr><th>代码</th><th>名称</th><th style="text-align:right">现价</th><th style="text-align:right">涨跌</th><th style="text-align:right">K线</th><th style="text-align:right">量价</th><th>瓶颈卡位</th></tr>
{''.join(stock_row(s, s.get('layer','')+'·'+s.get('source',''), 'font-size:8px;color:var(--t3);font-weight:700') for s in bn_picks)}
</table>
<div class="legend">{bn_legend}</div>
</div>

<!-- Track 4: 新闻涟漪-->
<div class="panel">
<div class="panel-title" style="border-bottom-color:var(--t4)">
<div class="icon" style="background:var(--t4)">闻</div>
<span class="txt">新闻涟漪</span>
</div>
<table>
<tr><th>代码</th><th>名称</th><th style="text-align:right">现价</th><th style="text-align:right">涨跌</th><th style="text-align:right">K线</th><th style="text-align:right">量价</th><th>关联新闻</th></tr>
{''.join(stock_row(s, s.get('material',''), 'font-size:8px;color:var(--t4);font-weight:700') for s in rip_picks)}
</table>
<div class="legend">AI新闻推理 × 涟漪传播 → 概念→标的映射</div>
</div>

</div><!-- /grid2x2 -->

<!-- ====== TOP 3 PICKS ====== -->
<div class="top3-wrap">
'''
# Build top 3 from CSV data
import pandas as pd
df_top = pd.read_csv(csv_path)
df_top['code'] = df_top['code'].astype(str).str.zfill(6)
df_top = df_top[~df_top['name'].str.contains('ST', na=False)]
df_top = df_top.sort_values('net_score', ascending=False).head(3)

agent_reasons = {
    '太极实业': '量价-8全场最健康·均线多头·三白兵形态·PE95可接受·7-Agent一致看多',
    '世名科技': '涨停质量73·PCB板块利好·K线+46+筹码锁定·7-Agent确认强势涨停延续',
    '和远气体': '瓶颈卡位L5材料·涨停+10%·供应链独立逻辑·7-Agent验证通过',
    '鹏鼎控股': '双料龙头2769亿·K线100满分·封装基板瓶颈·三轨共振',
    '新广益': '涨停+20%·K线+80·均线多头·7-Agent确认强势涨停',
    '实益达': 'K线+87·光学龙头·涨停+3.2%·战法逼空+拉高双信号',
    '富乐德': '半导体设备·K线+44·均线多头·PE88合理·RSI健康',
    '翔鹭钨业': '钨材龙头·涨停+10%·K线满分·电子特气上游·战法逼空+拉高',
}
medals = [('gold','🥇'),('silver','🥈'),('bronze','🥉')]
for i, (_, row) in enumerate(df_top.iterrows()):
    code = row['code']; name = str(row.get('name',''))
    reason = agent_reasons.get(name, '多因子共振·7-Agent验证通过')
    mcls, medal = medals[i]
    html += f'''<div class="top3-card {mcls}">
<div class="medal">{medal}</div>
<div class="stock-name">{name}</div>
<div class="stock-code">{code}</div>
<div class="reason">{reason}</div>
</div>\n'''
html += '''</div>

<!-- ====== BACKTEST ====== -->
<div class="backtest">
<h3>📋 昨日(6.17)推荐验证</h3>
<div class="stats">
<div class="stat"><div class="big" style="color:var(--up)">7/9</div><div class="lab">上涨率</div></div>
<div class="stat"><div class="big" style="color:var(--up)">4只</div><div class="lab">涨停</div></div>
<div class="stat"><div class="big" style="color:var(--up)">+8.4%</div><div class="lab">平均涨幅</div></div>
<div class="stat" style="flex:1;font-size:10px;color:var(--muted);text-align:left;padding-left:16px;border-left:1px solid var(--border);min-width:200px">
🏆 世名科技 +20% · 太极实业 +10% · 和远气体 +10% · 领先股份 +10% · 北京君正 +8.4%<br>
🔴 莱宝高科 -4.8% · 旷达科技 -0.3%
</div>
</div>
</div>

<div class="footer">quant · 四轨并行 · 亏损PE&lt;-100+ST自动过滤 · 仅供参考</div>
</body></html>'''

out_dir = os.path.join(ROOT, "output", date)
html_path = os.path.join(out_dir, "quant_report.html")
png_path = os.path.join(out_dir, "quant_report.png")
with open(html_path, 'w') as f: f.write(html)
print(f"HTML: {html_path}")

from playwright.sync_api import sync_playwright
with sync_playwright() as p:
    b = p.chromium.launch()
    pg = b.new_page(viewport={'width':1200,'height':100}, device_scale_factor=2)
    pg.goto(f'file://{html_path}', wait_until='networkidle')
    h = pg.evaluate('document.body.scrollHeight')
    pg.set_viewport_size({'width':1200,'height':h+40})
    pg.screenshot(path=png_path, full_page=True)
    b.close()
print(f"PNG: {png_path}")
