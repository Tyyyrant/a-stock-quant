#!/usr/bin/env python3
"""
组合分析 v2 — 战法诊断 + 多维数据 + 一次LLM深度分析

流程:
  1. analyze_stock.py (2s)  → 战法诊断 + 交易计划
  2. 并行拉数据 (5-8s)       → 融资融券/龙虎榜/北向/研报/产业链因果
  3. 一次DeepSeek调用 (15s)  → 深度分析报告
  4. PNG输出 (3s)            → 美观报告

总耗时 ~25s，v1 用 vibe-trading agent loop 要 180s。

用法:
  python3 scripts/combined_analysis.py 000977
  python3 scripts/combined_analysis.py 000977,002428
  python3 scripts/combined_analysis.py 000977 --date 2026-06-18
"""

import argparse, json, os, re, subprocess, sys, time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import numpy as np
import requests

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from data_loader import ensure_dirs

UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
EM_SESSION = requests.Session()
EM_SESSION.headers.update({"User-Agent": UA, "Referer": "https://data.eastmoney.com/"})
_em_last = [0.0]


def em_get(url, params=None, timeout=15):
    wait = 1.0 - (time.time() - _em_last[0])
    if wait > 0: time.sleep(wait + np.random.uniform(0.1, 0.5))
    try:
        r = EM_SESSION.get(url, params=params, timeout=timeout)
        _em_last[0] = time.time()
        return r
    except Exception:
        _em_last[0] = time.time()
        raise


def em_datacenter(report_name, filter_str="", page_size=10, sort_cols="", sort_types="-1"):
    params = {
        "reportName": report_name, "columns": "ALL",
        "filter": filter_str, "pageNumber": "1", "pageSize": str(page_size),
        "sortColumns": sort_cols, "sortTypes": sort_types,
        "source": "WEB", "client": "WEB",
    }
    r = em_get("https://datacenter-web.eastmoney.com/api/data/v1/get", params=params)
    d = r.json()
    return d.get("result", {}).get("data", []) if d.get("result") else []


# ══════════════════════════════════════════════════════════════
# 数据采集
# ══════════════════════════════════════════════════════════════

def _date_filter(date_str: str, field="TRADE_DATE") -> str:
    """生成东财 datacenter 的日期过滤条件"""
    return f"({field}>= '2020-01-01')({field}<= '{date_str}')"


def fetch_financing(code: str, date: str) -> dict:
    """融资融券 — 最近30日"""
    try:
        data = em_datacenter(
            "RPTA_WEB_SL_MARGINTRADINGDETAIL",
            filter_str=f'(SECURITY_CODE="{code}"){_date_filter(date)}',
            page_size=30, sort_cols="TRADE_DATE",
        )
        if data:
            latest = data[0]
            return {
                "latest_date": latest.get("TRADE_DATE", ""),
                "fin_balance_yi": round(float(latest.get("FIN_BALANCE", 0)) / 1e8, 1),
                "fin_buy_yi": round(float(latest.get("FIN_BUY_AMT", 0)) / 1e8, 1),
                "fin_sell_yi": round(float(latest.get("FIN_SELL_AMT", 0)) / 1e8, 1),
                "margin_balance_yi": round(float(latest.get("MARGIN_BALANCE", 0)) / 1e8, 1),
                "trend_30d": [round(float(d.get("FIN_BALANCE", 0)) / 1e8, 1) for d in data[:5]],
            }
    except Exception as e:
        return {"error": str(e)[:80]}
    return {"note": "无融资融券数据"}


def fetch_dragon_tiger(code: str, date: str) -> dict:
    """龙虎榜 — 最近上榜记录"""
    try:
        data = em_datacenter(
            "RPT_DAILYBILLBOARD_DETAILSNEW",
            filter_str=f'(SECURITY_CODE="{code}"){_date_filter(date)}',
            page_size=5, sort_cols="TRADE_DATE",
        )
        if data:
            records = []
            for d in data[:3]:
                records.append({
                    "date": d.get("TRADE_DATE", ""),
                    "reason": d.get("BILLBOARD_REASON", ""),
                    "net_buy_wan": round(float(d.get("BILLBOARD_NET_AMT", 0)) / 1e4, 0),
                    "buy_wan": round(float(d.get("BILLBOARD_BUY_AMT", 0)) / 1e4, 0),
                    "top5_buy": d.get("BILLBOARD_BUY_ORG", "")[:60],
                })
            return {"recent": records}
    except Exception as e:
        return {"error": str(e)[:80]}
    return {"note": "近期无龙虎榜"}


def fetch_northbound(date: str) -> dict:
    """北向资金 — 最近5日沪深股通流向（历史日期仅返回概览）"""
    try:
        headers = {"User-Agent": UA, "Referer": "https://data.hexin.cn/"}
        r = requests.get("https://data.hexin.cn/market/hsgtApi/method/dayChart/",
                         headers=headers, timeout=8)
        d = r.json()
        items = d.get("data", []) if isinstance(d, dict) else []
        # 过滤到目标日期之前
        recent = []
        for item in items:
            t = item.get("time", "")
            if t <= date:
                recent.append({
                    "time": t,
                    "hgt": round(float(item.get("hgt", 0)), 1),
                    "sgt": round(float(item.get("sgt", 0)), 1),
                })
        return {"recent_5d": recent[-5:], "source": "同花顺hsgtApi"}
    except Exception as e:
        return {"error": str(e)[:80]}
    except Exception as e:
        return {"error": str(e)[:80]}


def fetch_reports(code: str, date: str) -> dict:
    """研报 — 分析日期之前的研报摘要"""
    try:
        r = em_get("https://reportapi.eastmoney.com/report/list", params={
            "code": code, "beginTime": "2020-01-01", "endTime": date,
            "pageNo": "1", "pageSize": "5", "fields": "",
        })
        d = r.json()
        rows = d.get("data") or []
        reports = []
        for row in rows[:3]:
            reports.append({
                "date": (row.get("publishDate") or "")[:10],
                "org": row.get("orgSName", ""),
                "title": row.get("title", "")[:80],
                "rating": row.get("emRatingName", ""),
            })
        return {"recent": reports}
    except Exception as e:
        return {"error": str(e)[:80]}


def fetch_stock_news(code: str, name: str) -> dict:
    """个股新闻 — 东财 search-api-web"""
    try:
        import re as _re
        cb = "jQuery_news"
        inner = json.dumps({
            "uid": "", "keyword": code,
            "type": ["cmsArticleWebOld"], "client": "web", "clientType": "web",
            "clientVersion": "curr",
            "param": {"cmsArticleWebOld": {"searchScope": "default", "sort": "default",
                      "pageIndex": 1, "pageSize": 10, "preTag": "", "postTag": ""}},
        }, separators=(',', ':'))
        r = requests.get("https://search-api-web.eastmoney.com/search/jsonp",
                         params={"cb": cb, "param": inner},
                         headers={"User-Agent": UA, "Referer": "https://so.eastmoney.com/"},
                         timeout=10)
        text = r.text
        json_str = text[text.index("(") + 1: text.rindex(")")]
        d = json.loads(json_str)
        articles = d.get("result", {}).get("cmsArticleWebOld", []) or []
        news = []
        for a in articles[:5]:
            news.append({
                "title": _re.sub(r'<[^>]+>', '', a.get("title", "")),
                "time": a.get("date", ""),
                "source": a.get("mediaName", ""),
            })
        # 同时搜股票名称
        inner2 = json.dumps({
            "uid": "", "keyword": name,
            "type": ["cmsArticleWebOld"], "client": "web", "clientType": "web",
            "clientVersion": "curr",
            "param": {"cmsArticleWebOld": {"searchScope": "default", "sort": "default",
                      "pageIndex": 1, "pageSize": 10, "preTag": "", "postTag": ""}},
        }, separators=(',', ':'))
        r2 = requests.get("https://search-api-web.eastmoney.com/search/jsonp",
                          params={"cb": cb, "param": inner2},
                          headers={"User-Agent": UA, "Referer": "https://so.eastmoney.com/"},
                          timeout=10)
        text2 = r2.text
        json_str2 = text2[text2.index("(") + 1: text2.rindex(")")]
        d2 = json.loads(json_str2)
        articles2 = d2.get("result", {}).get("cmsArticleWebOld", []) or []
        for a in articles2[:5]:
            t = _re.sub(r'<[^>]+>', '', a.get("title", ""))
            if t not in {n["title"] for n in news}:
                news.append({"title": t, "time": a.get("date", ""),
                             "source": a.get("mediaName", "")})
        # 板块级新闻：用同花顺概念名搜索
        sector_news = {}
        board_names = []
        try:
            import re as _re2
            rr = requests.get(f"https://basic.10jqka.com.cn/new/{code}/",
                              headers={"User-Agent": UA}, timeout=6)
            rr.encoding = "gbk"
            concepts = _re2.findall(r'concept.*?>\s*([^<\s]{2,16}?)\s*<', rr.text, _re2.IGNORECASE)
            skip = {"融资融券", "深股通", "沪股通", "概念题材", "转融券标的"}
            seen = set()
            for c in concepts:
                c = c.strip()
                if c not in skip and c not in seen and len(c) >= 3:
                    seen.add(c); board_names.append(c)
        except Exception:
            pass
        for bn in board_names[:3]:
            try:
                inner3 = json.dumps({
                    "uid": "", "keyword": bn,
                    "type": ["cmsArticleWebOld"], "client": "web", "clientType": "web",
                    "clientVersion": "curr",
                    "param": {"cmsArticleWebOld": {"searchScope": "default", "sort": "default",
                              "pageIndex": 1, "pageSize": 3, "preTag": "", "postTag": ""}},
                }, separators=(',', ':'))
                r3 = requests.get("https://search-api-web.eastmoney.com/search/jsonp",
                                  params={"cb": cb, "param": inner3},
                                  headers={"User-Agent": UA, "Referer": "https://so.eastmoney.com/"},
                                  timeout=10)
                text3 = r3.text
                json_str3 = text3[text3.index("(") + 1: text3.rindex(")")]
                d3 = json.loads(json_str3)
                a3 = d3.get("result", {}).get("cmsArticleWebOld", []) or []
                sn = []
                for a in a3[:3]:
                    sn.append({"title": _re.sub(r'<[^>]+>', '', a.get("title", "")),
                               "time": a.get("date", "")})
                if sn:
                    sector_news[bn] = sn
                time.sleep(0.3)
            except Exception:
                pass
        # 产业链材料级新闻（查因果图谱中该股关联的材料名）
        material_news = {}
        try:
            from bottleneck_discovery import load_causal_graph
            graph = load_causal_graph()
            mat_name = ""
            for mn, md in graph.get("materials", {}).items():
                for p in md.get("domestic_players", []):
                    if p.get("code") == code:
                        mat_name = mn; break
                if mat_name: break
            if mat_name and mat_name not in board_names:
                inner_m = json.dumps({"uid":"","keyword":mat_name,"type":["cmsArticleWebOld"],"client":"web","clientType":"web","clientVersion":"curr","param":{"cmsArticleWebOld":{"searchScope":"default","sort":"default","pageIndex":1,"pageSize":3,"preTag":"","postTag":""}}}, separators=(',',':'))
                r_m = requests.get("https://search-api-web.eastmoney.com/search/jsonp", params={"cb":cb,"param":inner_m}, headers={"User-Agent":UA,"Referer":"https://so.eastmoney.com/"}, timeout=10)
                jm = r_m.text; jms = jm[jm.index("(")+1:jm.rindex(")")]
                am = json.loads(jms).get("result",{}).get("cmsArticleWebOld",[]) or []
                mn = []
                for a in am[:3]:
                    mn.append({"title": _re.sub(r'<[^>]+>','',a.get("title","")), "time": a.get("date","")})
                if mn: material_news[mat_name] = mn
        except Exception: pass

        return {
            "stock_news": sorted(news, key=lambda x: x.get("time", ""), reverse=True)[:6],
            "sector_news": sector_news,
            "material_news": material_news,
        }
    except Exception as e:
        return {"error": str(e)[:80]}


def fetch_sector_analysis(code: str, date: str) -> dict:
    """关联板块 — 同花顺basic页面(主) + 东财slist(备)"""
    boards = []
    source = ""

    # 主: 同花顺 basic 页面 (不封IP，零鉴权)
    try:
        import re as _re
        r = requests.get(f"https://basic.10jqka.com.cn/new/{code}/",
                         headers={"User-Agent": UA}, timeout=8)
        r.encoding = "gbk"
        concepts = _re.findall(r'concept.*?>\s*([^<\s]{2,16}?)\s*<', r.text, _re.IGNORECASE)
        # 去重去泛
        skip = {"融资融券", "深股通", "沪股通", "富时罗素", "标准普尔", "MSCI中国",
                "创业板综", "深成500", "上证380", "转融券标的", "养老金持股",
                "概念题材", "2025年报预增", "海峡两岸", "金属回收"}
        seen = set()
        for c in concepts:
            c = c.strip()
            if c not in skip and c not in seen and len(c) >= 3:
                seen.add(c)
                boards.append({"name": c, "source": "同花顺"})
        if boards:
            source = "同花顺basic"
    except Exception:
        pass

    # 备: 东财 slist
    if not boards:
        try:
            r = requests.get(
                "https://push2.eastmoney.com/api/qt/slist/get",
                params={"spt": "3", "fltt": "2", "invt": "2",
                        "fields": "f14,f3",
                        "secid": f"{'1' if code.startswith('6') else '0'}.{code}"},
                headers={"User-Agent": UA, "Referer": "https://quote.eastmoney.com/"},
                timeout=8
            )
            east_boards = r.json().get("data", {}).get("diff", []) if r.status_code == 200 else []
            skip2 = {"融资融券", "深股通", "沪股通", "标准普尔", "富时罗素"}
            for b in east_boards:
                name = b.get("f14", "")
                if name not in skip2 and not name.endswith("板块"):
                    boards.append({"name": name, "chg_pct": b.get("f3", 0), "source": "东财"})
            if boards:
                source = "东财slist"
        except Exception:
            pass

    return {"boards": boards[:8], "source": source or "无数据"}


def fetch_causal_context(code: str) -> dict:
    """产业链因果上下文"""
    try:
        from bottleneck_discovery import load_causal_graph, get_causal_context
        graph = load_causal_graph()
        materials = graph.get("materials", {})
        for mat_name, mat_data in materials.items():
            for player in mat_data.get("domestic_players", []):
                if player.get("code") == code:
                    return {
                        "material": mat_name,
                        "role": player.get("progress", ""),
                        "thesis": mat_data.get("investment_thesis", "")[:150],
                        "gap": mat_data.get("self_sufficiency", ""),
                        "chain": mat_data.get("causal_chain", ""),
                    }
    except Exception:
        pass
    return {}


# ══════════════════════════════════════════════════════════════
# 步骤1: 战法诊断
# ══════════════════════════════════════════════════════════════

def run_analyze_stock(code: str, date: str) -> dict:
    script = str(ROOT / "scripts" / "analyze_stock.py")
    result = subprocess.run(
        ["python3", script, code, "--date", date, "--json"],
        capture_output=True, text=True, timeout=60,
        env={**os.environ, "PYTHONPATH": str(ROOT / "scripts")},
    )
    if result.returncode != 0:
        return {"error": result.stderr[:200]}
    return json.loads(result.stdout)


# ══════════════════════════════════════════════════════════════
# 步骤2: LLM深度分析
# ══════════════════════════════════════════════════════════════

def _get_deepseek():
    api_key = os.environ.get("ANTHROPIC_AUTH_TOKEN", "")
    base_url = os.environ.get("ANTHROPIC_BASE_URL", "")
    model = os.environ.get("ANTHROPIC_DEFAULT_SONNET_MODEL", "deepseek-v4-pro")
    if not api_key:
        try:
            settings_path = os.path.expanduser("~/.claude/settings.json")
            if os.path.exists(settings_path):
                with open(settings_path) as f:
                    s = json.load(f)
                env = s.get("env", {})
                api_key = env.get("ANTHROPIC_AUTH_TOKEN", "")
                base_url = env.get("ANTHROPIC_BASE_URL", "")
                model = env.get("ANTHROPIC_DEFAULT_SONNET_MODEL", model)
        except Exception:
            pass
    if not api_key:
        return None
    import anthropic
    kwargs = {"api_key": api_key}
    if base_url:
        kwargs["base_url"] = base_url
    return anthropic.Anthropic(**kwargs), model


def deep_analysis(code: str, name: str, date: str, diag: dict, extra_data: dict) -> str:
    """一次 DeepSeek 调用完成深度分析"""
    client_info = _get_deepseek()
    if not client_info:
        return "(DeepSeek API 不可用)"

    client, model = client_info
    plan = diag.get("trading_plan", {})
    zone = diag.get("zone", {})
    summary = diag.get("summary", {})

    prompt = f"""你是一位A股分析师，当前分析日期为{date}。你只能使用下面提供的数据进行分析，严格禁止引用{date}之后的任何事件、价格或信息。如果某项数据为空，直接标注"无数据"，不要编造。

## 战法诊断
现价: ¥{diag.get('price', 0):.2f} | 涨跌: {diag.get('chg_pct', 0):+.1f}% | PE: {diag.get('pe', 0):.0f} | 市值: {diag.get('mcap', 0):.0f}亿
ABC三区: {zone.get('zone', '?')}区 ({zone.get('zone_score', 0)}分) — {zone.get('zone_reason', '')}
交易计划: 入场¥{plan.get('entry', 0):.2f} | 止损¥{plan.get('stop', 0):.2f} | 止盈¥{plan.get('target', 0):.2f} | 盈亏比1:{plan.get('rr_ratio', 0):.1f}
趋势类型: {plan.get('trend_type', '')}
战法裁决: {summary.get('verdict', '?')}（规则推算，不可随意推翻）
三度厚度: {json.dumps(diag.get('thickness', {}), ensure_ascii=False)}
主力意图: {json.dumps(diag.get('pullup_intent', {}), ensure_ascii=False)}
一票否决: {json.dumps([r[1] for r in summary.get('risks', []) if isinstance(r, tuple)] if summary.get('risks') else [], ensure_ascii=False)}
正面信号: {json.dumps([s[1] for s in summary.get('signals', []) if isinstance(s, tuple)] if summary.get('signals') else [], ensure_ascii=False)}

## 资金面数据
融资融券: {json.dumps(extra_data.get('financing', {}), ensure_ascii=False)}
龙虎榜: {json.dumps(extra_data.get('dragon_tiger', {}), ensure_ascii=False)}
北向资金: {json.dumps(extra_data.get('northbound', {}), ensure_ascii=False)}

## 研报与产业链
研报: {json.dumps(extra_data.get('reports', {}), ensure_ascii=False)}
产业链卡位: {json.dumps(extra_data.get('causal', {}), ensure_ascii=False)}

## 近期新闻
个股新闻: {json.dumps(extra_data.get('news', {}), ensure_ascii=False)}
板块归属: {json.dumps(extra_data.get('sector', {}), ensure_ascii=False)}

## 实时板块数据（来自搜索引擎，当日真实行情）
{extra_data.get("sector_context_json", "无实时板块数据")}

直接输出分析内容，不要任何开场白（如"好的"、"以下是分析"、"基于您提供的数据"），不要客套话，不要重复我已经给你的数据。

## 一、资金面
给出关键数字（融资余额、龙虎榜净买额、北向流向），判断主力意图。如果没有某类数据就标注"无数据"。150-200字。

## 二、情绪与催化剂
从近期新闻中提取1-2条最关键的事件或趋势，结合研报评级判断机构态度。如果新闻数据为空则根据板块和产业链逻辑推断。100-150字。

## 三、产业链卡位
公司在产业链中的位置、竞争壁垒、估值是否匹配地位。100-150字。

## 四、关联板块与市场环境
先引用"实时板块数据"中该股所属板块的当日真实涨跌幅、排名、资金流向、龙头股表现。
然后判断该股今日涨跌是板块系统性β（与板块同向且幅度接近）还是个股独立α（与板块相悖或幅度显著偏离）。
结合板块近5日走势判断是阶段性见顶还是正常回调。如果实时数据为空，则从板块新闻推断。120-180字。

## 五、操作建议
结合交易计划（入场{plan.get('entry', 0):.2f}/止损{plan.get('stop', 0):.2f}/止盈{plan.get('target', 0):.2f}/盈亏比1:{plan.get('rr_ratio', 0):.1f}），给出具体建议。
立场必须与战法裁决一致，不得偏离。
  🔥值得博弈 / ✅可以关注 → 看多
  👀再看看 → 观察
  ⛔不建议买入 → 看空
最后一行严格按此格式：
立场: 看多/观察/看空"""

    try:
        msg = client.messages.create(
            model=model,
            max_tokens=1200,
            messages=[{"role": "user", "content": prompt}],
            extra_body={"thinking": {"type": "disabled"}},
        )
        text = ""
        for block in msg.content:
            if hasattr(block, 'text') and block.text:
                text += block.text
        return text
    except Exception as e:
        return f"(DeepSeek调用失败: {e})"


# ══════════════════════════════════════════════════════════════
# 步骤4: PNG生成
# ══════════════════════════════════════════════════════════════

def generate_png(code: str, name: str, date: str, diag: dict, analysis_text: str):
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        return None

    plan = diag.get("trading_plan", {})
    zone = diag.get("zone", {})
    summary = diag.get("summary", {})
    price = diag.get("price", 0)
    chg = diag.get("chg_pct", 0)
    pe = diag.get("pe", 0)
    mcap = diag.get("mcap", 0)

    zone_map = {"A": "🟢 A区强势", "B": "🟡 B区震荡", "C": "🔴 C区风险"}
    zone_str = zone_map.get(zone.get("zone", "?"), zone.get("zone", "?"))
    verdict = summary.get("verdict", "?")

    rr = plan.get("rr_ratio", 0)
    rr_color = "#16a34a" if rr >= 2.5 else "#ca8a04" if rr >= 1.5 else "#dc2626"
    chg_cls = "up" if chg > 0 else "dn"

    # 提取立场
    standpoint = "—"
    st_color = "#6b7280"
    m = re.search(r'立场[：:]\s*(.+)', analysis_text)
    if m:
        standpoint = m.group(1).strip()[:20]
        if "看多" in standpoint: st_color = "#d4343e"
        elif "看空" in standpoint: st_color = "#1ca051"
        elif "观察" in standpoint: st_color = "#d97706"

    # 去掉开头客套话，按章节标题拆分
    clean = re.sub(r'^(好的|以下是|基于您|根据您)[^。\n]*[。\n]?\s*', '', analysis_text.strip())
    clean = re.sub(r'\n---+\n?', '\n', clean)
    # 匹配: # 一、 / ## 一、 / 一、(单独成行)
    chapters = re.split(r'\n(?:#{1,2}\s+)?(?=[一二三四五六七八九十]、)', clean)
    sections = []
    for ch in chapters:
        ch = ch.strip()
        if not ch or ch.startswith("好的") or ch.startswith("以下") or ch.startswith("基于") or ch.startswith("根据"):
            continue
        # 去掉开头的 ## 或 # 标记
        ch = re.sub(r'^#{1,3}\s*', '', ch)
        lines = ch.split('\n', 1)
        title = lines[0].strip()
        content = lines[1].strip() if len(lines) > 1 else ""
        content = re.sub(r'^(好的|以下是|基于您|根据您)[^。\n]*[。\n]?\s*', '', content)
        content = content.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        content = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', content)
        content = re.sub(r'^[-*]\s+(.+)$', r'<li>\1</li>', content, flags=re.MULTILINE)
        content = content.replace("\n\n", "</p><p>").replace("\n", "<br>")
        sections.append((title, content))

    section_html = ""
    for title, content in sections:
        # 立场行特殊渲染
        sm = re.search(r'立场[：:]\s*(.+)', content)
        if sm:
            standpoint = sm.group(1).strip()[:20]
            if "看多" in standpoint: st_color = "#d4343e"
            elif "看空" in standpoint: st_color = "#1ca051"
            elif "观察" in standpoint: st_color = "#d97706"

        section_html += f'''<div style="margin:14px 0;padding:14px 18px;background:#fafbfc;border-radius:8px;border-left:3px solid #e5e7eb">
<h3 style="margin:0 0 8px;font-size:13px;font-weight:800;color:#374151">{title}</h3>
<p style="color:#4b5563">{content}</p>
</div>'''

    html = f'''<!DOCTYPE html><html lang="zh"><head><meta charset="UTF-8"><style>
:root{{--bg:#f8f9fb;--card:#fff;--text:#1a1a2e;--muted:#8b8fa3;--border:#e8eaef;--up:#d4343e;--down:#1ca051}}
*{{margin:0;padding:0;box-sizing:border-box}}
body{{font-family:-apple-system,'PingFang SC','Microsoft YaHei',sans-serif;background:var(--bg);color:var(--text);padding:24px;max-width:780px;margin:0 auto;font-size:13px;line-height:1.75}}
.card{{background:var(--card);border-radius:10px;padding:20px 24px;margin-bottom:12px;box-shadow:0 1px 3px rgba(0,0,0,0.03)}}
h1{{font-size:20px;font-weight:800}}
h2{{font-size:14px;font-weight:800;margin:16px 0 8px;padding-bottom:6px;border-bottom:1.5px solid var(--border)}}
h3{{font-size:12px;font-weight:700;margin:10px 0 4px;color:#444}}
.sub{{font-size:10px;color:var(--muted)}}
.up{{color:var(--up);font-weight:700}}.dn{{color:var(--down);font-weight:700}}
.row{{display:flex;gap:16px;flex-wrap:wrap;margin:8px 0}}
.metric{{text-align:center;flex:1;min-width:60px}}
.metric .val{{font-size:18px;font-weight:900}}
.metric .lab{{font-size:9px;color:var(--muted)}}
.standpoint{{display:inline-block;font-size:15px;font-weight:900;padding:6px 18px;border-radius:6px;margin:8px 0;background:#f0fdf4;color:#059669}}
p{{margin:6px 0}}
li{{margin:2px 0 2px 16px}}
.footer{{text-align:center;color:var(--muted);font-size:9px;padding:16px}}
</style></head><body>

<div class="card">
  <h1>{name} <span style="font-size:10px;color:var(--muted);font-weight:400">{code}</span></h1>
  <div class="sub">{date} · 战法诊断 + 多维数据 + DeepSeek深度分析</div>
  <div class="row" style="margin-top:12px">
    <div class="metric"><div class="val">¥{price:.2f}</div><div class="lab">现价</div></div>
    <div class="metric"><div class="val {chg_cls}">{chg:+.1f}%</div><div class="lab">涨跌</div></div>
    <div class="metric"><div class="val">{pe:.0f}</div><div class="lab">PE</div></div>
    <div class="metric"><div class="val">{mcap:.0f}亿</div><div class="lab">市值</div></div>
    <div class="metric"><div class="val">{zone_str}</div><div class="lab">三区定位</div></div>
  </div>
</div>

<div class="card">
  <h2>📐 交易计划（规则推算）</h2>
  <div class="row">
    <div class="metric"><div class="val">¥{plan.get('entry', 0):.2f}</div><div class="lab">入场</div></div>
    <div class="metric"><div class="val" style="color:var(--down)">¥{plan.get('stop', 0):.2f}</div><div class="lab">止损</div></div>
    <div class="metric"><div class="val" style="color:var(--up)">¥{plan.get('target', 0):.2f}</div><div class="lab">止盈</div></div>
    <div class="metric"><div class="val" style="color:var(--down)">-{plan.get('risk_pct', 0):.1f}%</div><div class="lab">风险</div></div>
    <div class="metric"><div class="val" style="color:var(--up)">+{plan.get('reward_pct', 0):.1f}%</div><div class="lab">收益</div></div>
    <div class="metric"><div class="val" style="font-size:22px;color:{rr_color}">1:{rr:.1f}</div><div class="lab">盈亏比</div></div>
  </div>
  <div style="font-size:10px;color:var(--muted);margin-top:4px">{plan.get('trend_type', '')}</div>
  <div style="display:inline-block;font-size:15px;font-weight:900;padding:6px 18px;border-radius:6px;margin:8px 0;background:{st_color}15;color:{st_color}">{standpoint}</div>
</div>

<div class="card">
  <h2>🔍 深度分析</h2>
  {section_html}
</div>

<div class="footer">免责声明：以上分析基于公开数据和战法诊断框架，不构成投资建议。<br>analyze_stock + 多维数据 + DeepSeek · {date}</div>
</body></html>'''

    output_dir = ROOT / "output" / date
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_name = name.replace("*", "").replace("/", "").replace(" ", "")
    html_path = output_dir / f"{safe_name}_{code}.html"
    png_path = output_dir / f"{safe_name}_{code}.png"
    html_path.write_text(html, encoding="utf-8")

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch()
            page = browser.new_page(viewport={"width": 780, "height": 1600}, device_scale_factor=2)
            page.goto(f"file://{html_path}", wait_until="networkidle")
            page.screenshot(path=str(png_path), full_page=True)
            browser.close()
        return str(png_path)
    except Exception as e:
        print(f"  PNG失败: {e}")
        return None


# ══════════════════════════════════════════════════════════════
# 主流程
# ══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="组合分析 v2 — 战法+多维数据+DeepSeek")
    parser.add_argument("codes", type=str, help="股票代码(逗号分隔)")
    parser.add_argument("--date", type=str, default=None)
    parser.add_argument("--no-png", action="store_true", help="跳过PNG")
    args = parser.parse_args()

    date = args.date or datetime.now().strftime("%Y-%m-%d")
    codes = [c.strip().zfill(6) for c in args.codes.split(",") if c.strip()]

    for i, code in enumerate(codes):
        t0 = time.time()
        print(f"\n{'='*60}")
        print(f"  [{i+1}/{len(codes)}] {code} — 组合分析")
        print(f"{'='*60}")

        # Step 1: 战法诊断
        print("  [1/3] 战法诊断...", end=" ", flush=True)
        diag = run_analyze_stock(code, date)
        if "error" in diag:
            print(f"❌ {diag['error']}")
            continue
        name = diag.get("name", code)
        print(f"{name} ✓ ({time.time()-t0:.0f}s)")

        # Step 2: 并行拉数据
        print("  [2/3] 拉取多维数据...", end=" ", flush=True)
        extra = {}
        with ThreadPoolExecutor(max_workers=6) as ex:
            futures = {
                ex.submit(fetch_financing, code, date): "financing",
                ex.submit(fetch_dragon_tiger, code, date): "dragon_tiger",
                ex.submit(fetch_northbound, date): "northbound",
                ex.submit(fetch_reports, code, date): "reports",
                ex.submit(fetch_causal_context, code): "causal",
                ex.submit(fetch_sector_analysis, code, date): "sector",
                ex.submit(fetch_stock_news, code, name): "news",
            }
            for f in as_completed(futures, timeout=15):
                key = futures[f]
                try:
                    extra[key] = f.result()
                except Exception:
                    extra[key] = {}
        t_data = time.time() - t0
        print(f"✓ ({t_data:.0f}s)")

        # 加载外部板块实时数据（由搜索Agent写入）
        sector_ctx_path = "/tmp/sector_context.json"
        if os.path.exists(sector_ctx_path):
            try:
                with open(sector_ctx_path) as f:
                    extra["sector_context_json"] = json.dumps(json.load(f), ensure_ascii=False)
            except Exception:
                extra["sector_context_json"] = ""

        # Step 3: LLM深度分析
        print("  [3/3] DeepSeek深度分析...", end=" ", flush=True)
        analysis = deep_analysis(code, name, date, diag, extra)
        t_llm = time.time() - t0
        print(f"✓ ({t_llm:.0f}s 总)")

        # 输出
        print(f"\n{'─'*60}")
        print(analysis[:4000])
        print(f"{'─'*60}")

        # Step 4: PNG
        if not args.no_png:
            print(f"\n  生成PNG...", end=" ", flush=True)
            png = generate_png(code, name, date, diag, analysis)
            if png:
                print(f"✓ {png}")
                import subprocess; subprocess.run(["open", png])
            else:
                print("跳过")

        print(f"\n  总耗时: {time.time()-t0:.0f}s")

        if i < len(codes) - 1:
            time.sleep(1)


if __name__ == "__main__":
    main()
