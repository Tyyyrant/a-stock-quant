#!/usr/bin/env python3
"""
Phase 1: 热门主线自动发现引擎

数据源（按优先级）:
  1. 东财概念板块涨幅排名 (eastmoney push2)
  2. 同花顺题材归因 (ths hot reason)
  3. 北向资金行业流向 (hexin hsgtApi)
  4. 行业成交额占比变化 (eastmoney push2)
  5. 龙虎榜机构活跃板块 (eastmoney datacenter)

输出:
  1. 当前最强主线列表（含热度得分、持续性）
  2. 每条主线的成分股列表

用法:
  python3 scripts/theme_discovery.py                     # 发现今日主线
  python3 scripts/theme_discovery.py --top-n 5          # 只输出 Top 5
  python3 scripts/theme_discovery.py --output json      # JSON 输出
"""

import argparse
import json
import os
import sys
import time
import random
from datetime import datetime, timedelta
from pathlib import Path
from collections import Counter, defaultdict
from typing import Optional

import requests

ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / "output"
sys.path.insert(0, str(ROOT / "scripts"))

# ---- 复用 data_loader 的基础设施 ----
from data_loader import (
    em_get, eastmoney_datacenter, SESSION, UA,
    EM_SESSION, EM_MIN_INTERVAL, _em_last_call,
    ensure_dirs, get_north_flow_history, tencent_quote,
)

# ============================================================
# 1. 东财概念板块排名
# ============================================================

_CONCEPT_BOARD_CACHE = None  # 模块级缓存，避免重复失败调用

def fetch_concept_board_ranking() -> list[dict]:
    """
    东财概念板块近5日涨幅排名。
    URL: push2.eastmoney.com — 行业+概念板块行情
    结果模块级缓存（失败也缓存5分钟，避免重复超时）
    """
    global _CONCEPT_BOARD_CACHE
    if _CONCEPT_BOARD_CACHE is not None:
        return _CONCEPT_BOARD_CACHE

    # 概念板块: m:90+t:3 (f:3=概念)
    url = "https://push2.eastmoney.com/api/qt/clist/get"
    params = {
        "pn": "1", "pz": "100",
        "po": "1", "np": "1",
        "fltt": "2", "invt": "2",
        "fs": "m:90+t:3",  # 概念板块
        "fields": "f2,f3,f4,f12,f14,f104,f105,f128,f136,f140,f141,f207",
    }
    headers = {"User-Agent": UA, "Referer": "https://quote.eastmoney.com/"}
    try:
        r = em_get(url, params=params, headers=headers, timeout=15)
        data = r.json()
        items = (data.get("data") or {}).get("diff") or []
        results = []
        for it in items:
            results.append({
                "code": it.get("f12", ""),
                "name": it.get("f14", ""),
                "price": it.get("f2"),
                "change_pct_1d": it.get("f3", 0),
                "change_pct_5d": it.get("f104", 0) or it.get("f3", 0),
                "change_pct_20d": it.get("f105", 0),
                "up_count": it.get("f136", 0),
                "down_count": it.get("f140", 0),
                "lead_stock": it.get("f128", ""),
                "turnover": it.get("f141", 0),
                "amplitude": it.get("f207", 0),
            })
        _CONCEPT_BOARD_CACHE = results
        return results
    except Exception as e:
        print(f"  [WARN] 概念板块排名: {e}")
        _CONCEPT_BOARD_CACHE = []
        return []


_INDUSTRY_BOARD_CACHE = None

def fetch_industry_board_ranking() -> list[dict]:
    """东财行业板块涨幅排名"""
    url = "https://push2.eastmoney.com/api/qt/clist/get"
    params = {
        "pn": "1", "pz": "100",
        "po": "1", "np": "1",
        "fltt": "2", "invt": "2",
        "fs": "m:90+t:2",  # 行业板块
        "fields": "f2,f3,f4,f12,f14,f104,f105,f128,f136,f140,f141,f207",
    }
    headers = {"User-Agent": UA, "Referer": "https://quote.eastmoney.com/"}
    try:
        r = em_get(url, params=params, headers=headers, timeout=15)
        data = r.json()
        items = (data.get("data") or {}).get("diff") or []
        results = []
        for it in items:
            results.append({
                "code": it.get("f12", ""),
                "name": it.get("f14", ""),
                "change_pct_1d": it.get("f3", 0),
                "change_pct_5d": it.get("f104", 0) or it.get("f3", 0),
                "change_pct_20d": it.get("f105", 0),
                "up_count": it.get("f136", 0),
                "down_count": it.get("f140", 0),
                "lead_stock": it.get("f128", ""),
                "turnover": it.get("f141", 0),
            })
        return results
    except Exception as e:
        print(f"  [WARN] 行业板块排名: {e}")
        return []


# ============================================================
# 2. 同花顺题材归因 — 从当日最强个股反推题材
# ============================================================

def fetch_ths_hot_stocks() -> list[dict]:
    """
    同花顺涨停原因/热点个股。
    URL: zx.10jqka.com.cn — getHarden
    返回每只强势股及其涨停/大涨原因标签。
    """
    today = datetime.now().strftime("%Y-%m-%d")
    url = f"http://zx.10jqka.com.cn/event/api/getharden/date/{today}/orderby/date/orderway/desc/charset/GBK/"
    headers = {"User-Agent": UA, "Referer": "https://www.10jqka.com.cn/"}
    try:
        r = requests.get(url, headers=headers, timeout=15)
        r.encoding = "gbk"
        data = r.json()
        results = []
        for item in data if isinstance(data, list) else data.get("data", []):
            reason_str = item.get("reason", "") or item.get("reasonTitle", "")
            results.append({
                "code": item.get("code", ""),
                "name": item.get("name", ""),
                "change_pct": item.get("changePct", item.get("change", 0)),
                "reason": reason_str,
                "reasons": [t.strip() for t in reason_str.replace("+", ",").replace("、", ",").split(",") if t.strip()],
                "turnover_pct": item.get("turnoverRate", item.get("turn", 0)),
                "market": item.get("market", ""),
            })
        return results
    except Exception as e:
        print(f"  [WARN] 同花顺热点: {e}")
        return []


def aggregate_ths_topics(hot_stocks: list[dict]) -> dict[str, dict]:
    """
    从同花顺强势股中聚合题材标签。
    返回: {topic_name: {stock_count, total_change, stocks: [...]}}
    """
    topics = defaultdict(lambda: {"stock_count": 0, "total_change": 0.0, "stocks": []})

    for stock in hot_stocks:
        try:
            chg = float(stock.get("change_pct", 0))
        except (ValueError, TypeError):
            chg = 0.0

        for tag in stock.get("reasons", []):
            topics[tag]["stock_count"] += 1
            topics[tag]["total_change"] += chg
            topics[tag]["stocks"].append({
                "code": stock["code"],
                "name": stock["name"],
                "change_pct": chg,
            })

    # 计算平均涨幅
    for tag, info in topics.items():
        info["avg_change"] = round(info["total_change"] / info["stock_count"], 2)

    return dict(topics)


# ============================================================
# 3. 北向资金行业流向感知
# ============================================================

def get_northbound_sentiment() -> dict:
    """
    获取北向资金近期流向情绪。
    Returns: {direction: "inflow"|"outflow"|"neutral",
              net_total_5d, net_total_20d, trend}
    """
    df = get_north_flow_history(30)
    if df.empty:
        return {"direction": "neutral", "net_total_5d": 0, "net_total_20d": 0, "trend": "flat"}

    if "net_total" not in df.columns:
        df["net_total"] = df["hgt_yi"].fillna(0) + df["sgt_yi"].fillna(0)

    net_5d = df["net_total"].tail(5).sum()
    net_20d = df["net_total"].tail(20).sum()

    # 趋势判断
    if net_5d > 30 and net_20d > 50:
        direction = "inflow"
    elif net_5d < -30 and net_20d < -50:
        direction = "outflow"
    else:
        direction = "neutral"

    recent = df["net_total"].tail(10)
    if len(recent) >= 5:
        trend = "accelerating_in" if (recent.iloc[-1] > recent.iloc[-5] and net_5d > 0) else \
                "accelerating_out" if (recent.iloc[-1] < recent.iloc[-5] and net_5d < 0) else \
                "stable"
    else:
        trend = "unknown"

    return {
        "direction": direction,
        "net_total_5d": round(float(net_5d), 1),
        "net_total_20d": round(float(net_20d), 1),
        "trend": trend,
    }


# ============================================================
# 4. 龙虎榜活跃板块
# ============================================================

def fetch_dragon_tiger_daily(date: str = None) -> list[dict]:
    """获取当日龙虎榜全市场数据"""
    if date is None:
        date = datetime.now().strftime("%Y-%m-%d")

    data = eastmoney_datacenter(
        "RPT_DAILYBILLBOARD_DETAILSNEW",
        filter_str=f'(TRADE_DATE=\'{date}\')',
        page_size=200,
        sort_columns="BILLBOARD_NET_AMT", sort_types="-1",
    )
    results = []
    for row in data:
        results.append({
            "code": str(row.get("SECURITY_CODE", "")),
            "name": row.get("SECURITY_NAME_ABBR", ""),
            "reason": row.get("BILLBOARD_REASON", ""),
            "close": row.get("CLOSE_PRICE", 0),
            "change_pct": row.get("CHANGE_RATE", 0),
            "net_buy_wan": (row.get("BILLBOARD_NET_AMT") or 0) / 10000,
            "turnover_pct": row.get("TURNOVERRATE", 0),
        })
    return results


def aggregate_lhb_by_reason(daily_data: list[dict]) -> dict[str, dict]:
    """
    按龙虎榜上榜原因聚合，识别机构集中出现的板块。
    """
    reasons = defaultdict(lambda: {"count": 0, "total_net_buy": 0.0, "stocks": []})
    for item in daily_data:
        reason = item.get("reason", "其他")
        reasons[reason]["count"] += 1
        reasons[reason]["total_net_buy"] += item.get("net_buy_wan", 0)
        reasons[reason]["stocks"].append({
            "code": item["code"],
            "name": item["name"],
            "net_buy_wan": item.get("net_buy_wan", 0),
        })

    for r, info in reasons.items():
        info["avg_net_buy"] = round(info["total_net_buy"] / info["count"], 2)

    return dict(reasons)


# ============================================================
# 5. 主线发现算法 — 多源信号融合
# ============================================================

# 同义词映射：将不同来源的近似概念合并
CONCEPT_SYNONYMS = {
    "算力": ["算力租赁", "智算中心", "东数西算", "算力"],
    "AI电力": ["液冷", "数据中心电源", "HVDC", "算力电力", "AI电力", "电力电源"],
    "低空经济": ["低空经济", "飞行汽车", "eVTOL", "无人机"],
    "机器人": ["人形机器人", "机器人", "减速器", "执行器", "灵巧手"],
    "HBM": ["HBM", "先进封装", "高带宽内存", "存储芯片"],
    "半导体设备": ["半导体设备", "光刻机", "刻蚀", "薄膜沉积"],
    "自动驾驶": ["自动驾驶", "智能驾驶", "无人驾驶", "L4"],
    "商业航天": ["商业航天", "卫星互联网", "火箭回收"],
    "固态电池": ["固态电池", "全固态", "硫化物电解质"],
    "AI Agent": ["AI Agent", "智能体", "企业AI"],
    "核聚变": ["核聚变", "托卡马克", "高温超导"],
    "量子计算": ["量子计算", "量子芯片", "量子比特"],
    "氢能": ["氢能", "电解槽", "氢燃料电池"],
    # 新增映射 — 基于同花顺常用标签
    "AI算力": ["AI算力", "算力", "算力租赁", "智算中心", "Token工厂"],
    "光通信": ["光模块", "光通信", "CPO", "硅光", "光芯片"],
    "PCB": ["PCB", "印制电路板", "HDI", "高频高速"],
    "消费电子": ["消费电子", "果链", "MR", "VisionPro", "AI眼镜"],
    "新能源车": ["新能源车", "汽车零部件", "一体化压铸", "热管理"],
    "光伏": ["光伏", "钙钛矿", "TOPCon", "HJT", "BC电池"],
    "风电": ["风电", "海上风电", "海缆", "塔筒"],
    "储能": ["储能", "大储", "工商储", "户储", "构网型储能"],
    "军工": ["军工", "军工信息化", "军贸", "导弹", "无人装备"],
    "医药": ["创新药", "CRO", "CDMO", "减肥药", "GLP-1", "ADC"],
    "数据要素": ["数据要素", "数据确权", "数据交易", "可信数据空间"],
    "鸿蒙": ["鸿蒙", "鸿蒙原生", "开源鸿蒙", "华为生态"],
    "华为链": ["华为", "华为链", "昇腾", "鲲鹏", "华为汽车"],
    "新消费": ["谷子经济", "盲盒", "出海零售", "跨境电商"],
    "央企改革": ["央企", "中字头", "国企改革", "国企"],
}

# 非投资主线噪声关键词（财报日期/定增类型/ST等）
NOISE_TOPIC_KEYWORDS = [
    "一季报", "中报", "三季报", "年报", "业绩预", "业绩预告",
    "扭亏", "预增", "预减", "增长", "下降",
    "定增", "增发", "配股", "回购", "减持", "举牌",
    "连续涨停", "连续跌停", "首板", "二板", "三板",
    "最新", "今日", "昨日", "早盘", "尾盘",
    "次新", "破发", "解禁",
]


def _normalize_topic_name(name: str) -> str:
    """将不同来源的近似概念映射到统一名称"""
    name_lower = name.lower().replace(" ", "")
    for canonical, synonyms in CONCEPT_SYNONYMS.items():
        for syn in synonyms:
            if syn.lower().replace(" ", "") in name_lower or name_lower in syn.lower().replace(" ", ""):
                return canonical
    return name


def _new_candidate() -> dict:
    """创建空的候选主线数据结构"""
    return {
        "raw_names": [],
        "concept_rank": 999,
        "concept_change_5d": 0.0,
        "concept_change_20d": 0.0,
        "up_count": 0,
        "down_count": 0,
        "ths_stock_count": 0,
        "ths_avg_change": 0.0,
        "lhb_count": 0,
        "lhb_net_buy": 0.0,
        "constituent_stocks": [],
    }


def discover_themes(top_n: int = 10) -> dict:
    """
    主函数：多源信号融合，发现当前最强投资主线。

    Returns:
        {
            "date": "YYYY-MM-DD",
            "market_environment": {...},
            "themes": [
                {
                    "name": "主线名称",
                    "heat_score": 0-100,
                    "persistence": "rising"|"new"|"fading",
                    "signals": {
                        "concept_rank": int,
                        "concept_change_5d": float,
                        "ths_stock_count": int,
                        "northbound_aligned": bool,
                        "lhb_active": bool,
                    },
                    "constituent_stocks": [{"code": "", "name": ""}, ...],
                },
                ...
            ]
        }
    """
    today = datetime.now().strftime("%Y-%m-%d")
    print(f"\n{'='*60}")
    print(f"  热门主线发现引擎 — {today}")
    print(f"{'='*60}\n")

    # ---- 信号1: 概念板块涨幅 ----
    print("[1/5] 获取东财概念板块排名...")
    concept_boards = fetch_concept_board_ranking()
    print(f"  获取 {len(concept_boards)} 个概念板块")

    # ---- 信号2: 同花顺强势股反推题材 ----
    print("[2/5] 获取同花顺强势股题材...")
    hot_stocks = fetch_ths_hot_stocks()
    ths_topics = aggregate_ths_topics(hot_stocks)
    print(f"  获取 {len(hot_stocks)} 只强势股, {len(ths_topics)} 个题材标签")

    # ---- 信号3: 北向资金情绪 ----
    print("[3/5] 获取北向资金情绪...")
    northbound = get_northbound_sentiment()
    print(f"  北向: {northbound['direction']} (5日: {northbound['net_total_5d']}亿, 20日: {northbound['net_total_20d']}亿)")

    # ---- 信号4: 行业板块涨幅 ----
    print("[4/5] 获取行业板块排名...")
    industry_boards = fetch_industry_board_ranking()
    print(f"  获取 {len(industry_boards)} 个行业板块")

    # ---- 信号5: 龙虎榜活跃板块 ----
    print("[5/5] 获取龙虎榜活跃板块...")
    lhb_data = fetch_dragon_tiger_daily()
    lhb_by_reason = aggregate_lhb_by_reason(lhb_data)
    print(f"  龙虎榜 {len(lhb_data)} 条记录, {len(lhb_by_reason)} 个上榜原因")

    # ---- 融合打分 ----
    print("\n[融合] 多源信号打分...")

    theme_candidates = {}  # normalized_name -> signals
    has_concept_boards = len(concept_boards) > 0

    if has_concept_boards:
        # 主路径: 从概念板块构建候选主线
        for board in concept_boards[:80]:
            name = board["name"]
            norm = _normalize_topic_name(name)
            if norm not in theme_candidates:
                theme_candidates[norm] = _new_candidate()
            c = theme_candidates[norm]
            c["raw_names"].append(name)
            c["concept_rank"] = min(c["concept_rank"], concept_boards.index(board) + 1)
            c["concept_change_5d"] = max(c["concept_change_5d"],
                                         float(board.get("change_pct_5d", 0) or 0))
            c["concept_change_20d"] = max(c["concept_change_20d"],
                                          float(board.get("change_pct_20d", 0) or 0))
            c["up_count"] = max(c["up_count"], int(board.get("up_count", 0) or 0))
            c["down_count"] = max(c["down_count"], int(board.get("down_count", 0) or 0))
    else:
        # 降级路径: 东财 push2 不可用，从同花顺题材直接构建候选主线
        print("  ⚠️ 东财概念板块不可用，使用同花顺题材作为主信号")
        for topic_name, info in ths_topics.items():
            if info["stock_count"] < 2:
                continue
            norm = _normalize_topic_name(topic_name)
            if norm not in theme_candidates:
                theme_candidates[norm] = _new_candidate()
            c = theme_candidates[norm]
            c["raw_names"].append(topic_name)
            c["ths_stock_count"] = max(c["ths_stock_count"], info["stock_count"])
            c["ths_avg_change"] = max(c["ths_avg_change"], info["avg_change"])
            c["concept_change_5d"] = max(c["concept_change_5d"], info["avg_change"])
            c["concept_rank"] = min(c["concept_rank"], 30)  # 中等排名兜底

    # Step B: 匹配同花顺题材(补充/强化信号)
    for topic_name, info in ths_topics.items():
        norm = _normalize_topic_name(topic_name)
        if norm in theme_candidates:
            c = theme_candidates[norm]
            c["ths_stock_count"] = max(c["ths_stock_count"], info["stock_count"])
            c["ths_avg_change"] = max(c["ths_avg_change"], info["avg_change"])

    # Step C: 匹配龙虎榜活跃板块
    for reason, info in lhb_by_reason.items():
        norm = _normalize_topic_name(reason)
        if norm in theme_candidates:
            theme_candidates[norm]["lhb_count"] = info["count"]
            theme_candidates[norm]["lhb_net_buy"] = info["total_net_buy"]

    # Step D: 计算热度得分 (0-100)
    themes = []
    for norm_name, c in theme_candidates.items():
        if has_concept_boards:
            concept_score = min(c["concept_change_5d"] / 10 * 30, 30) if c["concept_change_5d"] > 0 else \
                            max(c["concept_change_5d"] / 10 * 30, -15)
            ths_score = min(c["ths_stock_count"] / 3 * 25, 25) if c["ths_stock_count"] > 0 else 0
            rank_score = max(20 - c["concept_rank"] * 0.25, 0) if c["concept_rank"] < 80 else 0
            total = c["up_count"] + c["down_count"]
            breadth_score = (c["up_count"] / max(total, 1)) * 15 if total > 0 else 7.5
            lhb_score = min(c["lhb_count"] / 2 * 10, 10) if c["lhb_count"] > 0 else 0
            heat_raw = concept_score + ths_score + rank_score + breadth_score + lhb_score
        else:
            # 降级评分: 同花顺热度60% + 龙虎榜30% + 涨跌幅10%
            ths_score = min(c["ths_stock_count"] / 2 * 60, 60) if c["ths_stock_count"] > 0 else 0
            lhb_score = min(c["lhb_count"] / 2 * 30, 30) if c["lhb_count"] > 0 else 0
            chg_score = min(max(c["ths_avg_change"], 0) / 10 * 10, 10)
            heat_raw = ths_score + lhb_score + chg_score

        north_bonus = 0
        north_aligned = False
        if northbound["direction"] == "inflow" and c["concept_change_5d"] > 3:
            north_bonus = 8; north_aligned = True
        elif northbound["direction"] == "outflow" and c["concept_change_5d"] < -3:
            north_bonus = 5; north_aligned = True

        heat_score = min(round(heat_raw + north_bonus), 100)

        persistence = "rising" if c["ths_stock_count"] >= 5 else \
                      "new" if c["ths_stock_count"] >= 2 else "fading"

        signal_count = sum([
            1 if c["concept_rank"] < 999 else 0,
            1 if c["ths_stock_count"] > 0 else 0,
            1 if c["lhb_count"] > 0 else 0,
        ])
        if signal_count < 1 or heat_score < 10:
            continue

        # 过滤噪声关键词（财报/定增/日期类）
        is_noise = False
        for kw in NOISE_TOPIC_KEYWORDS:
            if kw in norm_name:
                is_noise = True
                break
        if is_noise:
            continue

        themes.append({
            "name": norm_name,
            "display_names": c["raw_names"][:5],
            "heat_score": heat_score,
            "persistence": persistence,
            "signals": {
                "concept_rank": c["concept_rank"],
                "concept_change_5d": round(c["concept_change_5d"], 2),
                "concept_change_20d": round(c["concept_change_20d"], 2),
                "ths_stock_count": c["ths_stock_count"],
                "ths_avg_change": round(c["ths_avg_change"], 2),
                "northbound_aligned": north_aligned,
                "lhb_active": c["lhb_count"] > 0,
                "lhb_count": c["lhb_count"],
                "up_down_ratio": f"{c['up_count']}/{c['down_count']}",
            },
        })

    # 排序: 热度降序
    themes.sort(key=lambda t: t["heat_score"], reverse=True)
    themes = themes[:top_n]

    # ---- 获取每条主线的成分股 ----
    print("\n[成分股] 获取主线成分股...")
    for theme in themes:
        stocks = get_theme_constituents(theme["name"])
        theme["constituent_stocks"] = stocks
        theme["constituent_count"] = len(stocks)

    result = {
        "date": today,
        "market_environment": {
            "northbound": northbound,
            "total_concept_boards": len(concept_boards),
            "total_hot_stocks": len(hot_stocks),
            "total_lhb_records": len(lhb_data),
        },
        "themes": themes,
    }

    return result


# ============================================================
# 6. 主线成分股获取
# ============================================================

def get_theme_constituents(theme_name: str) -> list[dict]:
    """
    根据主线名称获取成分股。
    策略：
      1. 从东财概念板块匹配 → 获取板块内成分股
      2. 从同花顺题材匹配 → 获取相关个股
      3. 去重合并
    """
    # 先找到对应的概念板块 code
    concept_match = _find_concept_board(theme_name)
    stocks = {}

    if concept_match:
        board_code = concept_match["code"]
        board_name = concept_match["name"]
        constituents = _fetch_board_constituents(board_code)
        for s in constituents:
            stocks[s["code"]] = {
                "code": s["code"],
                "name": s.get("name", ""),
                "source": f"概念板块:{board_name}",
            }

    # 补充同花顺强势股中同题材的
    try:
        hot_stocks = fetch_ths_hot_stocks()
        for stock in hot_stocks:
            reasons = stock.get("reasons", [])
            for r in reasons:
                if _normalize_topic_name(r) == theme_name:
                    if stock["code"] not in stocks:
                        stocks[stock["code"]] = {
                            "code": stock["code"],
                            "name": stock.get("name", ""),
                            "source": "同花顺题材",
                        }
                    break
    except Exception:
        pass

    return list(stocks.values())


def _find_concept_board(theme_name: str) -> Optional[dict]:
    """根据主线名称查找最匹配的东财概念板块"""
    boards = fetch_concept_board_ranking()
    theme_lower = theme_name.lower().replace(" ", "")

    # 精确匹配
    for b in boards:
        name = b["name"].lower().replace(" ", "")
        if theme_lower == name:
            return b

    # 包含匹配
    for b in boards:
        name = b["name"].lower().replace(" ", "")
        if theme_lower in name or name in theme_lower:
            return b

    # 关键词匹配（取前2个字符）
    if len(theme_lower) >= 2:
        for b in boards:
            name = b["name"].lower().replace(" ", "")
            if theme_lower[:2] in name:
                return b

    return None


def _fetch_board_constituents(board_code: str) -> list[dict]:
    """获取概念板块的成分股（最多50只）"""
    # 通过东财 push2 获取板块成分股
    url = "https://push2.eastmoney.com/api/qt/clist/get"
    params = {
        "pn": "1", "pz": "50",
        "po": "1", "np": "1",
        "fltt": "2", "invt": "2",
        "fs": f"b:{board_code}+f:!50",  # 排除停牌
        "fields": "f2,f3,f4,f12,f14",
    }
    headers = {"User-Agent": UA, "Referer": "https://quote.eastmoney.com/"}
    try:
        r = em_get(url, params=params, headers=headers, timeout=10, max_retries=0)  # 不重试，避免8s延迟
        data = r.json()
        items = (data.get("data") or {}).get("diff") or []
        return [{"code": it.get("f12", ""), "name": it.get("f14", "")} for it in items]
    except Exception:
        return []  # silently fail for this fast path


# ============================================================
# 7. 输出格式化
# ============================================================

def format_themes_report(result: dict) -> str:
    """格式化为 Markdown 报告"""
    lines = []
    lines.append(f"# 热门主线发现报告 — {result['date']}")
    lines.append("")

    env = result["market_environment"]
    nb = env["northbound"]
    lines.append("## 市场环境")
    lines.append(f"- 概念板块总数: {env['total_concept_boards']}")
    lines.append(f"- 今日强势股: {env['total_hot_stocks']} 只")
    lines.append(f"- 龙虎榜上榜: {env['total_lhb_records']} 只")
    lines.append(f"- 北向资金: **{nb['direction']}** "
                 f"(5日 {nb['net_total_5d']}亿, 20日 {nb['net_total_20d']}亿, "
                 f"趋势: {nb['trend']})")
    lines.append("")

    lines.append(f"## 最强主线 Top {len(result['themes'])}")
    lines.append("")
    lines.append("| 排名 | 主线 | 热度 | 持续性 | 概念5日涨幅 | 强势股数 | 北向一致 | 涨跌比 | 成分股 |")
    lines.append("|------|------|------|--------|-----------|---------|---------|--------|--------|")
    for i, t in enumerate(result["themes"], 1):
        s = t["signals"]
        nb_icon = "✅" if s["northbound_aligned"] else "—"
        persistence_cn = {"rising": "🔥 上升", "new": "🆕 新出现", "fading": "🔻 衰减"}.get(t["persistence"], t["persistence"])
        lines.append(
            f"| {i} | **{t['name']}** | {t['heat_score']:.0f} | {persistence_cn} | "
            f"{s['concept_change_5d']:+.1f}% | {s['ths_stock_count']} | {nb_icon} | "
            f"{s['up_down_ratio']} | {t['constituent_count']} 只 |"
        )
    lines.append("")

    # 每条主线详情
    for i, t in enumerate(result["themes"], 1):
        lines.append(f"## {i}. {t['name']} (热度: {t['heat_score']:.0f})")
        lines.append("")
        s = t["signals"]
        lines.append(f"- 概念板块排名: #{s['concept_rank']}, "
                     f"5日涨幅: {s['concept_change_5d']:+.1f}%, "
                     f"20日涨幅: {s['concept_change_20d']:+.1f}%")
        lines.append(f"- 同花顺强势股: {s['ths_stock_count']} 只, "
                     f"平均涨幅: {s['ths_avg_change']:+.1f}%")
        lines.append(f"- 涨跌比: {s['up_down_ratio']}")
        if s["lhb_active"]:
            lines.append(f"- 龙虎榜活跃: {s['lhb_count']} 只上榜")
        if s["northbound_aligned"]:
            lines.append(f"- 北向资金方向一致 ✅")
        lines.append(f"- 关联概念: {', '.join(t['display_names'][:5])}")

        stocks = t.get("constituent_stocks", [])
        if stocks:
            lines.append(f"- 成分股 ({len(stocks)} 只):")
            # 显示前20只
            for stock in stocks[:20]:
                lines.append(f"  - {stock['code']} {stock.get('name','')} "
                            f"({stock.get('source','')})")
            if len(stocks) > 20:
                lines.append(f"  - ... 还有 {len(stocks)-20} 只")
        lines.append("")

    return "\n".join(lines)


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="热门主线自动发现引擎")
    parser.add_argument("--top-n", type=int, default=10, help="输出 Top N 主线 (default: 10)")
    parser.add_argument("--output", choices=["markdown", "json"], default="markdown",
                        help="输出格式 (default: markdown)")
    parser.add_argument("--save", action="store_true", help="保存到 output 目录")
    args = parser.parse_args()

    ensure_dirs()

    result = discover_themes(top_n=args.top_n)

    if args.output == "json":
        # 简化 JSON 输出
        json_result = {
            "date": result["date"],
            "market_environment": result["market_environment"],
            "themes": [
                {
                    "name": t["name"],
                    "heat_score": t["heat_score"],
                    "persistence": t["persistence"],
                    "signals": t["signals"],
                    "constituent_count": len(t.get("constituent_stocks", [])),
                    "constituent_stocks": t.get("constituent_stocks", []),
                }
                for t in result["themes"]
            ],
        }
        print(json.dumps(json_result, ensure_ascii=False, indent=2))
    else:
        report = format_themes_report(result)
        print(report)

    if args.save:
        date_dir = OUTPUT_DIR / result["date"]
        date_dir.mkdir(parents=True, exist_ok=True)
        # Save JSON
        with open(date_dir / "themes.json", "w") as f:
            json.dump(result, f, ensure_ascii=False, indent=2, default=str)
        # Save markdown
        report = format_themes_report(result)
        with open(date_dir / "theme_discovery_report.md", "w") as f:
            f.write(report)
        print(f"\n✅ 已保存到 {date_dir}/")


if __name__ == "__main__":
    main()
