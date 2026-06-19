#!/usr/bin/env python3
"""
新闻驱动 × 供应链涟漪引擎 v1

1. 拉取 东财全球资讯 + news项目数据
2. 提取结构化关键词（国家·动作·产品）
3. 在 material_graph 中匹配受影响材料
4. 涟漪传播: Layer1直接→Layer2关联→Layer3瓶颈扩散
5. 对候选标的做技术面验证

用法:
  python3 scripts/news_ripple.py                    # 分析近7天新闻
  python3 scripts/news_ripple.py --days 3           # 近3天
  python3 scripts/news_ripple.py --date 2026-06-17  # 指定日期
"""

import argparse, json, os, re, sys, time
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from data_loader import (
    eastmoney_global_news, ensure_dirs, load_universe_klines, get_stock_kline,
)
from candlestick_patterns import identify_all_patterns
from volume_price_analyzer import analyze_volume_price
from chip_distribution import estimate_chip_distribution

# ============================================================
# 关键词提取
# ============================================================

# 国家/地区
COUNTRIES = ["日本", "美国", "韩国", "荷兰", "德国", "台湾", "英国", "法国", "欧盟"]

# 限制/断供动词
RESTRICTION_VERBS = [
    "限制", "断供", "制裁", "禁止", "出口管制", "封锁", "停产",
    "事故", "爆炸", "火灾", "停工", "审查", "加征关税", "反倾销",
    "列入实体清单", "技术封锁", "供应紧张", "缺货", "涨价",
]

# 产品/材料关键词 → material_graph 匹配名
PRODUCT_KEYWORDS = {
    # 钨系
    "六氟化钨": "六氟化钨", "WF6": "六氟化钨",
    "钨靶材": "钨靶材", "钨溅射靶": "钨靶材",
    "钨丝": "钨丝", "钨粉": "钨粉", "碳化钨": "钨粉",
    # 稀土
    "稀土": "稀土", "钕铁硼": "钕铁硼", "永磁体": "钕铁硼",
    "镝": "重稀土", "铽": "重稀土",
    # 电子特气
    "光刻胶": "光刻胶", "氟化氢": "氟化氢", "氟化氩": "氟化氩",
    "氟聚酰亚胺": "氟聚酰亚胺", "高纯氖气": "氖气",
    # 半导体
    "EDA": "EDA软件", "离子注入": "离子注入机",
    "光刻机": "光刻机", "DUV": "光刻机", "EUV": "光刻机",
    "刻蚀机": "刻蚀机", "薄膜沉积": "薄膜沉积设备",
    # 高纯材料
    "高纯石英": "高纯石英", "高纯硅": "高纯多晶硅",
    "溅射靶材": "溅射靶材", "靶材": "溅射靶材",
    # 芯片/封装
    "HBM": "HBM", "高带宽内存": "HBM",
    "先进封装": "先进封装", "CoWoS": "先进封装",
    "chiplet": "chiplet", "芯粒": "chiplet",
    "封装基板": "封装基板", "IC基板": "封装基板", "ABF": "封装基板",
    # 设备/零部件
    "刻蚀机": "刻蚀机", "刻蚀": "刻蚀机",
    "CVD": "薄膜沉积设备", "PVD": "薄膜沉积设备", "ALD": "薄膜沉积设备",
    # PCB
    "PCB": "PCB", "印制电路板": "PCB", "HDI": "PCB",
    # 光通信材料
    "磷化铟": "磷化铟", "InP": "磷化铟", "铟磷": "磷化铟",
    "铌酸锂": "薄膜铌酸锂(TFLN)", "TFLN": "薄膜铌酸锂(TFLN)", "LNOI": "薄膜铌酸锂(TFLN)",
    "薄膜铌酸锂": "薄膜铌酸锂(TFLN)",
    # EDA
    "EDA": "EDA软件",
}


def extract_keywords(news_text: str) -> dict:
    """从新闻标题+摘要中提取结构化关键词"""
    result = {
        "countries": [], "actions": [], "products": [],
        "has_restriction": False, "confidence": 0,
    }
    text = news_text

    for c in COUNTRIES:
        if c in text:
            result["countries"].append(c)

    for v in RESTRICTION_VERBS:
        if v in text:
            result["actions"].append(v)
            result["has_restriction"] = True

    for kw, material in PRODUCT_KEYWORDS.items():
        if kw in text:
            if material not in result["products"]:
                result["products"].append(material)

    # 置信度
    if result["products"] and result["has_restriction"]:
        result["confidence"] = 0.8
    elif result["products"] and result["countries"]:
        result["confidence"] = 0.5
    elif result["products"]:
        result["confidence"] = 0.3
    elif result["has_restriction"]:
        result["confidence"] = 0.2

    return result


# ============================================================
# 材料知识图谱（内嵌简化版，完整版在 config/material_graph.yaml）
# ============================================================

MATERIAL_GRAPH = {
    # ═══════════════════════════════════════════════════════════
    # 钨系 (日本·韩国管制高风险)
    # ═══════════════════════════════════════════════════════════
    "六氟化钨": {
        "aliases": ["WF6", "六氟"],
        "category": "钨材料(电子特气上游)",
        "domestic_producers": [
            {"code": "600549", "name": "厦门钨业", "relevance": 0.9},
            {"code": "000657", "name": "中钨高新", "relevance": 0.9},
            {"code": "002378", "name": "章源钨业", "relevance": 0.7},
        ],
        "related_products": [
            {"product": "钨靶材", "relation": "日本同源出口管制", "ripple": 0.8},
            {"product": "钨粉", "relation": "上游粉体同材质", "ripple": 0.6},
            {"product": "钨丝", "relation": "光伏切割同材质", "ripple": 0.5},
        ],
    },
    "钨靶材": {
        "category": "溅射靶材",
        "domestic_producers": [
            {"code": "300666", "name": "江丰电子", "relevance": 1.0},
            {"code": "300706", "name": "阿石创", "relevance": 0.8},
            {"code": "300263", "name": "隆华科技", "relevance": 0.6},
        ],
    },
    "钨粉": {
        "category": "钨材料(电子特气上游)",
        "domestic_producers": [
            {"code": "002378", "name": "章源钨业", "relevance": 0.9},
            {"code": "002842", "name": "翔鹭钨业", "relevance": 0.8},
        ],
    },
    "钨丝": {
        "category": "钨材料(电子特气上游)",
        "domestic_producers": [
            {"code": "600549", "name": "厦门钨业", "relevance": 0.9},
            {"code": "000657", "name": "中钨高新", "relevance": 0.8},
        ],
    },
    "稀土": {
        "aliases": ["稀土永磁", "钕铁硼"],
        "category": "战略资源",
        "domestic_producers": [
            {"code": "600111", "name": "北方稀土", "relevance": 1.0},
            {"code": "000831", "name": "中国稀土", "relevance": 1.0},
            {"code": "600392", "name": "盛和资源", "relevance": 0.8},
        ],
        "related_products": [
            {"product": "钕铁硼", "relation": "下游永磁材料", "ripple": 0.9},
        ],
    },
    "钕铁硼": {
        "category": "永磁材料",
        "domestic_producers": [
            {"code": "300748", "name": "金力永磁", "relevance": 1.0},
            {"code": "300224", "name": "正海磁材", "relevance": 0.9},
            {"code": "000970", "name": "中科三环", "relevance": 0.9},
        ],
    },
    "光刻胶": {
        "category": "半导体材料",
        "domestic_producers": [
            {"code": "300576", "name": "容大感光", "relevance": 0.9},
            {"code": "300346", "name": "南大光电", "relevance": 0.9},
            {"code": "300655", "name": "晶瑞电材", "relevance": 0.8},
        ],
    },
    "溅射靶材": {
        "category": "半导体材料",
        "domestic_producers": [
            {"code": "300666", "name": "江丰电子", "relevance": 1.0},
            {"code": "300706", "name": "阿石创", "relevance": 0.8},
        ],
    },
    "HBM": {
        "category": "先进封装",
        "domestic_producers": [
            {"code": "002156", "name": "通富微电", "relevance": 0.9},
            {"code": "002185", "name": "华天科技", "relevance": 0.8},
            {"code": "600584", "name": "长电科技", "relevance": 0.8},
        ],
    },
    "先进封装": {
        "category": "半导体制造",
        "domestic_producers": [
            {"code": "600584", "name": "长电科技", "relevance": 1.0},
            {"code": "002156", "name": "通富微电", "relevance": 0.9},
            {"code": "002185", "name": "华天科技", "relevance": 0.9},
        ],
        "related_products": [
            {"product": "封装基板", "relation": "先进封装核心耗材", "ripple": 0.8},
            {"product": "EDA软件", "relation": "chiplet设计工具", "ripple": 0.5},
        ],
    },
    # ═══════════════════════════════════════════════════════════
    # 电子特气 (日本·韩国主导，断供高风险)
    # ═══════════════════════════════════════════════════════════
    "氟化氢": {
        "aliases": ["高纯氟化氢", "电子级氢氟酸"],
        "category": "电子特气",
        "source_countries": ["日本", "韩国"],
        "domestic_producers": [
            {"code": "002409", "name": "雅克科技", "relevance": 0.9},
            {"code": "002915", "name": "中欣氟材", "relevance": 0.7},
            {"code": "603379", "name": "三美股份", "relevance": 0.6},
        ],
        "related_products": [
            {"product": "光刻胶", "relation": "同属日韩垄断半导体材料", "ripple": 0.7},
        ],
    },
    "氟化氩": {
        "aliases": ["ArF", "氟化氩光刻气体"],
        "category": "电子特气",
        "domestic_producers": [
            {"code": "002409", "name": "雅克科技", "relevance": 0.9},
            {"code": "300346", "name": "南大光电", "relevance": 0.8},
        ],
    },
    "氟聚酰亚胺": {
        "aliases": ["含氟聚酰亚胺", "FPI"],
        "category": "半导体材料",
        "source_countries": ["日本"],
        "domestic_producers": [
            {"code": "002409", "name": "雅克科技", "relevance": 0.8},
        ],
    },
    "氖气": {
        "aliases": ["高纯氖气", "Ne"],
        "category": "电子特气",
        "domestic_producers": [
            {"code": "002549", "name": "凯美特气", "relevance": 0.9},
            {"code": "600378", "name": "昊华科技", "relevance": 0.9},
        ],
    },
    # ═══════════════════════════════════════════════════════════
    # 光刻胶 + 光刻相关 (日本绝对垄断)
    # ═══════════════════════════════════════════════════════════
    "光刻胶": {
        "aliases": ["光刻胶", "光阻剂", "photoresist", "ArF光刻胶", "KrF光刻胶", "EUV光刻胶"],
        "category": "半导体材料",
        "source_countries": ["日本"],
        "domestic_producers": [
            {"code": "300576", "name": "容大感光", "relevance": 0.9},
            {"code": "300346", "name": "南大光电", "relevance": 0.9},
            {"code": "300655", "name": "晶瑞电材", "relevance": 0.8},
            {"code": "300721", "name": "怡达股份", "relevance": 0.6},
        ],
        "related_products": [
            {"product": "溅射靶材", "relation": "同属日本垄断半导体材料", "ripple": 0.7},
            {"product": "高纯石英", "relation": "光刻机镜头材料", "ripple": 0.6},
        ],
    },
    # ═══════════════════════════════════════════════════════════
    # 高纯材料
    # ═══════════════════════════════════════════════════════════
    "高纯石英": {
        "aliases": ["高纯石英砂", "石英坩埚", "半导体石英"],
        "category": "高纯材料",
        "domestic_producers": [
            {"code": "603688", "name": "石英股份", "relevance": 1.0},
            {"code": "300554", "name": "三超新材", "relevance": 0.7},
        ],
        "related_products": [
            {"product": "高纯多晶硅", "relation": "同属高纯材料", "ripple": 0.5},
        ],
    },
    "高纯多晶硅": {
        "aliases": ["电子级多晶硅", "半导体硅料"],
        "category": "高纯材料",
        "domestic_producers": [
            {"code": "600438", "name": "通威股份", "relevance": 0.8},
            {"code": "601012", "name": "隆基绿能", "relevance": 0.7},
        ],
    },
    # ═══════════════════════════════════════════════════════════
    # 溅射靶材 (半导体·显示面板)
    # ═══════════════════════════════════════════════════════════
    "溅射靶材": {
        "aliases": ["靶材", "溅射靶", "PVD靶材"],
        "category": "半导体材料",
        "source_countries": ["日本", "美国"],
        "domestic_producers": [
            {"code": "300666", "name": "江丰电子", "relevance": 1.0},
            {"code": "300706", "name": "阿石创", "relevance": 0.8},
            {"code": "300263", "name": "隆华科技", "relevance": 0.7},
        ],
    },
    # ═══════════════════════════════════════════════════════════
    # 先进封装
    # ═══════════════════════════════════════════════════════════
    "HBM": {
        "aliases": ["高带宽内存", "HBM3", "HBM4"],
        "category": "先进封装",
        "domestic_producers": [
            {"code": "002156", "name": "通富微电", "relevance": 0.9},
            {"code": "002185", "name": "华天科技", "relevance": 0.8},
            {"code": "600584", "name": "长电科技", "relevance": 0.8},
        ],
        "related_products": [
            {"product": "先进封装", "relation": "HBM依赖先进封装", "ripple": 0.9},
        ],
    },
    "chiplet": {
        "aliases": ["芯粒", "Chiplet", "小芯片"],
        "category": "先进封装",
        "domestic_producers": [
            {"code": "002156", "name": "通富微电", "relevance": 0.9},
            {"code": "600584", "name": "长电科技", "relevance": 0.9},
            {"code": "603005", "name": "晶方科技", "relevance": 0.8},
        ],
    },
    "封装基板": {
        "aliases": ["IC基板", "ABF基板", "BT基板"],
        "category": "先进封装",
        "domestic_producers": [
            {"code": "002916", "name": "深南电路", "relevance": 0.9},
            {"code": "002938", "name": "鹏鼎控股", "relevance": 0.8},
            {"code": "002436", "name": "兴森科技", "relevance": 0.8},
        ],
    },
    # ═══════════════════════════════════════════════════════════
    # 半导体设备/零部件
    # ═══════════════════════════════════════════════════════════
    "光刻机": {
        "aliases": ["DUV", "EUV", "光刻", "lithography"],
        "category": "半导体设备",
        "source_countries": ["荷兰", "日本"],
        "domestic_producers": [],
        "related_products": [
            {"product": "光刻胶", "relation": "光刻耗材", "ripple": 0.9},
            {"product": "高纯石英", "relation": "光刻镜头材料", "ripple": 0.6},
            {"product": "刻蚀机", "relation": "光刻配套设备", "ripple": 0.5},
        ],
    },
    "刻蚀机": {
        "aliases": ["刻蚀", "etch", "等离子刻蚀"],
        "category": "半导体设备",
        "domestic_producers": [
            {"code": "002371", "name": "北方华创", "relevance": 1.0},
            {"code": "688012", "name": "中微公司", "relevance": 1.0},
        ],
    },
    "薄膜沉积设备": {
        "aliases": ["CVD", "PVD", "ALD", "薄膜"],
        "category": "半导体设备",
        "domestic_producers": [
            {"code": "002371", "name": "北方华创", "relevance": 0.9},
            {"code": "688012", "name": "中微公司", "relevance": 0.8},
        ],
    },
    "离子注入机": {
        "aliases": ["离子注入", "ion implant"],
        "category": "半导体设备",
        "source_countries": ["美国"],
        "domestic_producers": [
            {"code": "688596", "name": "正帆科技", "relevance": 0.7},
        ],
    },
    # ═══════════════════════════════════════════════════════════
    # PCB/铜箔
    # ═══════════════════════════════════════════════════════════
    "PCB": {
        "aliases": ["印制电路板", "HDI", "高频高速", "IC载板"],
        "category": "电子制造",
        "domestic_producers": [
            {"code": "002916", "name": "深南电路", "relevance": 0.9},
            {"code": "002938", "name": "鹏鼎控股", "relevance": 0.9},
            {"code": "002463", "name": "沪电股份", "relevance": 0.9},
        ],
    },
    # ═══════════════════════════════════════════════════════════
    # 磷化铟 (InP) — AI算力×光通信核心衬底材料
    # ═══════════════════════════════════════════════════════════
    "磷化铟": {
        "aliases": ["InP", "铟磷", "铟化物"],
        "category": "光通信材料",
        "domestic_producers": [
            {"code": "002428", "name": "云南锗业", "relevance": 1.0},
            {"code": "600703", "name": "三安光电", "relevance": 0.8},
            {"code": "600206", "name": "有研新材", "relevance": 0.7},
        ],
        "related_products": [
            {"product": "薄膜铌酸锂(TFLN)", "relation": "同属光通信上游芯片材料", "ripple": 0.9},
            {"product": "光模块", "relation": "磷化铟是EML激光器衬底", "ripple": 0.8},
        ],
    },
    # ═══════════════════════════════════════════════════════════
    # 薄膜铌酸锂 (TFLN) — 1.6T/3.2T光模块核心使能技术
    # ═══════════════════════════════════════════════════════════
    "薄膜铌酸锂(TFLN)": {
        "aliases": ["TFLN", "铌酸锂", "LNOI", "钽酸锂", "光波导"],
        "category": "光通信材料",
        "domestic_producers": [
            {"code": "002281", "name": "光迅科技", "relevance": 0.9},
            {"code": "688195", "name": "腾景科技", "relevance": 0.8},
            {"code": "688662", "name": "富信科技", "relevance": 0.6},
        ],
        "related_products": [
            {"product": "磷化铟", "relation": "同属光通信上游芯片材料", "ripple": 0.9},
            {"product": "光模块", "relation": "TFLN是1.6T光模块调制器核心", "ripple": 0.8},
        ],
    },
    # ═══════════════════════════════════════════════════════════
    # EDA / 工业软件
    # ═══════════════════════════════════════════════════════════
    "EDA软件": {
        "aliases": ["EDA", "电子设计自动化", "EDA工具"],
        "category": "工业软件",
        "source_countries": ["美国"],
        "domestic_producers": [
            {"code": "301269", "name": "华大九天", "relevance": 1.0},
            {"code": "688206", "name": "概伦电子", "relevance": 0.8},
        ],
    },
}


# ============================================================
# 涟漪传播
# ============================================================

def ripple_propagate(material_name: str, max_layer: int = 2) -> dict:
    """
    涟漪传播: 从受影响材料出发，找到Layer1直接标的 → Layer2关联标的

    Returns: {layer1: [{code, name, relevance, layer, logic}], layer2: [...]}
    """
    result = {"layer1": [], "layer2": [], "material": material_name}

    mat = MATERIAL_GRAPH.get(material_name)
    if not mat:
        return result

    # Layer 1: 直接国内替代
    for p in mat.get("domestic_producers", []):
        result["layer1"].append({
            "code": p["code"], "name": p["name"],
            "relevance": p.get("relevance", 0.5),
            "layer": 1,
            "logic": f"「{material_name}」国内替代",
        })

    # Layer 2: 关联产品涟漪
    for rp in mat.get("related_products", [])[:max_layer]:
        related_mat = MATERIAL_GRAPH.get(rp["product"])
        if related_mat:
            for p in related_mat.get("domestic_producers", [])[:3]:
                # 去重
                if p["code"] not in [x["code"] for x in result["layer1"]]:
                    result["layer2"].append({
                        "code": p["code"], "name": p["name"],
                        "relevance": p.get("relevance", 0.5) * rp.get("ripple", 0.5),
                        "layer": 2,
                        "logic": f"「{material_name}」→「{rp['product']}」{rp.get('relation','')}",
                    })

    return result


# ============================================================
# 技术面验证
# ============================================================

def tech_verify(code: str, target_date: str, kline_map: dict = None) -> dict:
    """对涟漪标的做快速技术面验证"""
    market = 1 if code.startswith("6") else 0
    df = get_stock_kline(code, market, refresh=False)
    if df is None or len(df) < 60:
        return {"pass": False, "reason": "无数据"}

    df = df[df["date"] <= target_date].copy()
    p = identify_all_patterns(df, ticker=code)
    v = analyze_volume_price(df)
    c = estimate_chip_distribution(df)

    price = float(df["close"].values[-1])
    chg = (price / float(df["close"].values[-2]) - 1) if len(df) >= 2 else 0

    # 简单评分
    score = p.pattern_score * 0.35 + v.get("volume_score", 0) * 0.3 + c.get("chip_score", 0) * 0.2

    # 上涨天数检测: ≤2天最佳（主力刚建仓），>5天已透支
    c_arr = df["close"].values
    up_days = 0
    for i in range(len(c_arr)-1, max(0, len(c_arr)-8), -1):
        if c_arr[i] > c_arr[i-1]: up_days += 1
        else: break
    if up_days <= 2: score += 8       # 刚启动，机构先知先觉
    elif up_days <= 4: score += 3     # 趋势中
    else: score -= 5                  # 涨太久，新闻可能已price in

    return {
        "pass": score >= 10,
        "score": round(score, 1),
        "up_days": up_days,
        "price": price, "chg_pct": round(chg * 100, 2),
        "k_score": round(p.pattern_score, 1),
        "v_score": v.get("volume_score", 0),
        "c_score": c.get("chip_score", 0),
        "patterns": [h.pattern for h in p.bullish_reversal[:2] + p.continuation[:1]],
    }


# ============================================================
# 主函数
# ============================================================

def analyze_news_ripple(days: int = 7, target_date: str = None) -> dict:
    """
    主入口: 拉取新闻 → 提取关键词 → 涟漪分析 → 技术验证 → 返回结果
    """
    if target_date is None:
        target_date = datetime.now().strftime("%Y-%m-%d")

    print(f"\n{'='*60}")
    print(f"  新闻驱动涟漪分析 — {target_date} (近{days}天)")
    print(f"{'='*60}")

    # 1. 拉取新闻
    print("\n[1/4] 拉取新闻...")
    news_items = eastmoney_global_news(page_size=100)
    print(f"  获取 {len(news_items)} 条快讯")

    # 2. 提取关键词
    print("\n[2/4] 提取关键词...")
    all_matches = defaultdict(list)
    for n in news_items:
        text = n["title"] + " " + n.get("summary", "")
        kw = extract_keywords(text)
        if kw["products"]:
            for prod in kw["products"]:
                all_matches[prod].append({
                    "title": n["title"],
                    "time": n.get("time", ""),
                    "countries": kw["countries"],
                    "actions": kw["actions"],
                    "confidence": kw["confidence"],
                })

    print(f"  发现 {len(all_matches)} 个相关材料: {list(all_matches.keys())}")

    # 3. 涟漪传播
    print("\n[3/4] 涟漪传播...")
    ripple_results = []
    all_ripple_codes = set()

    for material, news_refs in all_matches.items():
        max_conf = max(r["confidence"] for r in news_refs) if news_refs else 0
        if max_conf < 0.3:  # 过滤低置信度
            continue

        ripple = ripple_propagate(material)
        ripple["news_refs"] = news_refs[:3]
        ripple["confidence"] = max_conf

        l1_codes = [s["code"] for s in ripple["layer1"]]
        l2_codes = [s["code"] for s in ripple["layer2"]]
        all_ripple_codes.update(l1_codes + l2_codes)
        print(f"  {material}: L1={l1_codes} L2={l2_codes} (置信{max_conf:.0%})")
        ripple_results.append(ripple)

    # 4. 技术面验证
    print(f"\n[4/4] 技术面验证 ({len(all_ripple_codes)} 只)...")
    verified = []
    kline_map = load_universe_klines(watchlist_only=False, refresh=False)

    for code in all_ripple_codes:
        if code.startswith("688"): continue
        tv = tech_verify(code, target_date)
        tv["code"] = code
        # 找涟漪来源
        for rr in ripple_results:
            for s in rr["layer1"] + rr["layer2"]:
                if s["code"] == code:
                    tv["name"] = s["name"]
                    tv["layer"] = s["layer"]
                    tv["logic"] = s["logic"]
                    tv["material"] = rr["material"]
                    tv["confidence"] = rr.get("confidence", 0)
                    break
        if tv.get("pass"):
            verified.append(tv)
        print(f"  {code} {tv.get('name','?'):<8s}: {'✅' if tv['pass'] else '❌'} "
              f"得分{tv['score']} K{tv['k_score']} V{tv['v_score']} C{tv['c_score']}")

    verified.sort(key=lambda x: (x.get("layer", 9), -x.get("score", 0)))

    return {
        "date": target_date, "days": days,
        "total_news": len(news_items),
        "materials_found": len(all_matches),
        "ripple_results": ripple_results,
        "verified_stocks": verified,
    }


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="新闻驱动涟漪分析")
    parser.add_argument("--days", type=int, default=7, help="回溯天数")
    parser.add_argument("--date", type=str, default=None, help="目标日期")
    args = parser.parse_args()

    ensure_dirs()
    result = analyze_news_ripple(days=args.days, target_date=args.date)

    print(f"\n{'='*60}")
    print(f"  涟漪标的汇总 ({len(result['verified_stocks'])} 只通过技术验证)")
    print(f"{'='*60}")
    for s in result["verified_stocks"]:
        layer_tag = "L1直接" if s.get("layer") == 1 else "L2涟漪"
        print(f"  {s['code']} {s.get('name','?'):<8s} [{layer_tag}] "
              f"¥{s['price']:.2f} {s['chg_pct']:+.1f}% "
              f"K{s['k_score']:.0f} V{s['v_score']:.0f} "
              f"← {s.get('logic','')}")


if __name__ == "__main__":
    main()
