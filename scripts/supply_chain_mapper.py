#!/usr/bin/env python3
"""
Phase 2b: A股供应链瓶颈逆向映射引擎

输入: 投资主线名称（如"AI数据中心电源"）
输出:
  1. 5层供应链拆解图
  2. 瓶颈环节标注（9+3原型匹配）
  3. 瓶颈环节的A股候选标的

数据源:
  - supply_chain.yaml (预建知识库)
  - 东财概念板块成分股
  - iwencai 产业链关键词检索 (可选，需API Key)

用法:
  python3 scripts/supply_chain_mapper.py --theme "AI数据中心电源"
  python3 scripts/supply_chain_mapper.py --theme "AI数据中心电源" --output json
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import requests
import yaml

ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = ROOT / "config"
OUTPUT_DIR = ROOT / "output"
sys.path.insert(0, str(ROOT / "scripts"))

from data_loader import (
    em_get, UA, EM_SESSION, ensure_dirs, tencent_quote,
    eastmoney_concept_blocks, get_eastmoney_sector_map,
)
from theme_discovery import (
    _normalize_topic_name, _find_concept_board,
    _fetch_board_constituents, fetch_concept_board_ranking,
)


# ============================================================
# 瓶颈原型库 (9+3)
# ============================================================

ARCHETYPES = {
    "单源供应商": {
        "id": 1,
        "description": "全球/全国只有1-2家企业能生产的核心零部件/材料",
        "signal_strength": "极强",
        "typical_valuation_premium": "30-50%",
        "risk": "替代风险（技术路线变更或新进入者）",
    },
    "产能瓶颈": {
        "id": 2,
        "description": "扩产周期2年以上，供不应求将持续的结构性短缺",
        "signal_strength": "强",
        "typical_valuation_premium": "20-40%",
        "risk": "周期下行时产能过剩",
    },
    "认证壁垒": {
        "id": 3,
        "description": "客户认证周期超过1年，一旦进入就很难被替换",
        "signal_strength": "强",
        "typical_valuation_premium": "15-30%",
        "risk": "技术路线变化导致认证价值归零",
    },
    "耗材绑定": {
        "id": 4,
        "description": "设备销售后持续产生耗材收入的商业模式",
        "signal_strength": "中等",
        "typical_valuation_premium": "20-35%",
        "risk": "设备稼动率下降",
    },
    "设计锁定": {
        "id": 5,
        "description": "产品设计一旦选定该供应商就无法轻易更换",
        "signal_strength": "极强",
        "typical_valuation_premium": "25-50%",
        "risk": "客户自研替代",
    },
    "高转换成本": {
        "id": 6,
        "description": "客户切换供应商的直接+间接成本极高",
        "signal_strength": "强",
        "typical_valuation_premium": "15-25%",
        "risk": "新进入者大幅降价抢份额",
    },
    "稀缺牌照": {
        "id": 7,
        "description": "政府特许经营/牌照数量有限制",
        "signal_strength": "极强",
        "typical_valuation_premium": "20-40%",
        "risk": "政策放开牌照限制",
    },
    "工艺壁垒": {
        "id": 8,
        "description": "良率/工艺know-how积累极难被复制，即使有设备也做不出来",
        "signal_strength": "强",
        "typical_valuation_premium": "15-30%",
        "risk": "工艺被突破/良率被追上",
    },
    "规模效应": {
        "id": 9,
        "description": "规模越大单位成本越低的持续正循环",
        "signal_strength": "中等",
        "typical_valuation_premium": "10-20%",
        "risk": "需求萎缩导致规模效应反转",
    },
    # A股特有3原型
    "国产替代锁定": {
        "id": 10,
        "description": "卡脖子环节的国内唯一或技术最领先供应商，享受国家政策+下游导入双重驱动",
        "signal_strength": "极强",
        "typical_valuation_premium": "40-100%",
        "risk": "技术差距被拉大/其他国产厂商突破",
    },
    "政策壁垒": {
        "id": 11,
        "description": "核电/军工/卫星/烟草等准入资质壁垒，民营资本难以进入",
        "signal_strength": "极强",
        "typical_valuation_premium": "20-40%",
        "risk": "政策转向开放",
    },
    "国企改革套利": {
        "id": 12,
        "description": "央企整合/资产注入/混改预期，治理改善驱动价值重估",
        "signal_strength": "中等",
        "typical_valuation_premium": "10-30%",
        "risk": "改革进度不及预期",
    },
}


# ============================================================
# 供应链知识库加载
# ============================================================

def load_supply_chain_knowledge() -> dict:
    """加载预建的产业链知识库"""
    config_path = CONFIG_DIR / "supply_chain.yaml"
    if not config_path.exists():
        print(f"  [WARN] 未找到 supply_chain.yaml，使用空知识库")
        return {"themes": {}}

    with open(config_path) as f:
        data = yaml.safe_load(f)
    return data


def find_theme_chain(theme_name: str) -> Optional[dict]:
    """在知识库中查找主题的产业链映射"""
    kb = load_supply_chain_knowledge()
    themes = kb.get("themes", {})

    # 精确匹配
    if theme_name in themes:
        return themes[theme_name]

    # 模糊匹配
    norm = _normalize_topic_name(theme_name)
    if norm in themes:
        return themes[norm]

    # 遍历搜索
    for name, chain in themes.items():
        desc = chain.get("description", "")
        if theme_name in name or name in theme_name or theme_name in desc:
            return chain
        if norm in name or name in norm:
            return chain

    return None


# ============================================================
# 瓶颈层候选股发现
# ============================================================

def discover_bottleneck_stocks(chain: dict,
                              ths_hot_stocks: list[dict] = None) -> dict[str, list[dict]]:
    """
    对产业链的每个瓶颈层，找到对应的A股标的。

    策略（按优先级降级）：
      1. 东财 push2 概念板块成分股（最佳，但可能被网络拦截）
      2. 预建的 sample_stocks（来自 YAML）
      3. 同花顺强势股中匹配（兜底）
      4. 腾讯财经验证基本面（PE/PB等）

    Returns:
        {layer_name: [{code, name, pe, pb, mcap_yi, ...}]}
    """
    bottleneck_layers = {}
    all_codes = set()
    push2_available = True  # will be set to False if all attempts fail

    for layer_key, layer in chain.get("chain", {}).items():
        if not layer.get("is_bottleneck", False):
            continue

        sectors = layer.get("a_stock_sectors", [])
        layer_stocks = {}

        # Strategy 1: 东财 push2 概念板块成分股
        found_via_push2 = False
        for sector_name in sectors:
            board = _find_concept_board(sector_name)
            if board:
                found_via_push2 = True
                constituents = _fetch_board_constituents(board["code"])
                for s in constituents:
                    if s["code"] not in layer_stocks:
                        layer_stocks[s["code"]] = {
                            "code": s["code"], "name": s.get("name", ""),
                            "source_board": sector_name,
                        }

        if not found_via_push2 and sectors:
            push2_available = False

        # Strategy 2: sample_stocks (always add)
        for code in layer.get("sample_stocks", []):
            if code not in layer_stocks:
                layer_stocks[code] = {
                    "code": code, "name": "",
                    "source_board": "示例标的",
                }

        # Strategy 3: 同花顺强势股兜底（当 push2 不可用且 layer 无 sample 时）
        if not found_via_push2 and ths_hot_stocks and not layer_stocks:
            sector_kw = set()
            for s in sectors:
                for kw in s.replace("-", "").replace("/", "").split():
                    if len(kw) >= 2:
                        sector_kw.add(kw)
            for stock in ths_hot_stocks:
                reasons = stock.get("reasons", [])
                for r in reasons:
                    if any(kw in r for kw in sector_kw):
                        if stock["code"] not in layer_stocks:
                            layer_stocks[stock["code"]] = {
                                "code": stock["code"],
                                "name": stock.get("name", ""),
                                "source_board": f"同花顺:{r}",
                            }
                        break

        codes = list(layer_stocks.keys())
        all_codes.update(codes)

        # 获取基本面
        enriched = []
        if codes:
            try:
                quotes = tencent_quote(codes)
                for code, stock in layer_stocks.items():
                    q = quotes.get(code, {})
                    enriched.append({
                        "code": code,
                        "name": q.get("name") or stock["name"],
                        "price": q.get("price"),
                        "change_pct": q.get("change_pct"),
                        "pe_ttm": q.get("pe_ttm"),
                        "pb": q.get("pb"),
                        "mcap_yi": q.get("mcap_yi"),
                        "source_board": stock["source_board"],
                    })
            except Exception as e:
                print(f"    [WARN] 基本面获取 {layer_key}: {e}")
                enriched = list(layer_stocks.values())

        bottleneck_layers[layer_key] = {
            "name": layer.get("name", layer_key),
            "description": layer.get("description", ""),
            "archetypes": layer.get("archetypes", []),
            "stocks": enriched,
            "total_stocks": len(enriched),
        }

    if not push2_available and not all_codes:
        print("  ⚠️ 东财概念板块不可用，已使用 sample_stocks + 同花顺兜底")

    return bottleneck_layers


# ============================================================
# 动态供应链发现 (当知识库无匹配时)
# ============================================================

def dynamic_supply_chain_discovery(theme_name: str) -> Optional[dict]:
    """
    当预建知识库无匹配时，尝试通过概念板块+行业分类
    动态构建供应链映射。
    """
    print(f"  知识库未命中「{theme_name}」，尝试动态发现...")

    # Step 1: 找到最相关的概念板块
    boards = fetch_concept_board_ranking()
    related_boards = []
    theme_lower = theme_name.lower().replace(" ", "")

    for b in boards:
        name = b["name"].lower().replace(" ", "")
        if any(kw in name for kw in theme_lower[:3]):
            related_boards.append(b)
        elif theme_lower in name or name in theme_lower:
            related_boards.append(b)

    if not related_boards:
        print(f"  [WARN] 未找到相关概念板块")
        return None

    related_boards = related_boards[:10]  # 取前10个最相关的

    # Step 2: 基于板块构建简化的供应链
    # 由于没有预建知识库，按板块涨幅排序，涨幅最大的当作下游热点，涨幅较小的可能是上游
    related_boards.sort(key=lambda b: float(b.get("change_pct_5d", 0) or 0), reverse=True)

    chain = {"description": f"{theme_name} 动态发现产业链", "chain": {}}

    # L1: 涨幅最大的概念板块 → 下游热点
    # L2-3: 中间板块 → 中游
    # L4-5: 涨幅较小 + "材料/设备/零部件"关键字的 → 上游

    layer_keys = ["L1_下游热点", "L2_中游系统", "L3_中游器件", "L4_上游设备", "L5_上游材料"]
    for i, board in enumerate(related_boards[:5]):
        idx = min(i, 4)
        is_bottleneck = i >= 3  # 上游默认为瓶颈
        chain["chain"][layer_keys[idx]] = {
            "name": board["name"],
            "description": f"概念板块: {board['name']} (5日涨幅: {board.get('change_pct_5d', 'N/A')}%)",
            "is_bottleneck": is_bottleneck,
            "archetypes": ["工艺壁垒"] if is_bottleneck else [],
            "a_stock_sectors": [board["name"]],
            "sample_stocks": [],
        }

    return chain


# ============================================================
# 主函数
# ============================================================

def map_supply_chain(theme_name: str, use_dynamic: bool = True) -> dict:
    """
    主函数: 对给定主线做供应链拆解+瓶颈挖掘。

    Returns:
        {
            "theme": "主线名称",
            "date": "YYYY-MM-DD",
            "source": "knowledge_base" | "dynamic_discovery",
            "chain": { 5 layers },
            "bottleneck_layers": { layer_name: {name, archetypes, stocks} },
            "total_bottleneck_candidates": int,
            "archetype_summary": [str],
        }
    """
    today = datetime.now().strftime("%Y-%m-%d")
    print(f"\n{'='*60}")
    print(f"  供应链瓶颈挖掘 — {theme_name}")
    print(f"{'='*60}\n")

    # Step 1: 查知识库
    chain_data = find_theme_chain(theme_name)
    source = "knowledge_base"

    if chain_data is None and use_dynamic:
        chain_data = dynamic_supply_chain_discovery(theme_name)
        source = "dynamic_discovery"

    if chain_data is None:
        print(f"  [FAIL] 无法解析「{theme_name}」的供应链结构")
        return {
            "theme": theme_name,
            "date": today,
            "error": "无法解析供应链",
            "chain": {},
            "bottleneck_layers": {},
            "total_bottleneck_candidates": 0,
            "archetype_summary": [],
        }

    print(f"  数据来源: {source}")
    print(f"  产业链描述: {chain_data.get('description', '无')}")

    # Step 2: 标记瓶颈层
    print("\n[瓶颈分析]")
    chain = chain_data.get("chain", {})
    bottleneck_count = 0

    for layer_key, layer in chain.items():
        nb = "🔴 瓶颈" if layer.get("is_bottleneck") else "⚪ 非瓶颈"
        archetypes = layer.get("archetypes", [])
        arch_str = f" (原型: {', '.join(archetypes)})" if archetypes else ""
        stocks_str = f" 示例: {', '.join(layer.get('sample_stocks', []))}" if layer.get("sample_stocks") else ""
        print(f"  {layer_key}: {layer.get('name', '?')} [{nb}]{arch_str}{stocks_str}")
        if layer.get("is_bottleneck"):
            bottleneck_count += 1

    print(f"\n  瓶颈层: {bottleneck_count}/{len(chain)} 层")

    # Step 3: 发现瓶颈层候选股（传入同花顺数据做兜底）
    print("\n[候选股发现]")
    try:
        from theme_discovery import fetch_ths_hot_stocks
        ths_stocks = fetch_ths_hot_stocks()
    except Exception:
        ths_stocks = None
    bottleneck_layers = discover_bottleneck_stocks(chain_data, ths_hot_stocks=ths_stocks)

    total_candidates = sum(
        bl["total_stocks"]
        for bl in bottleneck_layers.values()
    )
    print(f"  瓶颈候选股总计: {total_candidates} 只")

    # Step 4: 汇总原型
    all_archetypes = set()
    for bl in bottleneck_layers.values():
        for arch in bl.get("archetypes", []):
            all_archetypes.add(arch)
    archetype_summary = [
        f"{a}: {ARCHETYPES.get(a, {}).get('description', '')}"
        for a in all_archetypes
    ]

    result = {
        "theme": theme_name,
        "date": today,
        "source": source,
        "description": chain_data.get("description", ""),
        "chain": chain,
        "bottleneck_layers": bottleneck_layers,
        "total_bottleneck_candidates": total_candidates,
        "archetype_summary": archetype_summary,
    }

    return result


def format_supply_chain_report(result: dict) -> str:
    """格式化为 Markdown 报告"""
    lines = []
    lines.append(f"# 供应链瓶颈分析 — {result['theme']}")
    lines.append(f"**日期**: {result['date']} | **数据来源**: {result['source']}")
    lines.append("")

    if result.get("error"):
        lines.append(f"❌ {result['error']}")
        return "\n".join(lines)

    lines.append(f"**描述**: {result.get('description', '')}")
    lines.append("")

    # 供应链层级图
    lines.append("## 五层供应链拆解")
    lines.append("")
    lines.append("```")
    for layer_key, layer in result.get("chain", {}).items():
        icon = "🔴" if layer.get("is_bottleneck") else "⚪"
        name = layer.get("name", layer_key)
        desc = layer.get("description", "")
        arch = layer.get("archetypes", [])
        arch_str = f" [{', '.join(arch)}]" if arch else ""
        lines.append(f"  {icon} [{layer_key}] {name}: {desc}{arch_str}")
    lines.append("```")
    lines.append("")

    # 瓶颈原型总览
    if result.get("archetype_summary"):
        lines.append("## 匹配瓶颈原型")
        lines.append("")
        for a in result["archetype_summary"]:
            lines.append(f"- {a}")
        lines.append("")

    # 各瓶颈层候选
    lines.append("## 瓶颈层候选股")
    lines.append(f"**总计**: {result.get('total_bottleneck_candidates', 0)} 只候选")
    lines.append("")

    for layer_key, bl in result.get("bottleneck_layers", {}).items():
        lines.append(f"### {bl['name']} ({layer_key})")
        lines.append(f"**原型**: {', '.join(bl.get('archetypes', [])) or '无'}")
        lines.append(f"**描述**: {bl.get('description', '')}")
        lines.append("")
        stocks = bl.get("stocks", [])
        if stocks:
            lines.append("| 代码 | 名称 | 价格 | PE(TTM) | PB | 市值(亿) | 来源 |")
            lines.append("|------|------|------|---------|-----|---------|------|")
            for s in stocks:
                pe = f"{s.get('pe_ttm',''):.1f}" if s.get('pe_ttm') else "N/A"
                pb = f"{s.get('pb',''):.1f}" if s.get('pb') else "N/A"
                mcap = f"{s.get('mcap_yi',''):.0f}" if s.get('mcap_yi') else "N/A"
                lines.append(
                    f"| {s['code']} | {s.get('name','')} | "
                    f"{s.get('price','')} | {pe} | {pb} | {mcap} | "
                    f"{s.get('source_board','')} |"
                )
        else:
            lines.append("(无候选股)")
        lines.append("")

    return "\n".join(lines)


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="A股供应链瓶颈逆向映射引擎")
    parser.add_argument("--theme", type=str, required=True, help="投资主线名称")
    parser.add_argument("--output", choices=["markdown", "json"], default="markdown")
    parser.add_argument("--no-dynamic", action="store_true",
                        help="禁用动态发现（仅使用预建知识库）")
    parser.add_argument("--save", action="store_true", help="保存到 output 目录")
    args = parser.parse_args()

    ensure_dirs()

    result = map_supply_chain(args.theme, use_dynamic=not args.no_dynamic)

    if args.output == "json":
        # 简化JSON（只保留关键信息）
        json_result = {
            "theme": result["theme"],
            "date": result["date"],
            "source": result.get("source"),
            "error": result.get("error"),
            "total_candidates": result.get("total_bottleneck_candidates", 0),
            "archetypes": result.get("archetype_summary", []),
            "bottleneck_layers": {
                k: {
                    "name": v["name"],
                    "archetypes": v.get("archetypes", []),
                    "total_stocks": v["total_stocks"],
                    "stocks": v["stocks"],
                }
                for k, v in result.get("bottleneck_layers", {}).items()
            },
        }
        print(json.dumps(json_result, ensure_ascii=False, indent=2))
    else:
        report = format_supply_chain_report(result)
        print(report)

    if args.save:
        date_dir = OUTPUT_DIR / result["date"] / f"theme_{result['theme']}"
        date_dir.mkdir(parents=True, exist_ok=True)
        with open(date_dir / "supply_chain.json", "w") as f:
            json.dump(result, f, ensure_ascii=False, indent=2, default=str)
        report = format_supply_chain_report(result)
        with open(date_dir / "supply_chain.md", "w") as f:
            f.write(report)
        print(f"\n✅ 已保存到 {date_dir}/")


if __name__ == "__main__":
    main()
