#!/usr/bin/env python3
"""
Phase 2a/2c: 双轮因子筛选管线

对给定候选池做因子打分+风控过滤，支持双模式:
  Mode A "trend"   — 主线直接股: 偏动量/技术形态/反转 (跟趋势)
  Mode B "value"   — 瓶颈卡点股: 偏质量/低波/价值     (找价值)

复用 quant 现有因子体系 (factor_library, indicator_engine, market_diagnostic...)

用法:
  python3 scripts/dual_selection.py --codes 600519,000858,300750 --mode trend
  python3 scripts/dual_selection.py --theme "AI数据中心电源" --mode both
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / "output"
sys.path.insert(0, str(ROOT / "scripts"))

from data_loader import (
    ensure_dirs, load_universe_klines, load_fundamentals_for_codes,
    load_news_sector_data, map_stocks_to_sectors, build_stock_universe,
    get_eastmoney_sector_map, tencent_quote, batch_fetch_klines,
)
from market_diagnostic import diagnose_market, select_regime_weights
from factor_library import (
    compute_factors, compute_factors_batch,
    process_factor_panel, compute_composite_score,
    load_factor_config, load_weights_config,
)
from risk_controller import apply_risk_controls

# ============================================================
# 双模式因子权重
# ============================================================

# Mode A: 主线直接 — 跟趋势，重动量+技术形态
TREND_WEIGHTS = {
    "momentum":     0.20,  # 动量因子 ↑
    "reversal":     0.15,
    "technical":    0.20,  # 技术形态 ↑
    "lowvol":       0.04,  # 低波 ↓
    "quality":      0.08,
    "north":        0.12,
    "money_flow":   0.12,
    "value":        0.03,  # 价值 ↓
    "analyst":      0.06,
}

# Mode B: 瓶颈卡点 — 找价值，重质量+低波+价值
VALUE_WEIGHTS = {
    "momentum":     0.08,  # 动量 ↓
    "reversal":     0.12,
    "technical":    0.10,  # 技术形态 ↓
    "lowvol":       0.22,  # 低波 ↑↑
    "quality":      0.18,  # 质量 ↑↑
    "north":        0.12,
    "money_flow":   0.05,  # 资金流 ↓
    "value":        0.08,  # 价值 ↑
    "analyst":      0.05,
}

# 市场环境 → 模式选择建议
def suggest_mode(regime: dict) -> str:
    """根据市场环境建议选股模式"""
    temp = regime.get("temperature", 50)
    vol_regime = regime.get("vol_regime", "normal")

    if temp > 65 and vol_regime != "high":
        return "trend"   # 强势市场用趋势模式
    elif temp < 35:
        return "value"   # 弱势市场用价值防御模式
    else:
        return "both"    # 中性市场双管齐下


# ============================================================
# 候选池数据准备
# ============================================================

def prepare_candidate_pool(codes: list[str],
                           refresh: bool = False) -> tuple[dict, dict, dict]:
    """
    为候选股票池准备数据: K线 + 基本面 + 板块分类。

    Returns: (kline_map, fundamentals_map, sector_map)
    """
    print(f"\n  准备 {len(codes)} 只候选股数据...")

    # 1. K 线（缓存在 data/stocks/）
    kline_map = {}
    for code in codes:
        market = 1 if code.startswith("6") else 0
        from data_loader import get_stock_kline
        df = get_stock_kline(code, market, refresh=refresh)
        if not df.empty:
            kline_map[code] = df

    print(f"    K线: {len(kline_map)}/{len(codes)} 只有数据")

    # 2. 基本面（腾讯财经，不封IP）
    fundamentals = load_fundamentals_for_codes(list(kline_map.keys()), refresh=refresh)
    print(f"    基本面: {len(fundamentals)} 只")

    # 3. 板块分类
    sector_map = get_eastmoney_sector_map(list(kline_map.keys()))
    print(f"    板块分类: {len(sector_map)} 只")

    return kline_map, fundamentals, sector_map


# ============================================================
# 因子筛选引擎
# ============================================================

def screen_candidates(kline_map: dict,
                      fundamentals: dict,
                      sector_map: dict,
                      mode: str = "trend",
                      top_n: int = 10,
                      regime_override: dict = None) -> pd.DataFrame:
    """
    对候选池做因子筛选+风控过滤。

    Args:
        kline_map: {code: DataFrame(OHLCV)}
        fundamentals: {code: {"pe": ..., "pb": ..., "roe": ..., ...}}
        sector_map: {code: {"primary_sector": ..., "sectors": [...]}}
        mode: "trend" | "value"
        top_n: 返回前N只
        regime_override: 市场环境（可选，否则自动诊断）

    Returns:
        DataFrame with columns: code, name, score, sector, pe, pb, change_pct, ...
    """
    codes = list(kline_map.keys())
    if not codes:
        return pd.DataFrame()

    print(f"\n  [{mode.upper()} 模式] 因子筛选 {len(codes)} 只候选...")

    # 1. 市场环境诊断
    if regime_override:
        regime = regime_override
        print(f"    使用给定市场环境: 温度={regime.get('temperature', '?')}")
    else:
        # 加载指数K线做诊断
        from data_loader import load_index_klines
        indices = load_index_klines()
        idx_df = indices.get("1000300") or indices.get("1000001")  # 沪深300/上证
        if idx_df is not None and not idx_df.empty:
            regime = diagnose_market(idx_df, kline_map)
            print(f"    市场温度: {regime.get('temperature', '?')}")
        else:
            regime = {"temperature": 50, "regime": "neutral"}
            print(f"    无指数数据，使用默认温度=50")

    # 2. 选择因子权重
    if mode == "trend":
        weights = TREND_WEIGHTS
    elif mode == "value":
        weights = VALUE_WEIGHTS
    else:
        weights = TREND_WEIGHTS  # default

    # 3. 计算因子值
    print(f"    计算因子...")
    try:
        factor_scores = _compute_factor_scores(
            codes, kline_map, fundamentals, sector_map, weights, regime
        )
    except Exception as e:
        print(f"    [WARN] 因子计算异常: {e}")
        # Fallback: 简单排序（按涨跌幅+PE）
        factor_scores = _simple_scores(codes, kline_map, fundamentals)

    # 4. 风控过滤
    print(f"    风控过滤...")
    passed = _apply_risk_filter(factor_scores, fundamentals, regime)

    # 5. 排序输出
    passed.sort(key=lambda x: x.get("score", 0), reverse=True)

    df = pd.DataFrame(passed[:top_n])
    if not df.empty:
        df = df.rename(columns={
            "code": "代码", "name": "名称", "score": "综合得分",
            "primary_sector": "板块", "price": "价格",
            "change_pct": "涨跌幅%", "pe": "PE", "pb": "PB",
        })

    return df


def _compute_factor_scores(codes, kline_map, fundamentals, sector_map, weights, regime):
    """使用 quant 因子库计算综合得分（简化版）"""
    results = []

    # 尝试加载完整因子系统
    try:
        factor_config = load_factor_config()
    except Exception:
        factor_config = None

    for code in codes:
        df = kline_map.get(code)
        if df is None or df.empty or len(df) < 30:
            continue

        fund = fundamentals.get(code, {})
        sector = sector_map.get(code, {})

        # 计算技术指标（精简版：直接算关键信号）
        score = _compute_simplified_score(df, fund, sector, weights, regime)

        results.append({
            "code": code,
            "name": fund.get("name", ""),
            "score": round(score, 1),
            "primary_sector": sector.get("primary_sector", "未分类"),
            "price": float(fund.get("price", 0) or 0),
            "pe": float(fund.get("pe", 0) or 0),
            "pb": float(fund.get("pb", 0) or 0),
            "change_pct": float(fund.get("change_pct", 0) or 0),
            "market_cap": float(fund.get("market_cap", 0) or 0) / 1e8,
            "turnover_pct": float(fund.get("turnover_pct", 0) or 0),
            "roe": float(fund.get("roe", 0) or 0),
            "revenue_yoy": float(fund.get("revenue_yoy", 0) or 0),
        })

    return results


def _compute_simplified_score(df: pd.DataFrame,
                               fund: dict,
                               sector: dict,
                               weights: dict,
                               regime: dict) -> float:
    """精简版因子打分（当完整因子库不可用时使用）"""
    if df.empty or len(df) < 30:
        return 0.0

    c = df["close"].values
    v = df["volume"].values
    latest = c[-1]
    score = 50.0  # 基准分

    # ---- 动量因子 (20%) ----
    try:
        ret_20d = (c[-1] - c[-20]) / c[-20] * 100
        mom_z = min(max(ret_20d / 10, -3), 3)  # 截断
        score += mom_z * 20 * weights.get("momentum", 0.15)
    except Exception:
        pass

    # ---- 反转因子 (15%) ----
    try:
        ret_5d = (c[-1] - c[-5]) / c[-5] * 100
        rev_z = min(max(-ret_5d / 5, -3), 3)
        score += rev_z * 15 * weights.get("reversal", 0.15)
    except Exception:
        pass

    # ---- 技术形态因子 (20%) ----
    try:
        ma20 = np.mean(c[-20:])
        vol_ratio = v[-1] / np.mean(v[-20:]) if np.mean(v[-20:]) > 0 else 1
        # 突破信号
        if latest > ma20 and vol_ratio > 1.5:
            score += 15 * weights.get("technical", 0.15)
        elif latest > ma20:
            score += 8 * weights.get("technical", 0.15)
        # 均线多头排列
        ma5 = np.mean(c[-5:])
        ma10 = np.mean(c[-10:])
        if ma5 > ma10 > ma20:
            score += 10 * weights.get("technical", 0.15)
    except Exception:
        pass

    # ---- 低波因子 (12%) ----
    try:
        rets = (c[1:] - c[:-1]) / c[:-1]
        vol_60d = np.std(rets[-60:]) * 100 if len(rets) >= 60 else np.std(rets) * 100
        vol_score = -min(max(vol_60d / 2 - 1, -3), 3)  # 低波加分
        score += vol_score * 12 * weights.get("lowvol", 0.12)
    except Exception:
        pass

    # ---- 质量因子 (10%) ----
    roe = float(fund.get("roe", 0) or 0)
    if roe > 15:
        score += 10 * weights.get("quality", 0.10)
    elif roe > 8:
        score += 5 * weights.get("quality", 0.10)
    elif roe < 0:
        score -= 5 * weights.get("quality", 0.10)

    # ---- 价值因子 (5%) ----
    pe = float(fund.get("pe", 0) or 0)
    if 0 < pe < 20:
        score += 5 * weights.get("value", 0.05)
    elif 0 < pe < 30:
        score += 2 * weights.get("value", 0.05)
    elif pe > 100:
        score -= 3 * weights.get("value", 0.05)

    # ---- 北向资金 (12%) ----
    # 使用市场级北向数据（简化）
    nb_signal = fund.get("north_signal", 0) if "north_signal" in fund else 0
    score += nb_signal * 10 * weights.get("north", 0.12)

    # ---- 资金流因子 (8%) ----
    turnover = float(fund.get("turnover_pct", 0) or 0)
    if 2 < turnover < 10:
        score += 5 * weights.get("money_flow", 0.08)
    elif turnover > 20:  # 异常高换手，减分
        score -= 3 * weights.get("money_flow", 0.08)

    # ---- 市场温度调整 ----
    temp = regime.get("temperature", 50)
    temp_mult = temp / 50  # 温度系数
    score = score * temp_mult

    return round(score, 1)


def _simple_scores(codes, kline_map, fundamentals) -> list[dict]:
    """简单排序 fallback：按最近涨跌幅+PE"""
    results = []
    for code in codes:
        fund = fundamentals.get(code, {})
        df = kline_map.get(code)
        chg = 0
        if df is not None and not df.empty and len(df) >= 5:
            c = df["close"].values
            chg = (c[-1] - c[-5]) / c[-5] * 100

        pe = float(fund.get("pe", 0) or 0)
        score = 50 + chg * 2 + (5 if 0 < pe < 30 else 0)

        results.append({
            "code": code,
            "name": fund.get("name", ""),
            "score": round(score, 1),
            "primary_sector": "未分类",
            "price": fund.get("price"),
            "pe": pe,
            "pb": fund.get("pb"),
            "change_pct": fund.get("change_pct"),
            "market_cap": (fund.get("market_cap") or 0) / 1e8,
        })

    results.sort(key=lambda x: x["score"], reverse=True)
    return results


def _apply_risk_filter(scores: list[dict],
                       fundamentals: dict,
                       regime: dict) -> list[dict]:
    """简单风控过滤"""
    passed = []
    for s in scores:
        code = s["code"]
        fund = fundamentals.get(code, {})

        # ST 过滤
        name = fund.get("name", "")
        if "ST" in name or "*ST" in name:
            continue

        # 日成交额过滤（< 3000万剔除）
        turnover = float(fund.get("turnover_pct", 0) or 0)
        mcap = float(fund.get("circ_market_cap", 0) or 0)
        daily_amount = mcap * turnover / 100 if mcap and turnover else None
        if daily_amount is not None and daily_amount < 3000 * 1e4:  # < 3000万
            continue

        # PE 极端值
        pe = float(fund.get("pe", 0) or 0)
        if pe < 0 or pe > 500:
            s["score"] = max(s["score"] - 15, 0)

        # 高质押风险
        debt_ratio = float(fund.get("debt_to_equity", 0) or 0)
        if debt_ratio > 90:
            s["score"] = max(s["score"] - 10, 0)

        passed.append(s)

    return passed


# ============================================================
# 双轮对比分析
# ============================================================

def compare_ab(result_a: pd.DataFrame, result_b: pd.DataFrame) -> dict:
    """对比 A/B 两组选股结果"""
    codes_a = set(result_a["代码"].tolist()) if not result_a.empty else set()
    codes_b = set(result_b["代码"].tolist()) if not result_b.empty else set()

    intersection = codes_a & codes_b
    a_only = codes_a - codes_b
    b_only = codes_b - codes_a

    # 详细差异
    def _get_info(df, code_col="代码"):
        if df.empty:
            return {}
        return {row[code_col]: {
            "name": row.get("名称", ""),
            "score": row.get("综合得分", ""),
            "sector": row.get("板块", ""),
        } for _, row in df.iterrows()}

    info_a = _get_info(result_a)
    info_b = _get_info(result_b)

    return {
        "total_a": len(codes_a),
        "total_b": len(codes_b),
        "intersection_count": len(intersection),
        "a_only_count": len(a_only),
        "b_only_count": len(b_only),
        "intersection": [{"code": c, **info_a.get(c, {})} for c in intersection],
        "a_only": [{"code": c, **info_a.get(c, {})} for c in a_only],
        "b_only": [{"code": c, **info_b.get(c, {})} for c in b_only],
    }


# ============================================================
# 主入口
# ============================================================

def run_dual_selection(codes: list[str],
                       mode: str = "both",
                       top_n: int = 10,
                       refresh: bool = False) -> dict:
    """
    主函数: 对候选池执行双轮/单轮因子筛选。

    Returns:
        {
            "date": "YYYY-MM-DD",
            "mode": "both"|"trend"|"value",
            "market_regime": {...},
            "result_a": DataFrame (trend),
            "result_b": DataFrame (value),
            "comparison": {...}  (仅当 mode="both")
        }
    """
    today = datetime.now().strftime("%Y-%m-%d")
    print(f"\n{'='*60}")
    print(f"  双轮因子筛选管线 — {today}")
    print(f"  候选池: {len(codes)} 只 | 模式: {mode}")
    print(f"{'='*60}")

    # 准备数据
    kline_map, fundamentals, sector_map = prepare_candidate_pool(codes, refresh=refresh)

    if not kline_map:
        print("  [FAIL] 无有效K线数据")
        return {"date": today, "error": "无有效K线数据"}

    # 市场环境诊断
    from data_loader import load_index_klines
    indices = load_index_klines()
    idx_df = indices.get("1000300")
    if idx_df is None or (hasattr(idx_df, 'empty') and idx_df.empty):
        idx_df = indices.get("1000001")
    regime = {"temperature": 50, "regime": "normal", "vol_regime": "normal"}
    if idx_df is not None and isinstance(idx_df, pd.DataFrame) and not idx_df.empty:
        try:
            regime = diagnose_market(idx_df, kline_map)
        except Exception:
            pass

    result = {
        "date": today,
        "mode": mode,
        "market_regime": regime,
        "suggested_mode": suggest_mode(regime),
        "candidate_count": len(codes),
        "valid_count": len(kline_map),
    }

    if mode in ("trend", "both"):
        print(f"\n  >>> 第一轮: 主线直接选股 (Trend)")
        df_a = screen_candidates(kline_map, fundamentals, sector_map,
                                 mode="trend", top_n=top_n, regime_override=regime)
        result["result_a"] = df_a.to_dict(orient="records") if not df_a.empty else []
        result["result_a_count"] = len(df_a)

    if mode in ("value", "both"):
        print(f"\n  >>> 第二轮: 瓶颈卡点选股 (Value)")
        df_b = screen_candidates(kline_map, fundamentals, sector_map,
                                 mode="value", top_n=top_n, regime_override=regime)
        result["result_b"] = df_b.to_dict(orient="records") if not df_b.empty else []
        result["result_b_count"] = len(df_b)

    if mode == "both":
        df_a = pd.DataFrame(result.get("result_a", []))
        df_b = pd.DataFrame(result.get("result_b", []))
        comparison = compare_ab(df_a, df_b)
        result["comparison"] = comparison

    return result


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="双轮因子筛选管线")
    parser.add_argument("--codes", type=str,
                        help="逗号分隔的A股代码列表")
    parser.add_argument("--theme", type=str,
                        help="主线名称（自动发现成分股）")
    parser.add_argument("--mode", choices=["trend", "value", "both"],
                        default="both", help="选股模式")
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--output", choices=["text", "json"], default="text")
    parser.add_argument("--save", action="store_true")
    args = parser.parse_args()

    ensure_dirs()

    # 获取候选股代码
    codes = []
    if args.codes:
        codes = [c.strip() for c in args.codes.split(",") if c.strip()]
    elif args.theme:
        from supply_chain_mapper import map_supply_chain
        chain_result = map_supply_chain(args.theme)
        for layer_key, bl in chain_result.get("bottleneck_layers", {}).items():
            for s in bl.get("stocks", []):
                if s["code"] not in codes:
                    codes.append(s["code"])
        if not codes:
            print(f"  未找到主题「{args.theme}」的候选股")
            sys.exit(1)
        print(f"  从主题「{args.theme}」获取 {len(codes)} 只候选")
    else:
        # 默认: 使用 quant 项目的 watchlist 全量
        universe = build_stock_universe(watchlist_only=True)
        codes = [s["code"] for s in universe]
        print(f"  使用 watchlist 全量: {len(codes)} 只")

    # 执行筛选
    result = run_dual_selection(codes, mode=args.mode, top_n=args.top_n)

    if args.output == "json":
        print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    else:
        # 文本输出
        print(f"\n  市场环境: 温度={result.get('market_regime',{}).get('temperature','?')} "
              f"建议模式={result.get('suggested_mode','?')}")

        if "result_a" in result:
            print(f"\n  === A组: 主线直接标的 (Trend) ===")
            for i, s in enumerate(result.get("result_a", [])[:10], 1):
                print(f"  A{i}. {s.get('code')} {s.get('name','')} "
                      f"得分={s.get('综合得分','')} PE={s.get('PE','')} PB={s.get('PB','')}")

        if "result_b" in result:
            print(f"\n  === B组: 瓶颈卡点标的 (Value) ===")
            for i, s in enumerate(result.get("result_b", [])[:10], 1):
                print(f"  B{i}. {s.get('code')} {s.get('name','')} "
                      f"得分={s.get('综合得分','')} PE={s.get('PE','')} PB={s.get('PB','')}")

        comp = result.get("comparison", {})
        if comp:
            print(f"\n  === A/B 对比 ===")
            print(f"  A组: {comp.get('total_a')} 只 | B组: {comp.get('total_b')} 只 "
                  f"| 交集: {comp.get('intersection_count')} 只")
            if comp.get("a_only_count"):
                print(f"  A有B无: {comp['a_only_count']} 只")
            if comp.get("b_only_count"):
                print(f"  B有A无: {comp['b_only_count']} 只")

    if args.save and "error" not in result:
        date_dir = OUTPUT_DIR / result["date"]
        date_dir.mkdir(parents=True, exist_ok=True)
        with open(date_dir / "dual_selection.json", "w") as f:
            json.dump(result, f, ensure_ascii=False, indent=2, default=str)
        print(f"\n✅ 已保存到 {date_dir}/")


if __name__ == "__main__":
    main()
