#!/usr/bin/env python3
"""
量价关系分析器 — 量价配合、量堆识别、主力意图推断

核心思想：同样的量，在不同位置意义完全不同。
低位的放量是吸筹，高位的放量是出货。

分析维度：
  1. 量价配合/背离（日频）
  2. 关键量事件（地量/天量/倍量）
  3. 放量位置判断（低位/突破位/高位）
  4. 缩量意义（回调洗盘/无人接盘）
  5. 量堆识别（连续放量区域）

用法:
  from volume_price_analyzer import analyze_volume_price
  result = analyze_volume_price(df)
  print(f"量价分: {result['volume_price_score']}")
"""

import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


def analyze_volume_price(df: pd.DataFrame) -> dict:
    """
    主入口：对 OHLCV DataFrame 做完整量价关系分析。

    Args:
        df: 含 [date, open, high, low, close, volume, amount] 的 DataFrame
            至少需要 60 条数据

    Returns:
        {
            "volume_trend": "expanding" | "contracting" | "normal",
            "price_vol_relation": "healthy" | "divergent" | "neutral",
            "recent_signal": "accumulation" | "distribution" | "churning" | "neutral",
            "key_volume_events": [...],
            "volume_score": -100 ~ 100,
            "risk_flags": [...],
            "analysis_summary": "...",
        }
    """
    if df.empty or len(df) < 60:
        return _empty_result("数据不足 (< 60条)")

    df = df.copy().reset_index(drop=True)
    c = df["close"].values.astype(float)
    v = df["volume"].values.astype(float)
    o = df["open"].values.astype(float)
    h = df["high"].values.astype(float)
    l = df["low"].values.astype(float)
    dates = df["date"].astype(str).values
    n = len(c)

    score = 0.0
    risk_flags = []
    key_events = []

    # --- 基准量 ---
    ma5_v = pd.Series(v).rolling(5).mean().values
    ma20_v = pd.Series(v).rolling(20).mean().values
    ma60_v = pd.Series(v).rolling(60).mean().values

    latest_vol = v[-1]
    latest_close = c[-1]
    latest_change = (c[-1] - c[-2]) / c[-2] if n >= 2 else 0

    # ================================================================
    # 维度1: 日频量价配合（最近5天）
    # ================================================================

    healthy_days = 0
    divergent_days = 0
    for i in range(max(0, n - 5), n):
        if i < 1:
            continue
        chg = (c[i] - c[i - 1]) / c[i - 1]
        vol_chg = (v[i] - v[i - 1]) / max(v[i - 1], 1)

        # 价涨量增 → 健康
        if chg > 0.005 and vol_chg > 0.05:
            healthy_days += 1
        # 价涨量缩 → 背离（上涨乏力）
        elif chg > 0.01 and vol_chg < -0.1:
            divergent_days += 1
            if i == n - 1:
                score -= 10
                risk_flags.append("当日量价背离(涨缩量)")
        # 价跌量缩 → 正常回调
        elif chg < -0.005 and vol_chg < -0.05:
            healthy_days += 0.5
        # 价跌量增 → 放量下跌
        elif chg < -0.01 and vol_chg > 0.1:
            divergent_days += 1
            if i == n - 1:
                score -= 15
                risk_flags.append("当日放量下跌")

    if healthy_days >= 3:
        price_vol_relation = "healthy"
    elif divergent_days >= 2:
        price_vol_relation = "divergent"
    else:
        price_vol_relation = "neutral"

    # ================================================================
    # 维度2: 关键量事件
    # ================================================================

    # 地量（近60日最低量）
    if n >= 60:
        min_vol_60d = np.min(v[-60:])
        if latest_vol <= min_vol_60d * 1.05:
            score += 12
            key_events.append({
                "type": "volume_dry",
                "date": dates[-1],
                "desc": "近60日地量，抛压枯竭，关注变盘",
                "score_impact": 12,
            })

    # 天量（近60日最高量）
    if n >= 60:
        max_vol_60d = np.max(v[-60:])
        if latest_vol >= max_vol_60d * 0.95:
            # 判断位置
            ret_20d = (c[-1] - c[-21]) / c[-21] if n >= 21 else 0
            if ret_20d < -0.15:
                key_events.append({
                    "type": "volume_climax_bottom",
                    "date": dates[-1],
                    "desc": "低位天量，恐慌盘涌出，可能是底部信号",
                    "score_impact": 8,
                })
                score += 8
            elif ret_20d > 0.30:
                key_events.append({
                    "type": "volume_climax_top",
                    "date": dates[-1],
                    "desc": "高位天量，分歧极大，警惕见顶",
                    "score_impact": -12,
                })
                score -= 12
                risk_flags.append("高位天量⚠️")
            else:
                key_events.append({
                    "type": "volume_climax",
                    "date": dates[-1],
                    "desc": "近60日天量，关注后续方向确认",
                    "score_impact": 0,
                })

    # 倍量（量比 > 2）
    if n >= 20 and ma20_v[-1] > 0:
        vol_ratio = latest_vol / ma20_v[-1]
        if vol_ratio > 2.5:
            ret_20d = (c[-1] - c[-21]) / c[-21] if n >= 21 else 0
            if latest_change > 0.03:
                if ret_20d < 0:
                    score += 15
                    key_events.append({
                        "type": "volume_surge_buy",
                        "date": dates[-1],
                        "desc": f"低位倍量阳线(量比{vol_ratio:.1f})，主力吸筹迹象",
                        "score_impact": 15,
                    })
                else:
                    score += 5
                    key_events.append({
                        "type": "volume_surge",
                        "date": dates[-1],
                        "desc": f"倍量阳线(量比{vol_ratio:.1f})",
                        "score_impact": 5,
                    })
            elif latest_change < -0.02:
                score -= 15
                risk_flags.append("放量下跌⚠️")
                key_events.append({
                    "type": "volume_surge_sell",
                    "date": dates[-1],
                    "desc": f"放量下跌(量比{vol_ratio:.1f})，资金出逃",
                    "score_impact": -15,
                })

    # ================================================================
    # 维度3: 放量位置判断
    # ================================================================

    if n >= 120:
        # 相对位置: 当前价在120日内的百分位
        price_percentile = (latest_close - np.min(c[-120:])) / max(np.max(c[-120:]) - np.min(c[-120:]), 0.01)
        latest_vol_ratio = latest_vol / max(ma20_v[-1], 0.01)

        if latest_vol_ratio > 1.5:
            if price_percentile < 0.25:
                score += 10
                key_events.append({
                    "type": "low_position_volume",
                    "date": dates[-1],
                    "desc": f"低位放量(位置{price_percentile*100:.0f}%)，吸筹区间",
                    "score_impact": 10,
                })
            elif price_percentile > 0.80:
                score -= 8
                risk_flags.append(f"高位放量(位置{price_percentile*100:.0f}%)")
                key_events.append({
                    "type": "high_position_volume",
                    "date": dates[-1],
                    "desc": f"高位放量(位置{price_percentile*100:.0f}%)，警惕出货",
                    "score_impact": -8,
                })

    # ================================================================
    # 维度4: 缩量分析
    # ================================================================

    if n >= 10 and ma20_v[-1] > 0:
        vol_ratio = latest_vol / ma20_v[-1]
        if vol_ratio < 0.5:
            ret_5d = (c[-1] - c[-6]) / c[-6] if n >= 6 else 0
            if ret_5d > 0.02:
                score += 8
                key_events.append({
                    "type": "shrink_uptrend",
                    "date": dates[-1],
                    "desc": "上升途中缩量，筹码锁定良好",
                    "score_impact": 8,
                })
            elif -0.03 < ret_5d < 0:
                score += 10
                key_events.append({
                    "type": "shrink_pullback",
                    "date": dates[-1],
                    "desc": "缩量回调，洗盘特征，抛压轻",
                    "score_impact": 10,
                })
            elif ret_5d < -0.05:
                score -= 8
                risk_flags.append("缩量阴跌，无人接盘")
                key_events.append({
                    "type": "shrink_downtrend",
                    "date": dates[-1],
                    "desc": "缩量持续下跌，多头无力",
                    "score_impact": -8,
                })

    # ================================================================
    # 维度5: 量堆识别（连续放量区域）
    # ================================================================

    volume_piles = _detect_volume_piles(v, dates, c, n)
    for pile in volume_piles[:3]:
        score += pile.get("score_impact", 0)
        key_events.append(pile)

    # ================================================================
    # 量能趋势
    # ================================================================

    if n >= 20:
        recent_avg = np.mean(v[-5:])
        mid_avg = np.mean(v[-20:-5])
        if recent_avg > mid_avg * 1.3:
            volume_trend = "expanding"
        elif recent_avg < mid_avg * 0.7:
            volume_trend = "contracting"
        else:
            volume_trend = "normal"
    else:
        volume_trend = "normal"

    # ================================================================
    # 主力意图推断
    # ================================================================

    recent_signal = _infer_intent(v, c, n)

    # ================================================================
    # 综合评分
    # ================================================================

    volume_score = round(max(-100, min(100, score)), 1)

    # 摘要
    summary_parts = [f"量能趋势:{volume_trend}", f"量价关系:{price_vol_relation}"]
    if recent_signal != "neutral":
        summary_parts.append(f"主力意图:{recent_signal}")
    if risk_flags:
        summary_parts.append(f"风险:{', '.join(risk_flags[:2])}")
    if key_events:
        recent_event = key_events[-1]
        summary_parts.append(recent_event.get("desc", ""))

    return {
        "volume_trend": volume_trend,
        "price_vol_relation": price_vol_relation,
        "recent_signal": recent_signal,
        "key_volume_events": key_events,
        "volume_score": volume_score,
        "risk_flags": risk_flags,
        "analysis_summary": "; ".join(summary_parts),
    }


def _detect_volume_piles(v, dates, c, n) -> list[dict]:
    """检测连续放量区域（量堆）"""
    if n < 30:
        return []

    ma20 = pd.Series(v).rolling(20).mean().values
    piles = []

    in_pile = False
    pile_start = 0
    for i in range(20, n):
        above_ma = v[i] > ma20[i] * 1.2
        if above_ma and not in_pile:
            in_pile = True
            pile_start = i
        elif not above_ma and in_pile:
            in_pile = False
            duration = i - pile_start
            if duration >= 3:
                # 量堆位置判断
                seg_before = c[max(0, pile_start - 10):pile_start]
                if len(seg_before) > 0:
                    pile_pos = (c[pile_start] - np.min(seg_before)) / max(np.max(seg_before) - np.min(seg_before), 0.01)
                else:
                    pile_pos = 0.5

                if pile_pos < 0.3:
                    label = "低位量堆(吸筹)"
                    score_impact = 12
                elif pile_pos > 0.8:
                    label = "高位量堆(警惕)"
                    score_impact = -10
                else:
                    label = "中位量堆"
                    score_impact = 3

                piles.append({
                    "type": "volume_pile",
                    "date": f"{dates[pile_start]}~{dates[i - 1]}",
                    "desc": f"{label}，持续{duration}天放量",
                    "score_impact": score_impact,
                })

    # 如果还在量堆中
    if in_pile and n - pile_start >= 3:
        duration = n - pile_start
        seg_before = c[max(0, pile_start - 10):pile_start]
        if len(seg_before) > 0:
            pile_pos = (c[pile_start] - np.min(seg_before)) / max(np.max(seg_before) - np.min(seg_before), 0.01)
        else:
            pile_pos = 0.5

        if pile_pos < 0.3:
            label, score_impact = "低位量堆(吸筹中)", 12
        elif pile_pos > 0.8:
            label, score_impact = "高位量堆(警惕中)", -10
        else:
            label, score_impact = "中位量堆", 3

        piles.append({
            "type": "volume_pile",
            "date": f"{dates[pile_start]}~至今",
            "desc": f"{label}，已持续{duration}天放量",
            "score_impact": score_impact,
        })

    return piles


def _infer_intent(v, c, n) -> str:
    """推断主力意图"""
    if n < 20:
        return "neutral"

    # 最近 10 天
    seg_c = c[-11:]
    chgs = np.diff(seg_c) / seg_c[:-1]
    vols = v[-10:]

    up_vol_avg = np.mean(vols[chgs > 0]) if sum(chgs > 0) > 0 else 0
    down_vol_avg = np.mean(vols[chgs < 0]) if sum(chgs < 0) > 0 else 0

    if up_vol_avg > down_vol_avg * 1.5:
        return "accumulation"  # 上涨放量 → 吸筹
    elif down_vol_avg > up_vol_avg * 1.5:
        return "distribution"  # 下跌放量 → 出货
    elif abs(up_vol_avg - down_vol_avg) < down_vol_avg * 0.2:
        return "churning"      # 量能均衡 → 换手
    else:
        return "neutral"


def _empty_result(reason: str = "") -> dict:
    return {
        "volume_trend": "unknown",
        "price_vol_relation": "unknown",
        "recent_signal": "neutral",
        "key_volume_events": [],
        "volume_score": 0.0,
        "risk_flags": [],
        "analysis_summary": reason,
    }


# ============================================================
# CLI 测试
# ============================================================

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from data_loader import ensure_dirs, get_stock_kline

    ensure_dirs()

    ticker = sys.argv[1] if len(sys.argv) > 1 else "300750"
    market = 1 if ticker.startswith("6") else 0
    df = get_stock_kline(ticker, market, refresh=False)

    if df.empty:
        print(f"无{ticker}的K线数据")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"  量价关系分析 — {ticker}")
    print(f"  K线数据: {len(df)} 条 ({df['date'].iloc[0]} ~ {df['date'].iloc[-1]})")
    print(f"{'='*60}")

    result = analyze_volume_price(df)

    print(f"\n📊 量价分: {result['volume_score']}")
    print(f"📈 量能趋势: {result['volume_trend']}")
    print(f"📉 量价关系: {result['price_vol_relation']}")
    print(f"🎯 主力意图: {result['recent_signal']}")

    if result["risk_flags"]:
        print(f"\n⚠️ 风险标记: {', '.join(result['risk_flags'])}")

    if result["key_volume_events"]:
        print(f"\n📋 关键量事件:")
        for ev in result["key_volume_events"][-8:]:
            icon = "🟢" if ev.get("score_impact", 0) > 0 else "🔴" if ev.get("score_impact", 0) < 0 else "⚪"
            print(f"  {icon} [{ev['date']}] {ev['type']}: {ev['desc']}")

    print(f"\n📝 摘要: {result['analysis_summary']}")
    print(f"\n--- JSON ---")
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
