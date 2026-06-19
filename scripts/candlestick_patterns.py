#!/usr/bin/env python3
"""
K线形态识别库 — 经典日本蜡烛图形态 + A股实战调整

识别 16 种核心形态，输出结构化信号和综合评分。
纯向量化计算，不做逐行循环（除必要的滚动窗口识别外）。

用法:
  from candlestick_patterns import identify_all_patterns, PatternResult
  result = identify_all_patterns(df)
  print(f"最新信号: {result.latest_signal}, 形态分: {result.pattern_score}")
"""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class PatternHit:
    """单次形态命中"""
    pattern: str
    date: str
    strength: float       # 0.0 ~ 1.0
    direction: str        # "bullish" | "bearish"
    category: str         # "reversal" | "continuation"
    days_ago: int         # 0 = 最新一期
    description: str


@dataclass
class PatternResult:
    """形态识别完整结果"""
    ticker: str = ""
    date: str = ""
    bullish_reversal: list[PatternHit] = field(default_factory=list)
    bearish_reversal: list[PatternHit] = field(default_factory=list)
    continuation: list[PatternHit] = field(default_factory=list)
    latest_signal: str = "neutral"  # "bullish" | "bearish" | "neutral"
    pattern_score: float = 0.0       # -100 ~ 100
    active_patterns: list[str] = field(default_factory=list)
    summary: str = ""


def identify_all_patterns(df: pd.DataFrame,
                           ticker: str = "",
                           lookback_days: int = 60) -> PatternResult:
    """
    主入口: 对 OHLCV DataFrame 做全量形态识别。

    Args:
        df: 含 [date, open, high, low, close, volume] 的 DataFrame
        ticker: 股票代码
        lookback_days: 回溯多少天（默认60）

    Returns:
        PatternResult 结构体
    """
    if df.empty or len(df) < 5:
        return PatternResult(ticker=ticker, summary="数据不足")

    df = df.tail(lookback_days).copy().reset_index(drop=True)
    o = df["open"].values.astype(float)
    h = df["high"].values.astype(float)
    l = df["low"].values.astype(float)
    c = df["close"].values.astype(float)
    v = df["volume"].values.astype(float)
    dates = df["date"].astype(str).values
    n = len(c)

    all_hits: list[PatternHit] = []

    # 趋势判断辅助
    def _is_downtrend(i, window=10):
        if i < window:
            return False
        return c[i] < np.mean(c[i - window:i])

    def _is_uptrend(i, window=10):
        if i < window:
            return False
        return c[i] > np.mean(c[i - window:i])

    # ================================================================
    # 看多反转形态
    # ================================================================

    for i in range(20, n):
        days_ago = n - 1 - i

        # --- 锤子线 (Hammer) ---
        if _is_downtrend(i):
            body = abs(c[i] - o[i])
            lower_shadow = min(o[i], c[i]) - l[i]
            upper_shadow = h[i] - max(o[i], c[i])
            real_body = c[i] - o[i]  # positive=阳, negative=阴

            if body > 0 and lower_shadow >= 2 * body and upper_shadow <= 0.3 * body:
                strength = min(lower_shadow / (2 * body), 3.0) / 3.0
                # 阳锤子 > 阴锤子
                if real_body > 0:
                    strength = min(strength * 1.2, 1.0)
                all_hits.append(PatternHit(
                    pattern="锤子线", date=dates[i], strength=round(strength, 2),
                    direction="bullish", category="reversal", days_ago=days_ago,
                    description=f"底部锤子线，下影线={lower_shadow/body:.1f}倍实体"
                ))

        # --- 倒锤子 (Inverted Hammer) ---
        if _is_downtrend(i):
            body = abs(c[i] - o[i])
            upper_shadow = h[i] - max(o[i], c[i])
            lower_shadow = min(o[i], c[i]) - l[i]
            if body > 0 and upper_shadow >= 2 * body and lower_shadow <= 0.3 * body:
                # 需要次日阳线确认
                if i + 1 < n and c[i + 1] > o[i + 1] and c[i + 1] > c[i]:
                    strength = min(upper_shadow / (2 * body), 3.0) / 3.0 * 1.1
                    all_hits.append(PatternHit(
                        pattern="倒锤子(已确认)", date=dates[i], strength=round(strength, 2),
                        direction="bullish", category="reversal", days_ago=days_ago - 1,
                        description=f"倒锤子线，次日阳线确认"
                    ))

        # --- 看涨吞没 (Bullish Engulfing) ---
        if i >= 1 and _is_downtrend(i - 1):
            prev_body = c[i - 1] - o[i - 1]  # 前阴
            curr_body = c[i] - o[i]
            if prev_body < 0 and curr_body > 0:
                if o[i] <= c[i - 1] and c[i] >= o[i - 1]:
                    engulf_ratio = abs(curr_body) / max(abs(prev_body), 0.001)
                    strength = min(engulf_ratio / 2.0, 1.0)
                    all_hits.append(PatternHit(
                        pattern="看涨吞没", date=dates[i], strength=round(strength, 2),
                        direction="bullish", category="reversal", days_ago=days_ago,
                        description=f"阳包阴，吞没比例={engulf_ratio:.1f}x"
                    ))

        # --- 穿刺线 (Piercing Line) ---
        if i >= 1 and _is_downtrend(i - 1):
            prev_body = c[i - 1] - o[i - 1]
            curr_body = c[i] - o[i]
            if prev_body < 0 and curr_body > 0:
                if o[i] < l[i - 1] and c[i] > (o[i - 1] + c[i - 1]) / 2:
                    penetration = (c[i] - (o[i - 1] + c[i - 1]) / 2) / max(abs(prev_body), 0.001)
                    strength = min(0.5 + penetration * 2, 1.0)
                    all_hits.append(PatternHit(
                        pattern="穿刺线", date=dates[i], strength=round(strength, 2),
                        direction="bullish", category="reversal", days_ago=days_ago,
                        description=f"低开高走穿刺前阴50%以上"
                    ))

        # --- 启明星 (Morning Star) ---
        if i >= 2 and _is_downtrend(i - 2):
            body1 = c[i - 2] - o[i - 2]       # 长阴
            body2 = abs(c[i - 1] - o[i - 1])   # 小实体星
            body3 = c[i] - o[i]                # 长阳
            avg_body = np.mean([abs(c[j] - o[j]) for j in range(max(0, i - 15), i)])
            if body1 < 0 and abs(body1) > avg_body * 0.8:
                if body2 < avg_body * 0.5:
                    if body3 > 0 and abs(body3) > avg_body * 0.6:
                        if c[i] > (o[i - 2] + c[i - 2]) / 2:
                            strength = min(abs(body3) / max(abs(body1), 0.001), 2.0) / 2.0
                            all_hits.append(PatternHit(
                                pattern="启明星", date=dates[i], strength=round(strength, 2),
                                direction="bullish", category="reversal", days_ago=days_ago,
                                description="三K线启明星形态，经典底部反转"
                            ))

        # --- 三白兵 (Three White Soldiers) ---
        if i >= 2:
            b1 = c[i - 2] - o[i - 2]
            b2 = c[i - 1] - o[i - 1]
            b3 = c[i] - o[i]
            if b1 > 0 and b2 > 0 and b3 > 0:
                if c[i - 2] > c[i - 3] and c[i - 1] > c[i - 2] and c[i] > c[i - 1]:
                    if abs(b2) >= abs(b1) * 0.7 and abs(b3) >= abs(b2) * 0.7:
                        strength = 0.8
                        if abs(b1) < abs(b2) < abs(b3):
                            strength = 0.95
                        # A股调整：需要量能配合
                        if v[i] > np.mean(v[max(0, i - 5):i]):
                            strength = min(strength * 1.1, 1.0)
                        all_hits.append(PatternHit(
                            pattern="三白兵", date=dates[i], strength=round(strength, 2),
                            direction="bullish", category="continuation", days_ago=days_ago,
                            description="连续三阳线，多头强势推进"
                        ))

    # ================================================================
    # 看空反转形态
    # ================================================================

    for i in range(20, n):
        days_ago = n - 1 - i

        # --- 吊颈线 (Hanging Man) ---
        if _is_uptrend(i):
            body = abs(c[i] - o[i])
            lower_shadow = min(o[i], c[i]) - l[i]
            upper_shadow = h[i] - max(o[i], c[i])
            if body > 0 and lower_shadow >= 2 * body and upper_shadow <= 0.3 * body:
                strength = min(lower_shadow / (2 * body), 3.0) / 3.0
                all_hits.append(PatternHit(
                    pattern="吊颈线", date=dates[i], strength=round(strength, 2),
                    direction="bearish", category="reversal", days_ago=days_ago,
                    description=f"高位吊颈线，下影线={lower_shadow/body:.1f}倍实体，需警惕"
                ))

        # --- 射击之星 (Shooting Star) ---
        if _is_uptrend(i):
            body = abs(c[i] - o[i])
            upper_shadow = h[i] - max(o[i], c[i])
            lower_shadow = min(o[i], c[i]) - l[i]
            if body > 0 and upper_shadow >= 2 * body and lower_shadow <= 0.3 * body:
                # 实体越小越危险
                strength = min(upper_shadow / (2 * body), 3.0) / 3.0
                if body < abs(c[i] - o[i]) * 0.3:
                    strength = min(strength * 1.2, 1.0)
                all_hits.append(PatternHit(
                    pattern="射击之星", date=dates[i], strength=round(strength, 2),
                    direction="bearish", category="reversal", days_ago=days_ago,
                    description=f"高位射击之星，上影线={upper_shadow/body:.1f}倍实体"
                ))

        # --- 看跌吞没 (Bearish Engulfing) ---
        if i >= 1 and _is_uptrend(i - 1):
            prev_body = c[i - 1] - o[i - 1]   # 前阳
            curr_body = c[i] - o[i]            # 后阴
            if prev_body > 0 and curr_body < 0:
                if o[i] >= c[i - 1] and c[i] <= o[i - 1]:
                    engulf_ratio = abs(curr_body) / max(abs(prev_body), 0.001)
                    strength = min(engulf_ratio / 2.0, 1.0)
                    all_hits.append(PatternHit(
                        pattern="看跌吞没", date=dates[i], strength=round(strength, 2),
                        direction="bearish", category="reversal", days_ago=days_ago,
                        description=f"阴包阳，吞没比例={engulf_ratio:.1f}x"
                    ))

        # --- 乌云盖顶 (Dark Cloud Cover) ---
        if i >= 1 and _is_uptrend(i - 1):
            prev_body = c[i - 1] - o[i - 1]
            curr_body = c[i] - o[i]
            if prev_body > 0 and curr_body < 0:
                if o[i] > h[i - 1] and c[i] < (o[i - 1] + c[i - 1]) / 2:
                    penetration = (c[i] - (o[i - 1] + c[i - 1]) / 2) / max(abs(prev_body), 0.001)
                    strength = min(0.5 + abs(penetration) * 2, 1.0)
                    all_hits.append(PatternHit(
                        pattern="乌云盖顶", date=dates[i], strength=round(strength, 2),
                        direction="bearish", category="reversal", days_ago=days_ago,
                        description=f"高开低走盖过前阳50%以上"
                    ))

        # --- 黄昏星 (Evening Star) ---
        if i >= 2 and _is_uptrend(i - 2):
            body1 = c[i - 2] - o[i - 2]       # 长阳
            body2 = abs(c[i - 1] - o[i - 1])   # 小实体星
            body3 = c[i] - o[i]                # 长阴
            avg_body = np.mean([abs(c[j] - o[j]) for j in range(max(0, i - 15), i)])
            if body1 > 0 and abs(body1) > avg_body * 0.8:
                if body2 < avg_body * 0.5:
                    if body3 < 0 and abs(body3) > avg_body * 0.6:
                        if c[i] < (o[i - 2] + c[i - 2]) / 2:
                            strength = min(abs(body3) / max(abs(body1), 0.001), 2.0) / 2.0
                            all_hits.append(PatternHit(
                                pattern="黄昏星", date=dates[i], strength=round(strength, 2),
                                direction="bearish", category="reversal", days_ago=days_ago,
                                description="三K线黄昏星形态，经典顶部反转"
                            ))

        # --- 三乌鸦 (Three Black Crows) ---
        if i >= 2:
            b1 = c[i - 2] - o[i - 2]
            b2 = c[i - 1] - o[i - 1]
            b3 = c[i] - o[i]
            if b1 < 0 and b2 < 0 and b3 < 0:
                if c[i - 2] < c[i - 3] and c[i - 1] < c[i - 2] and c[i] < c[i - 1]:
                    if abs(b2) >= abs(b1) * 0.7 and abs(b3) >= abs(b2) * 0.7:
                        strength = 0.8
                        if v[i] > np.mean(v[max(0, i - 5):i]):
                            strength = min(strength * 1.1, 1.0)
                        all_hits.append(PatternHit(
                            pattern="三乌鸦", date=dates[i], strength=round(strength, 2),
                            direction="bearish", category="reversal", days_ago=days_ago,
                            description="连续三阴线，空头强势推进"
                        ))

    # ================================================================
    # 持续形态（A股重点：平台突破）
    # ================================================================

    # --- 横盘整理后放量突破 (A股高频形态) ---
    for i in range(20, n):
        days_ago = n - 1 - i
        # 找前 15-5 天是否横盘
        lookback = min(15, i - 5)
        if lookback < 5:
            continue
        seg = c[i - lookback:i - 3]
        if len(seg) < 5:
            continue
        seg_range = (np.max(seg) - np.min(seg)) / np.mean(seg)
        if seg_range < 0.08:  # 振幅 < 8%
            # 今日突破
            if c[i] > np.max(seg) * 1.005:
                vol_ratio = v[i] / max(np.mean(v[i - lookback:i - 3]), 0.001)
                if vol_ratio > 1.5:
                    strength = min(vol_ratio / 3.0, 1.0)
                    consolidation_days = lookback
                    all_hits.append(PatternHit(
                        pattern="横盘突破", date=dates[i], strength=round(strength, 2),
                        direction="bullish", category="continuation", days_ago=days_ago,
                        description=f"横盘{consolidation_days}日后放量突破，振幅{seg_range*100:.1f}%，量比{vol_ratio:.1f}"
                    ))

        # --- 下方破位 ---
        if seg_range < 0.08:
            if c[i] < np.min(seg) * 0.995:
                vol_ratio = v[i] / max(np.mean(v[i - lookback:i - 3]), 0.001)
                if vol_ratio > 1.3:
                    all_hits.append(PatternHit(
                        pattern="平台破位", date=dates[i], strength=min(vol_ratio / 3.0, 1.0),
                        direction="bearish", category="reversal", days_ago=days_ago,
                        description=f"横盘后放量破位，振幅{seg_range*100:.1f}%，量比{vol_ratio:.1f}"
                    ))

    # ================================================================
    # 综合评分
    # ================================================================

    # 只取最近 20 天的形态
    recent_hits = [h for h in all_hits if h.days_ago <= 20]

    bullish_hits = [h for h in recent_hits if h.direction == "bullish"]
    bearish_hits = [h for h in recent_hits if h.direction == "bearish"]

    # 加权：越近权重越大
    def _weighted_score(hits):
        return sum(h.strength * max(0, 1 - h.days_ago / 20) for h in hits)

    bull_score = _weighted_score(bullish_hits)
    bear_score = _weighted_score(bearish_hits)

    # 形态分归一化到 -100 ~ 100
    raw_score = (bull_score - bear_score) * 50
    pattern_score = round(max(-100, min(100, raw_score)), 1)

    # 最新信号
    if pattern_score > 20:
        latest_signal = "bullish"
    elif pattern_score < -20:
        latest_signal = "bearish"
    else:
        latest_signal = "neutral"

    # 活跃形态（最近5天）
    active = [h.pattern for h in recent_hits if h.days_ago <= 5]
    active_patterns = list(dict.fromkeys(active))[:5]  # 去重保序

    # 摘要
    summary_parts = []
    if active_patterns:
        summary_parts.append(f"活跃形态: {', '.join(active_patterns)}")
    if bullish_hits:
        top_bull = sorted(bullish_hits, key=lambda h: h.strength * max(0, 1 - h.days_ago / 20), reverse=True)[:2]
        summary_parts.append(f"看多: {', '.join(h.pattern for h in top_bull)}")
    if bearish_hits:
        top_bear = sorted(bearish_hits, key=lambda h: h.strength * max(0, 1 - h.days_ago / 20), reverse=True)[:2]
        summary_parts.append(f"看空: {', '.join(h.pattern for h in top_bear)}")
    if not summary_parts:
        summary_parts.append("近期无显著形态信号")

    return PatternResult(
        ticker=ticker,
        date=str(dates[-1]),
        bullish_reversal=[h for h in recent_hits if h.direction == "bullish" and h.category == "reversal"],
        bearish_reversal=[h for h in recent_hits if h.direction == "bearish" and h.category == "reversal"],
        continuation=[h for h in recent_hits if h.category == "continuation"],
        latest_signal=latest_signal,
        pattern_score=pattern_score,
        active_patterns=active_patterns,
        summary="; ".join(summary_parts),
    )


def pattern_result_to_dict(result: PatternResult) -> dict:
    """将 PatternResult 转为可序列化的 dict"""
    def _hit_to_dict(h: PatternHit) -> dict:
        return {
            "pattern": h.pattern,
            "date": h.date,
            "strength": h.strength,
            "direction": h.direction,
            "category": h.category,
            "days_ago": h.days_ago,
            "description": h.description,
        }

    return {
        "ticker": result.ticker,
        "date": result.date,
        "bullish_reversal": [_hit_to_dict(h) for h in result.bullish_reversal],
        "bearish_reversal": [_hit_to_dict(h) for h in result.bearish_reversal],
        "continuation": [_hit_to_dict(h) for h in result.continuation],
        "latest_signal": result.latest_signal,
        "pattern_score": result.pattern_score,
        "active_patterns": result.active_patterns,
        "summary": result.summary,
    }


# ============================================================
# CLI 测试
# ============================================================

if __name__ == "__main__":
    import sys
    from pathlib import Path
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
    print(f"  K线形态识别 — {ticker}")
    print(f"  K线数据: {len(df)} 条 ({df['date'].iloc[0]} ~ {df['date'].iloc[-1]})")
    print(f"{'='*60}")

    result = identify_all_patterns(df, ticker=ticker)

    print(f"\n📊 最新收盘: {df['close'].iloc[-1]:.2f}  ({df['date'].iloc[-1]})")
    print(f"📈 形态分: {result.pattern_score}  →  {result.latest_signal.upper()}")
    print(f"\n{result.summary}")

    if result.bullish_reversal:
        print(f"\n🟢 看多反转 ({len(result.bullish_reversal)}):")
        for h in sorted(result.bullish_reversal, key=lambda x: x.days_ago)[:5]:
            print(f"  [{h.days_ago}d前] {h.pattern} (强度:{h.strength:.2f}) — {h.description}")

    if result.bearish_reversal:
        print(f"\n🔴 看空反转 ({len(result.bearish_reversal)}):")
        for h in sorted(result.bearish_reversal, key=lambda x: x.days_ago)[:5]:
            print(f"  [{h.days_ago}d前] {h.pattern} (强度:{h.strength:.2f}) — {h.description}")

    if result.continuation:
        print(f"\n🔵 持续形态 ({len(result.continuation)}):")
        for h in sorted(result.continuation, key=lambda x: x.days_ago)[:5]:
            print(f"  [{h.days_ago}d前] {h.pattern} (强度:{h.strength:.2f}) — {h.description}")

    # JSON 输出用于集成
    import json
    print(f"\n--- JSON ---")
    print(json.dumps(pattern_result_to_dict(result), ensure_ascii=False, indent=2))
