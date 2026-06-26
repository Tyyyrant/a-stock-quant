#!/usr/bin/env python3
"""
AB 价格行为引擎 — Al Brooks Price Action 裸K分析

7 个核心函数，纯裸K + EMA20，零外部依赖。
基于《高级趋势技术分析》— 急速与通道、信号K线、高/低点入场、高潮反转。

用法:
  from abprice import *
  result = is_always_in_long(df)
  spikes = identify_spike_channel(df)
"""

import numpy as np
import pandas as pd


# ══════════════════════════════════════════════════════════════
# 工具函数
# ══════════════════════════════════════════════════════════════

def _ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    """确保 DataFrame 有计算所需的列"""
    df = df.copy()
    df = df.sort_values("date").reset_index(drop=True)
    if "ema20" not in df.columns:
        df["ema20"] = df["close"].ewm(span=20, adjust=False).mean()
    if "body" not in df.columns:
        df["body"] = df["close"] - df["open"]
    if "body_pct" not in df.columns:
        df["body_pct"] = (df["body"] / df["open"] * 100)
    if "range" not in df.columns:
        df["range"] = df["high"] - df["low"]
    if "upper_wick" not in df.columns:
        df["upper_wick"] = df["high"] - df[["close", "open"]].max(axis=1)
    if "lower_wick" not in df.columns:
        df["lower_wick"] = df[["close", "open"]].min(axis=1) - df["low"]
    if "is_bull" not in df.columns:
        df["is_bull"] = df["close"] > df["open"]
    if "is_doji" not in df.columns:
        df["is_doji"] = abs(df["body"]) < df["range"] * 0.15
    return df


def _swings(df: pd.DataFrame, window: int = 3) -> pd.DataFrame:
    """检测摆动高点和低点"""
    df = df.copy()
    df["swing_high"] = False
    df["swing_low"] = False
    for i in range(window, len(df) - window):
        if df["high"].iloc[i] == df["high"].iloc[i - window:i + window + 1].max():
            df.loc[df.index[i], "swing_high"] = True
        if df["low"].iloc[i] == df["low"].iloc[i - window:i + window + 1].min():
            df.loc[df.index[i], "swing_low"] = True
    return df


# ══════════════════════════════════════════════════════════════
# 1. 市场结构分类
# ══════════════════════════════════════════════════════════════

def classify_market_structure(df: pd.DataFrame) -> dict:
    """
    判断当前市场结构：趋势 / 交易区间 / 反转。

    Returns:
        {structure, sub_type, confidence, reason}
        structure: "trend_up" | "trend_down" | "trading_range" | "reversal"
    """
    df = _ensure_columns(df)
    df = _swings(df)
    if len(df) < 30:
        return {"structure": "trading_range", "sub_type": "insufficient_data",
                "confidence": 0.0, "reason": "数据不足30根K线"}

    last20 = df.tail(20)
    last10 = df.tail(10)
    price = df["close"].iloc[-1]
    ema = df["ema20"].iloc[-1]

    # EMA20 斜率 (最近10根)
    ema_slope = (df["ema20"].iloc[-1] - df["ema20"].iloc[-10]) / df["ema20"].iloc[-10] * 100

    # K线在EMA之上比例
    above_ema_ratio = (last20["close"] > last20["ema20"]).sum() / 20

    # 摆动高低点序列
    recent_highs = df[df["swing_high"]].tail(5)
    recent_lows = df[df["swing_low"]].tail(5)

    # 判断趋势: 高点上移 + 低点上移 = 上升趋势
    if len(recent_highs) >= 2 and len(recent_lows) >= 2:
        hh = recent_highs["high"].iloc[-1] > recent_highs["high"].iloc[-2]
        hl = recent_lows["low"].iloc[-1] > recent_lows["low"].iloc[-2]

        if hh and hl and ema_slope > 0.3:
            return {"structure": "trend_up", "sub_type": "strong_trend",
                    "confidence": min(0.9, above_ema_ratio + 0.2),
                    "reason": f"高点上移+低点上移，EMA20斜率+{ema_slope:.1f}%，{above_ema_ratio:.0%}K线在EMA之上"}
        elif not hh and not hl and ema_slope < -0.3:
            return {"structure": "trend_down", "sub_type": "strong_trend",
                    "confidence": min(0.9, (1 - above_ema_ratio) + 0.2),
                    "reason": f"高点下移+低点下移，EMA20斜率{ema_slope:.1f}%，{1-above_ema_ratio:.0%}K线在EMA之下"}

    # 判断交易区间: EMA20 平缓 + 比例接近50%
    if abs(ema_slope) < 0.5 and 0.35 < above_ema_ratio < 0.65:
        return {"structure": "trading_range", "sub_type": "flat_range",
                "confidence": 0.7,
                "reason": f"EMA20近乎水平({ema_slope:+.1f}%)，K线分布均衡"}

    # 判断反转: 价格从一边极端到另一边
    if len(df) >= 40:
        prev20 = df.tail(40).head(20)
        prev_above = (prev20["close"] > prev20["ema20"]).sum() / 20
        if prev_above > 0.7 and above_ema_ratio < 0.3:
            return {"structure": "reversal", "sub_type": "bull_to_bear",
                    "confidence": 0.6,
                    "reason": f"前期{prev_above:.0%}在EMA之上→当前仅{above_ema_ratio:.0%}，疑似转空"}
        if prev_above < 0.3 and above_ema_ratio > 0.7:
            return {"structure": "reversal", "sub_type": "bear_to_bull",
                    "confidence": 0.6,
                    "reason": f"前期{prev_above:.0%}在EMA之上→当前{above_ema_ratio:.0%}，疑似转多"}

    # 默认: 趋势偏多还是偏空
    if price > ema and ema_slope > 0:
        return {"structure": "trend_up", "sub_type": "weak_trend",
                "confidence": 0.4, "reason": "价在EMA之上但趋势偏弱"}
    elif price < ema and ema_slope < 0:
        return {"structure": "trend_down", "sub_type": "weak_trend",
                "confidence": 0.4, "reason": "价在EMA之下但趋势偏弱"}
    else:
        return {"structure": "trading_range", "sub_type": "mixed",
                "confidence": 0.3, "reason": "信号混杂，无明确方向"}


# ══════════════════════════════════════════════════════════════
# 2. 急速与通道识别
# ══════════════════════════════════════════════════════════════

def identify_spike_channel(df: pd.DataFrame) -> dict:
    """
    识别最近的急速行情 (Spike) 及后续是否进入通道 (Channel)。

    急速 = 连续3根以上同向强趋势K线(实体>1%)，期间回调<急速幅度的38%。
    通道 = 急速后形成至少2波同向推动，斜率低于急速。

    Returns:
        {phase, direction, spike_start, spike_end, spike_size,
         spike_low, spike_high, target, channel_active, channel_waves}
    """
    df = _ensure_columns(df)
    if len(df) < 20:
        return {"phase": "unknown", "direction": None, "reason": "数据不足"}

    # 寻找最近的急速: 从末尾往前找连续强趋势K线
    bull_spike_bars = []
    bear_spike_bars = []
    current_bull = 0
    current_bear = 0

    for i in range(len(df) - 1, max(0, len(df) - 30), -1):
        r = df.iloc[i]
        if r["is_bull"] and r["body_pct"] > 1.0:
            current_bull += 1
            current_bear = 0
            bull_spike_bars.append(i)
        elif not r["is_bull"] and r["body_pct"] < -1.0:
            current_bear += 1
            current_bull = 0
            bear_spike_bars.append(i)
        else:
            current_bull = 0
            current_bear = 0

        if current_bull >= 3:
            break
        if current_bear >= 3:
            break

    # 取最近的满足条件的急速
    if len(bull_spike_bars) >= 3:
        spike_indices = sorted(bull_spike_bars[:10])  # 取最近10根中的
        # 找连续段
        spike_end_idx = spike_indices[-1]
        spike_start_idx = spike_end_idx
        for j in range(spike_end_idx - 1, max(0, spike_end_idx - 15), -1):
            if df.iloc[j]["is_bull"] and df.iloc[j]["body_pct"] > 1.0:
                spike_start_idx = j
            else:
                break

        spike_low = df["low"].iloc[spike_start_idx:spike_end_idx + 1].min()
        spike_high = df["high"].iloc[spike_start_idx:spike_end_idx + 1].max()
        spike_size = spike_high - spike_low
        target = spike_high + spike_size  # 等距向上投射

        # 检查急速后是否形成通道
        after_spike = df.iloc[spike_end_idx + 1:]
        channel_active = False
        channel_waves = 0
        if len(after_spike) >= 5:
            # 通道条件: 急速后有回调但不破急速低点, 然后继续推高
            after_low = after_spike["low"].min()
            pullback_pct = (spike_high - after_low) / spike_size if spike_size > 0 else 1
            if pullback_pct < 0.62 and after_spike["high"].max() > spike_high * 0.98:
                channel_active = True
                # 数通道波数
                ch_swings = _swings(after_spike, window=2)
                channel_waves = ch_swings["swing_high"].sum()

        return {
            "phase": "channel" if channel_active else "spike",
            "direction": "up",
            "spike_start_date": df["date"].iloc[spike_start_idx],
            "spike_end_date": df["date"].iloc[spike_end_idx],
            "spike_size": round(spike_size, 2),
            "spike_low": round(spike_low, 2),
            "spike_high": round(spike_high, 2),
            "spike_bars": spike_end_idx - spike_start_idx + 1,
            "target": round(target, 2),
            "channel_active": channel_active,
            "channel_waves": channel_waves,
            "target_method": "spike_height_projection"
        }

    if len(bear_spike_bars) >= 3:
        spike_indices = sorted(bear_spike_bars[:10])
        spike_end_idx = spike_indices[-1]
        spike_start_idx = spike_end_idx
        for j in range(spike_end_idx - 1, max(0, spike_end_idx - 15), -1):
            if not df.iloc[j]["is_bull"] and df.iloc[j]["body_pct"] < -1.0:
                spike_start_idx = j
            else:
                break

        spike_high = df["high"].iloc[spike_start_idx:spike_end_idx + 1].max()
        spike_low = df["low"].iloc[spike_start_idx:spike_end_idx + 1].min()
        spike_size = spike_high - spike_low
        target = spike_low - spike_size

        after_spike = df.iloc[spike_end_idx + 1:]
        channel_active = False
        channel_waves = 0
        if len(after_spike) >= 5:
            after_high = after_spike["high"].max()
            rally_pct = (after_high - spike_low) / spike_size if spike_size > 0 else 1
            if rally_pct < 0.62 and after_spike["low"].min() < spike_low * 1.02:
                channel_active = True
                ch_swings = _swings(after_spike, window=2)
                channel_waves = ch_swings["swing_low"].sum()

        return {
            "phase": "channel" if channel_active else "spike",
            "direction": "down",
            "spike_start_date": df["date"].iloc[spike_start_idx],
            "spike_end_date": df["date"].iloc[spike_end_idx],
            "spike_size": round(spike_size, 2),
            "spike_high": round(spike_high, 2),
            "spike_low": round(spike_low, 2),
            "spike_bars": spike_end_idx - spike_start_idx + 1,
            "target": round(target, 2),
            "channel_active": channel_active,
            "channel_waves": channel_waves,
            "target_method": "spike_height_projection"
        }

    # 没有检测到急速
    return {"phase": "no_spike", "direction": None,
            "reason": "近期无连续3根以上强趋势K线，处于常态波动"}


# ══════════════════════════════════════════════════════════════
# 3. 信号K线识别
# ══════════════════════════════════════════════════════════════

def identify_signal_bars(df: pd.DataFrame, lookback: int = 15) -> list[dict]:
    """
    识别最近的信号K线。

    信号类型:
      - bullish_engulfing: 阳包阴
      - bearish_engulfing: 阴包阳
      - doji: 十字星 (实体<振幅15%)
      - inside: 内包K线 (高低点都在前一根之内)
      - outside: 外包K线 (高低点都超出前一根)
      - marubozu_bull: 光头光脚阳线 (无上下影)
      - marubozu_bear: 光头光脚阴线
      - reversal_bull: 长下影阳线 (锤子线)
      - reversal_bear: 长上影阴线 (流星线)

    Returns:
        [{date, type, strength, position_vs_ema, note}]
    """
    df = _ensure_columns(df)
    signals = []

    for i in range(max(1, len(df) - lookback), len(df)):
        r = df.iloc[i]
        prev = df.iloc[i - 1] if i > 0 else None

        signal = None
        strength = 0

        # 阳包阴
        if prev is not None and r["is_bull"] and not prev["is_bull"] \
                and r["open"] <= prev["close"] and r["close"] >= prev["open"]:
            engulf_pct = abs(r["body_pct"]) + abs(prev["body_pct"])
            strength = min(10, engulf_pct)
            signal = "bullish_engulfing"

        # 阴包阳
        elif prev is not None and not r["is_bull"] and prev["is_bull"] \
                and r["open"] >= prev["close"] and r["close"] <= prev["open"]:
            engulf_pct = abs(r["body_pct"]) + abs(prev["body_pct"])
            strength = min(10, engulf_pct)
            signal = "bearish_engulfing"

        # 十字星
        if r["is_doji"] and signal is None:
            signal = "doji"
            strength = 3

        # 内包
        if prev is not None and r["high"] < prev["high"] and r["low"] > prev["low"]:
            if signal is None:
                signal = "inside"
                strength = 4

        # 外包
        if prev is not None and r["high"] > prev["high"] and r["low"] < prev["low"]:
            signal = "outside"
            strength = 6 if r["is_bull"] else 5

        # 光头阳线 (marubozu)
        if r["is_bull"] and r["upper_wick"] < r["range"] * 0.1 and r["lower_wick"] < r["range"] * 0.1:
            if signal is None or signal == "inside":
                signal = "marubozu_bull"
                strength = 8 if r["body_pct"] > 3 else 6

        # 光头阴线
        if not r["is_bull"] and r["upper_wick"] < r["range"] * 0.1 and r["lower_wick"] < r["range"] * 0.1:
            if signal is None or signal == "inside":
                signal = "marubozu_bear"
                strength = 8 if r["body_pct"] < -3 else 6

        # 锤子线 (长下影+小实体+在低位)
        if r["is_bull"] and r["lower_wick"] > r["body"] * 2 and r["upper_wick"] < r["body"] * 0.5:
            if signal is None:
                signal = "reversal_bull"
                strength = 6

        # 流星线 (长上影+小实体+在高位)
        if not r["is_bull"] and r["upper_wick"] > abs(r["body"]) * 2 and r["lower_wick"] < abs(r["body"]) * 0.5:
            if signal is None:
                signal = "reversal_bear"
                strength = 6

        if signal is None:
            # 普通K线也要记录位置
            if abs(r["body_pct"]) > 5:
                signal = "big_bull" if r["is_bull"] else "big_bear"
                strength = 5
            else:
                continue

        # 位置评估
        ema = r["ema20"]
        dist_to_ema = (r["close"] - ema) / ema * 100
        near_ema = abs(dist_to_ema) < 1.5
        above_ema = r["close"] > ema

        signals.append({
            "date": str(r["date"]),
            "type": signal,
            "strength": strength,
            "close": round(r["close"], 2),
            "ema20": round(ema, 2),
            "dist_to_ema_pct": round(dist_to_ema, 1),
            "near_ema": near_ema,
            "above_ema": above_ema,
            "note": f"{'靠近EMA20' if near_ema else ('EMA之上' if above_ema else 'EMA之下')}"
        })

    return signals[-10:]  # 最多返回最近10个


# ══════════════════════════════════════════════════════════════
# 4. 高/低点入场识别
# ══════════════════════════════════════════════════════════════

def find_high_low_entries(df: pd.DataFrame) -> dict:
    """
    识别高1/高2/低1/低2入场点。

    - 高1: 上升趋势中，回调后第一根阳线信号K线
    - 高2: 上升趋势中，回调后第二根阳线信号K线 (最可靠)
    - 低1: 下降趋势中，反弹后第一根阴线信号K线
    - 低2: 下降趋势中，反弹后第二根阴线信号K线 (最可靠)

    Returns:
        {high1, high2, low1, low2, best_entry, best_type, best_reason}
    """
    df = _ensure_columns(df)
    struct = classify_market_structure(df)
    trend = struct.get("structure", "trading_range")
    lookback = min(25, len(df))

    high1 = None; high2 = None
    low1 = None; low2 = None
    best_entry = None; best_type = None; best_reason = ""

    # 上升趋势中找高1/高2
    if trend in ("trend_up",):
        # 找最近的回调低点
        for i in range(len(df) - 1, len(df) - lookback, -1):
            if df["swing_low"].iloc[i] if "swing_low" in df.columns else df["low"].iloc[i] == df["low"].iloc[i-2:i+3].min():
                swing_low_idx = i
                # 从回调低点往后找阳线信号K线
                bull_count = 0
                for j in range(swing_low_idx + 1, len(df)):
                    r = df.iloc[j]
                    if r["is_bull"] and r["body_pct"] > 0.5:
                        bull_count += 1
                        if bull_count == 1:
                            high1 = {
                                "date": str(r["date"]),
                                "price": round(r["close"], 2),
                                "body_pct": round(r["body_pct"], 1),
                                "near_ema": abs(r["close"] - r["ema20"]) / r["close"] * 100 < 2,
                            }
                        elif bull_count == 2:
                            high2 = {
                                "date": str(r["date"]),
                                "price": round(r["close"], 2),
                                "body_pct": round(r["body_pct"], 1),
                                "near_ema": abs(r["close"] - r["ema20"]) / r["close"] * 100 < 2,
                            }
                            if high2["near_ema"]:
                                best_entry = high2
                                best_type = "high2_at_ema"
                                best_reason = "高2在EMA20附近 — AB最强买入信号"
                            elif best_entry is None:
                                best_entry = high2
                                best_type = "high2"
                                best_reason = "高2买入信号 (未在EMA附近，可靠性降低)"
                            break
                break

        # 如果没有找到高2，高1也行
        if high1 and best_entry is None:
            best_entry = high1
            best_type = "high1"
            best_reason = "高1买入信号 (早期入场，失败率较高)"

    # 下降趋势中找低1/低2
    elif trend in ("trend_down",):
        for i in range(len(df) - 1, len(df) - lookback, -1):
            if df["swing_high"].iloc[i] if "swing_high" in df.columns else df["high"].iloc[i] == df["high"].iloc[i-2:i+3].max():
                swing_high_idx = i
                bear_count = 0
                for j in range(swing_high_idx + 1, len(df)):
                    r = df.iloc[j]
                    if not r["is_bull"] and r["body_pct"] < -0.5:
                        bear_count += 1
                        if bear_count == 1:
                            low1 = {
                                "date": str(r["date"]),
                                "price": round(r["close"], 2),
                                "body_pct": round(r["body_pct"], 1),
                                "near_ema": abs(r["close"] - r["ema20"]) / r["close"] * 100 < 2,
                            }
                        elif bear_count == 2:
                            low2 = {
                                "date": str(r["date"]),
                                "price": round(r["close"], 2),
                                "body_pct": round(r["body_pct"], 1),
                                "near_ema": abs(r["close"] - r["ema20"]) / r["close"] * 100 < 2,
                            }
                            if low2["near_ema"]:
                                best_entry = low2
                                best_type = "low2_at_ema"
                                best_reason = "低2在EMA20附近 — AB最强卖出信号"
                            elif best_entry is None:
                                best_entry = low2
                                best_type = "low2"
                                best_reason = "低2卖出信号"
                            break
                break

        if low1 and best_entry is None:
            best_entry = low1
            best_type = "low1"
            best_reason = "低1卖出信号 (早期入场，失败率较高)"

    return {
        "high1": high1,
        "high2": high2,
        "low1": low1,
        "low2": low2,
        "best_entry": best_entry,
        "best_type": best_type,
        "best_reason": best_reason,
        "trend_context": trend,
    }


# ══════════════════════════════════════════════════════════════
# 5. 顺势回调入场评估
# ══════════════════════════════════════════════════════════════

def evaluate_trend_setup(df: pd.DataFrame) -> dict:
    """
    评估当前是否有顺势回调入场机会。

    做多条件: 上升趋势 + 价格回调至EMA20附近 + 出现阳线信号K线
    做空条件: 下降趋势 + 价格反弹至EMA20附近 + 出现阴线信号K线

    Returns:
        {setup_type, quality, entry_price, stop_price, signal_bar}
    """
    df = _ensure_columns(df)
    struct = classify_market_structure(df)
    signals = identify_signal_bars(df, lookback=5)
    entries = find_high_low_entries(df)
    price = df["close"].iloc[-1]
    ema = df["ema20"].iloc[-1]

    dist_to_ema = (price - ema) / ema * 100

    result = {
        "setup_type": "none",
        "quality": 0,
        "entry_price": None,
        "stop_price": None,
        "signal_bar": None,
        "reason": ""
    }

    # 做多条件评估
    if struct["structure"] in ("trend_up", "reversal"):
        near_ema = abs(dist_to_ema) < 2
        recent_signals = [s for s in signals if s["type"] in
                          ("bullish_engulfing", "marubozu_bull", "reversal_bull", "outside")]

        quality = 0
        if near_ema:
            quality += 3
        if recent_signals:
            quality += 3
            result["signal_bar"] = recent_signals[-1]
        if ((entries or {}).get("best_type") or "").startswith("high"):
            quality += 2
        if struct["structure"] == "trend_up" and struct.get("confidence", 0) > 0.5:
            quality += 2

        if quality >= 4:
            result["setup_type"] = "long_pullback"
            result["quality"] = min(10, quality)
            # 入场: 信号K线收盘价 或 EMA20 附近的限价单
            result["entry_price"] = round(
                recent_signals[-1]["close"] if recent_signals else ema, 2)
            # 止损: 最近摆动低点下方
            recent_lows = df["low"].tail(10)
            result["stop_price"] = round(recent_lows.min() * 0.99, 2)
            result["reason"] = f"做多回调: 价距EMA{dist_to_ema:+.1f}%，" \
                + (f"信号K线{'有' if recent_signals else '无'}，" ) \
                + f"质量{quality}/10"

    # 做空条件评估
    elif struct["structure"] in ("trend_down",):
        near_ema = abs(dist_to_ema) < 2
        recent_signals = [s for s in signals if s["type"] in
                          ("bearish_engulfing", "marubozu_bear", "reversal_bear", "outside")]

        quality = 0
        if near_ema:
            quality += 3
        if recent_signals:
            quality += 3
            result["signal_bar"] = recent_signals[-1]
        if ((entries or {}).get("best_type") or "").startswith("low"):
            quality += 2
        if struct.get("confidence", 0) > 0.5:
            quality += 2

        if quality >= 4:
            result["setup_type"] = "short_rally"
            result["quality"] = min(10, quality)
            result["entry_price"] = round(
                recent_signals[-1]["close"] if recent_signals else ema, 2)
            recent_highs = df["high"].tail(10)
            result["stop_price"] = round(recent_highs.max() * 1.01, 2)
            result["reason"] = f"做空反弹: 价距EMA{dist_to_ema:+.1f}%，" \
                + (f"信号K线{'有' if recent_signals else '无'}，" ) \
                + f"质量{quality}/10"

    if result["setup_type"] == "none":
        result["reason"] = f"无顺势回调机会 (结构:{struct['structure']}, 距EMA:{dist_to_ema:+.1f}%)"

    return result


# ══════════════════════════════════════════════════════════════
# 6. 高潮反转检测
# ══════════════════════════════════════════════════════════════

def detect_climax_reversal(df: pd.DataFrame) -> dict:
    """
    检测买人高潮/卖人高潮反转。

    买人高潮 = 大阳线(实体>3%) + 下一根为停止K线(十字星/内包/长上影)
    卖人高潮 = 大阴线(实体<-3%) + 下一根为停止K线(十字星/内包/长下影)

    Returns:
        {detected, type, climax_bar_date, stop_bar_date, note}
    """
    df = _ensure_columns(df)
    if len(df) < 3:
        return {"detected": False, "type": None, "note": "数据不足"}

    # 检查最近3根K线
    for i in range(len(df) - 2, len(df)):
        climax = df.iloc[i - 1] if i > 0 else None
        stop_bar = df.iloc[i]

        if climax is None:
            continue

        # 买人高潮
        if climax["is_bull"] and climax["body_pct"] > 3.0:
            is_stop = (stop_bar["is_doji"] or
                       (stop_bar["high"] < climax["high"] and stop_bar["low"] > climax["low"]) or
                       (stop_bar["upper_wick"] > stop_bar["range"] * 0.4))
            if is_stop:
                return {
                    "detected": True,
                    "type": "buying_climax",
                    "climax_bar_date": str(climax["date"]),
                    "stop_bar_date": str(stop_bar["date"]),
                    "climax_body_pct": round(climax["body_pct"], 1),
                    "stop_bar_type": "doji" if stop_bar["is_doji"]
                    else ("inside" if stop_bar["high"] < climax["high"] else "long_upper_wick"),
                    "note": "买人高潮 → 短期见顶风险，多单应收紧止损",
                }

        # 卖人高潮
        if not climax["is_bull"] and climax["body_pct"] < -3.0:
            is_stop = (stop_bar["is_doji"] or
                       (stop_bar["high"] < climax["high"] and stop_bar["low"] > climax["low"]) or
                       (stop_bar["lower_wick"] > stop_bar["range"] * 0.4))
            if is_stop:
                return {
                    "detected": True,
                    "type": "selling_climax",
                    "climax_bar_date": str(climax["date"]),
                    "stop_bar_date": str(stop_bar["date"]),
                    "climax_body_pct": round(climax["body_pct"], 1),
                    "stop_bar_type": "doji" if stop_bar["is_doji"]
                    else ("inside" if stop_bar["high"] < climax["high"] else "long_lower_wick"),
                    "note": "卖人高潮 → 短期见底信号，空单应收紧止损",
                }

    # 检查是否是单K线的高潮（超大实体K线自身蕴含反转风险）
    latest = df.iloc[-1]
    if latest["is_bull"] and latest["body_pct"] > 8:
        return {
            "detected": True,
            "type": "buying_climax",
            "climax_bar_date": str(latest["date"]),
            "stop_bar_date": None,
            "climax_body_pct": round(latest["body_pct"], 1),
            "stop_bar_type": "pending",
            "note": "单K线巨阳(>{:.0f}%)自身构成买人高潮，等待下一根停止K线确认".format(latest["body_pct"]),
        }
    if not latest["is_bull"] and latest["body_pct"] < -8:
        return {
            "detected": True,
            "type": "selling_climax",
            "climax_bar_date": str(latest["date"]),
            "stop_bar_date": None,
            "climax_body_pct": round(latest["body_pct"], 1),
            "note": "单K线巨阴(>{:.0f}%)自身构成卖人高潮，等待下一根停止K线确认".format(abs(latest["body_pct"])),
        }

    return {"detected": False, "type": None, "note": "无高潮反转信号"}


# ══════════════════════════════════════════════════════════════
# 7. 始终入场方向
# ══════════════════════════════════════════════════════════════

def is_always_in_long(df: pd.DataFrame) -> dict:
    """
    始终入场 (Always-In) 判断 — 含确认阶段。

    方向: LONG / SHORT / NEUTRAL
    阶段: confirmed(已确认) / unconfirmed_reversal(待确认V反) / developing(趋势发展中)

    核心理念:
    - V形反转不是入场信号，是"关注信号"——需要等回踩确认
    - 买人高潮后追涨是最差入场
    - HIGH2 at EMA20 才是最好的确认

    Returns:
        {direction, confidence, phase, confirmation, score, reason}
    """
    df = _ensure_columns(df)
    if len(df) < 20:
        return {"direction": "NEUTRAL", "confidence": 0.0, "phase": None,
                "confirmation": None, "score": 0, "reason": "数据不足"}

    price = df["close"].iloc[-1]
    ema = df["ema20"].iloc[-1]
    last5 = df.tail(5)
    last10 = df.tail(10)
    last20 = df.tail(20)

    score = 0
    reasons = []
    above_ema = price > ema

    # ── 信号收集 ──
    # 1. 价格与EMA20关系 (权重: ±3)
    if above_ema:
        score += 3; reasons.append("价在EMA20之上(+3)")
    else:
        score -= 3; reasons.append("价在EMA20之下(-3)")

    # 2. EMA20 斜率
    ema_slope = (df["ema20"].iloc[-1] - df["ema20"].iloc[-10]) / df["ema20"].iloc[-10] * 100
    ema_rising = ema_slope > 0.5
    ema_falling = ema_slope < -0.5
    if ema_rising:
        score += 2; reasons.append(f"EMA20上升+{ema_slope:.1f}%(+2)")
    elif ema_falling:
        score -= 2; reasons.append(f"EMA20下降{ema_slope:.1f}%(-2)")

    # 3. V 形反转检测 — 必须是"从长期下跌中反转"，而非上升趋势中的回调
    is_v_reversal = False
    if len(df) >= 20:
        prev_5_below_ema = df["close"].iloc[-6] < df["ema20"].iloc[-6]
        # 确认前期是下跌趋势: 过去20根K线中至少55%在EMA之下
        prev_20_check = df.tail(20).head(15)
        bearish_ratio = (prev_20_check["close"] < prev_20_check["ema20"]).sum() / len(prev_20_check)
        was_bearish = bearish_ratio > 0.55
        if "ma5" not in df.columns:
            df["ma5"] = df["close"].rolling(5).mean()
        ma5_series = df["ma5"]
        ma5_slope = (ma5_series.iloc[-1] - ma5_series.iloc[-5]) / ma5_series.iloc[-5] * 100 if ma5_series.iloc[-5] > 0 else 0
        if prev_5_below_ema and above_ema and ma5_slope > 3 and was_bearish:
            is_v_reversal = True
            score += 2
            reasons.append(f"V形反转突破EMA(+2) MA5急升{ma5_slope:.0f}% 前期{bearish_ratio:.0%}弱势")

    # 4. Higher Low 检查 (反转确认的关键) — 双路径检测
    has_higher_low = False
    if len(df) >= 15:
        # 方法1: 摆动低点比较
        swings = _swings(df.tail(30))
        recent_swing_lows = swings[swings["swing_low"]].tail(3)
        if len(recent_swing_lows) >= 2:
            if recent_swing_lows["low"].iloc[-1] > recent_swing_lows["low"].iloc[-2]:
                has_higher_low = True
        # 方法2: 区间低点比较 (近5日最低 vs 前6-15日最低)
        if not has_higher_low:
            recent_low = df["low"].tail(5).min()
            prior_low = df["low"].tail(15).head(10).min()
            if recent_low > prior_low:
                has_higher_low = True
        if has_higher_low:
            score += 2
            reasons.append(f"Higher Low已确认(+2)")

    # 5. 近5日多空比
    bull_count = last5["is_bull"].sum()
    if bull_count >= 3:
        score += 2; reasons.append(f"近5日{bull_count}阳(+2)")
    elif bull_count <= 1:
        score -= 2; reasons.append(f"近5日{bull_count}阳(-2)")

    # 6. 最后K线强度
    latest_body = df["body_pct"].iloc[-1]
    is_climax_bar = abs(latest_body) > 8  # >8% 实体 = 高潮级别
    if latest_body > 5:
        if is_climax_bar:
            score += 2; reasons.append(f"巨阳线{latest_body:+.1f}%(+2 高潮K线)")
        else:
            score += 2; reasons.append(f"大阳线{latest_body:+.1f}%(+2)")
    elif latest_body > 3:
        score += 2; reasons.append(f"阳线{latest_body:+.1f}%(+2)")
    elif latest_body < -5:
        score -= 3; reasons.append(f"巨阴线{latest_body:+.1f}%(-3)")
    elif latest_body < -3:
        score -= 2; reasons.append(f"大阴线{latest_body:+.1f}%(-2)")

    # 7. 连续强趋势K线
    consecutive_bull = 0; consecutive_bear = 0
    for i in range(len(last10) - 1, -1, -1):
        if last10["is_bull"].iloc[i] and last10["body_pct"].iloc[i] > 1:
            consecutive_bull += 1; consecutive_bear = 0
        elif not last10["is_bull"].iloc[i] and last10["body_pct"].iloc[i] < -1:
            consecutive_bear += 1; consecutive_bull = 0
        else:
            break
    if consecutive_bull >= 2:
        score += 2; reasons.append(f"连{consecutive_bull}根强阳(+2)")
    elif consecutive_bear >= 3:
        score -= 2; reasons.append(f"连{consecutive_bear}根强阴(-2)")

    # 8. 买人/卖人高潮惩罚
    climax = detect_climax_reversal(df)
    climax_penalty = 0
    if climax["detected"]:
        if climax["type"] == "buying_climax" and above_ema:
            climax_penalty = -2
            reasons.append(f"⚠️ 买人高潮(涨停{latest_body:+.0f}%)→短期超买 追涨风险大(-2)")
        elif climax["type"] == "selling_climax" and not above_ema:
            climax_penalty = -2
            reasons.append(f"⚠️ 卖人高潮→超卖(-2)")
    score += climax_penalty

    # 9. 摆动结构
    struct = classify_market_structure(df)
    if struct["structure"] == "trend_up":
        score += 1; reasons.append("摆动结构上升(+1)")
    elif struct["structure"] == "trend_down" and not is_v_reversal and not has_higher_low:
        score -= 1; reasons.append("摆动结构下降(-1)")

    # ── 裁决 ──
    if score >= 5:
        direction = "LONG"
    elif score <= -5:
        direction = "SHORT"
    else:
        direction = "NEUTRAL"

    # ── 确认阶段判定 ──
    phase = None
    confirmation = []

    if direction == "LONG":
        # 确认条件逐一检查
        checks = []

        # 条件1: EMA 走平或上升
        if ema_rising or ema_slope > -0.3:
            checks.append(True)
        else:
            checks.append(False)
            confirmation.append(f"EMA20走平或拐头(当前斜率{ema_slope:+.1f}%)")

        # 条件2: Higher Low 已出现
        if has_higher_low:
            checks.append(True)
        else:
            checks.append(False)
            recent_low_price = last20["low"].min()
            confirmation.append(f"出现Higher Low(当前最低¥{recent_low_price:.2f})")

        # 条件3: 无买人高潮 或 高潮已消化(出现停止K线)
        if not climax["detected"] or climax["stop_bar_date"]:
            checks.append(True)
        else:
            checks.append(False)
            confirmation.append(f"买人高潮消化(等停止K线/缩量星线确认高潮结束)")

        # 条件4: 至少站上MA5 且 MA5 不下降
        if "ma5" not in df.columns:
            df["ma5"] = df["close"].rolling(5).mean()
        above_ma5 = price > df["ma5"].iloc[-1]
        ma5_flat_or_rising = df["ma5"].iloc[-1] >= df["ma5"].iloc[-2]
        if above_ma5 and ma5_flat_or_rising:
            checks.append(True)
        else:
            checks.append(False)
            confirmation.append(f"站稳MA5(¥{df['ma5'].iloc[-1]:.2f})")

        passed = sum(checks)

        # V反需要全部通过才算确认；趋势股只需3/4 (高潮消化不是必要条件)
        if is_v_reversal:
            if passed >= 4:
                phase = "confirmed"
                confidence = 0.85
            elif passed >= 2:
                phase = "unconfirmed_reversal"
                confidence = 0.35
            else:
                phase = "unconfirmed_reversal"
                confidence = 0.25
            # V反的确认条件保持完整（4项全列）
        else:
            # 非V反趋势: EMA+HigherLow+MA5 三项通过即可确认
            structural_checks = checks[0:2] + [checks[3]]  # EMA, HigherLow, MA5
            structural_passed = sum(structural_checks)
            if structural_passed >= 3:
                phase = "confirmed"
                confidence = min(0.9, 0.6 + passed * 0.1)
                # 高潮消化只是警告，从确认条件中移除（不作为必须项）
                if "买人高潮" in str(confirmation):
                    confirmation = [c for c in confirmation if "高潮" not in c]
                    if confirmation:
                        confirmation.append("⚠️ 注意: 买人高潮未消化→追涨需等缩量星线")
            elif structural_passed >= 2:
                phase = "developing"
                confidence = 0.45
            else:
                phase = "developing"
                confidence = 0.3
    elif direction == "SHORT":
        # 类似逻辑…
        ema_strong_down = ema_slope < -1.0
        has_lower_high = False
        swings = _swings(df.tail(30))
        recent_highs = swings[swings["swing_high"]].tail(3)
        if len(recent_highs) >= 2:
            if recent_highs["high"].iloc[-1] < recent_highs["high"].iloc[-2]:
                has_lower_high = True
        if ema_strong_down and has_lower_high:
            phase = "confirmed"
            confidence = 0.8
        else:
            phase = "developing"
            confidence = 0.4
            if not ema_strong_down:
                confirmation.append("EMA20加速下行")
            if not has_lower_high:
                confirmation.append("出现Lower High确认")
    else:
        phase = None
        confidence = 0.2

    # 生成确认条件文本
    if confirmation:
        confirmation_text = " | ".join(confirmation)
        if phase == "unconfirmed_reversal":
            confirmation_text = "V反待确认: " + confirmation_text
        elif phase == "developing":
            confirmation_text = "趋势待强化: " + confirmation_text
    else:
        confirmation_text = None
        if phase == "confirmed":
            confirmation_text = "全部确认条件满足 ✅"
        elif phase == "developing":
            confirmation_text = "趋势发展中，关注量能持续"

    return {
        "direction": direction,
        "confidence": round(confidence, 2),
        "phase": phase,
        "confirmation": confirmation_text,
        "score": score,
        "reason": " | ".join(reasons),
    }


# ══════════════════════════════════════════════════════════════
# 综合AB分析
# ══════════════════════════════════════════════════════════════

def analyze(df: pd.DataFrame) -> dict:
    """一键运行全部7个AB分析函数，返回综合结果"""
    df = _ensure_columns(df)
    df = _swings(df)

    return {
        "market_structure": classify_market_structure(df),
        "spike_channel": identify_spike_channel(df),
        "signal_bars": identify_signal_bars(df),
        "high_low_entries": find_high_low_entries(df),
        "trend_setup": evaluate_trend_setup(df),
        "climax": detect_climax_reversal(df),
        "always_in": is_always_in_long(df),
    }
