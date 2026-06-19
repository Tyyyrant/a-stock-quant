#!/usr/bin/env python3
"""
筹码分布估算模块 — 基于日K线成交量加权价格分布近似

方法：用过去N天的 (VWAP, 成交量) 构建价格-筹码直方图，近似估算筹码分布。

核心指标:
  1. 筹码峰位置（支撑峰/套牢峰）
  2. 获利盘比例 / 套牢盘比例
  3. 筹码集中度趋势（股东户数变化）
  4. 距上方套牢盘的距离（上涨空间）

限制: 精确筹码分布需要 L2 逐笔数据，本模块做的是"近似估算"。
      在实际使用中标注为 "估算值"。

用法:
  from chip_distribution import estimate_chip_distribution
  result = estimate_chip_distribution(df)
  print(f"获利盘: {result['profit_ratio']:.1%}, 筹码分: {result['chip_score']}")
"""

import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


# 筹码直方图的价格区间数
CHIP_BINS = 80


def estimate_chip_distribution(df: pd.DataFrame,
                                lookback: int = 150,
                                ticker: str = "",
                                fundamentals: dict = None) -> dict:
    """
    主入口：估算筹码分布。

    算法：
      1. 取过去 lookback 天的日K线
      2. 每天将成交量分配到 [low, high] 价格区间（假设均匀分布）
      3. 加权衰减：越远的交易对当前筹码的影响越小（换手衰减）
      4. 构建筹码峰 → 识别支撑/压力位

    Args:
        df: OHLCV DataFrame，需含 amount（成交额）列，至少 60 条
        lookback: 回看天数
        ticker: 股票代码
        fundamentals: {shareholder_count_trend, top10_holding_pct, ...}

    Returns:
        {
            "current_price": float,
            "profit_ratio": float,       # 获利盘比例
            "loss_ratio": float,         # 套牢盘比例
            "chip_peaks": [...],         # 筹码峰列表
            "nearest_resistance": float,  # 最近上方筹码峰价格
            "nearest_support": float,     # 最近下方筹码峰价格
            "resistance_distance_pct": float,  # 距上方压力距离%
            "support_distance_pct": float,     # 距下方支撑距离%
            "concentration": "集中"|"分散"|"正常",
            "concentration_detail": {...},
            "chip_score": float,         # -100 ~ 100
            "risk_flags": [...],
            "analysis_summary": str,
        }
    """
    if df.empty or len(df) < 30:
        return _empty_result("数据不足")

    df = df.copy().reset_index(drop=True)
    n = len(df)
    lookback = min(lookback, n)

    c = df["close"].values.astype(float)
    h = df["high"].values.astype(float)
    l = df["low"].values.astype(float)
    v = df["volume"].values.astype(float)
    dates = df["date"].astype(str).values

    current_price = float(c[-1])
    score = 0.0
    risk_flags = []

    # ================================================================
    # 1. 构建筹码分布直方图
    # ================================================================

    # 价格范围
    seg_prices = np.concatenate([h[-lookback:], l[-lookback:]])
    price_min = np.percentile(seg_prices, 2)
    price_max = np.percentile(seg_prices, 98)
    if price_max <= price_min:
        price_max = price_min * 1.3
        price_min = price_min * 0.7

    price_range = price_max - price_min
    bin_edges = np.linspace(price_min, price_max, CHIP_BINS + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    chip_hist = np.zeros(CHIP_BINS)

    # 每日成交量分配到其振幅区间
    daily_turnover_rate = 0.02  # 默认日换手 2%，用于衰减

    for i in range(n - lookback, n):
        day_idx = i - (n - lookback)
        days_elapsed = n - 1 - i

        # 时间衰减：越远越不可靠（换手导致筹码转移）
        decay = np.exp(-days_elapsed * daily_turnover_rate)

        day_low = l[i]
        day_high = h[i]
        day_vol = v[i]

        if day_high <= day_low or day_vol <= 0:
            continue

        # 找到该日价格区间覆盖的 bins
        low_bin = max(0, int((day_low - price_min) / price_range * CHIP_BINS))
        high_bin = min(CHIP_BINS - 1, int((day_high - price_min) / price_range * CHIP_BINS))

        if high_bin >= low_bin:
            # 均匀分配到区间内的 bins
            vol_per_bin = day_vol * decay / (high_bin - low_bin + 1)
            for b in range(low_bin, high_bin + 1):
                if 0 <= b < CHIP_BINS:
                    chip_hist[b] += vol_per_bin

    total_chip = chip_hist.sum()
    if total_chip == 0:
        return _empty_result("无有效成交量数据")
    chip_hist_norm = chip_hist / total_chip  # 归一化

    # ================================================================
    # 2. 识别筹码峰
    # ================================================================

    chip_peaks = _find_chip_peaks(chip_hist_norm, bin_centers, current_price)
    chip_valleys = _find_chip_valleys(chip_hist_norm, bin_centers, chip_peaks)

    # ================================================================
    # 3. 获利盘 / 套牢盘
    # ================================================================

    current_bin = int((current_price - price_min) / price_range * CHIP_BINS)
    current_bin = max(0, min(CHIP_BINS - 1, current_bin))

    profit_mask = bin_centers <= current_price
    loss_mask = bin_centers > current_price

    profit_ratio = float(chip_hist_norm[profit_mask].sum())
    loss_ratio = float(chip_hist_norm[loss_mask].sum())

    # ================================================================
    # 4. 筹码峰评分
    # ================================================================

    # 找下方最近筹码峰（支撑）
    support_peaks = [p for p in chip_peaks if p["price"] < current_price]
    nearest_support = None
    if support_peaks:
        nearest_support = max(support_peaks, key=lambda p: p["price"])
        support_distance_pct = (current_price - nearest_support["price"]) / current_price
    else:
        support_distance_pct = 0.30  # 无支撑，视为远

    # 找上方最近筹码峰（压力）
    resistance_peaks = [p for p in chip_peaks if p["price"] > current_price]
    nearest_resistance = None
    if resistance_peaks:
        nearest_resistance = min(resistance_peaks, key=lambda p: p["price"])
        resistance_distance_pct = (nearest_resistance["price"] - current_price) / current_price
    else:
        resistance_distance_pct = 0.30  # 无压力，视为远

    # --- 评分逻辑 ---

    # 获利盘评分 (修正: 趋势中的高获利盘=筹码锁定好，不是风险)
    # 只有结合"趋势走弱"才是风险信号
    if profit_ratio > 0.80:
        # 趋势判断: 近5日是否还在涨
        ret_5d = (c[-1] - c[-6]) / c[-6] if n >= 6 else 0
        if ret_5d > 0.02:
            score += 8  # 趋势向上+满盘获利=筹码锁定良好
        elif ret_5d < -0.03:
            score -= 10
            risk_flags.append(f"获利盘{profit_ratio:.0%}且趋势转弱，回吐压力大")
        else:
            score += 2  # 横盘+满盘获利=中性
    elif profit_ratio > 0.60:
        score += 5  # 多数人赚钱，趋势向好
    elif profit_ratio < 0.20:
        score -= 12
        risk_flags.append(f"深度套牢{profit_ratio:.0%}获利，反弹阻力大")
    elif profit_ratio < 0.40:
        score -= 3

    # 支撑距离评分
    if nearest_support and support_distance_pct < 0.05:
        score += 12  # 下方有强支撑
    elif nearest_support and support_distance_pct < 0.10:
        score += 6
    elif not nearest_support or support_distance_pct > 0.20:
        score -= 5
        risk_flags.append("下方无筹码支撑")

    # 压力距离评分（重要！）
    if nearest_resistance and resistance_distance_pct < 0.03:
        score -= 15  # 紧贴套牢峰，一涨就有人解套抛
        risk_flags.append(f"上方{resistance_distance_pct*100:.0f}%有密集套牢盘⚠️")
    elif nearest_resistance and resistance_distance_pct < 0.08:
        score -= 5
    elif nearest_resistance and resistance_distance_pct > 0.15:
        score += 8  # 上方无压力，空间大

    # 筹码峰质量
    support_weight = nearest_support["volume_pct"] if nearest_support else 0
    resistance_weight = nearest_resistance["volume_pct"] if nearest_resistance else 0
    if support_weight > resistance_weight * 2:
        score += 10  # 下方筹码峰远大于上方，支撑强
    elif resistance_weight > support_weight * 2:
        score -= 10  # 上方筹码峰远大于下方，压力重

    # ================================================================
    # 5. 筹码集中度（如果有股东数据）
    # ================================================================

    concentration = "正常"
    concentration_detail = {}

    if fundamentals:
        shareholder_trend = fundamentals.get("shareholder_count_trend")
        if shareholder_trend:
            if shareholder_trend == "decreasing":
                score += 15
                concentration = "集中"
                concentration_detail["trend"] = "股东户数下降，筹码趋于集中"
            elif shareholder_trend == "increasing":
                score -= 15
                concentration = "分散"
                risk_flags.append("股东户数上升，筹码分散")
                concentration_detail["trend"] = "股东户数上升，筹码趋于分散"

        top10_pct = fundamentals.get("top10_holding_pct")
        if top10_pct:
            concentration_detail["top10_holding"] = top10_pct
            if top10_pct > 70:
                concentration = "集中"
                concentration_detail["level"] = "高集中"
            elif top10_pct < 35:
                concentration = "分散"
                concentration_detail["level"] = "低集中"

    # ================================================================
    # 6. 综合
    # ================================================================

    chip_score = round(max(-100, min(100, score)), 1)

    summary_parts = [
        f"获利盘{profit_ratio:.0%}",
        f"筹码{concentration}",
    ]
    if nearest_support:
        summary_parts.append(f"支撑@{nearest_support['price']:.1f}(距{support_distance_pct*100:.1f}%)")
    if nearest_resistance:
        summary_parts.append(f"压力@{nearest_resistance['price']:.1f}(距{resistance_distance_pct*100:.1f}%)")
    if risk_flags:
        summary_parts.append(f"风险: {'; '.join(risk_flags[:2])}")

    # 构建筹码峰返回信息
    def _peak_to_dict(p):
        return {
            "price": round(p["price"], 2),
            "volume_pct": round(p["volume_pct"], 3),
            "type": p["type"],
        }

    return {
        "current_price": current_price,
        "price_range": [round(price_min, 2), round(price_max, 2)],
        "profit_ratio": round(profit_ratio, 3),
        "loss_ratio": round(loss_ratio, 3),
        "chip_peaks": [_peak_to_dict(p) for p in chip_peaks[:5]],
        "nearest_resistance": round(nearest_resistance["price"], 2) if nearest_resistance else None,
        "nearest_support": round(nearest_support["price"], 2) if nearest_support else None,
        "resistance_distance_pct": round(resistance_distance_pct, 3),
        "support_distance_pct": round(support_distance_pct, 3),
        "concentration": concentration,
        "concentration_detail": concentration_detail,
        "chip_score": chip_score,
        "risk_flags": risk_flags,
        "analysis_summary": "; ".join(summary_parts),
    }


def _find_chip_peaks(hist, centers, current_price) -> list[dict]:
    """在筹码直方图中识别峰值"""
    n = len(hist)
    peaks = []

    # 找局部极大值
    for i in range(1, n - 1):
        if hist[i] > hist[i - 1] and hist[i] > hist[i + 1] and hist[i] > 0.005:
            # 峰的高度（相对总筹码的比例）
            peak_vol = float(hist[i])

            # 峰的类型
            price = float(centers[i])
            if price < current_price:
                ptype = "下方支撑峰" if peak_vol > 0.03 else "小支撑峰"
            elif price > current_price:
                ptype = "上方套牢峰" if peak_vol > 0.03 else "小套牢峰"
            else:
                ptype = "当前价位峰"

            peaks.append({
                "price": price,
                "volume_pct": peak_vol,
                "type": ptype,
            })

    # 按峰高排序
    peaks.sort(key=lambda p: p["volume_pct"], reverse=True)
    return peaks


def _find_chip_valleys(hist, centers, peaks) -> list[dict]:
    """识别筹码谷（两个峰之间的低点，股价容易快速通过）"""
    valleys = []
    if len(peaks) < 2:
        return valleys

    sorted_peaks = sorted(peaks, key=lambda p: p["price"])

    for i in range(len(sorted_peaks) - 1):
        p1 = sorted_peaks[i]
        p2 = sorted_peaks[i + 1]
        # 在两个峰之间找最低点
        mask = (centers > p1["price"]) & (centers < p2["price"])
        if mask.any():
            min_idx = np.argmin(hist[mask])
            valley_prices = centers[mask]
            valleys.append({
                "price_low": round(p1["price"], 2),
                "price_high": round(p2["price"], 2),
                "min_volume": round(float(hist[mask][min_idx]), 4),
            })

    return valleys


def _empty_result(reason: str = "") -> dict:
    return {
        "current_price": 0,
        "profit_ratio": 0.5,
        "loss_ratio": 0.5,
        "chip_peaks": [],
        "nearest_resistance": None,
        "nearest_support": None,
        "resistance_distance_pct": 0.3,
        "support_distance_pct": 0.3,
        "concentration": "未知",
        "concentration_detail": {},
        "chip_score": 0.0,
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
    print(f"  筹码分布估算 — {ticker}")
    print(f"  K线数据: {len(df)} 条 ({df['date'].iloc[0]} ~ {df['date'].iloc[-1]})")
    print(f"{'='*60}")

    result = estimate_chip_distribution(df, ticker=ticker)

    print(f"\n📊 当前价: {result['current_price']:.2f}")
    print(f"📈 筹码分: {result['chip_score']}")
    print(f"💰 获利盘: {result['profit_ratio']:.1%}")
    print(f"📉 套牢盘: {result['loss_ratio']:.1%}")
    print(f"🔒 筹码集中度: {result['concentration']}")

    if result["nearest_support"]:
        print(f"🟢 最近支撑: {result['nearest_support']:.2f} (距{result['support_distance_pct']*100:.1f}%)")
    if result["nearest_resistance"]:
        print(f"🔴 最近压力: {result['nearest_resistance']:.2f} (距{result['resistance_distance_pct']*100:.1f}%)")

    if result["chip_peaks"]:
        print(f"\n🏔 筹码峰:")
        for p in result["chip_peaks"]:
            icon = "🔴" if p["price"] > result["current_price"] else "🟢" if p["price"] < result["current_price"] else "⚪"
            print(f"  {icon} {p['price']:.2f} ({p['type']}, 占比{p['volume_pct']:.2%})")

    if result["risk_flags"]:
        print(f"\n⚠️ 风险标记: {', '.join(result['risk_flags'])}")

    print(f"\n📝 摘要: {result['analysis_summary']}")
    print(f"\n--- JSON ---")
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
