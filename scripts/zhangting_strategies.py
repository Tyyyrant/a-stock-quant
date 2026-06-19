#!/usr/bin/env python3
"""
《股是股非 1 — 猎取暴涨股》核心战法实现

作者: 一路奔行 (蒋文辉)
核心理念: 三度交易系统 (厚度·力度·速度) — "强势与安全共生"

实现策略:
  1. ABC三区分类 — A区(强势)/B区(次级)/C区(风险)
  2. 量价异动+均线归位 — 全书核心变盘信号
  3. 单日强硬洗盘 — 大阴→次日阳包阴的反包模式
  4. 缺口模式 — 向上/向下跳空缺口的跟踪
  5. 高位倒灌K线 — 出货识别
  6. 量能体叠加评分 — 多信号共振确认

用法:
  from zhangting_strategies import classify_abc_zone, detect_washout_reversal, score_signal_resonance
"""

import numpy as np
from typing import Optional


# ══════════════════════════════════════════════════════════════
# 策略 1: ABC 三区分类
# ══════════════════════════════════════════════════════════════
#
# A区 (强势区): MA5>MA10>MA20 且 价格在MA5上方 且 量能放大
# B区 (次级区): MA 纠缠或价格在MA10-20之间 量价正常
# C区 (风险区): MA5<MA10 且 价格在MA10下方 或 高位放量滞涨
#
# 操作: A区积极做多, B区选择性参与, C区坚决回避

def classify_abc_zone(kline_df) -> dict:
    """
    将当前K线位置分类为 A/B/C 三区。

    Returns:
        {zone, zone_score, zone_reason, features}
    """
    if kline_df is None or len(kline_df) < 60:
        return {"zone": "?", "zone_score": 0, "zone_reason": "数据不足"}

    close = kline_df["close"].values
    vol = kline_df["volume"].values
    high = kline_df["high"].values

    ma5 = np.mean(close[-5:])
    ma10 = np.mean(close[-10:])
    ma20 = np.mean(close[-20:])
    ma60 = np.mean(close[-60:]) if len(close) >= 60 else ma20

    price = close[-1]

    # 均线方向
    ma5_rising = ma5 > np.mean(close[-6:-1])  # MA5 向上
    ma10_rising = ma10 > np.mean(close[-11:-1])

    # 量能
    vol_5 = np.mean(vol[-5:])
    vol_20 = np.mean(vol[-20:])
    vol_ratio = vol_5 / max(vol_20, 1)

    # 价格位置 (与MA20的关系)
    price_vs_ma20 = (price / ma20 - 1) * 100

    # 60日高位判断
    high_60 = np.max(high[-60:])
    near_high = price > high_60 * 0.92  # 距60日高<8%

    features = {
        "ma5": round(ma5, 2), "ma10": round(ma10, 2), "ma20": round(ma20, 2),
        "price_vs_ma20": round(price_vs_ma20, 1),
        "vol_ratio": round(vol_ratio, 2),
        "ma5_rising": bool(ma5_rising), "ma10_rising": bool(ma10_rising),
        "near_60d_high": near_high,
    }

    # === C区 判断 (优先，风险最高) ===
    c_signals = 0
    c_reasons = []

    # C1: 高位倒灌 — 距60日高近+当日收阴+量放大
    if near_high and close[-1] < close[-2] and vol_ratio > 1.3:
        c_signals += 2
        c_reasons.append("高位放量收阴(倒灌)")

    # C2: 均线空头 — MA5 < MA10 且价格在MA10下方
    if ma5 < ma10 and price < ma10:
        c_signals += 2
        c_reasons.append("均线死叉+价在线下")

    # C3: 价格远离MA20下方 (>8%)
    if price_vs_ma20 < -8:
        c_signals += 1
        c_reasons.append(f"深度破位({price_vs_ma20:.0f}%)")

    # C4: MA5 和 MA10 都在下行
    if not ma5_rising and not ma10_rising:
        c_signals += 1
        c_reasons.append("均线下行趋势")

    if c_signals >= 2:
        return {"zone": "C", "zone_score": -c_signals,
                "zone_reason": "; ".join(c_reasons), "features": features}

    # === A区 判断 ===
    a_signals = 0
    a_reasons = []

    # A1: 均线多头 — MA5>MA10>MA20>MA60
    if ma5 > ma10 > ma20 > ma60:
        a_signals += 3
        a_reasons.append("完全多头排列")

    # A2: 价格在MA5上方 + MA5上升
    if price > ma5 and ma5_rising:
        a_signals += 2
        a_reasons.append("强势在线")

    # A3: 量能温和放大 (1.0-2.5倍)
    if 1.0 < vol_ratio < 2.5:
        a_signals += 1
        a_reasons.append("量价配合")

    # A4: 价格在MA20上方5-20% (安全距离)
    if 5 < price_vs_ma20 < 20:
        a_signals += 1
        a_reasons.append("适中偏离")

    if a_signals >= 4:
        return {"zone": "A", "zone_score": a_signals,
                "zone_reason": "; ".join(a_reasons), "features": features}

    # === 量价异动+均线归位 检测 (A区特殊信号) ===
    # 即使未达A区标准，量价异动+均线归位也是强A信号
    anomaly_return = detect_volume_price_anomaly(kline_df)
    ma_return = detect_ma_realignment(kline_df)
    if anomaly_return["has_anomaly"] and ma_return["realigning"]:
        return {"zone": "A", "zone_score": 6,
                "zone_reason": f"量价异动+均线归位 ({anomaly_return['type']})",
                "features": features}

    # === B区 ===
    b_reasons = []
    if ma5 > ma10 or price > ma20:
        b_reasons.append("偏多震荡")
    else:
        b_reasons.append("中性整理")
    return {"zone": "B", "zone_score": 1,
            "zone_reason": "; ".join(b_reasons), "features": features}


# ══════════════════════════════════════════════════════════════
# 策略 2: 量价异动 + 均线归位 (全书核心信号)
# ══════════════════════════════════════════════════════════════
#
# 逻辑: 量价同时出现异常变化 → 带动均线系统回归合理位置
#   = 强势行情启动的经典变盘信号
#
# 量价异动类型:
#   a) 底部放量突破 — 低位突然放量2倍+ 突破MA20
#   b) 缩量回踩不破 — 回调缩量<0.5倍 回踩MA20不破
#   c) 突破前高放量 — 突破近期高点+量>1.5倍
#   d) 地量后的倍量 — 地量(最低量)后次日2倍+

def detect_volume_price_anomaly(kline_df) -> dict:
    """检测量价异动信号"""
    if len(kline_df) < 60:
        return {"has_anomaly": False, "type": "", "strength": 0}

    close = kline_df["close"].values
    vol = kline_df["volume"].values
    high = kline_df["high"].values

    price = close[-1]
    vol_today = vol[-1]
    vol_20 = np.mean(vol[-21:-1])
    vol_ratio = vol_today / max(vol_20, 1)
    vol_min_20 = np.min(vol[-21:-1])

    ma20 = np.mean(close[-20:])

    # a) 底部放量突破
    low_20 = np.min(close[-21:-1])
    bottom_breakout = (price > ma20 and close[-2] <= ma20
                       and vol_ratio > 1.5 and price > low_20 * 1.05)

    # b) 缩量回踩不破
    pullback_hold = (vol_ratio < 0.6 and price > ma20 * 0.98
                     and close[-1] > close[-2])

    # c) 突破前高
    high_20 = np.max(high[-21:-1])
    breakout_high = (price > high_20 and vol_ratio > 1.3)

    # d) 地量后倍量
    vol_yesterday = vol[-2]
    volume_burst = (vol_yesterday <= vol_min_20 * 1.1
                    and vol_today > vol_yesterday * 1.8)

    if bottom_breakout:
        return {"has_anomaly": True, "type": "底部放量突破MA20",
                "strength": 7, "vol_ratio": round(vol_ratio, 1)}
    elif breakout_high:
        return {"has_anomaly": True, "type": "放量突破前高",
                "strength": 6, "vol_ratio": round(vol_ratio, 1)}
    elif volume_burst:
        return {"has_anomaly": True, "type": "地量后倍量异动",
                "strength": 5, "vol_ratio": round(vol_ratio, 1)}
    elif pullback_hold:
        return {"has_anomaly": True, "type": "缩量回踩不破MA20",
                "strength": 4, "vol_ratio": round(vol_ratio, 1)}

    return {"has_anomaly": False, "type": "", "strength": 0, "vol_ratio": round(vol_ratio, 1)}


def detect_ma_realignment(kline_df) -> dict:
    """
    检测均线归位信号。
    归位 = 此前均线散乱/纠缠后，重新形成多头有序排列
    """
    if len(kline_df) < 30:
        return {"realigning": False, "reason": ""}

    close = kline_df["close"].values
    price = close[-1]

    def calc_ma(arr, n):
        return np.mean(arr[-n:]) if len(arr) >= n else arr[-1]

    ma5_now = calc_ma(close, 5)
    ma10_now = calc_ma(close, 10)
    ma20_now = calc_ma(close, 20)

    # 5日前均线状态 (用于比较)
    ma5_5d = calc_ma(close[:-5], 5)
    ma10_5d = calc_ma(close[:-5], 10)
    ma20_5d = calc_ma(close[:-5], 20)

    # 当前: 多头排列
    now_bull = ma5_now > ma10_now > ma20_now
    # 5日前: 非多头排列 (纠缠或空头)
    past_not_bull = not (ma5_5d > ma10_5d > ma20_5d)

    if now_bull and past_not_bull:
        return {"realigning": True, "reason": "均线从散乱归位多头排列"}
    elif now_bull:
        return {"realigning": True, "reason": "均线维持多头排列"}

    # 价格突破MA20归位
    if price > ma20_now and close[-2] < ma20_now:
        return {"realigning": True, "reason": "价格突破MA20归位"}

    return {"realigning": False, "reason": ""}


# ══════════════════════════════════════════════════════════════
# 策略 3: 单日强硬洗盘 (第五章·第3模式)
# ══════════════════════════════════════════════════════════════
#
# 特征: 前日大阴线(跌>4%) → 次日阳包阴(涨>阴线实体的60%)
#   = 主力故意打压洗盘，次日强力拉回
#
# 确认条件:
#   1. 前日跌幅 > 4%
#   2. 今日收阳 且 收盘 > 前日开盘 (完全反包)
#   3. 今日量 > 前日量 (真金白银拉回)
#   4. 处于A区或B区偏多

def detect_washout_reversal(kline_df) -> dict:
    """检测单日强硬洗盘后的反包信号"""
    if len(kline_df) < 20:
        return {"is_washout": False}

    close = kline_df["close"].values
    open_ = kline_df["open"].values
    vol = kline_df["volume"].values

    if len(close) < 3:
        return {"is_washout": False}

    # 前日
    prev_open = open_[-2]
    prev_close = close[-2]
    prev_vol = vol[-2]
    prev_chg = (prev_close / close[-3] - 1) * 100 if len(close) >= 3 else 0

    # 今日
    today_open = open_[-1]
    today_close = close[-1]
    today_vol = vol[-1]

    # 条件1: 前日大阴 (>4%跌幅)
    if prev_chg > -4:
        return {"is_washout": False}

    # 条件2: 今日阳包阴 (收阳 且 收盘超过前日开盘)
    is_yang = today_close > today_open
    full_engulf = today_close > prev_open

    if not (is_yang and full_engulf):
        return {"is_washout": False}

    # 条件3: 量能确认 (今日量 > 前日量或均值)
    vol_confirm = today_vol > prev_vol

    # 强度评分
    engulf_ratio = (today_close / prev_open - 1) * 100  # 超越前日开盘的幅度
    strength = 5 + min(engulf_ratio * 2, 10) + (3 if vol_confirm else 0)

    return {
        "is_washout": True,
        "strength": round(strength, 1),
        "prev_chg": round(prev_chg, 1),
        "engulf_pct": round(engulf_ratio, 1),
        "vol_confirm": vol_confirm,
        "pattern": "单日强硬洗盘反包",
    }


# ══════════════════════════════════════════════════════════════
# 策略 4: 缺口模式 (第五章·第4模式)
# ══════════════════════════════════════════════════════════════

def detect_gap_signal(kline_df) -> dict:
    """
    检测跳空缺口。

    向上缺口: 今日最低 > 昨日最高 → 强势突破
    向下缺口: 今日最高 < 昨日最低 → 风险信号

    缺口三日不回补 = 强趋势确认
    """
    if len(kline_df) < 5:
        return {"has_gap": False}

    high = kline_df["high"].values
    low = kline_df["low"].values
    close = kline_df["close"].values

    # 今日向上缺口
    gap_up = low[-1] > high[-2]
    # 前日向上缺口 (看3天是否回补)
    gap_up_3d = low[-3] > high[-4] if len(low) >= 4 else False
    gap_filled = gap_up_3d and min(low[-2:]) < high[-4]  # 回补

    result = {"has_gap": False}

    if gap_up:
        gap_pct = (low[-1] / high[-2] - 1) * 100
        result = {"has_gap": True, "direction": "up",
                  "gap_pct": round(gap_pct, 2),
                  "type": "今日向上缺口"}
    elif gap_up_3d and not gap_filled:
        gap_pct = (low[-3] / high[-4] - 1) * 100
        result = {"has_gap": True, "direction": "up",
                  "gap_pct": round(gap_pct, 2),
                  "type": "三日不补缺口(强势确认)"}

    # 向下缺口 — 风险信号
    gap_down = high[-1] < low[-2]
    if gap_down:
        gap_pct = (high[-1] / low[-2] - 1) * 100
        result = {"has_gap": True, "direction": "down",
                  "gap_pct": round(gap_pct, 2),
                  "type": "向下缺口(C区风险)"}

    return result


# ══════════════════════════════════════════════════════════════
# 策略 5: 高位倒灌K线 (第六章·卖出信号)
# ══════════════════════════════════════════════════════════════

def detect_distribution_signal(kline_df) -> dict:
    """
    检测出货/见顶信号。

    1. 高位倒灌: 高开→低走收阴，量放大 → 主力出货
    2. 阳奉阴违: 低开收阳但收盘<昨日收盘 → 表面强实则弱
    3. 放量滞涨: 量>1.5倍但涨幅<1% → 抛压沉重
    """
    if len(kline_df) < 60:
        return {"is_distribution": False}

    close = kline_df["close"].values
    open_ = kline_df["open"].values
    high = kline_df["high"].values
    vol = kline_df["volume"].values

    price = close[-1]
    high_60 = np.max(high[-60:])
    vol_20 = np.mean(vol[-21:-1])
    vol_ratio = vol[-1] / max(vol_20, 1)

    near_high = price > high_60 * 0.90

    result = {"is_distribution": False, "signals": []}

    # 1. 高位倒灌
    if near_high and open_[-1] > close[-1] and close[-1] < close[-2] and vol_ratio > 1.3:
        result["is_distribution"] = True
        result["signals"].append({
            "type": "高位倒灌", "severity": "高",
            "desc": f"60日高位+高开低走+放量{vol_ratio:.1f}倍"
        })

    # 2. 阳奉阴违 (假阳线)
    if close[-1] > open_[-1] and close[-1] < close[-2] and vol_ratio > 1.2:
        result["is_distribution"] = True
        result["signals"].append({
            "type": "阳奉阴违(假阳线)", "severity": "中",
            "desc": "收阳但低于昨收+放量"
        })

    # 3. 放量滞涨
    if abs((close[-1] / close[-2] - 1) * 100) < 1 and vol_ratio > 1.5:
        result["is_distribution"] = True
        result["signals"].append({
            "type": "放量滞涨", "severity": "中",
            "desc": f"涨幅微弱+量{vol_ratio:.1f}倍"
        })

    return result


# ══════════════════════════════════════════════════════════════
# 策略 6: 量能体叠加评分 (第七章·资金安全护城河)
# ══════════════════════════════════════════════════════════════
#
# "单打独斗的信号不可靠，多重共振才安全"
#
# 综合上述所有信号 + 已有战法信号 + 技术面 → 共振总分

def score_signal_resonance(kline_df, existing_signals: dict = None) -> dict:
    """
    多信号共振评分 — 量能体叠加。

    Args:
        kline_df: K线数据
        existing_signals: 已有的战法/技术面信号 (可选)

    Returns:
        {resonance_score, active_signals, recommendation}
    """
    signals = []
    total = 0

    # 1. ABC 三区
    zone = classify_abc_zone(kline_df)
    if zone["zone"] == "A":
        total += 10
        signals.append(f"A区强势({zone['zone_reason']})")
    elif zone["zone"] == "B":
        total += 3
        signals.append(f"B区次级({zone['zone_reason']})")
    else:  # C
        total -= 10
        signals.append(f"⚠C区风险({zone['zone_reason']})")

    # 2. 量价异动
    anomaly = detect_volume_price_anomaly(kline_df)
    if anomaly["has_anomaly"]:
        bonus = anomaly["strength"]
        total += bonus
        signals.append(f"量价异动:{anomaly['type']}(+{bonus})")

    # 3. 均线归位
    ma_re = detect_ma_realignment(kline_df)
    if ma_re["realigning"]:
        total += 5
        signals.append(f"均线归位({ma_re['reason']})")

    # 4. 洗盘反包
    washout = detect_washout_reversal(kline_df)
    if washout["is_washout"]:
        bonus = int(washout["strength"])
        total += bonus
        signals.append(f"洗盘反包(强度{bonus})")

    # 5. 缺口
    gap = detect_gap_signal(kline_df)
    if gap["has_gap"]:
        if gap["direction"] == "up":
            total += 6
            signals.append(f"向上缺口({gap['type']})")
        else:
            total -= 5
            signals.append(f"⚠向下缺口")

    # 6. 出货检测 (负面影响)
    dist = detect_distribution_signal(kline_df)
    if dist["is_distribution"]:
        for s in dist["signals"]:
            penalty = 8 if s["severity"] == "高" else 5
            total -= penalty
            signals.append(f"⚠{s['type']}(-{penalty})")

    # 7. 叠加已有信号
    if existing_signals:
        for sig_name, sig_score in existing_signals.items():
            total += sig_score
            if sig_score > 0:
                signals.append(f"已有:{sig_name}(+{sig_score})")

    # 综合建议
    if total >= 15:
        rec = "STRONG_BUY"
    elif total >= 8:
        rec = "BUY"
    elif total >= 2:
        rec = "WATCH"
    elif total >= -5:
        rec = "HOLD"
    else:
        rec = "AVOID"

    return {
        "resonance_score": total,
        "active_signals": signals,
        "n_signals": len(signals),
        "recommendation": rec,
        "zone": zone["zone"],
    }


# ══════════════════════════════════════════════════════════════
# CLI 测试
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from data_loader import get_stock_kline

    code = sys.argv[1] if len(sys.argv) > 1 else "600667"
    date = sys.argv[2] if len(sys.argv) > 2 else "2026-06-18"

    market = 1 if code.startswith("6") else 0
    dk = get_stock_kline(code, market, refresh=False)
    if dk is not None:
        dk = dk[dk["date"] <= date]
        resonance = score_signal_resonance(dk)
        print(f"\n{code} {date}")
        print(f"  共振总分: {resonance['resonance_score']}")
        print(f"  区域: {resonance['zone']}")
        print(f"  建议: {resonance['recommendation']}")
        print(f"  信号 ({resonance['n_signals']}个):")
        for s in resonance['active_signals']:
            print(f"    - {s}")
