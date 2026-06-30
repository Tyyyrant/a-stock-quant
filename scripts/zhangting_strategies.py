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
    # A5: 均线密集聚拢 (MA间距<3%)
    ma_spread = (max(ma5, ma10, ma20) / min(ma5, ma10, ma20) - 1)
    if ma_spread < 0.03:
        a_signals += 1; a_reasons.append("均线密集聚拢")
    # A6: 均线向上散发 (MA间距扩大30%+)
    g5 = abs(ma5/ma10 - 1)
    g5_5d = abs(np.mean(close[-5:]) / np.mean(close[-10:-5]) - 1) if len(close) >= 10 else 0
    if g5 > g5_5d * 1.3:
        a_signals += 1; a_reasons.append("均线向上散发")

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

    # === B区 (蓄势待发) ===
    b_signals = 0; b_reasons = []
    if ma5 > ma10: b_signals += 2; b_reasons.append("短线偏多")
    elif price > ma10: b_signals += 1; b_reasons.append("站上MA10")
    if price > ma20: b_signals += 1; b_reasons.append("站上MA20")
    if abs(ma5/ma10 - 1) < 0.02: b_signals += 1; b_reasons.append("均线纠缠蓄势")
    if price_vs_ma20 > 0 and np.mean(close[-5:]) < ma20: b_signals += 2; b_reasons.append("低位突破MA20")
    if 0.8 < vol_ratio < 1.5: b_signals += 1; b_reasons.append("量能正常")
    if b_signals >= 2:
        return {"zone": "B", "zone_score": b_signals,
                "zone_reason": "; ".join(b_reasons), "features": features}
    return {"zone": "—", "zone_score": 0,
            "zone_reason": "均线混乱无方向", "features": features}


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
    """检测量价异动信号 (V2: +再收集 +超常规短资)"""
    if len(kline_df) < 60:
        return {"has_anomaly": False, "type": "", "strength": 0}

    close = kline_df["close"].values
    vol = kline_df["volume"].values
    high = kline_df["high"].values
    open_ = kline_df["open"].values

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

    # e) 再收集 (V2新增): 前期有洗盘→缩量→再放量收集
    re_accumulation = False
    # 条件: 前20日有过上升→有过缩量回调→现在放量回升
    mid_price = np.mean(close[-41:-21])
    had_rise = np.max(close[-41:-21]) > mid_price * 1.10  # 前期有10%+拉升
    had_pullback = (np.min(vol[-31:-11]) < np.mean(vol[-41:-21]) * 0.5)  # 中期有缩量
    now_volume_up = vol_ratio > 1.2 and close[-1] > close[-2]  # 现在放量回升
    if had_rise and had_pullback and now_volume_up:
        re_accumulation = True

    # f) 超常规短资最后投入 (V2新增): 均线多头排列中突然冒出显著增加的阳量堆
    last_capital = False
    ma5 = np.mean(close[-5:])
    ma10 = np.mean(close[-10:])
    vol_burst_days = 0
    for i in range(-1, -8, -1):
        if vol[i] > vol_20 * 1.5 and close[i] > open_[i]:
            vol_burst_days += 1
    if ma5 > ma10 and vol_burst_days >= 2 and vol_ratio > 1.3:
        last_capital = True

    result = {"has_anomaly": False, "type": "", "strength": 0,
              "vol_ratio": round(vol_ratio, 1),
              "re_accumulation": re_accumulation,
              "last_capital": last_capital}

    if bottom_breakout:
        result.update({"has_anomaly": True, "type": "底部放量突破MA20", "strength": 7})
    elif breakout_high:
        result.update({"has_anomaly": True, "type": "放量突破前高", "strength": 6})
    elif volume_burst:
        result.update({"has_anomaly": True, "type": "地量后倍量异动", "strength": 5})
    elif pullback_hold:
        result.update({"has_anomaly": True, "type": "缩量回踩不破MA20", "strength": 4})
    elif last_capital and not result["has_anomaly"]:
        result.update({"has_anomaly": True, "type": "超常规短资最后投入", "strength": 8})
    elif re_accumulation and not result["has_anomaly"]:
        result.update({"has_anomaly": True, "type": "主力再收集", "strength": 6})

    return result


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
    """检测单日强硬洗盘后的反包信号 (V2: +4种位置模式识别)"""
    if len(kline_df) < 20:
        return {"is_washout": False}

    close = kline_df["close"].values
    open_ = kline_df["open"].values
    vol = kline_df["volume"].values
    high = kline_df["high"].values
    low = kline_df["low"].values

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

    # 三种洗盘模式
    is_yang = today_close > today_open
    full_engulf = today_close > prev_open
    prev_high_v = high[-2]
    prev_low_v = low[-2]
    prev_body = abs(prev_close - prev_open)
    prev_upper_shadow = prev_high_v - max(prev_open, prev_close)
    is_long_upper = prev_upper_shadow > prev_body * 1.5 and prev_upper_shadow > 0.02 * prev_close
    is_big_vol = prev_vol > np.mean(vol[-20:]) * 1.5
    is_fake_yang = prev_close > prev_open and (prev_close - prev_low_v) < prev_body * 0.5

    result = {"is_washout": False, "position_type": ""}

    # ── 确定反包模式 ──
    if prev_chg < -4 and is_yang and full_engulf:
        vc = today_vol > prev_vol
        er = (today_close / prev_open - 1) * 100
        result.update({"is_washout": True, "pattern": "大阴洗盘反包",
                       "strength": round(5 + min(er * 2, 10) + (3 if vc else 0), 1),
                       "engulf_pct": round(er, 1), "vol_confirm": vc})
    elif is_long_upper and is_big_vol and is_yang and today_close > prev_close:
        result.update({"is_washout": True, "pattern": "长上影洗盘反包",
                       "strength": round(7 + (3 if today_vol > prev_vol else 0), 1),
                       "engulf_pct": round((today_close / prev_close - 1) * 100, 1),
                       "vol_confirm": today_vol > prev_vol})
    elif is_fake_yang and is_big_vol and is_yang and today_close > prev_high_v:
        result.update({"is_washout": True, "pattern": "黑太阳洗盘反包",
                       "strength": round(8 + (3 if today_vol > prev_vol else 0), 1),
                       "engulf_pct": round((today_close / prev_high_v - 1) * 100, 1),
                       "vol_confirm": today_vol > prev_vol})
    else:
        # 量价严重背离洗盘 (V2新增): 缩量大阴 + 次日阳反转
        prev_vol_ratio = prev_vol / max(np.mean(vol[-21:-1]), 1)
        if prev_chg < -3 and prev_vol_ratio < 0.7 and is_yang and today_close > prev_close:
            result.update({"is_washout": True, "pattern": "量价背离强硬洗盘",
                           "strength": 9,
                           "engulf_pct": round((today_close / prev_close - 1) * 100, 1),
                           "vol_confirm": True})
        else:
            return {"is_washout": False}

    if not result["is_washout"]:
        return {"is_washout": False}

    # ── 位置模式判定 (V2新增) ──
    price = close[-1]
    ma5 = np.mean(close[-5:])
    ma10 = np.mean(close[-10:])
    ma20 = np.mean(close[-20:])
    high_30 = np.max(high[-31:-1])

    zone = classify_abc_zone(kline_df)["zone"]
    if zone in ("A", "B"):
        result["position_type"] = "强势A区B区洗盘"
        result["strength"] = min(result["strength"] + 2, 10)
    elif price > high_30 * 0.95:
        result["position_type"] = "突破前高位置洗盘"
        result["strength"] = min(result["strength"] + 1, 10)
    elif abs(ma5 / ma10 - 1) < 0.02 and abs(ma10 / ma20 - 1) < 0.02:
        result["position_type"] = "均线结点区洗盘"
        result["strength"] = min(result["strength"] + 1, 10)
    else:
        result["position_type"] = "整理平台洗盘"

    return result


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
    检测出货/见顶信号 (V2: +射击之星 +吊颈星线 +高位三星)。

    1. 高位倒灌: 高开→低走收阴，量放大 → 主力出货
    2. 阳奉阴违: 低开收阳但收盘<昨日收盘 → 表面强实则弱
    3. 放量滞涨: 量>1.5倍但涨幅<1% → 抛压沉重
    4. 射击之星: 高位长上影小实体星线 → 见顶 (V2)
    5. 吊颈星线: 高位长下影小实体星线 → 诱多见顶 (V2)
    6. 高位三星: 同价位反复长上/下影星线+阴量 → 暴跌前奏 (V2)
    """
    if len(kline_df) < 60:
        return {"is_distribution": False}

    close = kline_df["close"].values
    open_ = kline_df["open"].values
    high = kline_df["high"].values
    low = kline_df["low"].values
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

    # ── V2新增: 星线现顶信号 ──
    body = abs(close[-1] - open_[-1])
    range_ = high[-1] - low[-1]
    upper_wick = high[-1] - max(close[-1], open_[-1])
    lower_wick = min(close[-1], open_[-1]) - low[-1]
    is_star = range_ > 0 and (body / range_) < 0.4

    pos_pct = (price - np.min(low[-60:])) / max(high_60 - np.min(low[-60:]), 0.01)

    if is_star and pos_pct > 0.80:
        # 4. 射击之星(天针)
        if upper_wick > body * 3 and upper_wick > lower_wick and range_ > price * 0.02:
            result["is_distribution"] = True
            result["signals"].append({
                "type": "射击之星(天针)", "severity": "高",
                "desc": f"高位长上影星线，上影/实体={upper_wick/max(body,0.01):.0f}x"
            })

        # 5. 吊颈星线
        if lower_wick > body * 3 and lower_wick > upper_wick and range_ > price * 0.02:
            result["is_distribution"] = True
            result["signals"].append({
                "type": "吊颈星线", "severity": "高",
                "desc": f"高位长下影星线，收于高位有诱导性"
            })

    # 6. 高位三星顶部
    star_count = 0
    for i in range(-1, -min(12, len(close)), -1):
        b = abs(close[i] - open_[i])
        r = high[i] - low[i]
        uw = high[i] - max(close[i], open_[i])
        lw = min(close[i], open_[i]) - low[i]
        if r > 0 and b / r < 0.4 and abs(close[i] - price) / price < 0.03:
            if uw > b * 2 or lw > b * 2:
                star_count += 1
    yin_vol_count = sum(1 for i in range(-1, -min(12, len(close)), -1)
                       if close[i] < open_[i] and vol[i] > vol_20)
    if star_count >= 2 and pos_pct > 0.75 and yin_vol_count >= 1:
        result["is_distribution"] = True
        result["signals"].append({
            "type": "高位三星顶部", "severity": "高",
            "desc": f"同价区{star_count+1}根星线+阴量冒出→暴跌前奏"
        })

    # 天量阴线补充: 历史最大量+收阴
    if vol[-1] > np.max(vol[:-1]) * 1.1 and close[-1] < open_[-1]:
        result["is_distribution"] = True
        result["signals"].append({
            "type": "天量阴线", "severity": "高",
            "desc": "历史最大量+收阴"
        })

    return result


# ══════════════════════════════════════════════════════════════
# 策略 6: 三阳控三阴 (书2 — 成交量形态核心体系)
# ══════════════════════════════════════════════════════════════
#
# 三种量形态:
#   阳众阴寡 — 阳量多、阴量少 → 买盘主导
#   阳放阴缩 — 阳线放量、阴线缩量 → 最健康
#   阳聚阴散 — 阳量聚集堆积、阴量散乱 → 资金有组织

def detect_three_yang_control(kline_df, lookback: int = 30) -> dict:
    """
    三阳控三阴检测。

    Returns:
        {pattern, score, yang_ratio, yang_vol_ratio, details}
    """
    if len(kline_df) < lookback:
        return {"pattern": "unknown", "score": 0, "reason": "数据不足"}

    close = kline_df["close"].values[-lookback:]
    open_ = kline_df["open"].values[-lookback:]
    vol = kline_df["volume"].values[-lookback:]

    # 统计阳线/阴线
    is_yang = close > open_
    yang_count = np.sum(is_yang)
    yin_count = lookback - yang_count
    yang_ratio = yang_count / lookback

    # 阳量 vs 阴量 均值
    yang_vol = np.mean(vol[is_yang]) if yang_count > 0 else 0
    yin_vol = np.mean(vol[~is_yang]) if yin_count > 0 else 1
    yang_vol_ratio = yang_vol / max(yin_vol, 1)

    # 阳量标准差 (判断是否聚集)
    yang_vol_std = np.std(vol[is_yang]) if yang_count > 2 else 0
    yin_vol_std = np.std(vol[~is_yang]) if yin_count > 2 else 0

    score = 0
    patterns = []
    details = {"yang_count": int(yang_count), "yin_count": int(yin_count),
               "yang_ratio": round(yang_ratio, 2), "yang_vol_ratio": round(yang_vol_ratio, 2)}

    # 1. 阳众阴寡 (阳线数量占优)
    if yang_ratio >= 0.55:
        score += 2
        patterns.append("阳众阴寡")
        details["zhong"] = f"阳线{yang_ratio:.0%}"

    # 2. 阳放阴缩 (阳线放量、阴线缩量)
    if yang_vol_ratio > 1.2:
        score += 3
        patterns.append("阳放阴缩")
        details["fang"] = f"阳量/阴量={yang_vol_ratio:.1f}x"

    # 3. 阳聚阴散 (阳量集中、阴量分散)
    if yang_vol_std > yin_vol_std * 1.3 and yang_count >= 3:
        score += 2
        patterns.append("阳聚阴散")
        details["ju"] = f"阳量std={yang_vol_std:.0f} vs 阴量std={yin_vol_std:.0f}"

    # 综合判断
    pattern_name = "+".join(patterns) if patterns else "量形态混乱"
    if score >= 5:
        grade = "完美三阳控三阴"
    elif score >= 3:
        grade = "偏多量形态"
    elif score >= 1:
        grade = "弱偏多量形态"
    else:
        grade = "量形态偏弱"

    return {
        "pattern": pattern_name,
        "score": score,
        "grade": grade,
        "details": details,
    }


# ══════════════════════════════════════════════════════════════
# 策略 7: 星线全体系分类 (书3 核心扩展)
# ══════════════════════════════════════════════════════════════
#
# 四大类:
#   调整星线: 缓冲星线、震荡星线、巨星
#   止跌星线: 同步止跌、背离止跌
#   蓄势星线: 诱空蓄势、平台蓄势
#   现顶星线: 射击之星(天针)、吊颈星线、高位三星

def classify_star_pattern(kline_df, lookback: int = 10) -> dict:
    """
    星线全体系分类。

    Returns:
        {star_type, sub_type, direction, confidence, signals}
    """
    if len(kline_df) < 30:
        return {"star_type": "unknown", "sub_type": "", "direction": "neutral", "confidence": 0}

    close = kline_df["close"].values
    open_ = kline_df["open"].values
    high = kline_df["high"].values
    low = kline_df["low"].values
    vol = kline_df["volume"].values

    price = close[-1]
    body = abs(close[-1] - open_[-1])
    range_ = high[-1] - low[-1]
    upper_wick = high[-1] - max(close[-1], open_[-1])
    lower_wick = min(close[-1], open_[-1]) - low[-1]

    # 是否星线: 实体/振幅 < 0.4
    is_star = range_ > 0 and (body / range_) < 0.4 and range_ > 0
    if not is_star:
        # 宽松条件: 振幅>1.8% 且 实体/收盘<1.5%
        body_pct = body / price * 100 if price > 0 else 0
        range_pct = range_ / price * 100 if price > 0 else 0
        is_star = range_pct > 1.8 and body_pct < 1.5

    if not is_star:
        return {"star_type": "非星线", "sub_type": "", "direction": "neutral", "confidence": 0}

    ma5 = np.mean(close[-5:])
    ma10 = np.mean(close[-10:])
    ma20 = np.mean(close[-20:]) if len(close) >= 20 else ma10
    ma60 = np.mean(close[-60:]) if len(close) >= 60 else ma20
    vol_5 = np.mean(vol[-6:-1])
    vol_ratio = vol[-1] / max(vol_5, 1)

    high_60 = np.max(high[-60:]) if len(high) >= 60 else price * 1.5
    low_60 = np.min(low[-60:]) if len(low) >= 60 else price * 0.5
    pos_pct = (price - low_60) / max(high_60 - low_60, 0.01)

    # 前5天趋势
    prev_5_chg = (close[-6] / close[-11] - 1) * 100 if len(close) >= 11 else 0

    signals = []
    star_type = "unknown"
    sub_type = ""
    direction = "neutral"
    confidence = 0

    # ── 1. 现顶星线 (优先级最高——高位风险) ──
    if pos_pct > 0.80:
        # 射击之星(天针): 长上影+小实体+高位
        if upper_wick > body * 3 and upper_wick > lower_wick * 2 and range_ > price * 0.03:
            star_type = "现顶星线"
            sub_type = "射击之星(天针)"
            direction = "bearish"
            confidence = 0.8 if vol_ratio > 1.2 else 0.6
            signals.append(f"高位长上影天针(上影/实体={upper_wick/max(body,0.01):.0f}x)")

        # 吊颈星线: 长下影+小实体+高位+收高位
        elif lower_wick > body * 3 and lower_wick > upper_wick * 2 and range_ > price * 0.03:
            star_type = "现顶星线"
            sub_type = "吊颈星线"
            direction = "bearish"
            confidence = 0.7 if vol_ratio > 1.0 else 0.5
            signals.append(f"高位吊颈(下影/实体={lower_wick/max(body,0.01):.0f}x)")

    # 高位三星 (看过去是否同价位反复出现长上影或长下影)
    star_count_high = 0
    for i in range(-1, -min(lookback + 1, len(close)), -1):
        if i == -1:
            continue  # skip current
        b = abs(close[i] - open_[i])
        r = high[i] - low[i]
        uw = high[i] - max(close[i], open_[i])
        lw = min(close[i], open_[i]) - low[i]
        if r > 0 and b / r < 0.4 and abs(close[i] - price) / price < 0.03:
            if uw > b * 2 or lw > b * 2:
                star_count_high += 1
    if star_count_high >= 2 and pos_pct > 0.75:
        star_type = "现顶星线"
        sub_type = "高位三星顶部"
        direction = "bearish"
        confidence = 0.85
        signals.append(f"同一价区{star_count_high+1}根星线→暴跌前奏")

    # ── 2. 蓄势星线 ──
    if star_type == "unknown" and pos_pct > 0.20 and pos_pct < 0.80:
        # 诱空蓄势: 前期有强势异动+股价主动向下+在关键位置止跌+横向蓄势
        had_strong = False
        for i in range(-11, -2):
            chg_i = (close[i] / close[i-1] - 1) * 100
            if chg_i > 5 or (vol[i] > np.mean(vol[-20:]) * 1.8 and close[i] > open_[i]):
                had_strong = True
                break

        if had_strong and prev_5_chg < -3 and price > ma20:
            star_type = "蓄势星线"
            sub_type = "诱空蓄势星线"
            direction = "bullish_pending"
            confidence = 0.55
            signals.append("前期强资+主动向下+关键位止跌蓄势")

        # 平台蓄势: 均线上方横盘、星线排列
        elif price > ma10 and abs(prev_5_chg) < 2 and vol_ratio < 0.8:
            star_type = "蓄势星线"
            sub_type = "平台蓄势星线"
            direction = "bullish_pending"
            confidence = 0.5
            signals.append("MA上方横盘蓄势缩量")

    # ── 3. 止跌星线 ──
    if star_type == "unknown" and pos_pct < 0.40:
        vol_shrinking = vol_ratio < 0.7

        if vol_shrinking:
            # 同步止跌: K线收敛 + 量能同步收敛
            star_type = "止跌星线"
            sub_type = "同步止跌星线"
            direction = "bullish"
            confidence = 0.5
            signals.append("量价同步收敛止跌")
        else:
            # 背离止跌: K线收敛但量能暗流涌动
            star_type = "止跌星线"
            sub_type = "背离止跌星线"
            direction = "bullish"
            confidence = 0.65
            signals.append("价缩量不缩→主力暗中吸筹")

    # ── 4. 调整星线 ──
    if star_type == "unknown":
        if range_ > price * 0.05 and vol_ratio > 1.2:
            star_type = "调整星线"
            sub_type = "巨星(剧烈博弈)"
            direction = "bullish_pending"
            confidence = 0.55
            signals.append("巨量宽幅震荡→大行情前兆")
        elif range_ > price * 0.03 and vol_ratio > 0.9:
            star_type = "调整星线"
            sub_type = "震荡星线"
            direction = "neutral"
            confidence = 0.4
            signals.append("宽幅震荡洗盘")
        else:
            star_type = "调整星线"
            sub_type = "缓冲星线"
            direction = "bullish_pending"
            confidence = 0.45
            signals.append("动量后缓冲蓄力")

    # 弱势现顶补充: 偏离EMA20过大(>25%)的小星线=警惕
    if star_type == "unknown" and (price / ma20 - 1) * 100 > 25:
        star_type = "现顶星线"
        sub_type = "高位停滞星线"
        direction = "bearish"
        confidence = 0.5
        signals.append(f"偏离MA20 {(price/ma20-1)*100:.0f}%的星线")

    if star_type == "unknown":
        star_type = "普通星线"
        sub_type = ""
        direction = "neutral"
        confidence = 0.2

    return {
        "star_type": star_type,
        "sub_type": sub_type,
        "direction": direction,
        "confidence": round(confidence, 2),
        "is_star": True,
        "body_pct": round(body / price * 100, 2) if price > 0 else 0,
        "range_pct": round(range_ / price * 100, 2) if price > 0 else 0,
        "upper_wick_ratio": round(upper_wick / max(body, 0.01), 1),
        "lower_wick_ratio": round(lower_wick / max(body, 0.01), 1),
        "vol_ratio": round(vol_ratio, 2),
        "pos_pct": round(pos_pct * 100),
        "signals": signals,
    }


# ══════════════════════════════════════════════════════════════
# 策略 8: 量能体叠加评分 (书2 — 多重共振)
# ══════════════════════════════════════════════════════════════
#
# "单打独斗的信号不可靠，多重共振才安全"
#
# 综合上述所有信号 + 已有战法信号 + 技术面 → 共振总分

def score_signal_resonance(kline_df, existing_signals: dict = None) -> dict:
    """
    多信号共振评分 — 量能体叠加 (V2 升级版)。

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
        total += 5
        signals.append(f"B区蓄势({zone['zone_reason']})")
    elif zone["zone"] == "—":
        signals.append(f"无明确区间({zone['zone_reason']})")
    else:  # C
        total -= 10
        signals.append(f"⚠C区风险({zone['zone_reason']})")

    # 2. 量价异动
    anomaly = detect_volume_price_anomaly(kline_df)
    if anomaly["has_anomaly"]:
        bonus = anomaly["strength"]
        total += bonus
        signals.append(f"量价异动:{anomaly['type']}(+{bonus})")
        # 再收集信号
        if anomaly.get("re_accumulation"):
            total += 4
            signals.append("再收集(+4)")
        # 超常规短资
        if anomaly.get("last_capital"):
            total += 5
            signals.append("超常规短资(+5)")

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
        pos_label = washout.get("position_type", "")
        signals.append(f"洗盘反包[{pos_label}](强度{bonus})")

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

    # 7. 三阳控三阴 (V2新增)
    tyc = detect_three_yang_control(kline_df)
    if tyc["score"] >= 3:
        total += tyc["score"]
        signals.append(f"三阳控三阴:{tyc['grade']}(+{tyc['score']})")

    # 8. 星线信号 (V2新增)
    star = classify_star_pattern(kline_df)
    if star["star_type"] != "非星线" and star["star_type"] != "普通星线":
        if star["direction"] == "bearish":
            total -= int(star["confidence"] * 8)
            signals.append(f"⚠{star['sub_type']}(-{int(star['confidence']*8)})")
        elif star["direction"] == "bullish":
            total += int(star["confidence"] * 5)
            signals.append(f"{star['sub_type']}(+{int(star['confidence']*5)})")
        elif star["direction"] == "bullish_pending":
            total += int(star["confidence"] * 3)
            signals.append(f"{star['sub_type']}[待确认](+{int(star['confidence']*3)})")

    # 9. 底部积累 (V2新增)
    vol_acc = detect_volume_accumulation(kline_df)
    if vol_acc.get("giant_yang") and vol_acc.get("giant_yang", {}).get("detected"):
        total += 4
        signals.append("巨量阳线识底(+4)")
    if vol_acc.get("yinyang_embrace") and vol_acc.get("yinyang_embrace", {}).get("detected"):
        total += 3
        signals.append("阴阳合抱见底(+3)")

    # 10. 叠加已有信号
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

def detect_volume_accumulation(kline_df) -> dict:
    """
    三度之"厚度" — 量形态的高度·宽度·密集度 × 位置系数 (V2: +巨量阳线识底 +阴阳合抱)

    原著定义: 厚度 = 量形态之高度、宽度、密集度
    关键前提: 低位堆量=收集(加分)，高位堆量=出货(警惕)
    """
    if len(kline_df) < 60: return {"has_thickness": False, "score": 0}
    c = kline_df["close"].values; v = kline_df["volume"].values
    o = kline_df["open"].values; h = kline_df["high"].values; l = kline_df["low"].values

    v7 = np.mean(v[-8:-1]) if len(v) >= 8 else np.mean(v)
    v_prev7 = np.mean(v[-14:-7]) if len(v) >= 14 else v7

    # ── 三维评分 (各0-2分，总共0-6) ──
    # 高度: 近7天大阳量根数 (量>1.5x基线+阳线)
    tb = sum(1 for i in range(-7, -1) if v[i] > v7*1.5 and c[i] > o[i])
    score_h = 2 if tb >= 3 else (1 if tb >= 1 else 0)

    # 宽度: 近7天放量天数 (量>1.2x基线)
    wd = sum(1 for i in range(-7, -1) if v[i] > v7*1.2)
    score_w = 2 if wd >= 5 else (1 if wd >= 3 else 0)

    # 密集度: 近7天阳量占比
    yr = sum(1 for i in range(-7, -1) if c[i] > o[i]) / 7
    streak = 0
    for i in range(-1, -8, -1):
        if c[i] > o[i]: streak += 1
        else: break
    score_d = 2 if yr >= 0.6 else (1 if yr >= 0.45 or streak >= 3 else 0)

    raw = score_h + score_w + score_d  # 0-6

    # ── 位置系数 ──
    price = c[-1]
    high_60 = np.max(h[-60:]); low_60 = np.min(l[-60:])
    pos_pct = (price - low_60) / max(high_60 - low_60, 0.01)
    if pos_pct < 0.30:
        pos_coef = 1.2; pos_label = "低位"
    elif pos_pct < 0.70:
        pos_coef = 1.0; pos_label = "中位"
    else:
        pos_coef = 0.5; pos_label = "高位"

    final = round(raw * pos_coef, 1)

    # ── 加速度 (加分项，外围) ──
    v5 = np.mean(v[-6:-1]) if len(v) >= 6 else v7
    acc = v5 > v_prev7*1.3
    if acc and final > 0: final = min(final + 1, 6)

    # ── 原因 ──
    rs = []
    if score_h: rs.append(f"高度{score_h}分({tb}根大阳量)")
    if score_w: rs.append(f"宽度{score_w}分({wd}天放量)")
    if score_d: rs.append(f"密集度{score_d}分(阳量{yr:.0%})")
    rs.append(f"{pos_label}×{pos_coef}")
    if acc: rs.append("量能加速")

    # ── V2新增: 巨量阳线识底 ──
    giant_yang = {"detected": False, "position": "", "days_ago": 0}
    v_60 = np.mean(v[-61:-1])
    for i in range(-1, -min(15, len(c)), -1):
        chg_i = (c[i] / c[i-1] - 1) * 100
        if v[i] > v_60 * 3 and c[i] > o[i] and chg_i > 3:
            giant_yang = {
                "detected": True,
                "position": "低位巨量阳线" if pos_pct < 0.35 else "中位放量阳线",
                "days_ago": abs(i + 1),
                "vol_ratio": round(v[i] / v_60, 1),
                "chg_pct": round(chg_i, 1),
            }
            # 巨量后是否缩量蓄势? (观察价值)
            if abs(i + 1) > 1:
                post_vol = np.mean(v[i+1:-1])
                if post_vol < v[i] * 0.3:
                    giant_yang["post_shrink"] = True
                    giant_yang["position"] = "巨量+缩量蓄势完成"
            break

    # ── V2新增: 阴阳合抱见底 ──
    yinyang_embrace = {"detected": False, "strength": 0}
    for i in range(-3, -min(10, len(c)), -1):
        # 前日大阴
        prev_big_bear = (c[i-1] < o[i-1]
                        and abs(c[i-1] / c[i-2] - 1) * 100 > 4
                        and v[i-1] > np.mean(v[-20:]) * 1.2)
        # 今日大阳反包
        today_big_bull = (c[i] > o[i]
                         and c[i] > o[i-1]  # 收盘超过前阴开盘 = 完全反包
                         and v[i] > v[i-1] * 0.8)  # 量能确认
        if prev_big_bear and today_big_bull:
            embrace_pos = (c[i] - low_60) / max(high_60 - low_60, 0.01)
            if embrace_pos < 0.40:  # 低位阴阳合抱
                yinyang_embrace = {
                    "detected": True,
                    "strength": round(5 + (1 - embrace_pos) * 3, 1),
                    "days_ago": abs(i),
                    "position": "低位阴阳合抱见底",
                    "bearish_day": abs(i-1),
                    "bullish_day": abs(i),
                }
                break

    return {
        "has_thickness": final >= 3, "score": final,
        "raw_score": raw, "position_coef": pos_coef, "position_pct": round(pos_pct*100),
        "height": score_h, "width": score_w, "density": score_d,
        "reasons": rs, "tall_bars": tb, "wide_days": wd,
        "streak": streak, "yang_ratio": round(yr, 2),
        "giant_yang": giant_yang,  # V2
        "yinyang_embrace": yinyang_embrace,  # V2
    }


def classify_pullup_intent(kline_df) -> dict:
    """主力意图四分类"""
    if len(kline_df) < 60: return {"intent": "unknown", "confidence": 0, "price_position_pct": 0}
    c = kline_df["close"].values; v = kline_df["volume"].values
    h = kline_df["high"].values; l = kline_df["low"].values; o = kline_df["open"].values
    p = c[-1]; h60 = np.max(h[-60:]); l60 = np.min(l[-60:])
    pct = (p - l60) / max(h60 - l60, 0.01)
    v20 = np.mean(v[-21:-1]); vr = v[-1] / max(v20, 1)
    bb = sum(1 for i in range(-40, -10) if c[i] < np.mean(c[-60:])*1.1 and v[i] > v20*1.5)
    m5 = np.mean(c[-5:]); m10 = np.mean(c[-10:]); pos = round(pct*100)
    if pct > 0.85 and vr > 1.5 and c[-1] < o[-1]:
        return {"intent": "拉高出货", "confidence": 0.7, "price_position_pct": pos, "detail": "高位放量收阴，主力派发"}
    elif bb >= 3 and pct < 0.5:
        return {"intent": "抢筹建仓", "confidence": 0.6+min(bb*0.05,0.3), "price_position_pct": pos, "detail": f"底部堆量{int(bb)}天"}
    elif pct > 0.5 and vr > 1.2 and m5 > m10:
        return {"intent": "拉高冲刺", "confidence": 0.55, "price_position_pct": pos, "detail": "中高位放量推进，加速拉升"}
    elif pct < 0.3 and np.mean(v[-5:]) < v20*0.7:
        return {"intent": "收筹硬盘", "confidence": 0.5, "price_position_pct": pos, "detail": "底部缩量磨盘"}
    elif vr > 1.5 and c[-1] > o[-1] and pct < 0.65:
        return {"intent": "探测盘面", "confidence": 0.5, "price_position_pct": pos, "detail": "放量大阳，可能试盘"}
    return {"intent": "常规运作", "confidence": 0.3, "price_position_pct": pos, "detail": "无特殊意图"}


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
