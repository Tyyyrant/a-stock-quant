#!/usr/bin/env python3
"""
九大战法信号识别器 — 《股是股非》全书体系

1. 逼空星线: MACD强势 + 缩量星线洗盘 + 等次日大阳确认 (仅主板600/000)
2. 猎取B区: MA10首次金叉MA30趋势反转
3. A区起涨: 量能MA7金叉MA35 + 逐日放量 + 价站翘MA10 (作者自定义"A区")
4. 拉高抢筹: 量比>2 + 涨幅>4.5% + 收高位98% + 站MA5 + 均线多头 (仅主板600/000)
5. C区风险过滤: 高位倒灌/均线死叉/深度破位 → 硬过滤
6. 量价异动+均线归位: 全书核心变盘信号 (底部放量突破/地量倍量/缩量回踩)
7. 单日强硬洗盘: 前日大阴→次日阳包阴+量确认 = 反包买点
8. 缺口模式: 向上缺口三日不补=强势 / 向下缺口=风险
9. 高位倒灌出货: 60日高位高开低走放量/阳奉阴违/放量滞涨
"""

import numpy as np
import pandas as pd


def detect_all_warfare(code: str, df: pd.DataFrame) -> dict:
    """
    严格按文档公式检测四大战法。

    Args:
        code: 6位股票代码
        df: OHLCV DataFrame, date列为索引或列
    Returns:
        {战法名: {triggered, score, conditions_met, conditions_missed, detail}}
    """
    if df.empty or len(df) < 60:
        return {k: {"triggered": False, "score": 0, "conditions": "数据不足"}
                for k in ["逼空星线", "猎取B区", "A区起涨", "拉高抢筹"]}

    o = df["open"].values.astype(float)
    h = df["high"].values.astype(float)
    l = df["low"].values.astype(float)
    c = df["close"].values.astype(float)
    v = df["volume"].values.astype(float)
    n = len(c)
    idx = -1
    result = {}

    # ================================================================
    # 公共指标
    # ================================================================
    ma5 = pd.Series(c).rolling(5).mean().values
    ma10 = pd.Series(c).rolling(10).mean().values
    ma20 = pd.Series(c).rolling(20).mean().values
    mv5 = pd.Series(v).rolling(5).mean().values
    mv7 = pd.Series(v).rolling(7).mean().values
    mv35 = pd.Series(v).rolling(35).mean().values

    # MACD
    ema12 = pd.Series(c).ewm(span=12, adjust=False).mean().values
    ema26 = pd.Series(c).ewm(span=26, adjust=False).mean().values
    dif = ema12 - ema26
    dea = pd.Series(dif).ewm(span=9, adjust=False).mean().values
    macd_bar = 2 * (dif - dea)

    is_main_board = code.startswith("60") or code.startswith("00")

    # ================================================================
    # 1. 逼空星线 (严格按文档)
    # ================================================================
    bsx_met = []; bsx_miss = []

    # A区: MACD>0 AND MACD>REF(MACD,1)
    a_zone = (macd_bar > 0) & (np.diff(macd_bar, prepend=macd_bar[0]) > 0)
    # B区: MACD>0 AND MACD<REF(MACD,1)
    b_zone = (macd_bar > 0) & (np.diff(macd_bar, prepend=macd_bar[0]) < 0)
    ab_days = np.sum((a_zone | b_zone)[-10:])
    strong_ab = ab_days >= 7
    (bsx_met if strong_ab else bsx_miss).append(f"强拉伸:{ab_days}/10天")

    # 有涨幅: HHV(H,10)/LLV(L,10)-1 >= 10%
    hhv10 = np.max(h[-10:]); llv10 = np.min(l[-10:])
    range_pct = hhv10 / llv10 * 100 - 100
    has_range = range_pct >= 10
    (bsx_met if has_range else bsx_miss).append(f"振幅{range_pct:.0f}%")

    # 未透支: DIFF>DEA AND MACD>0
    not_overbought = dif[idx] > dea[idx] and macd_bar[idx] > 0
    (bsx_met if not_overbought else bsx_miss).append(f"DIFF>DEA(未透支)")

    # 星线: 实体/(H-L)<0.4 AND 实体/C*100<1.5 AND 振幅>1.8
    body = abs(c[idx] - o[idx])
    hl_range = h[idx] - l[idx]
    body_ratio = body / (hl_range + 0.001)
    body_pct = body / c[idx] * 100
    amplitude = hl_range / c[idx-1] * 100 if n >= 2 else 0  # 文档: (H-L)/REF(C,1)*100
    is_star = body_ratio < 0.4 and body_pct < 1.5 and amplitude > 1.8
    (bsx_met if is_star else bsx_miss).append(
        f"星线:体比{body_ratio:.1f}/体幅{body_pct:.1f}/振幅{amplitude:.1f}")

    # 缩量: VOL < MA(VOL,5) * 0.8
    shrinking = v[idx] < mv5[idx] * 0.8 if mv5[idx] > 0 else False
    (bsx_met if shrinking else bsx_miss).append(f"缩量:{v[idx]/max(mv5[idx],1):.1f}x")

    # 逼空星线: C>=MA5 AND C>REF(C,1)*0.995 AND C/O>0.995
    bkx_star = c[idx] >= ma5[idx] and c[idx] > c[idx-1]*0.995 and c[idx]/o[idx] > 0.995
    # 弱回调星线: C<REF(C,1) AND C>MA10
    weak_pullback_star = c[idx] < c[idx-1] and c[idx] > ma10[idx]
    star_position = bkx_star or weak_pullback_star
    (bsx_met if star_position else bsx_miss).append(
        f"位置:{'逼空站MA5' if bkx_star else ('弱回调至MA10' if weak_pullback_star else '不在位')}")

    # 主板
    (bsx_met if is_main_board else bsx_miss).append("主板")

    # 月涨幅 < 40%
    ret_20d = (c[idx] / c[-21] * 100 - 100) if n >= 21 else 0
    monthly_ok = ret_20d < 40
    (bsx_met if monthly_ok else bsx_miss).append(f"月涨幅{ret_20d:.0f}%")

    bsx_triggered = strong_ab and has_range and not_overbought and is_star and shrinking and star_position and is_main_board and monthly_ok
    result["逼空星线"] = {
        "triggered": bsx_triggered,
        "conditions_met": bsx_met,
        "conditions_missed": bsx_miss,
        "score": len(bsx_met),
        "detail": " | ".join(bsx_met) if bsx_met else "未触发",
        "wait_for": "次日放量阳线确认" if bsx_triggered else "",
    }

    # ================================================================
    # 2. 猎取B区 (金叉2日内)
    # ================================================================
    lb_met = []; lb_miss = []

    # 多头排列: MA10 > MA30 且在金叉2日内
    ma30 = pd.Series(c).rolling(30).mean().values
    bull_align = ma10[idx] > ma30[idx]
    # 检查当日/前1日/前2日是否发生金叉
    cross_within_2d = False
    cross_day = -1
    for d in [0, 1, 2]:
        if idx-d < 1: continue
        today = ma10[idx-d] > ma30[idx-d]
        yesterday = ma10[idx-d-1] > ma30[idx-d-1]
        if today and not yesterday:
            cross_within_2d = True
            cross_day = d
            break
    if cross_within_2d:
        lb_met.append(f"金叉{cross_day}日内")
    elif bull_align:
        lb_miss.append(f"已金叉超2日(仍在MA30上)")
    else:
        lb_miss.append("未金叉MA30")

    # 趋势确认: MA10 > REF(MA10,1)
    ma10_up = ma10[idx] > ma10[idx-1] if idx >= 1 else False
    (lb_met if ma10_up else lb_miss).append("MA10上翘" if ma10_up else "MA10未翘")

    lb_triggered = cross_within_2d and ma10_up
    result["猎取B区"] = {
        "triggered": lb_triggered,
        "conditions_met": lb_met,
        "conditions_missed": lb_miss,
        "score": 15 if lb_triggered else 0,
        "detail": " | ".join(lb_met) if lb_met else "未触发",
        "wait_for": "注意若高位票需谨慎" if lb_triggered else "",
    }

    # ================================================================
    # 3. A区起涨 (严格按文档 — 作者自定义"A区"，非MACD A区)
    # ================================================================
    aq_met = []; aq_miss = []

    # 价格A区: MA10上翘 AND C>MA10 AND C>O AND C>REF(C,1)
    price_a = ma10[idx] > ma10[idx-1] and c[idx] > ma10[idx] and c[idx] > o[idx] and c[idx] > c[idx-1]
    (aq_met if price_a else aq_miss).append(
        f"价格A区(翘{ma10[idx]>ma10[idx-1]}/站{c[idx]>ma10[idx]}/涨{c[idx]>c[idx-1]})")

    # 量A区: CROSS(MV7, MV35)
    mv7_now = mv7[idx]; mv35_now = mv35[idx]
    mv7_prev = mv7[idx-1]; mv35_prev = mv35[idx-1]
    vol_cross = mv7_prev <= mv35_prev and mv7_now > mv35_now
    (aq_met if vol_cross else aq_miss).append(
        f"量A区(金叉:{mv7_prev:.0f}≤{mv35_prev:.0f}→{mv7_now:.0f}>{mv35_now:.0f})")

    # 量递增: REF(V,1)>REF(V,2) AND REF(V,2)>REF(V,3)
    vol_inc = v[idx-1] > v[idx-2] and v[idx-2] > v[idx-3] if idx >= 3 else False
    (aq_met if vol_inc else aq_miss).append(
        f"量递增({v[idx-1]:.0f}>{v[idx-2]:.0f}>{v[idx-3]:.0f})" if idx>=3 else "量递增(N/A)")

    # 日放量: V > REF(V,1)
    vol_today_up = v[idx] > v[idx-1] if idx >= 1 else False
    (aq_met if vol_today_up else aq_miss).append(
        f"日放量({v[idx]:.0f}>{v[idx-1]:.0f})" if idx>=1 else "日放量(N/A)")

    aq_triggered = price_a and vol_cross and vol_inc and vol_today_up
    result["A区起涨"] = {
        "triggered": aq_triggered,
        "conditions_met": aq_met,
        "conditions_missed": aq_miss,
        "score": len(aq_met) * 5 if aq_triggered else max(0, (len(aq_met) - 1) * 5),
        "detail": " | ".join(aq_met) if aq_met else "未触发",
        "wait_for": "可做底仓介入" if aq_triggered else "",
    }

    # ================================================================
    # 4. 拉高抢筹 (放宽版 — 仅主板600/000, 5/7触发)
    # ================================================================
    lg_met = []; lg_miss = []

    # 量比: VOL/REF(MA(VOL,5),1) > 1.5 (放宽: 原2.0→1.5)
    prev_mv5 = mv5[idx-1] if idx >= 1 else mv5[idx]
    vol_ratio = v[idx] / prev_mv5 if prev_mv5 > 0 else 0
    huge_vol = vol_ratio > 1.5
    (lg_met if huge_vol else lg_miss).append(f"量比{vol_ratio:.1f}")

    # 单日涨幅: C/REF(C,1) > 1.035 (放宽: 原4.5%→3.5%)
    gain = c[idx] / c[idx-1] - 1 if idx >= 1 else 0
    big_gain = gain > 0.035
    (lg_met if big_gain else lg_miss).append(f"涨幅{gain*100:.1f}%")

    # 突破阳线实体: C/O > 1.03
    yang_body = c[idx] / o[idx] > 1.03 if o[idx] > 0 else False
    (lg_met if yang_body else lg_miss).append(f"实体{c[idx]/o[idx]-1:.1f}%")

    # 收于高位: C/HIGH > 0.98
    close_high = c[idx] / h[idx] > 0.98 if h[idx] > 0 else False
    (lg_met if close_high else lg_miss).append(f"收高位{c[idx]/h[idx]:.3f}")

    # 均线多头排列: MA5>MA10>MA20
    bull_ma = ma5[idx] > ma10[idx] > ma20[idx] if idx >= 0 else False
    (lg_met if bull_ma else lg_miss).append("多头排列" if bull_ma else "均线未多头")

    # 站稳5日线: C>MA5
    stand_ma5 = c[idx] > ma5[idx]
    (lg_met if stand_ma5 else lg_miss).append("站MA5" if stand_ma5 else "未站MA5")

    # 主板过滤
    (lg_met if is_main_board else lg_miss).append("主板600/000")

    # 触发: 5/7条件满足即可 (放宽: 原7/7→5/7)
    lg_triggered = len(lg_met) >= 5
    result["拉高抢筹"] = {
        "triggered": lg_triggered,
        "conditions_met": lg_met,
        "conditions_missed": lg_miss,
        "score": len(lg_met) * 3 if lg_triggered else max(0, (len(lg_met) - 2) * 3),
        "detail": " | ".join(lg_met) if lg_met else "未触发",
        "wait_for": "结合颈线/箱体突破" if lg_triggered else "",
    }

    # ================================================================
    # 5. C区风险过滤 (《股是股非》第四章·风险回避)
    # ================================================================
    cz_met = []; cz_miss = []
    c_signals = 0

    # C1: 高位放量收阴(倒灌) — 距60日高<8% + 收阴 + 量>1.3倍
    hhv60 = np.max(h[-60:])
    near_high = c[idx] > hhv60 * 0.92
    is_yin = c[idx] < o[idx]
    vol_20_avg = np.mean(v[-21:-1])
    vol_ratio_c = v[idx] / max(vol_20_avg, 1)
    if near_high and is_yin and vol_ratio_c > 1.3:
        c_signals += 2; cz_met.append(f"高位放量收阴(量{vol_ratio_c:.1f}x)")
    else:
        cz_miss.append("高位倒灌(否)")

    # C2: 均线死叉 + 价在线下
    if ma5[idx] < ma10[idx] and c[idx] < ma10[idx]:
        c_signals += 2; cz_met.append("均线死叉+价在线下")
    else:
        cz_miss.append("死叉(否)")

    # C3: 深度破位 (>8%低于MA20)
    price_vs_ma20 = (c[idx] / ma20[idx] - 1) * 100 if ma20[idx] > 0 else 0
    if price_vs_ma20 < -8:
        c_signals += 1; cz_met.append(f"深度破位({price_vs_ma20:.0f}%)")
    else:
        cz_miss.append(f"破位(否:{price_vs_ma20:.0f}%)")

    # C4: 均线下行
    ma5_rising_w = ma5[idx] > ma5[idx-1] if idx >= 1 else False
    ma10_rising_w = ma10[idx] > ma10[idx-1] if idx >= 1 else False
    if not ma5_rising_w and not ma10_rising_w:
        c_signals += 1; cz_met.append("均线下行趋势")
    else:
        cz_miss.append("下行(否)")

    cz_triggered = c_signals >= 2
    result["C区风险"] = {
        "triggered": cz_triggered,
        "conditions_met": cz_met,
        "conditions_missed": cz_miss,
        "score": -8 if cz_triggered else 0,
        "detail": " | ".join(cz_met) if cz_met else "安全",
        "action": "硬过滤·不参与" if cz_triggered else "",
    }

    # ================================================================
    # 6. 量价异动+均线归位 (《股是股非》第三章·核心变盘信号)
    # ================================================================
    yj_met = []; yj_miss = []; yj_score = 0

    # 底部放量突破MA20
    low_20d = np.min(c[-21:-1])
    if c[idx] > ma20[idx] and c[idx-1] <= ma20[idx] and vol_ratio_c > 1.5 and c[idx] > low_20d * 1.05:
        yj_score += 7; yj_met.append(f"底部放量突破MA20(量{vol_ratio_c:.1f}x)")
    else:
        yj_miss.append("底突破(否)")

    # 放量突破前高
    hhv20 = np.max(h[-21:-1])
    if c[idx] > hhv20 and vol_ratio_c > 1.3:
        yj_score += 6; yj_met.append(f"放量突破前高(量{vol_ratio_c:.1f}x)")
    else:
        yj_miss.append("前高突破(否)")

    # 地量后倍量
    vol_min_20 = np.min(v[-21:-1])
    if v[idx-1] <= vol_min_20 * 1.1 and v[idx] > v[idx-1] * 1.8:
        yj_score += 5; yj_met.append(f"地量后倍量({v[idx]/v[idx-1]:.1f}x)")
    else:
        yj_miss.append("地量倍量(否)")

    # 缩量回踩MA20不破
    if vol_ratio_c < 0.6 and c[idx] > ma20[idx] * 0.98 and c[idx] > c[idx-1]:
        yj_score += 4; yj_met.append(f"缩量回踩MA20(量{vol_ratio_c:.1f}x)")
    else:
        yj_miss.append("回踩不破(否)")

    # 均线归位: 此前纠缠→多头有序
    ma5_5d = np.mean(c[-6:-1])
    ma10_5d = np.mean(c[-11:-1])
    now_bull = ma5[idx] > ma10[idx] > ma20[idx]
    past_bull = ma5_5d > ma10_5d > ma20[idx]
    if now_bull and not past_bull:
        yj_score += 5; yj_met.append("均线从散乱归位多头")
    elif now_bull:
        yj_met.append("均线保持多头")
    else:
        yj_miss.append("均线未归位")
    # 价格突破MA20归位
    if c[idx] > ma20[idx] and c[idx-1] <= ma20[idx]:
        yj_score += 5; yj_met.append("价格突破MA20归位")

    yj_triggered = yj_score >= 5
    result["量价异动均线归位"] = {
        "triggered": yj_triggered,
        "conditions_met": yj_met,
        "conditions_missed": yj_miss,
        "score": min(yj_score, 15),
        "detail": " | ".join(yj_met) if yj_met else "未触发",
        "action": "全书核心变盘信号·重点介入" if yj_triggered else "",
    }

    # ================================================================
    # 7. 单日强硬洗盘 (《股是股非》第五章③)
    # ================================================================
    xp_met = []; xp_miss = []

    # 前日大阴: 跌>4%
    prev_chg = (c[idx-1] / c[idx-2] - 1) * 100 if idx >= 2 else 0
    prev_big_yin = prev_chg < -4
    (xp_met if prev_big_yin else xp_miss).append(f"前日跌幅{prev_chg:.1f}%")

    # 今日阳包阴: 收阳 + 收盘>前日开盘
    today_yang = c[idx] > o[idx]
    engulf = c[idx] > o[idx-1] if idx >= 1 else False
    (xp_met if (today_yang and engulf) else xp_miss).append(
        f"{'阳包阴' if (today_yang and engulf) else '未反包'}")

    # 量确认: 今日量>前日量
    vol_confirm = v[idx] > v[idx-1] if idx >= 1 else False
    (xp_met if vol_confirm else xp_miss).append("量确认" if vol_confirm else "量未确认")

    xp_triggered = prev_big_yin and today_yang and engulf and vol_confirm
    engulf_pct = (c[idx] / o[idx-1] - 1) * 100 if idx >= 1 else 0
    result["单日洗盘反包"] = {
        "triggered": xp_triggered,
        "conditions_met": xp_met,
        "conditions_missed": xp_miss,
        "score": 10 + min(engulf_pct * 2, 10) if xp_triggered else 0,
        "detail": " | ".join(xp_met) if xp_met else "未触发",
        "action": f"反包{engulf_pct:.0f}%·最佳买点" if xp_triggered else "",
    }

    # ================================================================
    # 8. 缺口模式 (《股是股非》第五章④)
    # ================================================================
    qk_met = []; qk_miss = []

    # 向上缺口: 今日最低 > 昨日最高
    gap_up = l[idx] > h[idx-1] if idx >= 1 else False
    gap_up_pct = (l[idx] / h[idx-1] - 1) * 100 if gap_up else 0

    # 三日向上缺口不回补
    gap_up_3d = l[idx-2] > h[idx-3] if idx >= 3 else False
    gap_filled = gap_up_3d and min(l[-2:]) < h[-3]

    # 向下缺口
    gap_down = h[idx] < l[idx-1] if idx >= 1 else False
    gap_down_pct = (h[idx] / l[idx-1] - 1) * 100 if gap_down else 0

    if gap_up:
        qk_met.append(f"向上缺口+{gap_up_pct:.1f}%")
    elif gap_up_3d and not gap_filled:
        qk_met.append(f"三日不补缺口·强势确认")
    else:
        qk_miss.append("上缺口(否)")

    if gap_down:
        qk_met.append(f"⚠向下缺口{gap_down_pct:.1f}%")
    else:
        qk_miss.append("下缺口(否)")

    qk_triggered = gap_up or (gap_up_3d and not gap_filled)
    result["缺口模式"] = {
        "triggered": qk_triggered,
        "conditions_met": qk_met,
        "conditions_missed": qk_miss,
        "score": 8 if qk_triggered else (-6 if gap_down else 0),
        "detail": " | ".join(qk_met) if qk_met else "无缺口",
        "action": "强势突破·跟进" if qk_triggered else ("风险缺口·回避" if gap_down else ""),
    }

    # ================================================================
    # 9. 高位倒灌出货 (《股是股非》第六章·卖出信号)
    # ================================================================
    cg_met = []; cg_miss = []; cg_penalty = 0

    # 高位倒灌: 60日高位+高开低走+放量
    if near_high and o[idx] > c[idx] and c[idx] < c[idx-1] and vol_ratio_c > 1.3:
        cg_penalty += 8; cg_met.append(f"⚠高位倒灌({vol_ratio_c:.1f}x)")
    else:
        cg_miss.append("倒灌(否)")

    # 阳奉阴违: 收阳但低于昨收 + 放量
    if c[idx] > o[idx] and c[idx] < c[idx-1] and vol_ratio_c > 1.2:
        cg_penalty += 5; cg_met.append(f"⚠阳奉阴违(量{vol_ratio_c:.1f}x)")
    else:
        cg_miss.append("假阳线(否)")

    # 放量滞涨: 量>1.5倍 但 涨幅<1%
    chg_today = (c[idx] / c[idx-1] - 1) * 100 if idx >= 1 else 0
    if abs(chg_today) < 1 and vol_ratio_c > 1.5:
        cg_penalty += 5; cg_met.append(f"⚠放量滞涨(量{vol_ratio_c:.1f}x)")
    else:
        cg_miss.append("滞涨(否)")

    cg_triggered = cg_penalty > 0
    result["高位倒灌出货"] = {
        "triggered": cg_triggered,
        "conditions_met": cg_met,
        "conditions_missed": cg_miss,
        "score": -cg_penalty,
        "detail": " | ".join(cg_met) if cg_met else "无出货信号",
        "action": "减仓/回避" if cg_penalty >= 8 else ("警惕" if cg_triggered else ""),
    }

    return result


# ============================================================
# 便捷函数: 供 quick_trade 直接调用
# ============================================================

def apply_warfare_filters(code, df):
    """
    在 deep_analyze 中应用战法过滤。
    返回 (signal_override, score_boost, reasons_add)

    signal_override: None=不覆盖, "PASS"=毙掉
    score_boost: 加减分
    reasons_add: [(方向, 理由), ...]
    """
    w = detect_all_warfare(code, df)

    boost = 0
    reasons = []

    # C区 → 硬毙
    if w["C区风险"]["triggered"]:
        return "PASS", 0, [("bear", f"⚠C区风险: {w['C区风险']['detail']}")]

    # 高位倒灌 → 硬毙 (严重度8)
    cg = w["高位倒灌出货"]
    if cg["triggered"] and cg["score"] <= -8:
        return "PASS", 0, [("bear", f"⚠高位倒灌出货: {cg['detail']}")]

    # 量价异动 + 均线归位 → 加分
    yj = w["量价异动均线归位"]
    if yj["triggered"]:
        boost += yj["score"]
        reasons.append(("bull", f"量价异动: {yj['detail']}"))

    # 洗盘反包 → 加分
    xp = w["单日洗盘反包"]
    if xp["triggered"]:
        boost += xp["score"]
        reasons.append(("bull", f"洗盘反包: {xp['detail']}"))

    # 缺口 → 加分/扣分
    qk = w["缺口模式"]
    if qk["triggered"]:
        boost += qk["score"]
        reasons.append(("bull", f"缺口突破: {qk['detail']}"))
    elif w["缺口模式"]["score"] < 0:
        boost += w["缺口模式"]["score"]
        reasons.append(("bear", f"向下缺口: {qk['detail']}"))

    # 高位出货 → 扣分 (非硬毙)
    if cg["triggered"] and cg["score"] > -8:
        boost += cg["score"]
        reasons.append(("bear", f"出货预警: {cg['detail']}"))

    return None, boost, reasons


# ============================================================
# CLI测试
# ============================================================
if __name__ == "__main__":
    import sys, json
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from data_loader import get_stock_kline

    code = sys.argv[1] if len(sys.argv) > 1 else "300223"
    market = 1 if code.startswith("6") else 0
    df = get_stock_kline(code, market, refresh=False)
    if df is None or df.empty:
        print(f"无数据: {code}"); sys.exit(1)
    df = df[df["date"] <= "2026-06-17"]

    result = detect_all_warfare(code, df)
    print(f"\n{'='*60}")
    print(f"  四大战法 — {code} (主板={code.startswith('60') or code.startswith('00')})")
    print(f"{'='*60}")
    for name, r in result.items():
        icon = "🔥" if r["triggered"] else "⚪"
        print(f"\n  {icon} {name} (得分{r['score']}):")
        print(f"     ✅ {r.get('conditions_met', [])}")
        if r.get('conditions_missed'):
            print(f"     ❌ {r['conditions_missed']}")
        if r.get('wait_for'):
            print(f"     → {r['wait_for']}")
