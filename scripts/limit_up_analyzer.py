#!/usr/bin/env python3
"""
涨停板分析模块 — 涨停质量评分 + 次日延续概率 + 板块龙头识别

六维评分:
  1. 封板时间 (越早越好) · 2. 封单强度 · 3. 涨停位置 (低位首板最佳)
  4. 量价配合 · 5. 板块地位 (龙头/跟风) · 6. K线形态

短线专用: MA5/MA10 偏离判断，不用 MA20

用法:
  from limit_up_analyzer import analyze_limit_up
  result = analyze_limit_up(code, df, sector_context)
"""

import numpy as np
import pandas as pd


def analyze_limit_up(code: str, df: pd.DataFrame,
                     sector_context: dict = None) -> dict:
    """
    分析一只涨停股票的质量。

    Args:
        code: 股票代码
        df: 日K线 DataFrame (含 OHLCV + date列), 至少60条
        sector_context: 同板块其他涨停票的信息
            {other_codes: [...], sector_name: str, first_limit_up_time: str}

    Returns:
        {quality_score, quality_label, continuation_prob,
         is_leader, leader_notes, next_day_signal, detail}
    """
    if df.empty or len(df) < 30:
        return _empty_result()

    o = df["open"].values.astype(float)
    h = df["high"].values.astype(float)
    l = df["low"].values.astype(float)
    c = df["close"].values.astype(float)
    v = df["volume"].values.astype(float)
    n = len(c)
    idx = -1

    price = c[idx]
    prev_close = c[idx-1] if n >= 2 else price
    chg_pct = (price / prev_close - 1) * 100 if prev_close > 0 else 0

    # 判断是否涨停 — 直接用涨停价比较当日最高价，不受复权影响
    is_gem = code.startswith("300") or code.startswith("301")
    limit_pct = 20.0 if is_gem else 10.0

    # 涨停价 = round(昨收 × (1 + limit_pct), 2)
    calc_limit_price = round(prev_close * (1 + limit_pct / 100), 2)
    # 涨停价比较(>99%) + 涨幅兜底(>9.5%) 双保险覆盖复权偏差
    is_limit_up = (h[idx] >= calc_limit_price * 0.99) or (chg_pct >= 9.5)

    if not is_limit_up:
        return {"is_limit_up": False, "quality_score": 0,
                "quality_label": "非涨停", "continuation_prob": 0}

    # ================================================================
    # 维度1: 封板时间估算 (用振幅和收盘位置近似推断)
    # ================================================================
    today_open = o[idx]; today_high = h[idx]; today_low = l[idx]
    body = price - today_open
    upper_shadow = today_high - max(price, today_open)

    # 光头阳线(收盘=最高) → 很可能是早盘封板
    # 有上影线 → 可能炸过板或尾盘封板
    if upper_shadow / max(today_high - today_low, 0.01) < 0.1:
        seal_time_score = 90       # 光头涨停 → 大概率早盘封板
        seal_time_est = "早盘封板(估)"
    elif upper_shadow / max(today_high - today_low, 0.01) < 0.3:
        seal_time_score = 70       # 小上影 → 可能上午封板
        seal_time_est = "上午封板(估)"
    else:
        seal_time_score = 40       # 长上影 → 可能炸板或尾盘
        seal_time_est = "尾盘/炸板(估)"

    # ================================================================
    # 维度2: 封单强度 (成交量相对判断)
    # ================================================================
    avg_vol_5 = np.mean(v[-6:-1]) if n >= 6 else v[idx]
    avg_vol_20 = np.mean(v[-21:-1]) if n >= 21 else avg_vol_5
    vol_ratio = v[idx] / max(avg_vol_5, 1)

    # 涨停日成交量: 太小=一字板买不到, 太大=出货
    if vol_ratio < 0.3:
        seal_strength_score = 30   # 一字无量板 (买不到)
        vol_note = "一字无量"
    elif vol_ratio < 1.0:
        seal_strength_score = 80   # 缩量涨停 (筹码锁定好)
        vol_note = "缩量封板"
    elif vol_ratio < 2.5:
        seal_strength_score = 95   # 温和放量涨停 (最佳)
        vol_note = "温和放量"
    elif vol_ratio < 5:
        seal_strength_score = 60   # 放量涨停
        vol_note = "放量封板"
    else:
        seal_strength_score = 30   # 巨量涨停 (出货嫌疑)
        vol_note = "巨量可疑"

    # ================================================================
    # 维度3: 涨停位置 (短线用 MA5/MA10)
    # ================================================================
    ma5 = np.mean(c[-5:]) if n >= 5 else price
    ma10 = np.mean(c[-10:]) if n >= 10 else price
    low_20 = np.min(l[-20:]) if n >= 20 else price
    high_20 = np.max(h[-20:]) if n >= 20 else price

    position_20d = (price - low_20) / max(high_20 - low_20, 0.01)
    deviation_ma5 = (price - ma5) / ma5 * 100
    deviation_ma10 = (price - ma10) / ma10 * 100

    # 连续涨停天数
    limit_up_streak = 0
    for i in range(idx, max(0, idx-5), -1):
        day_chg = (c[i] - c[i-1]) / c[i-1] * 100 if i > 0 else 0
        if day_chg >= limit_pct * 0.95: limit_up_streak += 1
        else: break

    if limit_up_streak <= 1 and position_20d < 0.3:
        position_score = 100       # 低位首板！最佳
        position_label = "低位首板"
        continuation_bonus = 15
    elif limit_up_streak <= 2 and position_20d < 0.5:
        position_score = 80        # 中低位连板
        position_label = f"中位{limit_up_streak}连板"
        continuation_bonus = 10
    elif limit_up_streak <= 3:
        position_score = 55        # 中高位
        position_label = f"{limit_up_streak}连板"
        continuation_bonus = 5
    else:
        position_score = 25        # 高位加速
        position_label = f"高位{limit_up_streak}连板"
        continuation_bonus = -5

    # ================================================================
    # 维度4: 量价配合
    # ================================================================
    if vol_ratio < 0.3:
        vol_price_score = 40       # 一字板→次日可能继续一字
    elif 0.5 <= vol_ratio <= 2.5:
        vol_price_score = 90       # 温和放量涨停最好
    elif 2.5 < vol_ratio <= 4:
        vol_price_score = 60       # 放量偏大但可接受
    else:
        vol_price_score = 25       # 巨量

    # ================================================================
    # 维度5: 板块地位
    # ================================================================
    is_leader = False
    leader_score = 50
    leader_notes = ""

    if sector_context and sector_context.get("other_codes"):
        # 同板块有其他涨停票 → 比较谁更早/更强
        others = sector_context.get("other_codes", [])
        my_strength = vol_ratio * (1 + chg_pct/100)
        all_strengths = sector_context.get("all_strengths", [])
        if all_strengths:
            rank = sum(1 for s in all_strengths if s > my_strength) + 1
            if rank == 1:
                leader_score = 100
                is_leader = True
                leader_notes = f"板块{len(others)+1}只涨停中最早最强→龙头"
                continuation_bonus += 10
            elif rank <= 3:
                leader_score = 75
                leader_notes = f"板块涨停排名#{rank}→前排跟风"
                continuation_bonus += 3
            else:
                leader_score = 45
                leader_notes = f"板块涨停排名#{rank}→后排跟风"

    # ================================================================
    # 维度6: K线形态
    # ================================================================
    # 涨停日K线: 光头光脚最佳
    body_ratio = abs(body) / max(today_high - today_low, 0.01)
    k_score = 0
    if price > today_open and body_ratio > 0.7:
        k_score += 40   # 实体大阳线
    if upper_shadow / max(today_high - today_low, 0.01) < 0.15:
        k_score += 30   # 光头
    if (price - today_low) / max(today_high - today_low, 0.01) < 0.15:
        k_score += 30   # 光脚
    k_score = min(k_score, 100)

    # ==== 涨停基因: 近3天是否有过涨停 (high >= 涨停价) ====
    recent_limit_up = False
    for i in range(idx-1, max(0, idx-4), -1):
        if i <= 0: continue
        day_limit_price = round(c[i-1] * (1 + limit_pct / 100), 2)
        if h[i] >= day_limit_price * 0.999:
            recent_limit_up = True
            break
    if recent_limit_up:
        seal_time_score += 10  # 有涨停基因，封板时间估分上调
        continuation_bonus += 5

    # ================================================================
    # 综合质量分
    # ================================================================
    quality_score = (
        seal_time_score * 0.20 +
        seal_strength_score * 0.10 +
        position_score * 0.25 +
        vol_price_score * 0.15 +
        leader_score * 0.15 +
        k_score * 0.15
    )

    if quality_score >= 80: label = "龙头首板"
    elif quality_score >= 65: label = "强势涨停"
    elif quality_score >= 50: label = "跟风涨停"
    elif quality_score >= 35: label = "可疑涨停"
    else: label = "陷阱涨停"

    # ================================================================
    # 次日延续概率
    # ================================================================
    # 基于历史统计的先验概率 + 调整
    base_cont = 55  # 涨停次日高开概率基础值

    # 质量调整
    if label == "龙头首板": base_cont += 20
    elif label == "强势涨停": base_cont += 10
    elif label == "可疑涨停": base_cont -= 15
    elif label == "陷阱涨停": base_cont -= 25

    # 位置调整
    base_cont += continuation_bonus

    # 偏离调整 (短线用MA5/MA10)
    if deviation_ma5 < 8:
        base_cont += 5   # 未大幅偏离MA5
    elif deviation_ma5 > 20:
        base_cont -= 10  # 严重偏离MA5

    # 连板调整
    if limit_up_streak >= 3:
        base_cont -= 10  # 3连板后延续概率下降

    continuation_prob = max(5, min(95, base_cont))

    # ================================================================
    # 次日操作建议
    # ================================================================
    if quality_score >= 80:
        next_day_signal = "次日高开<3%可追，高开>5%等回踩MA5"
    elif quality_score >= 65:
        next_day_signal = "次日高开3-5%轻仓，破涨停价止盈"
    elif quality_score >= 50:
        next_day_signal = "次日高开不追，等回踩MA10确认"
    else:
        next_day_signal = "不建议次日参与"

    return {
        "is_limit_up": True,
        "limit_pct": limit_pct,
        "limit_up_streak": limit_up_streak,
        "quality_score": round(quality_score, 1),
        "quality_label": label,
        "continuation_prob": continuation_prob,
        "is_leader": is_leader,
        "leader_notes": leader_notes,
        "next_day_signal": next_day_signal,
        "seal_time_est": seal_time_est,
        "position_label": position_label,
        "vol_note": vol_note,
        "deviation_ma5": round(deviation_ma5, 1),
        "deviation_ma10": round(deviation_ma10, 1),
        "detail": {
            "seal_time": seal_time_score,
            "seal_strength": seal_strength_score,
            "position": position_score,
            "vol_price": vol_price_score,
            "leader": leader_score,
            "k_pattern": k_score,
        }
    }


def analyze_sector_limit_ups(codes: list, kline_map: dict) -> dict:
    """
    分析一个板块内的涨停梯队。

    Returns: {leaders: [...], followers: [...], sector_leader: code}
    """
    results = []
    for code in codes:
        if code not in kline_map: continue
        item = kline_map[code]
        kline = item["kline"] if isinstance(item, dict) else item
        result = analyze_limit_up(code, kline)
        if result["is_limit_up"]:
            results.append((code, result))

    results.sort(key=lambda x: x[1]["quality_score"], reverse=True)

    return {
        "total_limit_ups": len(results),
        "sector_leader": results[0][0] if results else None,
        "leader_name": results[0][1].get("quality_label", "") if results else "",
        "leaders": [c for c, r in results if r["quality_score"] >= 65],
        "followers": [c for c, r in results if 50 <= r["quality_score"] < 65],
        "weak": [c for c, r in results if r["quality_score"] < 50],
        "all_results": {c: r for c, r in results},
    }


def _empty_result():
    return {"is_limit_up": False, "quality_score": 0,
            "quality_label": "非涨停", "continuation_prob": 0}


# ============================================================
# CLI测试
# ============================================================
if __name__ == "__main__":
    import sys, json
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from data_loader import get_stock_kline

    codes = sys.argv[1:] if len(sys.argv) > 1 else ["600110","002741","603324","300522"]
    for code in codes:
        market = 1 if code.startswith("6") else 0
        df = get_stock_kline(code, market, refresh=False)
        if df is None or len(df) < 60: continue
        df = df[df["date"] <= "2026-06-17"]
        r = analyze_limit_up(code, df)
        if r["is_limit_up"]:
            print(f"\n{code}: {r['quality_label']} 得分{r['quality_score']:.0f} "
                  f"延续{r['continuation_prob']}% {r['position_label']} "
                  f"{r['seal_time_est']} {r['vol_note']} "
                  f"偏离MA5:{r['deviation_ma5']:.0f}% MA10:{r['deviation_ma10']:.0f}%")
            if r['leader_notes']: print(f"  🏆 {r['leader_notes']}")
            print(f"  📋 次日: {r['next_day_signal']}")
