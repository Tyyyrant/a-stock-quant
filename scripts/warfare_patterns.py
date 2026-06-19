#!/usr/bin/env python3
"""
四大战法信号识别器 — 严格按文档公式实现

1. 逼空星线: MACD强势 + 缩量星线洗盘 + 等次日大阳确认 (仅主板600/000)
2. 猎取B区: MA10首次金叉MA30趋势反转
3. A区起涨: 量能MA7金叉MA35 + 逐日放量 + 价站翘MA10 (作者自定义"A区")
4. 拉高抢筹: 量比>2 + 涨幅>4.5% + 收高位98% + 站MA5 + 均线多头 (仅主板600/000)
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
    # 2. 猎取B区 (严格按文档)
    # ================================================================
    lb_met = []; lb_miss = []

    # 多头排列: MA10 > MA30
    ma30 = pd.Series(c).rolling(30).mean().values
    bull_align = ma10[idx] > ma30[idx]
    prev_bull_align = ma10[idx-1] > ma30[idx-1] if idx >= 1 else False
    # CROSS(多头排列, 0.5): 今日为真且昨日为假
    first_cross = bull_align and not prev_bull_align
    (lb_met if first_cross else lb_miss).append("首次金叉" if first_cross else f"非首次(已在MA30上)")

    # 趋势确认: MA10 > REF(MA10,1)
    ma10_up = ma10[idx] > ma10[idx-1] if idx >= 1 else False
    (lb_met if ma10_up else lb_miss).append("MA10上翘")

    lb_triggered = first_cross and ma10_up
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
    # 4. 拉高抢筹 (严格按文档 — 仅主板600/000)
    # ================================================================
    lg_met = []; lg_miss = []

    # 量比: VOL/REF(MA(VOL,5),1) > 2.0
    prev_mv5 = mv5[idx-1] if idx >= 1 else mv5[idx]
    vol_ratio = v[idx] / prev_mv5 if prev_mv5 > 0 else 0
    huge_vol = vol_ratio > 2.0
    (lg_met if huge_vol else lg_miss).append(f"量比{vol_ratio:.1f}")

    # 单日涨幅: C/REF(C,1) > 1.045
    gain = c[idx] / c[idx-1] - 1 if idx >= 1 else 0
    big_gain = gain > 0.045
    (lg_met if big_gain else lg_miss).append(f"涨幅{gain*100:.1f}%")

    # 突破阳线实体: C/O > 1.03
    yang_body = c[idx] / o[idx] > 1.03 if o[idx] > 0 else False
    (lg_met if yang_body else lg_miss).append(f"实体{c[idx]/o[idx]-1:.1f}%")

    # 收于高位: C/HIGH > 0.98
    close_high = c[idx] / h[idx] > 0.98 if h[idx] > 0 else False
    (lg_met if close_high else lg_miss).append(f"收高位{c[idx]/h[idx]:.3f}")

    # 均线多头排列: MA5>MA10>MA20
    bull_ma = ma5[idx] > ma10[idx] > ma20[idx] if idx >= 0 else False
    (lg_met if bull_ma else lg_miss).append(f"多头排列")

    # 站稳5日线: C>MA5
    stand_ma5 = c[idx] > ma5[idx]
    (lg_met if stand_ma5 else lg_miss).append(f"站MA5")

    # 主板过滤
    (lg_met if is_main_board else lg_miss).append("主板600/000")

    # 非ST
    lg_triggered = huge_vol and big_gain and yang_body and close_high and stand_ma5 and bull_ma and is_main_board
    result["拉高抢筹"] = {
        "triggered": lg_triggered,
        "conditions_met": lg_met,
        "conditions_missed": lg_miss,
        "score": len(lg_met) * 3 if lg_triggered else max(0, (len(lg_met) - 2) * 3),
        "detail": " | ".join(lg_met) if lg_met else "未触发",
        "wait_for": "结合颈线/箱体突破，注意量时空大压" if lg_triggered else "",
    }

    return result


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
