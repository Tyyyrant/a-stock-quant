#!/usr/bin/env python3
"""
个股战法分析工具 — 《股是股非》6大战法 + 9大猎取战法 一键诊断

用法:
  python3 scripts/analyze_stock.py 002426                    # 单只
  python3 scripts/analyze_stock.py 002426 --date 2026-06-18  # 指定日期
  python3 scripts/analyze_stock.py 002426,000657,300666       # 批量
  python3 scripts/analyze_stock.py 002426 --full              # 含K线图数据

输出:
  1. ABC三区定位 (A区强势 / B区次级 / C区风险)
  2. 量价异动检测 (4种异常类型)
  3. 均线归位判断
  4. 单日洗盘反包检测
  5. 缺口模式追踪
  6. 出货/见顶信号预警
  7. 猎取战法匹配 (逼空星线/猎取B区/A区起涨/拉高抢筹)
  8. 信号共振总分
"""

import argparse, json, os, sys, time
from pathlib import Path
from datetime import datetime

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from data_loader import get_stock_kline, tencent_quote, ensure_dirs


def analyze_stock(code: str, target_date: str = None, with_agent: bool = False) -> dict:
    """
    对单只股票做完整战法分析。

    Args:
        code: 股票代码
        target_date: 目标日期
        with_agent: 是否调用7-Agent辩论生成交易计划

    Returns:
        dict with keys: code, name, price, pe, zone, volume_anomaly, ma_alignment,
                        washout, gap, distribution, warfare, resonance_score, summary,
                        agent_plan (if with_agent=True)
    """
    if target_date is None:
        target_date = datetime.now().strftime("%Y-%m-%d")

    market = 1 if code.startswith("6") else 0
    dk = get_stock_kline(code, market, refresh=False)

    if dk is None or len(dk) < 60:
        return {"code": code, "error": "K线数据不足(<60条)"}

    dk = dk[dk["date"] <= target_date].copy()
    if len(dk) < 20:
        return {"code": code, "error": "目标日期之前K线不足(<20条)"}

    price = float(dk["close"].values[-1])
    chg = (price / float(dk["close"].values[-2]) - 1) if len(dk) >= 2 else 0

    # 基本面
    name = code
    pe = 0
    mcap = 0
    try:
        q = tencent_quote([code])
        if code in q:
            name = q[code].get("name", code)
            pe = q[code].get("pe_ttm", 0) or 0
            mcap = q[code].get("mcap_yi", 0) or 0
    except Exception:
        pass

    result = {
        "code": code, "name": name, "date": target_date,
        "price": price, "chg_pct": round(chg * 100, 2),
        "pe": round(pe, 0), "mcap": round(mcap, 0),
    }

    # ── 策略1: ABC三区 ──
    try:
        from zhangting_strategies import classify_abc_zone
        zone = classify_abc_zone(dk)
        result["zone"] = zone
    except Exception as e:
        result["zone"] = {"zone": "?", "zone_score": 0, "zone_reason": str(e)}

    # ── 策略2: 量价异动+均线归位 ──
    try:
        from zhangting_strategies import detect_volume_price_anomaly, detect_ma_realignment
        vpa = detect_volume_price_anomaly(dk)
        ma_re = detect_ma_realignment(dk)
        result["volume_anomaly"] = vpa
        result["ma_alignment"] = ma_re
    except Exception as e:
        result["volume_anomaly"] = {"error": str(e)}
        result["ma_alignment"] = {"error": str(e)}

    # ── 策略3: 单日洗盘反包 ──
    try:
        from zhangting_strategies import detect_washout_reversal
        washout = detect_washout_reversal(dk)
        result["washout"] = washout
    except Exception as e:
        result["washout"] = {"error": str(e)}

    # ── 策略4: 缺口模式 ──
    try:
        from zhangting_strategies import detect_gap_signal
        gap = detect_gap_signal(dk)
        result["gap"] = gap
    except Exception as e:
        result["gap"] = {"error": str(e)}

    # ── 策略5: 出货/见顶 ──
    try:
        from zhangting_strategies import detect_distribution_signal
        dist = detect_distribution_signal(dk)
        result["distribution"] = dist
    except Exception as e:
        result["distribution"] = {"error": str(e)}

    # ── 猎取战法 (逼空星线/猎取B区/A区起涨/拉高抢筹) ──
    try:
        from warfare_patterns import detect_all_warfare
        wf_all = detect_all_warfare(code, dk)
        warfare = {}
        for wf_name in ["逼空星线", "猎取B区", "A区起涨", "拉高抢筹",
                         "C区风险过滤", "量价异动均线归位", "单日洗盘反包",
                         "缺口模式", "高位倒灌出货"]:
            wf = wf_all.get(wf_name, {})
            warfare[wf_name] = {
                "triggered": wf.get("triggered", False),
                "score": wf.get("score", 0),
                "conditions": wf.get("conditions_met", []),
                "detail": wf.get("detail", ""),
            }
        result["warfare"] = warfare
    except Exception as e:
        result["warfare"] = {"error": str(e)}

    # ── 策略6: 共振总分 ──
    try:
        from zhangting_strategies import score_signal_resonance
        resonance = score_signal_resonance(dk)
        result["resonance"] = resonance
    except Exception as e:
        result["resonance"] = {"error": str(e)}

    # ── 三度厚度 ──
    try:
        from zhangting_strategies import detect_volume_accumulation
        result["thickness"] = detect_volume_accumulation(dk)
    except Exception as e:
        result["thickness"] = {"error": str(e)}

    # ── 主力意图 ──
    try:
        from zhangting_strategies import classify_pullup_intent
        result["pullup_intent"] = classify_pullup_intent(dk)
    except Exception as e:
        result["pullup_intent"] = {"error": str(e)}

    # ── 综合判定 ──
    result["summary"] = _build_summary(result)

    # ── 交易计划 (规则推算，始终输出) ──
    result["trading_plan"] = _compute_trading_plan(result, dk)

    # ── 7-Agent 定性判断 (可选) ──
    if with_agent:
        result["agent_verdict"] = _run_agent_debate(result, dk)

    return result


def _compute_trading_plan(r: dict, dk) -> dict:
    """
    纯规则推算入场/止损/止盈。根据趋势方向区分进攻型/防守型，保证价格在技术面自洽范围内。

    核心区分:
      - 进攻型(A区或从低位突破的B区): 追涨，入场贴近现价，止损紧
      - 防守型(从高位回落的B区): 等回调，入场在MA10/MA20，止损宽
    """
    close = dk["close"].values.astype(float)
    high = dk["high"].values.astype(float)
    low = dk["low"].values.astype(float)

    price = r["price"]
    zone = r.get("zone", {}).get("zone", "?")

    ma5 = np.mean(close[-5:])
    ma10 = np.mean(close[-10:])
    ma20 = np.mean(close[-20:])
    ma5_5d_ago = np.mean(close[-10:-5]) if len(close) >= 10 else ma5
    ma20_5d_ago = np.mean(close[-25:-5]) if len(close) >= 25 else ma20

    # 趋势方向：近期是否有过一段在MA20下的修正，现在回到了MA20上方
    recent_lows = close[-10:] < ma20  # 过去10天有价格低于MA20的天
    had_recent_correction = any(recent_lows) and not all(recent_lows)  # 有过低位但并非全程在下面
    price_above_ma20 = price > ma20
    ma5_rising = ma5 > np.mean(close[-8:-3]) if len(close) >= 8 else False

    # 进攻型：A区强趋势 或 B区但刚从MA20下V型反弹上来(MA5加速上升)
    is_offensive = (zone == "A") or (zone == "B" and price_above_ma20 and had_recent_correction and ma5_rising)

    # ATR
    tr = []
    for i in range(1, min(15, len(close))):
        h_l = high[-i] - low[-i]
        h_pc = abs(high[-i] - close[-i-1])
        l_pc = abs(low[-i] - close[-i-1])
        tr.append(max(h_l, h_pc, l_pc))
    atr = np.mean(tr) if tr else price * 0.03

    high_60 = np.max(high[-60:]) if len(high) >= 60 else np.max(high)
    low_60 = np.min(low[-60:]) if len(low) >= 60 else np.min(low)

    # 涨停日特殊处理：现价=天花板，不能也不该今天追
    is_limit_up = r["chg_pct"] >= 9.5
    next_day_note = ""

    trend_type = ""
    if is_limit_up:
        # 涨停日：次日开盘等回踩MA5，不想等就开盘价附近追
        entry = round(float(ma5), 2)  # 等回到MA5
        stop = round(float(max(price * 0.93, ma20 * 0.97)), 2)  # 破MA20或涨停价-7%
        target = round(float(max(high_60 * 0.95, price * 1.15)), 2)
        next_day_note = f"⚠️今日涨停封板，现价¥{price:.2f}买不到。次日若高开3%+不追等回踩MA5(¥{ma5:.2f})，若平开/低开可直接轻仓试探"
        if zone == "A":
            trend_type = "A区涨停→次日等回踩MA5，破MA20走"
        else:
            trend_type = "进攻型B区涨停→次日等回踩MA5，破MA20走"
    elif zone == "A":
        entry = round(float(np.clip(ma5, price * 0.97, price * 1.02)), 2)
        stop = round(float(max(ma10 * 0.98, entry - 1.0 * atr)), 2)
        target = round(float(max(high_60 * 0.95, entry + 3 * atr)), 2)
        trend_type = "A区强趋势→追涨，止损=MA10(紧)"
    elif is_offensive:
        entry = round(float(price), 2)
        stop = round(float(max(ma20 * 0.98, entry - 1.2 * atr)), 2)
        target = round(float(max(high_60 * 0.95, entry + 3 * atr)), 2)
        trend_type = "进攻型B区→低位突破现价追涨，止损=MA20"
    else:
        entry = round(float(np.clip(ma10, ma20, price * 1.01)), 2)
        stop = round(float(max(low_60 * 0.99, entry - 1.0 * atr)), 2)
        target = round(float(max(high_60 * 0.95, entry + 3 * atr)), 2)
        trend_type = "防守型B区→等回调MA10，止损=60日前低"

    stop = min(stop, round(entry * 0.95, 2))
    target = max(target, round(price * 1.05, 2))
    risk = entry - stop
    reward = target - entry
    rr_ratio = round(reward / risk, 1) if risk > 0 else 0

    return {
        "entry": entry, "stop": stop, "target": target,
        "atr": round(float(atr), 2), "rr_ratio": rr_ratio,
        "risk_pct": round(risk / entry * 100, 1),
        "reward_pct": round(reward / entry * 100, 1),
        "trend_type": trend_type,
        "is_offensive": is_offensive,
        "anchors": {
            "ma5": round(float(ma5), 2), "ma10": round(float(ma10), 2),
            "ma20": round(float(ma20), 2),
            "high_60": round(float(high_60), 2), "low_60": round(float(low_60), 2),
            "atr": round(float(atr), 2),
        },
        "logic": f"{trend_type} | ATR{atr:.1f} | 前高{high_60:.1f} | 前低{low_60:.1f}",
        "next_day_note": next_day_note,
    }


def _run_agent_debate(r: dict, dk) -> dict:
    """调用7-Agent辩论 — 只做定性判断(论点+判决)，不管价格"""
    try:
        from agent_debate import debate
        from volume_price_analyzer import analyze_volume_price
        from chip_distribution import estimate_chip_distribution
        from candlestick_patterns import identify_all_patterns
    except ImportError as e:
        return {"error": f"模块导入失败: {e}"}

    # 量价+筹码
    v = analyze_volume_price(dk)
    c = estimate_chip_distribution(dk)
    p = identify_all_patterns(dk, ticker=r["code"])

    zone = r.get("zone", {})
    is_a = zone.get("zone") == "A"
    is_c = zone.get("zone") == "C"
    dist = r.get("distribution", {})

    reasons_bull = []
    reasons_bear = []

    if is_a:
        reasons_bull.append(f"A区强势({zone.get('zone_reason','')})")
    if is_c:
        reasons_bear.append(f"C区风险({zone.get('zone_reason','')})")

    vs_val = v.get("volume_score", 0)
    cs_val = c.get("chip_score", 0)
    ks_val = p.pattern_score
    profit_r = c.get("profit_ratio", 0)

    if vs_val > 10:
        reasons_bull.append(f"量价健康({vs_val:.0f}分)")
    elif vs_val < -20:
        reasons_bear.append(f"量价异常({vs_val:.0f}分)")

    if cs_val > 10:
        reasons_bull.append(f"筹码锁定({cs_val:.0f}分)")
    elif cs_val < -15:
        reasons_bear.append(f"筹码压力({cs_val:.0f}分)")

    if ks_val > 30:
        reasons_bull.append(f"K线偏多({ks_val:.0f}分)")
    elif ks_val < -30:
        reasons_bear.append(f"K线偏空({ks_val:.0f}分)")

    if dist.get("has_distribution"):
        reasons_bear.append(f"出货预警:{dist.get('distribution_type','')}")

    for wf_name, wf in r.get("warfare", {}).items():
        if isinstance(wf, dict) and wf.get("triggered"):
            reasons_bull.append(f"战法:{wf_name}({wf.get('score',0)}分)")

    net = len(reasons_bull) * 3 - len(reasons_bear) * 3

    # 产业链因果上下文
    causal_ctx = None
    try:
        from bottleneck_discovery import load_causal_graph
        graph = load_causal_graph()
        materials = graph.get("materials", {})
        code = r["code"]
        for mat_name, mat_data in materials.items():
            if "domestic_players" not in mat_data:
                continue
            for player in mat_data.get("domestic_players", []):
                if player.get("code") == code:
                    thesis = mat_data.get("investment_thesis", "")
                    demand = [d.get("mechanism", "")[:80] for d in mat_data.get("demand_drivers", [])]
                    supply = [s.get("mechanism", "")[:80] for s in mat_data.get("supply_constraints", [])]
                    causal_ctx = {
                        "primary_material": mat_name,
                        "demand_drivers": demand,
                        "supply_constraints": supply,
                        "gap_summary": mat_data.get("self_sufficiency", ""),
                        "investment_thesis": thesis,
                        "causal_chain": mat_data.get("causal_chain", ""),
                        "key_events": mat_data.get("key_events", []),
                        "player_role": player.get("progress", ""),
                        "is_named_player": True,
                    }
                    break
            if causal_ctx:
                break
    except Exception:
        pass

    debate_input = {
        "code": r["code"], "name": r["name"],
        "sector": causal_ctx.get("primary_material", "") if causal_ctx else "",
        "close": r["price"], "change_pct": r["chg_pct"] / 100,
        "pe": r["pe"], "mcap": r["mcap"],
        "candlestick_score": ks_val, "volume_score": vs_val,
        "chip_score": cs_val, "profit_ratio": profit_r,
        "reasons_bull": reasons_bull, "reasons_bear": reasons_bear,
        "net_score": net, "limit_up": None,
        "causal_context": causal_ctx,
    }

    try:
        return debate(debate_input)
    except Exception as e:
        return {"error": f"Agent辩论失败: {e}"}
    try:
        from agent_debate import debate
        from volume_price_analyzer import analyze_volume_price
        from chip_distribution import estimate_chip_distribution
        from candlestick_patterns import identify_all_patterns
        from bottleneck_discovery import get_causal_context
    except ImportError as e:
        return {"error": f"模块导入失败: {e}"}

    # 量价+筹码
    v = analyze_volume_price(dk)
    c = estimate_chip_distribution(dk)
    p = identify_all_patterns(dk, ticker=r["code"])

    # 构建 debate 输入
    zone = r.get("zone", {})
    is_a = zone.get("zone") == "A"
    is_c = zone.get("zone") == "C"
    dist = r.get("distribution", {})

    reasons_bull = []
    reasons_bear = []

    if is_a:
        reasons_bull.append(f"A区强势({zone.get('zone_reason','')})")
    if is_c:
        reasons_bear.append(f"C区风险({zone.get('zone_reason','')})")

    vs_val = v.get("volume_score", 0)
    cs_val = c.get("chip_score", 0)
    ks_val = p.pattern_score
    profit_r = c.get("profit_ratio", 0)

    if vs_val > 10:
        reasons_bull.append(f"量价健康({vs_val:.0f}分)")
    elif vs_val < -20:
        reasons_bear.append(f"量价异常({vs_val:.0f}分)")

    if cs_val > 10:
        reasons_bull.append(f"筹码锁定({cs_val:.0f}分)")
    elif cs_val < -15:
        reasons_bear.append(f"筹码压力({cs_val:.0f}分)")

    if ks_val > 30:
        reasons_bull.append(f"K线偏多({ks_val:.0f}分)")
    elif ks_val < -30:
        reasons_bear.append(f"K线偏空({ks_val:.0f}分)")

    if dist.get("has_distribution"):
        reasons_bear.append(f"出货预警:{dist.get('distribution_type','')}")

    # 战法触发
    for wf_name, wf in r.get("warfare", {}).items():
        if isinstance(wf, dict) and wf.get("triggered"):
            reasons_bull.append(f"战法:{wf_name}({wf.get('score',0)}分)")

    net = len(reasons_bull) * 3 - len(reasons_bear) * 3

    # 产业链因果上下文
    causal_ctx = None
    try:
        from bottleneck_discovery import load_causal_graph
        graph = load_causal_graph()
        materials = graph.get("materials", {})
        # 尝试匹配股票代码到因果图谱
        code = r["code"]
        for mat_name, mat_data in materials.items():
            if "domestic_players" not in mat_data:
                continue
            for player in mat_data.get("domestic_players", []):
                if player.get("code") == code:
                    thesis = mat_data.get("investment_thesis", "")
                    demand = [d.get("mechanism", "")[:80] for d in mat_data.get("demand_drivers", [])]
                    supply = [s.get("mechanism", "")[:80] for s in mat_data.get("supply_constraints", [])]
                    causal_ctx = {
                        "primary_material": mat_name,
                        "demand_drivers": demand,
                        "supply_constraints": supply,
                        "gap_summary": mat_data.get("self_sufficiency", ""),
                        "investment_thesis": thesis,
                        "causal_chain": mat_data.get("causal_chain", ""),
                        "key_events": mat_data.get("key_events", []),
                        "player_role": player.get("progress", ""),
                        "is_named_player": True,
                    }
                    break
            if causal_ctx:
                break
    except Exception:
        pass

    debate_input = {
        "code": r["code"], "name": r["name"],
        "sector": causal_ctx.get("primary_material", "") if causal_ctx else "",
        "close": r["price"], "change_pct": r["chg_pct"] / 100,
        "pe": r["pe"], "mcap": r["mcap"],
        "candlestick_score": ks_val, "volume_score": vs_val,
        "chip_score": cs_val, "profit_ratio": profit_r,
        "reasons_bull": reasons_bull, "reasons_bear": reasons_bear,
        "net_score": net, "limit_up": None,
        "causal_context": causal_ctx,
    }

    try:
        agent_result = debate(debate_input)
        return agent_result
    except Exception as e:
        return {"error": f"Agent辩论失败: {e}"}


def _build_summary(r: dict) -> dict:
    """综合所有信号给出判定"""
    risks = []
    signals = []

    # ABC 区判定
    zone = r.get("zone", {})
    zone_name = zone.get("zone", "?")
    if zone_name == "A":
        signals.append(("强势", f"A区强势: {zone.get('zone_reason', '')}"))
    elif zone_name == "B":
        signals.append(("中性", f"B区次级: {zone.get('zone_reason', '')}"))
    elif zone_name == "C":
        risks.append(("高风险", f"C区风险: {zone.get('zone_reason', '')}"))

    # 出货信号
    dist = r.get("distribution", {})
    if dist.get("has_distribution"):
        dist_type = dist.get("distribution_type", "")
        risks.append(("出货预警", f"{dist_type}: {dist.get('detail', '')}"))

    # 洗盘反包
    washout = r.get("washout", {})
    if washout.get("is_washout_reversal"):
        signals.append(("洗盘反包", f"大阴→阳包阴, 主力故意打压后反包"))

    # 量价异动
    vpa = r.get("volume_anomaly", {})
    if vpa.get("has_anomaly"):
        atype = vpa.get("anomaly_type", "")
        detail = vpa.get("detail", "") or vpa.get("anomaly_type", "")
        signals.append(("量价异动", f"{atype}: {detail}"))

    # 均线归位
    ma = r.get("ma_alignment", {})
    if ma.get("is_realigning"):
        signals.append(("均线归位", "均线从散乱→多头有序, 变盘确认"))

    # 缺口
    gap = r.get("gap", {})
    if gap.get("up_gap_unfilled"):
        signals.append(("向上缺口", f"已{gap.get('up_gap_days', 0)}日未补, 强势确认"))

    # 猎取战法
    triggered_wf = []
    for wf_name, wf in r.get("warfare", {}).items():
        if isinstance(wf, dict) and wf.get("triggered"):
            triggered_wf.append(f"{wf_name}({wf.get('score', 0)}分)")
    if triggered_wf:
        signals.append(("战法触发", ", ".join(triggered_wf)))

    # 共振分
    resonance = r.get("resonance", {})
    resonance_score = resonance.get("total_score", 0)

    # 综合判定
    fatal = len(risks)
    positive = len(signals)

    if fatal >= 2:
        verdict = "⛔ 回避"
        action = "多重风险信号叠加, 不建议参与"
    elif fatal == 1:
        verdict = "⚠️ 谨慎"
        action = f"存在风险{risks[0][0]}但仍有{positive}个正面信号, 轻仓或等待风险解除"
    elif positive >= 3 and resonance_score >= 60:
        verdict = "🔥 强烈关注"
        action = f"多信号共振(共振分{resonance_score}), 可积极建仓"
    elif positive >= 2:
        verdict = "✅ 关注"
        action = f"信号偏多(共振分{resonance_score}), 回调或确认后参与"
    elif positive >= 1:
        verdict = "👀 观察"
        action = "信号不够强, 等待更多确认"
    else:
        verdict = "➖ 无信号"
        action = "当前无明确技术信号, 不建议操作"

    return {
        "verdict": verdict,
        "action": action,
        "risks": risks,
        "signals": signals,
        "resonance_score": resonance_score,
        "fatal_count": fatal,
        "positive_count": positive,
    }


def format_report(r: dict) -> str:
    """格式化为可读报告"""
    if "error" in r:
        return f"❌ {r['code']}: {r['error']}"

    s = r["summary"]
    lines = []
    lines.append("=" * 60)
    lines.append(f"  📊 {r['name']}({r['code']}) 战法诊断报告")
    lines.append(f"  日期: {r['date']}  |  现价: ¥{r['price']:.2f}  |  涨跌: {r['chg_pct']:+.1f}%  |  PE: {r['pe']:.0f}  |  市值: {r['mcap']:.0f}亿")
    lines.append("=" * 60)
    lines.append(f"  综合判定: {s['verdict']}  —  {s['action']}")
    lines.append(f"  共振总分: {s['resonance_score']}  |  正面信号: {s['positive_count']}  |  风险信号: {s['fatal_count']}")
    lines.append("")

    # ABC三区
    zone = r.get("zone", {})
    zmap = {"A": "🟢", "B": "🟡", "C": "🔴", "?": "⚪"}
    lines.append(f"  {zmap.get(zone.get('zone','?'), '?')} ABC三区: {zone.get('zone','?')}区 (得分{zone.get('zone_score',0)}) — {zone.get('zone_reason','')}")
    lines.append("")

    # 出货检测
    dist = r.get("distribution", {})
    if dist.get("has_distribution"):
        lines.append(f"  🚨 出货预警: {dist.get('distribution_type','')} (严重度{dist.get('severity',0)})")
        lines.append(f"      {dist.get('detail','')}")
        lines.append("")

    # 量价异动
    vpa = r.get("volume_anomaly", {})
    if vpa.get("has_anomaly"):
        lines.append(f"  📈 量价异动: {vpa.get('anomaly_type','')}")
        lines.append(f"      {vpa.get('detail','')}")
        lines.append("")

    # 均线归位
    ma = r.get("ma_alignment", {})
    lines.append(f"  📐 均线状态: {'归位→多头有序' if ma.get('is_realigning') else '正常'} | 多头排列: {'是' if ma.get('is_bullish_aligned') else '否'}")
    lines.append("")

    # 洗盘
    washout = r.get("washout", {})
    if washout.get("is_washout_reversal"):
        lines.append(f"  🔄 洗盘反包: 检测到! 主力故意打压后反包")
        lines.append(f"      {washout.get('detail','')}")
        lines.append("")

    # 缺口
    gap = r.get("gap", {})
    if gap.get("up_gap_unfilled"):
        lines.append(f"  ⬆️ 向上缺口: {gap.get('up_gap_days',0)}日未补, 强势确认")
        lines.append("")
    if gap.get("down_gap"):
        lines.append(f"  ⬇️ 向下缺口: 风险信号 (出现在{zone.get('zone','?')}区)")
        lines.append("")

    # 猎取战法
    lines.append("  ── 猎取战法匹配 ──")
    any_triggered = False
    for wf_name in ["逼空星线", "猎取B区", "A区起涨", "拉高抢筹", "C区风险过滤", "高位倒灌出货"]:
        wf = r.get("warfare", {}).get(wf_name, {})
        if isinstance(wf, dict):
            triggered = wf.get("triggered", False)
            score = wf.get("score", 0)
            conds = wf.get("conditions_met", [])
            mark = "✅" if triggered else "  "
            if triggered or score >= 3:
                any_triggered = True
                lines.append(f"  {mark} {wf_name:<12s} {score:>3d}分  {' | '.join(conds[:4]) if conds else ''}")
    if not any_triggered:
        lines.append(f"     (无触发)")

    # 正面/风险信号汇总
    lines.append("")
    lines.append("  ── 信号汇总 ──")
    if s["signals"]:
        for sig_type, sig_text in s["signals"]:
            lines.append(f"  ✅ [{sig_type}] {sig_text}")
    if s["risks"]:
        for risk_type, risk_text in s["risks"]:
            lines.append(f"  ⚠️ [{risk_type}] {risk_text}")

    # 厚度
    thick = r.get("thickness", {})
    if thick and "error" not in thick:
        score = thick.get("score", 0)
        icon = "✅" if thick.get("has_thickness") else "⚠️"
        lines.append(f"  {icon} 三度厚度: {score}分 {'底部扎实' if score >= 3 else '厚度不足'} "
                     f"(阳量{thick.get('yang_ratio',0):.0%} | 放量{thick.get('wide_days',0)}天 | 大阳量{thick.get('tall_bars',0)}根)")

    # 主力意图
    intent = r.get("pullup_intent", {})
    if intent and "error" not in intent:
        lines.append(f"  🎯 主力意图: {intent.get('intent','?')} (置信{intent.get('confidence',0):.0%}) — {intent.get('detail','')}")

    # ── 交易计划 (规则推算，始终有) ──
    plan = r.get("trading_plan", {})
    if plan:
        rr = plan.get("rr_ratio", 0)
        risk_pct = plan.get("risk_pct", 0)
        reward_pct = plan.get("reward_pct", 0)
        atr = plan.get("atr", 0)
        trend = plan.get("trend_type", "")

        if rr >= 2.5:
            rr_grade = "🟢 优秀"
        elif rr >= 1.5:
            rr_grade = "🟡 合格"
        elif rr >= 1.0:
            rr_grade = "🟠 勉强"
        else:
            rr_grade = "🔴 不值"

        lines.append("")
        lines.append("  ── 📐 交易计划 (规则推算) ──")
        next_day = plan.get("next_day_note", "")
        if next_day:
            lines.append(f"  {next_day}")
        lines.append(f"  趋势: {trend}")
        lines.append(f"  入场: ¥{plan.get('entry', r['price']):.2f}  |  "
                     f"止损: ¥{plan.get('stop', 0):.2f}  |  "
                     f"止盈: ¥{plan.get('target', 0):.2f}")
        lines.append(f"  止损: -{risk_pct:.1f}%  |  "
                     f"止盈: +{reward_pct:.1f}%  |  "
                     f"ATR: ¥{atr:.2f}")
        lines.append(f"  ═══ 盈亏比 1:{rr:.1f} → {rr_grade} ═══")

    # ── Agent 定性判断 (可选) ──
    agent = r.get("agent_verdict")
    if agent and "error" not in agent:
        lines.append("")
        lines.append("  ── 🤖 Agent 定性 ──")
        lines.append(f"  判决: {agent.get('final', '?')}  |  {agent.get('verdict', '')[:100]}")
        if agent.get("bull"):
            lines.append(f"  看多: {agent['bull'][:120]}")
        if agent.get("bear"):
            lines.append(f"  看空: {agent['bear'][:120]}")
    elif agent and "error" in agent:
        lines.append(f"")
        lines.append(f"  🤖 Agent: ❌ {agent['error']}")

    # ── 综合评估 ──
    zone_name = r.get("zone", {}).get("zone", "?")
    dist = r.get("distribution", {})
    plan = r.get("trading_plan", {})
    rr = plan.get("rr_ratio", 0)
    trend_type = plan.get("trend_type", "")
    is_off = plan.get("is_offensive", False)
    triggered_wf = any(
        isinstance(w, dict) and w.get("triggered")
        for w in r.get("warfare", {}).values()
    )

    fatal = []
    if zone_name == "C":
        fatal.append("C区风险区，坚决回避")
    if dist.get("has_distribution"):
        fatal.append(f"出货信号:{dist.get('distribution_type','')}")
    if r.get("pe", 0) < 0:
        fatal.append("亏损股")
    if rr < 1.0:
        fatal.append(f"盈亏比1:{rr:.1f}不值博")

    positives = []
    if zone_name == "A":
        positives.append("A区强势")
    elif zone_name == "B" and (is_off or "进攻" in trend_type):
        positives.append("进攻型B区")
    if triggered_wf:
        positives.append("战法触发")
    if rr >= 2.5:
        positives.append(f"盈亏比优秀(1:{rr:.1f})")
    elif rr >= 1.5:
        positives.append(f"盈亏比合格(1:{rr:.1f})")

    if fatal:
        verdict = "⛔ 不建议买入"
        reason = "; ".join(fatal)
    elif len(positives) >= 2 and rr >= 2.0:
        verdict = "🔥 值得博弈"
        reason = " + ".join(positives)
    elif len(positives) >= 1 and rr >= 1.5:
        verdict = "✅ 可以关注"
        reason = " + ".join(positives)
    else:
        verdict = "👀 再看看"
        reason = "信号不够强，等更多确认"

    lines.append("")
    lines.append(f"  ═══ 综合: {verdict} ═══")
    lines.append(f"  理由: {reason}")
    lines.append("")
    lines.append("=" * 60)
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="个股战法分析 —《股是股非》战法 + 猎取战法一键诊断")
    parser.add_argument("codes", type=str, help="股票代码(逗号分隔)")
    parser.add_argument("--date", type=str, default=None, help="目标日期 YYYY-MM-DD")
    parser.add_argument("--agent", action="store_true", help="调用7-Agent辩论")
    parser.add_argument("--json", action="store_true", help="JSON输出")
    parser.add_argument("--full", action="store_true", help="输出完整JSON(含K线数据)")
    args = parser.parse_args()

    ensure_dirs()
    codes = [c.strip().zfill(6) for c in args.codes.split(",") if c.strip()]

    results = []
    for i, code in enumerate(codes):
        if len(codes) > 1:
            print(f"\n[{i+1}/{len(codes)}] {code}...", file=sys.stderr)
        r = analyze_stock(code, args.date, with_agent=args.agent)
        results.append(r)

        if args.json:
            if not args.full:
                r.pop("kline_data", None)
        else:
            print(format_report(r))
            if len(codes) > 1 and i < len(codes) - 1:
                print()

    if args.json:
        print(json.dumps(results if len(results) > 1 else results[0],
                         ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
