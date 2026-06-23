#!/usr/bin/env python3
"""
AB × 战法 融合分析 — AB价格行为 + 《股是股非》战法 双轨融合

通过独立调用现有 analyze_stock.py (战法) 和新建 abprice.py (AB引擎)，
输出统一结论：看多 / 看空 / 观望。

用法:
  python3 scripts/fuse_analysis.py 300085
  python3 scripts/fuse_analysis.py 300085,600141,002156
  python3 scripts/fuse_analysis.py 300085 --date 2026-06-22

  from fuse_analysis import fuse_analysis
  result = fuse_analysis("300085", "2026-06-22")
"""

import argparse, json, os, subprocess, sys
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from abprice import (
    classify_market_structure,
    identify_spike_channel,
    identify_signal_bars,
    find_high_low_entries,
    evaluate_trend_setup,
    detect_climax_reversal,
    is_always_in_long,
    analyze as ab_analyze,
    _ensure_columns,
)


# ══════════════════════════════════════════════════════════════
# 战法调用
# ══════════════════════════════════════════════════════════════

def run_warfare(code: str, date: str) -> dict:
    """调用 analyze_stock.py 获取战法诊断"""
    script = str(ROOT / "scripts" / "analyze_stock.py")
    result = subprocess.run(
        ["python3", script, code, "--date", date, "--json"],
        capture_output=True, text=True, timeout=60,
        env={**os.environ, "PYTHONPATH": str(ROOT / "scripts")},
    )
    if result.returncode != 0:
        return {"error": result.stderr[:200]}
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError:
        return {"error": f"JSON解析失败: {result.stdout[:200]}"}


# ══════════════════════════════════════════════════════════════
# 战法方向映射
# ══════════════════════════════════════════════════════════════

def map_warfare_direction(warfare_diag: dict) -> dict:
    """将战法六档裁决映射为看多/看空/观望"""
    summary = warfare_diag.get("summary", {})
    verdict = summary.get("verdict", "➖ 无信号")
    fatal_count = summary.get("fatal_count", 0)

    bullish = ["🔥 强烈关注", "✅ 关注"]
    bearish = ["⛔ 回避", "⚠️ 谨慎"]
    neutral = ["👀 观察", "➖ 无信号"]

    if verdict in bullish:
        direction = "看多"
        confidence = "高" if verdict == "🔥 强烈关注" else "中"
    elif verdict in bearish:
        direction = "看空"
        confidence = "高" if fatal_count >= 2 else "中"
    elif verdict in neutral:
        direction = "观望"
        confidence = "低"
    else:
        direction = "观望"
        confidence = "低"

    return {
        "direction": direction,
        "confidence": confidence,
        "original_verdict": verdict,
        "fatal_count": fatal_count,
    }


# ══════════════════════════════════════════════════════════════
# 融合矩阵
# ══════════════════════════════════════════════════════════════

def fuse_direction(ab_direction: str, ab_phase: str,
                   wf_direction: str, wf_confidence: str) -> dict:
    """
    AB × 战法 方向融合矩阵。
    AB 确认状态影响最终标签:
      - confirmed+LONG + 战法看多 → 强烈看多
      - unconfirmed+LONG + 战法看多 → 看多(待AB确认)
    """
    # 基础方向矩阵 (只看方向，不看置信度)
    # 注意: ab_phase 影响 score_bonus，待确认时减半
    base = {
        ("LONG", "看多"):   ("看多", 20),
        ("LONG", "观望"):   ("看多", 10),
        ("LONG", "看空"):   ("观望", 0),
        ("SHORT", "看空"):  ("看空", 20),
        ("SHORT", "观望"):  ("看空", 10),
        ("SHORT", "看多"):  ("观望", 0),
        ("NEUTRAL", "看多"): ("看多", 10),
        ("NEUTRAL", "看空"): ("看空", 10),
        ("NEUTRAL", "观望"): ("观望", 0),
    }

    key = (ab_direction, wf_direction)
    if key in base:
        direction, score_bonus = base[key]
    else:
        direction, score_bonus = "观望", 0

    # 确认阶段打折
    if ab_phase == "unconfirmed_reversal":
        score_bonus = score_bonus // 2
    elif ab_phase == "developing":
        score_bonus = int(score_bonus * 0.75)

    # 分歧检测
    conflict = None
    is_conflict = (ab_direction == "LONG" and wf_direction == "看空") or \
                  (ab_direction == "SHORT" and wf_direction == "看多")
    if is_conflict:
        direction = "观望"
        score_bonus = 0
        conflict = {
            "ab_direction": ab_direction,
            "warfare_direction": wf_direction,
            "label": f"⚡ 分歧:AB{ab_direction}×战法{wf_direction}",
        }

    # 标签生成: 简单规则 — AB确认+方向一致=强烈
    # 终端用纯文本，PNG里用emoji（在generate_png中处理）
    if not is_conflict:
        if direction == "看多":
            if ab_phase == "confirmed":
                label = "强烈看多"
                confidence = "高"
            elif ab_phase == "unconfirmed_reversal":
                label = "看多(待AB确认)"
                confidence = "中"
            elif ab_direction == "LONG":
                label = "看多(AB驱动)"
                confidence = "中"
            else:
                label = "看多(战法驱动)"
                confidence = "中"
        elif direction == "看空":
            if ab_phase == "confirmed":
                label = "强烈看空"
                confidence = "高"
            else:
                label = "看空"
                confidence = "中"
        else:
            label = "观望"
            confidence = "低"
    else:
        label = conflict["label"]
        confidence = "分歧"

    return {
        "direction": direction,
        "confidence": confidence,
        "label": label,
        "score_bonus": score_bonus,
        "conflict": conflict,
    }


# ══════════════════════════════════════════════════════════════
# 入场/止损/止盈 合并
# ══════════════════════════════════════════════════════════════

def combine_entry_stop_target(ab_result: dict, warfare_diag: dict,
                               fusion_dir: str, confidence: str,
                               price: float) -> dict:
    """合并 AB 和战法的入场/止损/止盈"""

    plan = warfare_diag.get("trading_plan", {})
    wf_entry = plan.get("entry", price)
    wf_stop = plan.get("stop", price * 0.93)
    wf_target = plan.get("target", price * 1.1)

    # AB 入场: 从趋势评估取
    trend_setup = ab_result.get("trend_setup", {})
    ab_entry = trend_setup.get("entry_price", price)

    # AB 止损: 最近摆动低点
    ab_stop = trend_setup.get("stop_price", price * 0.95)

    # AB 目标: 从急速通道取
    spike = ab_result.get("spike_channel", {})
    ab_target = spike.get("target", price * 1.1)

    # AB 紧止损: 最近摆动低点/信号K线低点
    entries = ab_result.get("high_low_entries", {})
    best_entry_ab = entries.get("best_entry", {}) or {}
    ab_tight_stop = best_entry_ab.get("price", price * 0.95)

    # --- 入场价合并 ---
    if confidence == "高" and fusion_dir in ("看多", "看空"):
        # 强烈信号：取更接近现价的（激进）
        if fusion_dir == "看多":
            entry = max(ab_entry, wf_entry) if ab_entry and wf_entry else (wf_entry or ab_entry)
        else:
            entry = min(ab_entry, wf_entry) if ab_entry and wf_entry else (wf_entry or ab_entry)
        entry_style = "激进"
    else:
        # 中等/弱信号：取更远离现价的（保守，等更好价格）
        if fusion_dir == "看多":
            entry = min(ab_entry, wf_entry) if ab_entry and wf_entry else (wf_entry or ab_entry)
        else:
            entry = max(ab_entry, wf_entry) if ab_entry and wf_entry else (wf_entry or ab_entry)
        entry_style = "保守"

    if entry is None or entry == 0:
        entry = price

    # --- 双层止损 ---
    if fusion_dir == "看多":
        # 做多: 紧止损=较高的(离入场近), 宽止损=较低的(离入场远)
        stops_below = []
        for s in [ab_tight_stop, ab_stop, wf_stop]:
            if s and s < entry:
                stops_below.append(s)
        if len(stops_below) >= 2:
            stops_below.sort(reverse=True)  # 从高到低: [紧, 宽]
            stop_tight = stops_below[0]
            stop_wide = stops_below[-1]
        elif len(stops_below) == 1:
            stop_tight = stop_wide = stops_below[0]
        else:
            stop_tight = stop_wide = round(entry * 0.93, 2)
    else:
        # 做空: 紧止损=较低的(离入场近), 宽止损=较高的(离入场远)
        stops_above = []
        for s in [ab_tight_stop, ab_stop, wf_stop]:
            if s and s > entry:
                stops_above.append(s)
        if len(stops_above) >= 2:
            stops_above.sort()  # 从低到高: [紧, 宽]
            stop_tight = stops_above[0]
            stop_wide = stops_above[-1]
        elif len(stops_above) == 1:
            stop_tight = stop_wide = stops_above[0]
        else:
            stop_tight = stop_wide = round(entry * 1.07, 2)

    # 宽止损上限: 不超过入场价的 ±15%
    if fusion_dir == "看多" and stop_wide:
        stop_wide = max(stop_wide, entry * 0.85)
    elif fusion_dir == "看空" and stop_wide:
        stop_wide = min(stop_wide, entry * 1.15)

    # 合并判断: 两止损差距 < 2% 则合并
    merged = False
    if stop_tight and stop_wide and stop_wide > 0:
        if abs(stop_tight - stop_wide) / price < 0.02:
            merged = True
            stop_tight = stop_wide  # 用宽止损替代紧止损

    # --- 双目标止盈 ---
    if fusion_dir == "看多":
        targets = sorted([t for t in [ab_target, wf_target] if t and t > price])
        if not targets:
            targets = [price * 1.1]
    else:
        targets = sorted([t for t in [ab_target, wf_target] if t and t < price], reverse=True)
        if not targets:
            targets = [price * 0.9]

    t1 = targets[0] if targets else price
    t2 = targets[1] if len(targets) > 1 else t1

    return {
        "entry": round(entry, 2),
        "entry_style": entry_style,
        "stop_tight": round(stop_tight, 2) if stop_tight else None,
        "stop_wide": round(stop_wide, 2) if stop_wide else None,
        "stop_merged": merged,
        "target_t1": round(t1, 2),
        "target_t2": round(t2, 2),
        "ab_entry": round(ab_entry, 2) if ab_entry else None,
        "wf_entry": round(wf_entry, 2) if wf_entry else None,
    }


# ══════════════════════════════════════════════════════════════
# 综合评分
# ══════════════════════════════════════════════════════════════

def compute_fusion_score(ab_result: dict, wf_direction: str, score_bonus: int,
                          ab_phase: str = None) -> int:
    """计算 0-100 综合评分。AB确认阶段影响得分——待确认时降低权重。"""

    # AB得分 (0-40) — 基础分 × 确认系数
    spike = ab_result.get("spike_channel", {})
    always_in = ab_result.get("always_in", {})
    struct = ab_result.get("market_structure", {})

    ab_base = 0
    if spike.get("phase") == "spike" and spike.get("direction") == "up":
        ab_base = 40
    elif spike.get("phase") == "channel" and spike.get("direction") == "up":
        ab_base = 30
    elif always_in.get("direction") == "LONG":
        ab_base = 20
    elif always_in.get("direction") == "SHORT":
        ab_base = 0
    elif struct.get("structure") in ("trend_up",):
        ab_base = 25
    elif struct.get("structure") in ("trend_down",):
        ab_base = 5
    else:
        ab_base = 10

    # 确认系数: 待确认 → 打7折
    if ab_phase == "confirmed":
        phase_coef = 1.0
    elif ab_phase in ("unconfirmed_reversal",):
        phase_coef = 0.5
    else:
        phase_coef = 0.75

    ab_score = int(ab_base * phase_coef)

    # 战法得分 (0-40)
    wf_score = {"看多": 40, "观望": 20, "看空": 0}.get(wf_direction, 10)

    return min(100, ab_score + wf_score + score_bonus)


# ══════════════════════════════════════════════════════════════
# 分歧诊断
# ══════════════════════════════════════════════════════════════

def diagnose_conflict(ab_result: dict, warfare_diag: dict,
                       ab_dir: str, wf_dir: str) -> str:
    """分析AB与战法分歧的原因"""
    lines = []
    lines.append(f"⚡ AB{ab_dir} ⇄ 战法{wf_dir}")

    # AB看到什么
    ab_reasons = []
    ai = ab_result.get("always_in", {})
    sc = ab_result.get("spike_channel", {})
    cl = ab_result.get("climax", {})

    ab_reasons.append(f"Always-In: {ai.get('direction', '?')} ({ai.get('reason', '')[:60]})")
    if sc.get("phase") not in ("no_spike", "unknown", None):
        ab_reasons.append(f"急速: {sc.get('direction','')}向 {sc.get('phase','')}, 幅度¥{sc.get('spike_size',0)}")
    if cl.get("detected"):
        ab_reasons.append(f"高潮: {cl.get('type','')} — {cl.get('note','')[:60]}")

    lines.append(f"  AB看到: {'; '.join(ab_reasons)}")

    # 战法看到什么
    wf_reasons = []
    zone = warfare_diag.get("zone", {})
    summary = warfare_diag.get("summary", {})

    wf_reasons.append(f"{zone.get('zone','?')}区 ({zone.get('zone_reason','')[:60]})")
    if summary.get("risks"):
        risk_texts = [r[1][:40] if isinstance(r, tuple) else str(r)[:40]
                       for r in summary["risks"][:2]]
        wf_reasons.append(f"风险: {'; '.join(risk_texts)}")
    if summary.get("signals"):
        sig_texts = [s[1][:40] if isinstance(s, tuple) else str(s)[:40]
                      for s in summary["signals"][:2]]
        wf_reasons.append(f"信号: {'; '.join(sig_texts)}")

    lines.append(f"  战法看到: {'; '.join(wf_reasons)}")

    # 建议
    if ab_dir == "LONG" and wf_dir == "看空":
        lines.append("  建议: 短期有反弹动能但结构风险大，等战法风险信号消退再做多")
    elif ab_dir == "SHORT" and wf_dir == "看多":
        lines.append("  建议: 短期有回调压力但结构偏多，等AB空头信号消退再做多")

    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════
# 主融合函数
# ══════════════════════════════════════════════════════════════

def fuse_analysis(code: str, date: str = None) -> dict:
    """
    AB + 战法 融合分析。

    Args:
        code: 股票代码 (6位)
        date: 分析日期 YYYY-MM-DD，默认今天

    Returns:
        {code, name, price, ab: {...}, warfare: {...}, fusion: {...}}
    """
    if date is None:
        date = pd.Timestamp.now().strftime("%Y-%m-%d")

    # 1. 加载K线
    kline_path = ROOT / "data" / "stocks" / f"{code}.parquet"
    if not kline_path.exists():
        return {"code": code, "error": f"K线数据不存在: {kline_path}"}
    df = pd.read_parquet(kline_path)
    df = df[df["date"] <= date].copy()
    if df.empty:
        return {"code": code, "error": f"无{date}之前的K线数据"}

    price = float(df["close"].iloc[-1])

    # 2. 战法诊断
    warfare_diag = run_warfare(code, date)
    if "error" in warfare_diag:
        return {"code": code, "error": f"战法诊断失败: {warfare_diag['error']}"}
    name = warfare_diag.get("name", code)
    wf_mapped = map_warfare_direction(warfare_diag)

    # 3. AB分析
    try:
        ab_result = ab_analyze(df)
    except Exception as e:
        return {"code": code, "error": f"AB分析失败: {e}"}

    ab_direction = ab_result["always_in"]["direction"]
    ab_phase = ab_result["always_in"].get("phase")
    ab_confirmation = ab_result["always_in"].get("confirmation")

    # 4. 融合
    fusion = fuse_direction(ab_direction, ab_phase,
                             wf_mapped["direction"], wf_mapped["confidence"])
    score = compute_fusion_score(ab_result, wf_mapped["direction"],
                                  fusion["score_bonus"], ab_phase)

    # 5. 入场/止损/止盈
    ets = combine_entry_stop_target(ab_result, warfare_diag,
                                     fusion["direction"], fusion["confidence"],
                                     price)

    # 6. 分歧诊断
    conflict_detail = None
    if fusion["conflict"]:
        conflict_detail = diagnose_conflict(ab_result, warfare_diag,
                                             ab_direction, wf_mapped["direction"])

    # 7. PE/市值
    pe = warfare_diag.get("pe", 0)
    mcap = warfare_diag.get("mcap", 0)

    return {
        "code": code,
        "name": name,
        "date": date,
        "price": price,
        "chg_pct": warfare_diag.get("chg_pct", 0),
        "pe": pe,
        "mcap": mcap,
        "ab": {
            "direction": ab_direction,
            "phase": ab_phase,
            "confidence": round(ab_result["always_in"]["confidence"], 2),
            "market_structure": ab_result["market_structure"]["structure"],
            "spike_phase": ab_result["spike_channel"].get("phase", "none"),
            "spike_target": ab_result["spike_channel"].get("target"),
            "climax": ab_result["climax"]["type"],
            "climax_note": ab_result["climax"].get("note", ""),
            "confirmation": ab_confirmation,
        },
        "warfare": {
            "direction": wf_mapped["direction"],
            "verdict": wf_mapped["original_verdict"],
            "zone": warfare_diag.get("zone", {}).get("zone", "?"),
            "zone_score": warfare_diag.get("zone", {}).get("zone_score", 0),
            "rr_ratio": warfare_diag.get("trading_plan", {}).get("rr_ratio", 0),
        },
        "fusion": {
            "direction": fusion["direction"],
            "confidence": fusion["confidence"],
            "label": fusion["label"],
            "score": score,
            "entry": ets["entry"],
            "entry_style": ets["entry_style"],
            "stop_tight": ets["stop_tight"],
            "stop_wide": ets["stop_wide"],
            "stop_merged": ets["stop_merged"],
            "target_t1": ets["target_t1"],
            "target_t2": ets["target_t2"],
            "conflict": fusion["conflict"],
            "conflict_detail": conflict_detail,
        },
    }


# ══════════════════════════════════════════════════════════════
# 终端报告
# ══════════════════════════════════════════════════════════════

def print_report(result: dict):
    """打印单只股票的融合分析报告"""
    if "error" in result:
        print(f"  ❌ {result['code']}: {result['error']}")
        return

    f = result["fusion"]
    a = result["ab"]
    w = result["warfare"]
    price = result["price"]
    entry = f["entry"]

    # 方向颜色
    dir_colors = {"看多": "\033[91m", "看空": "\033[92m", "观望": "\033[93m"}
    c = dir_colors.get(f["direction"], "")
    reset = "\033[0m"

    print(f"\n{'='*65}")
    print(f"  {result['name']} ({result['code']})  —  AB×战法 融合分析")
    print(f"  {result['date']}  |  ¥{price:.2f}  |  PE:{result['pe']:.0f}  |  市值:{result['mcap']:.0f}亿")
    print(f"{'='*65}")

    # 融合结论
    print(f"  {c}融合结论: {f['label']}{reset}")
    print(f"  综合评分: {f['score']}/100  |  置信度: {f['confidence']}")

    # 双轨诊断
    phase_labels = {
        "confirmed": "已确认",
        "unconfirmed_reversal": "V反待确认",
        "developing": "趋势发展中",
    }
    ab_phase_label = phase_labels.get(a.get("phase", ""), "")

    print(f"\n  AB: {a['direction']} [{ab_phase_label}] | 结构:{a['market_structure']} | "
          f"急速:{a['spike_phase']} | 高潮:{a['climax'] or '无'}")
    if a.get("climax_note"):
        print(f"    ↳ {a['climax_note'][:80]}")

    # AB 确认条件
    if a.get("confirmation") and "全部确认条件满足" not in a["confirmation"]:
        print(f"    确认条件: {a['confirmation']}")

    print(f"  战法: {w['direction']} | 裁定:{w['verdict']} | "
          f"{w['zone']}区({w['zone_score']}分) | 盈亏比1:{w['rr_ratio']:.1f}")

    # 分歧详情
    if f["conflict"]:
        print(f"\n  {f['conflict_detail']}")

    # 交易计划
    if f["direction"] != "观望":
        print(f"\n  ┌─ 交易计划 ─────────────────────────────")
        print(f"  │ 入场: ¥{entry:.2f} ({f['entry_style']})")
        print(f"  │ 止损: ¥{f['stop_tight']:.2f} (紧) / ¥{f['stop_wide']:.2f} (宽)"
              + (" [合并]" if f["stop_merged"] else ""))
        print(f"  │ 止盈: T1=¥{f['target_t1']:.2f} → T2=¥{f['target_t2']:.2f}")
        if entry and entry != price:
            dist_to_entry = abs(price - entry) / price * 100
            dir_word = "回踩" if entry < price else "突破"
            print(f"  │ 距入场: {dir_word}{dist_to_entry:.1f}%")
        print(f"  └──────────────────────────────────────────")

    # 观望原因
    if f["direction"] == "观望" and not f["conflict"]:
        print(f"\n  💤 双方均无明确信号，建议观望")


def print_summary(results: list[dict]):
    """多只股票汇总表格"""
    print(f"\n{'='*95}")
    print(f"  AB×战法 融合分析汇总")
    print(f"{'='*95}")
    header = f"  {'代码':<8} {'名称':<8} {'现价':>8} {'涨跌':>8} {'AB':>6} {'战法':>6} {'融合':<16} {'评分':>5} {'入场':>8}"
    print(header)
    print(f"  {'-'*90}")

    for r in results:
        if "error" in r:
            print(f"  {r['code']:<8} ❌ {r['error'][:50]}")
            continue
        f = r["fusion"]
        a = r["ab"]
        w = r["warfare"]
        chg = r.get("chg_pct", 0)  # analyze_stock already returns percentage
        entry_str = f"¥{f['entry']:.2f}" if f['entry'] else "—"

        print(f"  {r['code']:<8} {r['name']:<8} {r['price']:>8.2f} {chg:>+7.1f}% "
              f"{a['direction']:>6} {w['direction']:>6} {f['label']:<16} {f['score']:>4}  {entry_str:>8}")

    print(f"{'='*95}")


# ══════════════════════════════════════════════════════════════
# PNG 报告生成
# ══════════════════════════════════════════════════════════════

def generate_png(result: dict) -> str:
    """为单只股票生成精美的融合分析 PNG 报告。返回 PNG 文件路径。"""
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("  ⚠️ Playwright 未安装，跳过 PNG 生成")
        return None

    f = result["fusion"]
    a = result["ab"]
    w = result["warfare"]
    code = result["code"]
    name = result["name"]
    date = result["date"]
    price = result["price"]
    chg = result.get("chg_pct", 0)
    pe = result.get("pe", 0)
    mcap = result.get("mcap", 0)

    # ── 颜色方案 ──
    dir_colors = {
        "看多": ("#d4343e", "#fef2f2", "#fecaca"),
        "看空": ("#1ca051", "#f0fdf4", "#bbf7d0"),
        "观望": ("#d97706", "#fffbeb", "#fde68a"),
    }
    primary, bg_light, bg_mid = dir_colors.get(f["direction"], ("#6b7280", "#f9fafb", "#e5e7eb"))

    # 评分条颜色
    score = f["score"]
    if score >= 90:   bar_color = "#d4343e"
    elif score >= 70: bar_color = "#e87400"
    elif score >= 50: bar_color = "#d97706"
    else:             bar_color = "#9ca3af"

    # ── 确认清单 HTML ──
    confirm_html = ""
    if a.get("confirmation") and "全部确认条件满足" not in a["confirmation"]:
        conditions = a["confirmation"].split(" | ")
        for cond in conditions:
            cond = cond.strip()
            if cond.startswith("V反待确认:") or cond.startswith("趋势待强化:"):
                label, _, rest = cond.partition(": ")
                confirm_html += f'<div class="confirm-header">{label}</div>'
                cond = rest
            # 判断状态
            if cond.startswith("⚠️"):
                icon = "⏳"; cls = "confirm-pending"
                text = cond[2:].strip()
            elif "待" in cond or "未" in cond or "消化" in cond or "走平" in cond or "拐头" in cond or "Higher Low" in cond or "站稳" in cond:
                icon = "⏳"; cls = "confirm-pending"
                text = cond
            else:
                icon = "✅"; cls = "confirm-done"
                text = cond
            confirm_html += f'<div class="confirm-row {cls}"><span class="confirm-icon">{icon}</span> {text}</div>'
    elif a.get("confirmation") and "全部确认条件满足" in a["confirmation"]:
        confirm_html = '<div class="confirm-row confirm-done"><span class="confirm-icon">✅</span> 全部确认条件满足 — 可执行交易计划</div>'
    else:
        confirm_html = '<div class="confirm-row confirm-done"><span class="confirm-icon">✅</span> 无需额外确认</div>'

    if f["direction"] == "观望":
        confirm_html = '<div class="confirm-row confirm-na"><span class="confirm-icon">➖</span> 方向不明确，无确认条件</div>'

    # ── 交易计划 HTML ──
    plan_html = ""
    if f["direction"] != "观望" and f["entry"]:
        entry_icon = "⚡" if f["entry_style"] == "激进" else "🎯"
        dist_text = ""
        if f["entry"] and price:
            dist = abs(price - f["entry"]) / price * 100
            if dist > 0.5:
                dir_word = "回踩" if f["entry"] < price else "突破"
                dist_text = f'<span class="plan-dist">{dir_word} {dist:.1f}%</span>'

        stop_tight = f['stop_tight']
        stop_wide = f['stop_wide']
        merged_note = ' <span class="plan-note">[已合并]</span>' if f.get('stop_merged') else ''

        plan_html = f'''
        <div class="plan-grid">
          <div class="plan-item">
            <div class="plan-label">{entry_icon} 入场 <span class="plan-style">{f['entry_style']}</span></div>
            <div class="plan-value">{f['entry']:.2f}</div>
            {dist_text}
          </div>
          <div class="plan-item">
            <div class="plan-label">🛑 止损 紧/宽{merged_note}</div>
            <div class="plan-value stop">¥{stop_tight:.2f} / ¥{stop_wide:.2f}</div>
          </div>
          <div class="plan-item">
            <div class="plan-label">🎯 止盈 T1→T2</div>
            <div class="plan-value target">¥{f['target_t1']:.2f} → ¥{f['target_t2']:.2f}</div>
            <div class="plan-sub">T1: AB等距目标 | T2: 战法目标</div>
          </div>
        </div>'''

    # ── 阶段标签 ──
    phase_labels = {
        "confirmed": "✅ 已确认",
        "unconfirmed_reversal": "⏳ V反待确认",
        "developing": "🔄 趋势发展中",
    }
    ab_phase_label = phase_labels.get(a.get("phase", ""), "")

    # ── 融合理由 ──
    ab_wf_reason = ""
    if a["direction"] == "LONG" and w["direction"] == "看多":
        if a.get("phase") == "confirmed":
            ab_wf_reason = "AB趋势确认 + 战法看多 — 双轨共振"
        elif a.get("phase") == "unconfirmed_reversal":
            ab_wf_reason = "AB V反待确认 + 战法看多 — 方向一致待技术确认"
        else:
            ab_wf_reason = "AB趋势发展中 + 战法看多"
    elif a["direction"] == "LONG" and w["direction"] == "观望":
        ab_wf_reason = "AB趋势确认 + 战法观望 — AB驱动"
    elif a["direction"] == "NEUTRAL" and w["direction"] == "看多":
        ab_wf_reason = "AB中性 + 战法看多 — 战法驱动"
    else:
        ab_wf_reason = f"AB {a['direction']} + 战法 {w['direction']}"

    # ── HTML ──
    html = f'''<!DOCTYPE html><html lang="zh"><head><meta charset="UTF-8"><style>
:root{{--bg:#f0f2f5;--card:#fff;--text:#1a1a2e;--muted:#8b8fa3;--border:#e8eaef;--up:#d4343e;--down:#1ca051}}
*{{margin:0;padding:0;box-sizing:border-box}}
body{{font-family:-apple-system,'PingFang SC','Microsoft YaHei',sans-serif;background:var(--bg);color:var(--text);padding:20px;max-width:800px;margin:0 auto;font-size:13px;line-height:1.6}}
.card{{background:var(--card);border-radius:12px;padding:22px 26px;margin-bottom:14px;box-shadow:0 1px 4px rgba(0,0,0,0.04)}}

/* Header */
.header{{background:linear-gradient(135deg,#1a1a2e 0%,#2d2d44 100%);color:#fff;padding:22px 28px;border-radius:12px 12px 0 0;display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:12px}}
.header h1{{font-size:21px;font-weight:800;letter-spacing:.5px}}
.header .sub{{font-size:10px;color:#8890a8;margin-top:3px}}
.metrics{{display:flex;gap:0}}
.met{{text-align:center;padding:0 16px;border-left:1px solid rgba(255,255,255,0.1)}}
.met:first-child{{border-left:none}}
.met .val{{font-size:24px;font-weight:900;line-height:1.2}}
.met .lab{{font-size:9px;color:#8890a8;text-transform:uppercase;letter-spacing:.3px;font-weight:600}}
.up{{color:var(--up)}}.dn{{color:var(--down)}}

/* Verdict */
.verdict{{text-align:center;padding:18px 0 8px}}
.verdict-badge{{display:inline-block;font-size:20px;font-weight:900;padding:8px 28px;border-radius:10px;background:{bg_light};color:{primary};border:2px solid {bg_mid};margin-bottom:6px}}
.score-section{{display:flex;align-items:center;justify-content:center;gap:12px;margin:8px 0}}
.score-bar-bg{{flex:1;max-width:300px;height:10px;background:#e5e7eb;border-radius:5px;overflow:hidden}}
.score-bar-fg{{height:100%;width:{score}%;background:{bar_color};border-radius:5px;transition:width .3s}}
.score-num{{font-size:28px;font-weight:900;color:{bar_color}}}
.confidence-line{{font-size:11px;color:var(--muted);margin-top:4px}}

/* Dual Track */
.dual-track{{display:grid;grid-template-columns:1fr 1fr;gap:1px;background:var(--border);border-radius:10px;overflow:hidden}}
.track{{background:var(--card);padding:16px 18px}}
.track-title{{font-size:12px;font-weight:800;margin-bottom:10px;padding-bottom:8px;border-bottom:2px solid}}
.track-title.ab{{border-bottom-color:#7c3aed}}
.track-title.wf{{border-bottom-color:#e87400}}
.track .row{{display:flex;justify-content:space-between;padding:3px 0;font-size:11px}}
.track .row .lbl{{color:var(--muted)}}
.track .row .val{{font-weight:700}}
.phase-badge{{display:inline-block;font-size:9px;font-weight:700;padding:1px 6px;border-radius:3px;margin-left:4px}}
.phase-confirmed{{background:#f0fdf4;color:#16a34a}}
.phase-unconfirmed{{background:#fffbeb;color:#d97706}}
.phase-developing{{background:#eff6ff;color:#2563eb}}

/* Confirmation */
.confirm-header{{font-size:11px;font-weight:800;color:var(--text);margin:8px 0 4px}}
.confirm-row{{font-size:12px;padding:6px 10px;margin:3px 0;border-radius:6px;display:flex;align-items:center;gap:8px}}
.confirm-done{{background:#f0fdf4;color:#166534}}
.confirm-pending{{background:#fffbeb;color:#92400e}}
.confirm-na{{background:#f9fafb;color:#6b7280}}
.confirm-icon{{font-size:14px;flex-shrink:0}}

/* Plan */
.plan-grid{{display:grid;grid-template-columns:1fr 1fr 1fr;gap:14px;margin-top:4px}}
.plan-item{{text-align:center;padding:10px 8px;background:#fafbfc;border-radius:8px}}
.plan-label{{font-size:10px;color:var(--muted);margin-bottom:4px;font-weight:600}}
.plan-style{{display:inline-block;font-size:9px;padding:1px 5px;border-radius:3px;background:#e5e7eb;color:#374151;margin-left:4px}}
.plan-value{{font-size:20px;font-weight:900}}
.plan-value.stop{{color:#1ca051}}
.plan-value.target{{color:#d4343e}}
.plan-dist{{font-size:10px;color:var(--muted);display:block;margin-top:2px}}
.plan-sub{{font-size:9px;color:var(--muted);margin-top:2px}}
.plan-note{{font-size:9px;color:#d97706;font-weight:600}}

/* Footer */
.footer{{text-align:center;color:var(--muted);font-size:10px;padding:12px;letter-spacing:.3px}}
.footer-card{{border-radius:0 0 12px 12px;padding:14px 28px;border-top:1px solid var(--border)}}

h3{{font-size:13px;font-weight:800;margin-bottom:8px}}
</style></head><body>

<div class="header">
  <div>
    <h1>{name} <span style="font-size:11px;color:#8890a8;font-weight:400">{code}</span></h1>
    <div class="sub">{date} · AB×战法 融合分析 · PE:{pe:.0f} · 市值:{mcap:.0f}亿</div>
  </div>
  <div class="metrics">
    <div class="met"><div class="val">¥{price:.2f}</div><div class="lab">现价</div></div>
    <div class="met"><div class="val" class="{'up' if chg>0 else 'dn'}">{chg:+.1f}%</div><div class="lab">涨跌</div></div>
    <div class="met"><div class="val">{f['score']}</div><div class="lab">评分</div></div>
  </div>
</div>

<div class="card">
  <div class="verdict">
    <div class="verdict-badge">{f['label']}</div>
    <div class="score-section">
      <span style="font-size:12px;font-weight:700;color:var(--muted)">综合评分</span>
      <div class="score-bar-bg"><div class="score-bar-fg"></div></div>
      <span class="score-num">{score}</span>
    </div>
    <div class="confidence-line">置信度: {f['confidence']} · {ab_wf_reason}</div>
  </div>
</div>

<div class="dual-track">
  <div class="track">
    <div class="track-title ab">📊 AB 价格行为</div>
    <div class="row"><span class="lbl">方向</span><span class="val" style="color:{'#d4343e' if a['direction']=='LONG' else '#1ca051' if a['direction']=='SHORT' else '#d97706'}">{'🟢' if a['direction']=='LONG' else '🔴' if a['direction']=='SHORT' else '🟡'} {a['direction']}<span class="phase-badge phase-{'confirmed' if a.get('phase')=='confirmed' else 'unconfirmed' if a.get('phase')=='unconfirmed_reversal' else 'developing'}">{ab_phase_label}</span></span></div>
    <div class="row"><span class="lbl">市场结构</span><span class="val">{a['market_structure']}</span></div>
    <div class="row"><span class="lbl">急速/通道</span><span class="val">{a['spike_phase']}</span></div>
    <div class="row"><span class="lbl">高潮信号</span><span class="val">{a['climax'] or '无'}</span></div>
    <div class="row"><span class="lbl">评分</span><span class="val">{a.get('confidence',0):.0%}</span></div>
  </div>
  <div class="track">
    <div class="track-title wf">⚔️ 战法系统</div>
    <div class="row"><span class="lbl">方向</span><span class="val">{'🟢' if w['direction']=='看多' else '🔴' if w['direction']=='看空' else '🟡'} {w['direction']}</span></div>
    <div class="row"><span class="lbl">裁定</span><span class="val">{w['verdict']}</span></div>
    <div class="row"><span class="lbl">三区定位</span><span class="val">{w['zone']}区 ({w['zone_score']}分)</span></div>
    <div class="row"><span class="lbl">盈亏比</span><span class="val" style="color:{'#16a34a' if w['rr_ratio']>=2.5 else '#d97706' if w['rr_ratio']>=1.5 else '#dc2626'}">1:{w['rr_ratio']:.1f}</span></div>
    <div class="row"><span class="lbl">状态</span><span class="val">{'进攻型' if f['entry_style']=='激进' else '防守型'}</span></div>
  </div>
</div>

<div class="card">
  <h3>🔍 确认清单</h3>
  {confirm_html}
</div>

<div class="card">
  <h3>📐 交易计划</h3>
  {plan_html if plan_html else '<div style="text-align:center;color:var(--muted);padding:12px">方向不明确，不设交易计划</div>'}
</div>

<div class="card">
  <h3>📋 AB信号详情</h3>
  <div style="font-size:11px;color:var(--muted);line-height:1.8">
    {a.get('climax_note', '')}<br>
    Always-In理由: {a.get('reason', '')[:200]}
  </div>
</div>

<div class="card footer-card">
  <div class="footer">AB×战法 融合分析 · {date} · 基于裸K+EMA20+战法诊断 · 仅供参考不构成投资建议</div>
</div>

</body></html>'''

    # ── 写入 HTML → Playwright 截图 ──
    output_dir = ROOT / "output" / date
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_name = name.replace("*", "").replace("/", "").replace(" ", "")
    html_path = output_dir / f"{safe_name}_{code}_fusion.html"
    png_path = output_dir / f"{safe_name}_{code}_fusion.png"
    html_path.write_text(html, encoding="utf-8")

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch()
            page = browser.new_page(viewport={"width": 800, "height": 1200}, device_scale_factor=2)
            page.goto(f"file://{html_path}", wait_until="networkidle")
            page.screenshot(path=str(png_path), full_page=True)
            browser.close()
        return str(png_path)
    except Exception as e:
        print(f"  ⚠️ PNG截图失败: {e}")
        return None


# ══════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="AB×战法 融合分析")
    parser.add_argument("codes", type=str, help="股票代码(逗号分隔)")
    parser.add_argument("--date", type=str, default=None)
    parser.add_argument("--png", action="store_true", default=True, help="生成PNG报告(默认开启)")
    parser.add_argument("--no-png", action="store_true", help="不生成PNG")
    args = parser.parse_args()

    date = args.date or pd.Timestamp.now().strftime("%Y-%m-%d")
    codes = [c.strip().zfill(6) for c in args.codes.split(",") if c.strip()]
    do_png = args.png and not args.no_png

    results = []
    for i, code in enumerate(codes):
        if len(codes) > 1:
            print(f"\n  [{i+1}/{len(codes)}] 分析 {code}...", end=" ", flush=True)
        result = fuse_analysis(code, date)
        results.append(result)

        if len(codes) > 1:
            f = result.get("fusion", {})
            print(f"{f.get('label', '❌')}")
        else:
            print_report(result)

        # PNG 生成
        if do_png and "error" not in result:
            png_path = generate_png(result)
            if png_path:
                print(f"  PNG: {png_path}")

    if len(codes) > 1:
        for r in results:
            if "error" not in r:
                print_report(r)
        print_summary(results)


if __name__ == "__main__":
    main()
