#!/usr/bin/env python3
"""
Phase 5: 统一三阶段流水线 — 一键运行

  Step 1: 数据驱动发现当前热门主线 (theme_discovery)
  Step 2: 供应链瓶颈挖掘 + 双轮因子筛选 (supply_chain_mapper + dual_selection)
  Step 3: 多Agent深度分析 (multi_agent_analyzer)

最终输出: Markdown 日报

用法:
  python3 scripts/unified_pipeline.py                              # 全自动: 发现→拆链→双轮选股→agent分析
  python3 scripts/unified_pipeline.py --theme "AI数据中心电源"      # 指定主线
  python3 scripts/unified_pipeline.py --skip-agent                 # 跳过agent分析(更快速)
  python3 scripts/unified_pipeline.py --top-themes 3 --top-n 10    # Top3主线，每主线Top10
  python3 scripts/unified_pipeline.py --save                       # 保存所有中间+最终报告
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

from collections import Counter

ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / "output"
sys.path.insert(0, str(ROOT / "scripts"))

from data_loader import ensure_dirs

# ============================================================
# 流水线编排
# ============================================================

def run_pipeline(themes: list[str] = None,
                 top_themes: int = 3,
                 top_n: int = 10,
                 run_agent: bool = True,
                 save: bool = True) -> dict:
    """
    主流水线。

    Args:
        themes: 指定主线列表，None = 自动发现
        top_themes: 自动发现时取前N条主线
        top_n: 每轮选股Top N
        run_agent: 是否运行多Agent分析
        save: 是否保存报告

    Returns:
        完整的流水线结果字典
    """
    today = datetime.now().strftime("%Y-%m-%d")
    print(f"\n{'='*70}")
    print(f"  量化主线双轮选股系统 — {today}")
    print(f"{'='*70}")

    # ================================================================
    # Step 1: 主线发现
    # ================================================================
    if themes is None:
        print(f"\n{'='*70}")
        print(f"  STEP 1: 热门主线自动发现")
        print(f"{'='*70}")

        from theme_discovery import discover_themes
        discovery = discover_themes(top_n=top_themes)
        themes = [t["name"] for t in discovery.get("themes", [])]

        if not themes:
            print("\n  [FAIL] 未发现任何主线，请检查数据源连接")
            return {"date": today, "error": "未发现主线", "step": "theme_discovery"}

        print(f"\n  发现 {len(themes)} 条主线:")
        for i, t in enumerate(discovery.get("themes", []), 1):
            print(f"    {i}. {t['name']} (热度: {t['heat_score']:.0f})")

        result = {"date": today, "discovery": discovery}
    else:
        print(f"\n  使用指定主线: {themes}")
        result = {"date": today, "themes": themes}

    if save:
        date_dir = OUTPUT_DIR / today
        date_dir.mkdir(parents=True, exist_ok=True)

    # ================================================================
    # Step 2: 对每条主线做供应链拆链 + 双轮选股
    # ================================================================
    print(f"\n{'='*70}")
    print(f"  STEP 2: 供应链瓶颈挖掘 + 双轮因子筛选")
    print(f"{'='*70}")

    all_results = {}
    all_codes_a = set()
    all_codes_b = set()

    for theme in themes:
        print(f"\n  --- 主线: {theme} ---")

        # 2a: 供应链拆链
        from supply_chain_mapper import map_supply_chain
        sc_result = map_supply_chain(theme)
        all_results[theme] = {"supply_chain": sc_result}

        # 收集所有候选股（瓶颈层 + 主线直接板块）
        bottleneck_codes = set()
        for layer_key, bl in sc_result.get("bottleneck_layers", {}).items():
            for s in bl.get("stocks", []):
                bottleneck_codes.add(s["code"])

        # 加上主线直接成分股
        from theme_discovery import get_theme_constituents
        direct_stocks = get_theme_constituents(theme)
        direct_codes = {s["code"] for s in direct_stocks}

        all_candidate_codes = list(bottleneck_codes | direct_codes)

        if not all_candidate_codes:
            print(f"    主线「{theme}」无候选股，跳过")
            continue

        print(f"    瓶颈候选: {len(bottleneck_codes)} 只 | 直接候选: {len(direct_codes)} 只 "
              f"| 合计: {len(all_candidate_codes)} 只")

        # 2b: 双轮因子筛选
        from dual_selection import run_dual_selection
        ds_result = run_dual_selection(
            all_candidate_codes,
            mode="both",
            top_n=top_n,
        )
        all_results[theme]["dual_selection"] = ds_result

        # 收集代码用于 Agent 分析
        for s in ds_result.get("result_a", []):
            all_codes_a.add(s.get("代码", s.get("code", "")))
        for s in ds_result.get("result_b", []):
            all_codes_b.add(s.get("代码", s.get("code", "")))

    # ================================================================
    # Step 3: 多Agent深度分析
    # ================================================================
    if run_agent:
        print(f"\n{'='*70}")
        print(f"  STEP 3: 多Agent深度分析")
        print(f"{'='*70}")

        # 对 A+B 去重后的标的做分析
        agent_codes = list(all_codes_a | all_codes_b)[:15]  # 最多15只
        print(f"  分析标的: {len(agent_codes)} 只 (A组: {len(all_codes_a)}, B组: {len(all_codes_b)})")

        if agent_codes:
            from multi_agent_analyzer import analyze_stocks, format_agent_prompt
            agent_results = analyze_stocks(agent_codes, today)
            result["agent_analysis"] = agent_results
        else:
            result["agent_analysis"] = []
    else:
        result["agent_analysis"] = []

    result["theme_results"] = all_results

    # --- 保存 ---
    if save:
        _save_results(result, today)

    # --- 打印最终报告 ---
    _print_final_report(result)

    return result


def _save_results(result: dict, today: str):
    """保存所有结果到 output/{date}/"""
    date_dir = OUTPUT_DIR / today
    date_dir.mkdir(parents=True, exist_ok=True)

    # 主线发现结果
    if "discovery" in result:
        with open(date_dir / "themes.json", "w") as f:
            json.dump(result["discovery"], f, ensure_ascii=False, indent=2, default=str)

    # 每条主线的详细结果
    for theme, tr in result.get("theme_results", {}).items():
        theme_dir = date_dir / f"theme_{theme}"
        theme_dir.mkdir(parents=True, exist_ok=True)

        sc = tr.get("supply_chain", {})
        if sc and "error" not in sc:
            from supply_chain_mapper import format_supply_chain_report
            with open(theme_dir / "supply_chain.md", "w") as f:
                f.write(format_supply_chain_report(sc))
            with open(theme_dir / "supply_chain.json", "w") as f:
                json.dump(sc, f, ensure_ascii=False, indent=2, default=str)

        ds = tr.get("dual_selection", {})
        if ds and "error" not in ds:
            # Save CSV
            import pandas as pd
            for key in ["result_a", "result_b"]:
                records = ds.get(key, [])
                if records:
                    pd.DataFrame(records).to_csv(theme_dir / f"{key}.csv", index=False)
            with open(theme_dir / "dual_selection.json", "w") as f:
                json.dump(ds, f, ensure_ascii=False, indent=2, default=str)

    # Agent 分析
    agent_results = result.get("agent_analysis", [])
    if agent_results:
        agent_dir = date_dir / "agent_analysis"
        agent_dir.mkdir(parents=True, exist_ok=True)
        from multi_agent_analyzer import format_agent_prompt
        for ar in agent_results:
            if "error" not in ar:
                report = format_agent_prompt(ar)
                with open(agent_dir / f"{ar['ticker']}_report.md", "w") as f:
                    f.write(report)
        with open(agent_dir / "analysis.json", "w") as f:
            json.dump(agent_results, f, ensure_ascii=False, indent=2, default=str)

    # 最终日报
    from _report_builder import build_markdown_report
    report = build_markdown_report(result)
    with open(date_dir / "daily_report.md", "w") as f:
        f.write(report)

    print(f"\n{'='*70}")
    print(f"  ✅ 报告已保存至 {date_dir}/")
    print(f"{'='*70}")


def _print_final_report(result: dict):
    """打印摘要到终端"""
    today = result["date"]

    print(f"\n{'='*70}")
    print(f"  最终报告摘要 — {today}")
    print(f"{'='*70}")

    # 主线总览
    if "discovery" in result:
        themes = result["discovery"].get("themes", [])
        print(f"\n  📊 当前最强主线 Top {len(themes)}:")
        for i, t in enumerate(themes, 1):
            print(f"    {i}. {t['name']} (热度: {t['heat_score']:.0f}, "
                  f"成分股: {t.get('constituent_count', 0)} 只)")

    # 选股结果
    print(f"\n  📈 选股结果:")
    for theme, tr in result.get("theme_results", {}).items():
        ds = tr.get("dual_selection", {})
        if not ds or "error" in ds:
            continue
        comp = ds.get("comparison", {})
        print(f"\n  ◆ {theme}:")
        print(f"    A组(主线直接): {ds.get('result_a_count', 0)} 只")
        print(f"    B组(瓶颈卡点): {ds.get('result_b_count', 0)} 只")
        if comp:
            print(f"    交集: {comp.get('intersection_count', 0)} 只 | "
                  f"A独有: {comp.get('a_only_count', 0)} 只 | B独有: {comp.get('b_only_count', 0)} 只")

    # Agent分析
    agent_results = result.get("agent_analysis", [])
    if agent_results:
        print(f"\n  🤖 Agent深度分析: {len(agent_results)} 只标的")
        for ar in agent_results[:5]:
            comp = ar.get("composite", {})
            sig = comp.get("signals", {})
            print(f"    {ar['ticker']} {ar.get('name','')} "
                  f"技术:{comp.get('technical_signal','?')} "
                  f"基本面:{comp.get('fundamental_signal','?')} "
                  f"看多信号:{sig.get('bullish',0)} 看空信号:{sig.get('bearish',0)}")


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="统一流水线: 主线发现→供应链拆链→双轮选股→Agent分析")
    parser.add_argument("--theme", type=str, help="指定主线名称（多个用逗号分隔）")
    parser.add_argument("--top-themes", type=int, default=3, help="自动发现Top N主线")
    parser.add_argument("--top-n", type=int, default=10, help="每轮选股Top N")
    parser.add_argument("--skip-agent", action="store_true", help="跳过多Agent分析")
    parser.add_argument("--save", action="store_true", default=True, help="保存报告")
    args = parser.parse_args()

    ensure_dirs()

    themes = None
    if args.theme:
        themes = [t.strip() for t in args.theme.split(",") if t.strip()]

    run_pipeline(
        themes=themes,
        top_themes=args.top_themes,
        top_n=args.top_n,
        run_agent=not args.skip_agent,
        save=args.save,
    )


if __name__ == "__main__":
    main()
