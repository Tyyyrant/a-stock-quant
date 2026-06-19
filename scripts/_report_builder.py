#!/usr/bin/env python3
"""
最终 Markdown 日报生成器
由 unified_pipeline.py 内部调用
"""

import json
from datetime import datetime


def build_markdown_report(result: dict) -> str:
    """从统一流水线结果生成 Markdown 日报"""
    lines = []
    today = result.get("date", datetime.now().strftime("%Y-%m-%d"))

    lines.append(f"# 量化主线双轮选股日报 — {today}")
    lines.append("")
    lines.append("> 自动生成，仅供参考研究，不构成投资建议")
    lines.append("")

    # ================================================================
    # 1. 市场环境
    # ================================================================
    lines.append("## 1. 当前市场环境")
    lines.append("")

    if "discovery" in result:
        env = result["discovery"].get("market_environment", {})
    else:
        env = {}

    nb = env.get("northbound", {})
    lines.append(f"- 概念板块总数: {env.get('total_concept_boards', 'N/A')}")
    lines.append(f"- 今日强势股: {env.get('total_hot_stocks', 'N/A')} 只")
    lines.append(f"- 龙虎榜上榜: {env.get('total_lhb_records', 'N/A')} 只")
    lines.append(f"- 北向资金: **{nb.get('direction', 'N/A')}** "
                 f"(5日 {nb.get('net_total_5d', 'N/A')}亿, "
                 f"20日 {nb.get('net_total_20d', 'N/A')}亿, "
                 f"趋势: {nb.get('trend', 'N/A')})")
    lines.append("")

    # ================================================================
    # 2. 当前最强主线
    # ================================================================
    lines.append("## 2. 当前最强主线")
    lines.append("")

    if "discovery" in result:
        themes = result["discovery"].get("themes", [])
        if themes:
            lines.append("| 排名 | 主线 | 热度 | 持续性 | 概念5日涨幅 | 强势股数 | 涨跌比 | 成分股 |")
            lines.append("|------|------|------|--------|-----------|---------|--------|--------|")
            for i, t in enumerate(themes, 1):
                s = t["signals"]
                persistence_cn = {"rising": "🔥 上升", "new": "🆕 新出现", "fading": "🔻 衰减"}.get(
                    t["persistence"], t["persistence"])
                lines.append(
                    f"| {i} | **{t['name']}** | {t['heat_score']:.0f} | {persistence_cn} | "
                    f"{s.get('concept_change_5d', 0):+.1f}% | {s.get('ths_stock_count', 0)} | "
                    f"{s.get('up_down_ratio', '-')} | {t.get('constituent_count', 0)} 只 |"
                )
            lines.append("")
        else:
            lines.append("(未发现显著主线)")
            lines.append("")
    else:
        themes_result = result.get("themes", [])
        if themes_result:
            lines.append(f"**指定主线**: {', '.join(themes_result)}")
            lines.append("")

    # ================================================================
    # 3. 每条主线的双轮选股结果
    # ================================================================
    lines.append("## 3. 双轮选股结果")
    lines.append("")

    for theme, tr in result.get("theme_results", {}).items():
        lines.append(f"### 3.{list(result['theme_results'].keys()).index(theme)+1} {theme}")
        lines.append("")

        # 供应链拆链
        sc = tr.get("supply_chain", {})
        if sc and "error" not in sc:
            lines.append("#### 供应链拆解")
            lines.append("")
            lines.append("```")
            for layer_key, layer in sc.get("chain", {}).items():
                icon = "🔴" if layer.get("is_bottleneck") else "⚪"
                name = layer.get("name", layer_key)
                arch = layer.get("archetypes", [])
                arch_str = f" [{', '.join(arch)}]" if arch else ""
                lines.append(f"  {icon} [{layer_key}] {name}{arch_str}")
            lines.append("```")
            lines.append("")

            # 瓶颈原型
            arch_summary = sc.get("archetype_summary", [])
            if arch_summary:
                lines.append("**匹配瓶颈原型**:")
                for a in arch_summary[:5]:
                    lines.append(f"- {a}")
                lines.append("")

        # 双轮选股
        ds = tr.get("dual_selection", {})
        if ds and "error" not in ds:
            # A组
            result_a = ds.get("result_a", [])
            if result_a:
                lines.append("#### A组: 主线直接标的 (跟趋势)")
                lines.append("")
                lines.append("| 排名 | 代码 | 名称 | 综合得分 | PE | PB | 板块 |")
                lines.append("|------|------|------|---------|-----|-----|------|")
                for i, s in enumerate(result_a[:10], 1):
                    lines.append(
                        f"| A{i} | {s.get('代码', s.get('code',''))} | "
                        f"{s.get('名称', s.get('name',''))} | "
                        f"{s.get('综合得分', s.get('score',''))} | "
                        f"{s.get('PE', s.get('pe',''))} | "
                        f"{s.get('PB', s.get('pb',''))} | "
                        f"{s.get('板块', s.get('primary_sector',''))} |"
                    )
                lines.append("")

            # B组
            result_b = ds.get("result_b", [])
            if result_b:
                lines.append("#### B组: 瓶颈卡点标的 (找价值)")
                lines.append("")
                lines.append("| 排名 | 代码 | 名称 | 综合得分 | PE | PB | 板块 |")
                lines.append("|------|------|------|---------|-----|-----|------|")
                for i, s in enumerate(result_b[:10], 1):
                    lines.append(
                        f"| B{i} | {s.get('代码', s.get('code',''))} | "
                        f"{s.get('名称', s.get('name',''))} | "
                        f"{s.get('综合得分', s.get('score',''))} | "
                        f"{s.get('PE', s.get('pe',''))} | "
                        f"{s.get('PB', s.get('pb',''))} | "
                        f"{s.get('板块', s.get('primary_sector',''))} |"
                    )
                lines.append("")

            # A/B 对比
            comp = ds.get("comparison", {})
            if comp:
                lines.append("#### A/B 对比分析")
                lines.append("")
                lines.append(f"- A组: {comp.get('total_a', 0)} 只 | B组: {comp.get('total_b', 0)} 只 "
                             f"| 交集(A∩B): {comp.get('intersection_count', 0)} 只")
                lines.append(f"- A独有: {comp.get('a_only_count', 0)} 只 "
                             f"| B独有: {comp.get('b_only_count', 0)} 只")

                a_only = comp.get("a_only", [])
                if a_only:
                    lines.append("")
                    lines.append("**A有B无** (主线直接但不在瓶颈层):")
                    for s in a_only[:5]:
                        lines.append(f"- {s['code']} {s.get('name','')} (得分: {s.get('score','')})")

                b_only = comp.get("b_only", [])
                if b_only:
                    lines.append("")
                    lines.append("**B有A无** (瓶颈层但不在主线直接):")
                    for s in b_only[:5]:
                        lines.append(f"- {s['code']} {s.get('name','')} (得分: {s.get('score','')})")

                intersect = comp.get("intersection", [])
                if intersect:
                    lines.append("")
                    lines.append("**A∩B 双重确认标的**:")
                    for s in intersect[:5]:
                        lines.append(f"- {s['code']} {s.get('name','')} (得分: {s.get('score','')})")
                lines.append("")

        lines.append("---")
        lines.append("")

    # ================================================================
    # 4. 多Agent深度分析
    # ================================================================
    agent_results = result.get("agent_analysis", [])
    if agent_results:
        lines.append("## 4. 多Agent深度分析")
        lines.append("")
        lines.append(f"对 {len(agent_results)} 只标的进行了技术面/基本面/新闻/宏观四维数据采集")
        lines.append("")

        lines.append("| 代码 | 名称 | 技术信号 | 基本面信号 | 新闻情绪 | 宏观 | 看多 | 看空 |")
        lines.append("|------|------|---------|-----------|---------|------|------|------|")
        for ar in agent_results:
            if "error" in ar:
                continue
            comp = ar.get("composite", {})
            sig = comp.get("signals", {})
            lines.append(
                f"| {ar['ticker']} | {ar.get('name','')} | "
                f"{comp.get('technical_signal','?')} | "
                f"{comp.get('fundamental_signal','?')} | "
                f"{comp.get('news_signal','?')} | "
                f"{comp.get('macro_signal','?')} | "
                f"{sig.get('bullish', 0)} | "
                f"{sig.get('bearish', 0)} |"
            )
        lines.append("")

        # 详细Agent报告（前5只）
        lines.append("### 详细分析报告")
        lines.append("")
        lines.append("> 以下为各标的多维数据摘要，完整Agent辩论请运行 Claude Code Agent 管线")
        lines.append("")

        # 引用各标的报告文件
        for ar in agent_results[:5]:
            if "error" in ar:
                continue
            lines.append(f"#### {ar['ticker']} {ar.get('name','')}")
            lines.append(f"- 详见: `output/{result['date']}/agent_analysis/{ar['ticker']}_report.md`")
            comp = ar.get("composite", {})
            sig = comp.get("signals", {})
            lines.append(f"- 技术面信号: **{comp.get('technical_signal','?')}** | "
                         f"基本面信号: **{comp.get('fundamental_signal','?')}**")
            lines.append(f"- 多空信号比: 看多 {sig.get('bullish',0)} / 看空 {sig.get('bearish',0)}")
            lines.append("")
    else:
        lines.append("## 4. 多Agent深度分析")
        lines.append("")
        lines.append("(跳过Agent分析 — 使用 `--no-skip-agent` 启用)")
        lines.append("")

    # ================================================================
    # 5. 风控提示
    # ================================================================
    lines.append("## 5. 风控提示")
    lines.append("")
    lines.append("- ⚠️ 本报告仅供研究教育用途，不构成投资建议")
    lines.append("- ⚠️ 所有标的均经过基础风控过滤（ST/流动性/质押），但不保证无其他风险")
    lines.append("- ⚠️ 因子打分基于历史数据，不代表未来表现")
    lines.append("- ⚠️ B组瓶颈股可能距离催化剂兑现还有较长时间，请结合自身持仓周期判断")
    lines.append("- ⚠️ A组主线直接股已经有一定涨幅，追高需谨慎")
    lines.append("")

    lines.append(f"*报告生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")

    return "\n".join(lines)
