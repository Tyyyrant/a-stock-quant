#!/usr/bin/env python3
"""
多Agent深度分析编排器 v2 — 真正集成 7-Agent 辩论管线

改造要点（vs v1）:
  1. 接入 3 个新分析模块: K线形态 + 量价关系 + 筹码分布
  2. 为每只股票生成完整的多维度分析数据
  3. 为 7 个 Agent 各生成定制化的分析简报 (Agent Brief)
  4. 支持被 unified_pipeline / run_pipeline 调用

Agent 辩论流程 (由 Claude Code Agent 工具执行):
  Phase 1 (并行): Technical → News → Fundamentals → Macro  (4 agents)
  Phase 2 (串行): Bull → Bear(反驳Bull) → Risk(压力测试)  (3 agents)
  Phase 3: Research Manager → 综合推荐
  Phase 4: Trader → 入场/止损/仓位
  Phase 5: Portfolio Manager → 最终 BUY/HOLD/SELL

用法:
  # 单只股票完整数据准备
  python3 scripts/multi_agent_analyzer.py --ticker 600519

  # 生成 Agent 辩论所需的全部 Brief
  python3 scripts/multi_agent_analyzer.py --ticker 600519 --agent-briefs

  # 批量分析 + 保存
  python3 scripts/multi_agent_analyzer.py --tickers 600519,000858,300750 --save
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / "output"
sys.path.insert(0, str(ROOT / "scripts"))

from data_loader import ensure_dirs
from fetch_a_share_data import (
    fetch_technical, fetch_fundamentals, fetch_news, fetch_a_share_macro,
)

# ---- 新分析模块 ----
from candlestick_patterns import identify_all_patterns, pattern_result_to_dict, PatternResult
from volume_price_analyzer import analyze_volume_price
from chip_distribution import estimate_chip_distribution


# ============================================================
# 主分析函数
# ============================================================

def analyze_single_stock(ticker: str,
                          date: str = None,
                          kline_df: pd.DataFrame = None,
                          fundamentals: dict = None,
                          enrich: bool = True) -> dict:
    """
    对单只股票做全方位数据采集 + 深度分析。

    Args:
        ticker: 股票代码
        date: 目标日期
        kline_df: 可传入已有的 K线 DataFrame（避免重复拉取）
        fundamentals: 可传入已有的基本面数据
        enrich: 是否运行新分析模块（K线形态+量价+筹码）

    Returns:
        完整的多维度分析报告
    """
    if date is None:
        date = datetime.now().strftime("%Y-%m-%d")

    print(f"  [{ticker}] 采集多维度数据...")

    # ---- 基础数据采集 ----
    tech = fetch_technical(ticker, date)
    fund = fetch_fundamentals(ticker, date)
    news = fetch_news(ticker, date)
    macro = fetch_a_share_macro("MACRO", date)

    name = fund.get("name", ticker)

    result = {
        "ticker": ticker,
        "name": name,
        "date": date,
        "technical": tech,
        "fundamentals": fund,
        "news": news,
        "macro": macro,
    }

    # ---- 深度分析模块 ----
    if enrich:
        # 获取完整 K线用于深度分析
        if kline_df is None:
            from data_loader import get_stock_kline
            market = 1 if ticker.startswith("6") else 0
            kline_df = get_stock_kline(ticker, market, refresh=False)

        if kline_df is not None and not kline_df.empty and len(kline_df) >= 60:
            kline_df = kline_df[kline_df["date"] <= date].copy()

            # K线形态
            patterns = identify_all_patterns(kline_df, ticker=ticker)
            result["candlestick"] = pattern_result_to_dict(patterns)

            # 量价关系
            vol_price = analyze_volume_price(kline_df)
            result["volume_price"] = vol_price

            # 筹码分布
            fdata = fundamentals or {}
            chip_fund = {}
            # 尝试从基本面提取筹码相关数据
            if fdata:
                if "shareholder_count" in fdata:
                    chip_fund["shareholder_count"] = fdata["shareholder_count"]
            chip = estimate_chip_distribution(kline_df, ticker=ticker, fundamentals=chip_fund)
            result["chip_distribution"] = chip

    # ---- 综合信号 ----
    result["composite"] = _build_composite_v2(result)

    return result


def _build_composite_v2(data: dict) -> dict:
    """v2 综合信号计算 — 纳入新分析模块"""

    tech = data.get("technical", {})
    fund = data.get("fundamentals", {})
    news = data.get("news", {})
    macro = data.get("macro", {})
    candlestick = data.get("candlestick", {})
    vol_price = data.get("volume_price", {})
    chip = data.get("chip_distribution", {})

    bullish = 0
    bearish = 0
    reasons_bull = []
    reasons_bear = []

    # --- 技术面 ---
    try:
        trend = tech.get("trend", {})
        price_vs_sma20 = trend.get("price_vs_sma20", 0)
        ma_align = trend.get("ma_alignment", "mixed")
        momentum = tech.get("momentum", {})
        rsi = momentum.get("rsi14")

        if ma_align == "bullish":
            bullish += 2
            reasons_bull.append("均线多头排列")
        elif ma_align == "bearish":
            bearish += 1
            reasons_bear.append("均线空头排列")

        if rsi is not None:
            if 30 < rsi < 65:
                bullish += 1
                reasons_bull.append(f"RSI={rsi:.0f} 健康区间")
            elif rsi > 80:
                bearish += 2
                reasons_bear.append(f"RSI={rsi:.0f} 严重超买")
            elif rsi < 25:
                bearish += 1
                reasons_bear.append(f"RSI={rsi:.0f} 弱势")
    except Exception:
        pass

    # --- 基本面 ---
    try:
        val = fund.get("valuation", {})
        pe = val.get("pe_ttm")
        if pe is not None and pe > 0:
            if pe < 20:
                bullish += 2
                reasons_bull.append(f"PE={pe:.0f} 低估值")
            elif pe > 100:
                bearish += 1
                reasons_bear.append(f"PE={pe:.0f} 高估值")
    except Exception:
        pass

    # --- 新闻面 ---
    sent = news.get("sentiment", "neutral")
    if sent == "positive":
        bullish += 1
        reasons_bull.append(f"新闻情绪偏正面")
    elif sent == "negative":
        bearish += 1
        reasons_bear.append("新闻情绪偏负面")

    # --- 宏观：risk-off下个股仍强 = 相对强度加分，不自动扣分 ---
    env = macro.get("environment", "neutral")
    tech_bullish_now = False  # 先占位
    if env == "risk_on":
        bullish += 1
        reasons_bull.append("宏观风险偏好")
    # risk_off 不直接加分/扣分，留给后面"逆势判断"

    # --- K线形态 ---
    if candlestick:
        cs = candlestick.get("latest_signal", "neutral")
        cscore = candlestick.get("pattern_score", 0)
        if cs == "bullish":
            bullish += 2
            reasons_bull.append(f"K线形态偏多({cscore:.0f}分)")
        elif cs == "bearish":
            bearish += 2
            reasons_bear.append(f"K线形态偏空({cscore:.0f}分)")

    # --- 量价关系 ---
    if vol_price:
        vs = vol_price.get("volume_score", 0)
        if vs > 15:
            bullish += 2
            reasons_bull.append(f"量价健康({vs:.0f}分)")
        elif vs < -15:
            bearish += 2
            reasons_bear.append(f"量价异常({vs:.0f}分)")

    # --- 筹码分布：趋势中高获利盘=健康，不是风险 ---
    if chip:
        cs = chip.get("chip_score", 0)
        profit_ratio = chip.get("profit_ratio", 0.5)

        # 关键修正: 高获利盘+趋势向上=筹码锁定好，非回吐风险
        # 只有高获利盘+趋势走弱才是风险
        has_strong_trend = (
            candlestick.get("latest_signal") == "bullish" or
            (vol_price.get("volume_score", 0) > 0)
        )

        if profit_ratio > 0.8 and has_strong_trend:
            bullish += 2
            reasons_bull.append(f"筹码高度锁定(获利{profit_ratio:.0%})")
        elif profit_ratio > 0.8:
            bearish += 2
            reasons_bear.append(f"获利盘{profit_ratio:.0%}，趋势若破有回吐风险")

        if cs > 15:
            bullish += 1
            reasons_bull.append(f"筹码结构优({cs:.0f}分)")
        elif cs < -15:
            bearish += 1
            reasons_bear.append(f"筹码压力({cs:.0f}分)")

    # --- 逆势加分：宏观risk-off下个股仍强 = 强庄/强逻辑 ---
    if env == "risk_off":
        # 如果技术面+形态+量价综合偏多，逆势上涨是加分
        tech_bull_points = bullish - bearish
        if tech_bull_points >= 3:
            bullish += 3
            reasons_bull.append("🔥逆势走强，主力实力突出")
        elif tech_bull_points >= 1:
            bullish += 1
            reasons_bull.append("大盘偏弱但个股抗跌")

    # 风险标记汇总
    all_risks = []
    if vol_price:
        all_risks.extend(vol_price.get("risk_flags", []))
    if chip:
        all_risks.extend(chip.get("risk_flags", []))

    # --- 最终信号 ---
    net = bullish - bearish
    if net >= 5:
        tech_signal = "STRONG_BULLISH"
    elif net >= 2:
        tech_signal = "BULLISH"
    elif net <= -5:
        tech_signal = "STRONG_BEARISH"
    elif net <= -2:
        tech_signal = "BEARISH"
    else:
        tech_signal = "NEUTRAL"

    return {
        "technical_signal": tech_signal,
        "net_score": net,
        "bullish_count": bullish,
        "bearish_count": bearish,
        "reasons_bull": reasons_bull,
        "reasons_bear": reasons_bear,
        "risk_flags": all_risks,
        "dimension_scores": {
            "candlestick": candlestick.get("pattern_score", 0) if candlestick else 0,
            "volume_price": vol_price.get("volume_score", 0) if vol_price else 0,
            "chip": chip.get("chip_score", 0) if chip else 0,
        },
    }


# ============================================================
# Agent Brief 生成 — 为 7 个 Agent 各生成定制化简报
# ============================================================

def generate_agent_briefs(stock_data: dict) -> dict:
    """
    为 7 个 Agent 角色生成定制化的分析简报。

    每个 Brief 只包含该 Agent 需要的数据，避免信息过载。
    供 Claude Code Agent 工具使用。

    Returns:
        {
            "technical_analyst": "...prompt...",
            "news_analyst": "...prompt...",
            "fundamentals_analyst": "...prompt...",
            "macro_analyst": "...prompt...",
            "bull_analyst": "...prompt...",    # 需先运行前4个
            "bear_analyst": "...prompt...",    # 需先运行Bull
            "risk_analyst": "...prompt...",    # 需前6个
        }
    """
    ticker = stock_data["ticker"]
    name = stock_data.get("name", ticker)
    date = stock_data["date"]

    tech = stock_data.get("technical", {})
    fund = stock_data.get("fundamentals", {})
    news = stock_data.get("news", {})
    macro = stock_data.get("macro", {})
    candlestick = stock_data.get("candlestick", {})
    vol_price = stock_data.get("volume_price", {})
    chip = stock_data.get("chip_distribution", {})
    composite = stock_data.get("composite", {})

    briefs = {}

    # --- Technical Analyst Brief ---
    briefs["technical_analyst"] = f"""你是一位A股技术分析师，分析 {name}({ticker}) 截至 {date}。

## 技术面数据
```json
{json.dumps(tech, ensure_ascii=False, indent=2)}
```

## K线形态分析
```json
{json.dumps(candlestick, ensure_ascii=False, indent=2)}
```

## 量价关系分析
```json
{json.dumps(vol_price, ensure_ascii=False, indent=2)}
```

## 筹码分布估算
```json
{json.dumps(chip, ensure_ascii=False, indent=2)}
```

请写一份150-200字的技术分析报告，覆盖:
- 趋势: 均线排列、价格vs关键均线
- 动量: RSI/MACD状态
- K线形态: 最近出现的形态信号及其含义
- 量价关系: 量能健康度、主力意图推断
- 筹码: 获利盘比例、支撑/压力位
- 关键支撑/阻力位

结尾必须写: TECHNICAL SIGNAL: BULLISH, BEARISH, or NEUTRAL"""

    # --- News Analyst Brief ---
    briefs["news_analyst"] = f"""你是一位A股新闻情绪分析师，分析 {name}({ticker}) 截至 {date}。

## 个股新闻
情绪: {news.get('sentiment', 'unknown')}
正面新闻数: {news.get('positive_count', 0)}
负面新闻数: {news.get('negative_count', 0)}
```json
{json.dumps(news, ensure_ascii=False, indent=2)}
```

## 当前最热概念板块
```json
{json.dumps(macro.get('top_concepts', []), ensure_ascii=False, indent=2)}
```

请写一份100-150字的新闻情绪报告，覆盖:
- 最影响股价的2-3条新闻
- 整体情绪: positive/negative/mixed
- 所属概念板块是否有催化剂/利空
- 是否有业绩、政策、订单相关的信号

结尾必须写: SENTIMENT SIGNAL: POSITIVE, NEGATIVE, or NEUTRAL"""

    # --- Fundamentals Analyst Brief ---
    briefs["fundamentals_analyst"] = f"""你是一位A股基本面分析师，分析 {name}({ticker})。

## 基本面数据
```json
{json.dumps(fund, ensure_ascii=False, indent=2)}
```

请写一份150-200字的基本面报告，覆盖:
- 估值: PE/PB 在同行业中的位置
- 盈利质量: ROE水平(如有)
- 交易面: 换手率、量比、振幅的含义
- 风险指标: 是否高质押、是否亏损

结尾必须写: FUNDAMENTAL SIGNAL: STRONG, FAIR, or WEAK"""

    # --- Macro Analyst Brief ---
    briefs["macro_analyst"] = f"""你是一位A股宏观环境分析师，截至 {date}。

## 宏观数据
```json
{json.dumps(macro, ensure_ascii=False, indent=2)}
```

请写一份100-150字的宏观环境报告，覆盖:
- 北向资金: 流入/流出趋势及含义
- 市场宽度: 概念板块涨跌比
- 当前最热主线: 哪些概念在领涨
- 整体环境: risk-on, risk-off, or neutral

结尾必须写: MACRO SIGNAL: RISK-ON, RISK-OFF, or NEUTRAL"""

    # --- Bull Analyst Brief ---
    briefs["bull_analyst"] = f"""你是看多分析师，为 {name}({ticker}) 构建最强的看多论证。

核心原则:
- A股中"大盘跌个股涨"恰恰是强庄/强逻辑的最有力证明，不是风险
- 趋势中"获利盘高"=筹码锁定好，不是回吐信号
- 技术面(量价/K线/趋势)的权重大于宏观背景

综合以下数据:
- 技术面 + K线形态 + 量价 + 筹码 (见 Technical Analyst 报告)
- 新闻情绪 (见 News Analyst 报告)
- 基本面估值 (见 Fundamentals Analyst 报告)
- 宏观环境 (见 Macro Analyst 报告)

最新综合信号:
- 多头({composite.get('bullish_count', 0)}): {', '.join(composite.get('reasons_bull', []))}
- 空头({composite.get('bearish_count', 0)}): {', '.join(composite.get('reasons_bear', []))}

请写150-200字看多论证。如果技术面明显偏强，不要因为PE高或宏观弱就降低信心。
结尾: BULL CONVICTION: HIGH, MEDIUM, or LOW"""

    # --- Bear Analyst Brief ---
    briefs["bear_analyst"] = f"""你是看空分析师，为 {name}({ticker}) 构建看空论证。

重要提醒:
- 仅凭"PE高"或"宏观弱"不足以反驳强势技术面
- 必须找到技术面本身的裂缝: 量价背离? 顶部K线形态? 关键阻力位?
- 如果没有实质性的技术面看空证据，不要强行看空

综合数据:
- 技术面 + K线形态 + 量价 + 筹码
- 新闻情绪 / 基本面估值 / 宏观环境
- 筹码风险: {json.dumps(composite.get('risk_flags', []), ensure_ascii=False)}

请写150-200字看空论证。如果技术面没有实质裂缝，应降低CONVICTION。
结尾: BEAR CONVICTION: HIGH, MEDIUM, or LOW"""

    # --- Risk Analyst Brief ---
    briefs["risk_analyst"] = f"""你是风控分析师，对 {name}({ticker}) 做压力测试。

注意: 趋势中的满盘获利是正常现象，不是风险。关注真正的风险: 量价背离、顶部反转形态、关键支撑破位。

筹码数据: {json.dumps(chip, ensure_ascii=False, indent=2)}
量价数据: {json.dumps(vol_price, ensure_ascii=False, indent=2)}

请聚焦:
- 下行风险场景（最大可能亏损）
- 趋势逆转的技术面触发条件（不是宏观猜测）
- 止损位应该设在哪里
- Bull/Bear 都没提到的尾部风险

写100-150字。结尾: RISK LEVEL: HIGH, MEDIUM, or LOW"""

    return briefs


# ============================================================
# 批量分析
# ============================================================

def analyze_stocks(tickers: list[str], date: str = None,
                    enrich: bool = True) -> list[dict]:
    """批量分析多只股票（含新分析模块）"""
    results = []
    for i, ticker in enumerate(tickers):
        if i > 0:
            time.sleep(1.5)  # 节流
        try:
            result = analyze_single_stock(ticker, date, enrich=enrich)
            results.append(result)
        except Exception as e:
            print(f"  [ERROR] {ticker}: {e}")
            results.append({"ticker": ticker, "error": str(e)})
    return results


# ============================================================
# Agent 管线入口 (供 unified_pipeline 调用)
# ============================================================

def prepare_agent_manifest(tickers: list[str],
                            date: str = None,
                            output_dir: Path = None) -> dict:
    """
    为多只股票准备 Agent 分析所需的一切数据。

    这个函数被 unified_pipeline 的 Step 3 调用。
    它不执行 Agent（Agent 由 Claude Code 执行），但生成:
      1. 每只股票的完整分析数据
      2. 每只股票的 7 个 Agent Brief
      3. Agent 执行清单 (manifest)

    Returns:
        manifest dict，供 Claude Code 读取并执行 Agent
    """
    if date is None:
        date = datetime.now().strftime("%Y-%m-%d")

    if output_dir is None:
        output_dir = OUTPUT_DIR / date / "agent_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Agent 分析数据准备 — {len(tickers)} 只标的")
    print(f"{'='*60}")

    # Step 1: 采集所有标的的完整数据
    full_data = analyze_stocks(tickers, date, enrich=True)

    # Step 2: 为每只股票生成 Agent Briefs + 保存
    manifest = {
        "date": date,
        "total_stocks": len(full_data),
        "stocks": [],
    }

    for sd in full_data:
        if "error" in sd:
            continue

        ticker = sd["ticker"]
        name = sd.get("name", ticker)

        # 生成 Briefs
        briefs = generate_agent_briefs(sd)

        # 保存完整数据和 Briefs
        stock_dir = output_dir / ticker
        stock_dir.mkdir(parents=True, exist_ok=True)

        with open(stock_dir / "full_analysis.json", "w") as f:
            json.dump(sd, f, ensure_ascii=False, indent=2, default=str)

        with open(stock_dir / "agent_briefs.json", "w") as f:
            json.dump(briefs, f, ensure_ascii=False, indent=2, default=str)

        # 保存供人阅读的 Markdown
        report = format_full_report(sd)
        with open(stock_dir / "analysis_report.md", "w") as f:
            f.write(report)

        comp = sd.get("composite", {})
        manifest["stocks"].append({
            "ticker": ticker,
            "name": name,
            "signal": comp.get("technical_signal", "NEUTRAL"),
            "net_score": comp.get("net_score", 0),
            "data_dir": str(stock_dir),
            "has_agent_briefs": True,
        })

    # 保存 manifest
    with open(output_dir / "agent_manifest.json", "w") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2, default=str)

    print(f"\n  ✅ 数据已保存到 {output_dir}/")
    print(f"  📋 Agent Manifest: {output_dir}/agent_manifest.json")
    print(f"  📂 每只股票的数据目录包含: full_analysis.json + agent_briefs.json + analysis_report.md")

    return manifest


# ============================================================
# 输出格式化
# ============================================================

def format_full_report(stock_data: dict) -> str:
    """格式化为完整的 Markdown 分析报告"""
    ticker = stock_data["ticker"]
    name = stock_data.get("name", ticker)
    date = stock_data["date"]
    comp = stock_data.get("composite", {})

    lines = []
    lines.append(f"# {name} ({ticker}) 深度分析报告")
    lines.append(f"日期: {date}")
    lines.append("")

    # 综合信号
    lines.append("## 综合信号")
    lines.append(f"- 信号: **{comp.get('technical_signal', '?')}** (净分: {comp.get('net_score', 0)})")
    lines.append(f"- 看多理由 ({comp.get('bullish_count', 0)}): {'; '.join(comp.get('reasons_bull', []))}")
    lines.append(f"- 看空理由 ({comp.get('bearish_count', 0)}): {'; '.join(comp.get('reasons_bear', []))}")
    if comp.get("risk_flags"):
        lines.append(f"- ⚠️ 风险标记: {'; '.join(comp['risk_flags'])}")
    dscores = comp.get("dimension_scores", {})
    if dscores:
        lines.append(f"- 维度分: K线形态={dscores.get('candlestick',0):.0f} | 量价={dscores.get('volume_price',0):.0f} | 筹码={dscores.get('chip',0):.0f}")
    lines.append("")

    # 技术面
    tech = stock_data.get("technical", {})
    if tech and "error" not in tech:
        lines.append("## 技术面")
        price = tech.get("price", {})
        trend = tech.get("trend", {})
        momentum = tech.get("momentum", {})
        sr = tech.get("support_resistance", {})
        lines.append(f"| 指标 | 值 |")
        lines.append(f"|------|-----|")
        lines.append(f"| 最新价 | {price.get('latest', '?')} |")
        lines.append(f"| 涨跌幅 | {price.get('change_1d_pct', '?')}% |")
        lines.append(f"| MA20 | {trend.get('sma20', '?')} (偏离{trend.get('price_vs_sma20', '?')}%) |")
        lines.append(f"| MA60 | {trend.get('sma50', '?')} |")
        lines.append(f"| 均线排列 | {trend.get('ma_alignment', '?')} |")
        lines.append(f"| RSI(14) | {momentum.get('rsi14', '?')} |")
        lines.append(f"| 量比 | {momentum.get('vol_ratio', '?')} |")
        lines.append(f"| 60日高 | {sr.get('high_60d', '?')} |")
        lines.append(f"| 60日低 | {sr.get('low_60d', '?')} |")
        lines.append("")

    # K线形态
    candlestick = stock_data.get("candlestick", {})
    if candlestick:
        lines.append("## K线形态")
        lines.append(f"- 形态分: {candlestick.get('pattern_score', 0):.0f} → {candlestick.get('latest_signal', '?').upper()}")
        lines.append(f"- 活跃形态: {', '.join(candlestick.get('active_patterns', [])) or '无'}")
        lines.append(f"- 摘要: {candlestick.get('summary', '')}")
        lines.append("")

    # 量价关系
    vol_price = stock_data.get("volume_price", {})
    if vol_price:
        lines.append("## 量价关系")
        lines.append(f"- 量价分: {vol_price.get('volume_score', 0):.0f}")
        lines.append(f"- 量能趋势: {vol_price.get('volume_trend', '?')}")
        lines.append(f"- 量价关系: {vol_price.get('price_vol_relation', '?')}")
        lines.append(f"- 主力意图: {vol_price.get('recent_signal', '?')}")
        if vol_price.get("risk_flags"):
            lines.append(f"- ⚠️ 风险: {'; '.join(vol_price['risk_flags'])}")
        lines.append(f"- 摘要: {vol_price.get('analysis_summary', '')}")
        lines.append("")

    # 筹码分布
    chip = stock_data.get("chip_distribution", {})
    if chip:
        lines.append("## 筹码分布 (估算)")
        lines.append(f"- 筹码分: {chip.get('chip_score', 0):.0f}")
        lines.append(f"- 获利盘: {chip.get('profit_ratio', 0)*100:.0f}%")
        lines.append(f"- 套牢盘: {chip.get('loss_ratio', 0)*100:.0f}%")
        ns = chip.get("nearest_support")
        nr = chip.get("nearest_resistance")
        if ns:
            lines.append(f"- 下方支撑: {ns:.1f} (距{chip.get('support_distance_pct', 0)*100:.1f}%)")
        if nr:
            lines.append(f"- 上方压力: {nr:.1f} (距{chip.get('resistance_distance_pct', 0)*100:.1f}%)")
        lines.append(f"- 集中度: {chip.get('concentration', '?')}")
        if chip.get("risk_flags"):
            lines.append(f"- ⚠️ 风险: {'; '.join(chip['risk_flags'])}")
        lines.append("")

    # 基本面
    fund = stock_data.get("fundamentals", {})
    if fund and "error" not in fund:
        lines.append("## 基本面")
        val = fund.get("valuation", {})
        trade = fund.get("trading", {})
        lines.append(f"| 指标 | 值 |")
        lines.append(f"|------|-----|")
        lines.append(f"| PE(TTM) | {val.get('pe_ttm', '?')} |")
        lines.append(f"| PB | {val.get('pb', '?')} |")
        lines.append(f"| 总市值(亿) | {val.get('mcap_yi', '?')} |")
        lines.append(f"| 换手率 | {trade.get('turnover_pct', '?')}% |")
        lines.append(f"| 振幅 | {trade.get('amplitude_pct', '?')}% |")
        lines.append("")

    # 新闻情绪
    news = stock_data.get("news", {})
    if news:
        lines.append("## 新闻情绪")
        lines.append(f"- 情绪: {news.get('sentiment', '?')}")
        lines.append(f"- 正面: {news.get('positive_count', 0)}, 负面: {news.get('negative_count', 0)}")
        for art in news.get("articles", [])[:5]:
            lines.append(f"- {art.get('title', '')} ({art.get('time', '')})")
        lines.append("")

    return "\n".join(lines)


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="多Agent深度分析编排器 v2 (A股)")
    parser.add_argument("--ticker", type=str, help="单只股票代码")
    parser.add_argument("--tickers", type=str, help="逗号分隔的多只股票代码")
    parser.add_argument("--top-n", type=int, default=10, help="分析前N只")
    parser.add_argument("--agent-briefs", action="store_true", help="输出 Agent Briefs")
    parser.add_argument("--prepare-manifest", action="store_true", help="准备 Agent 执行清单")
    parser.add_argument("--output", choices=["json", "markdown"], default="json")
    parser.add_argument("--save", action="store_true", help="保存到 output/ 目录")
    parser.add_argument("--no-enrich", action="store_true", help="跳过深度分析模块")
    args = parser.parse_args()

    ensure_dirs()

    if args.ticker:
        tickers = [args.ticker.strip()]
    elif args.tickers:
        tickers = [c.strip() for c in args.tickers.split(",") if c.strip()]
    else:
        print("请提供 --ticker 或 --tickers")
        sys.exit(1)

    tickers = tickers[:args.top_n]
    date = datetime.now().strftime("%Y-%m-%d")

    if args.prepare_manifest:
        # 准备 Agent 执行清单
        manifest = prepare_agent_manifest(tickers, date)
        print(f"\n  可供 {len(manifest['stocks'])} 只标的启动 Agent 辩论")
    else:
        # 标准分析
        print(f"\n  多Agent分析 {len(tickers)} 只标的 — {date}\n")
        results = analyze_stocks(tickers, date, enrich=not args.no_enrich)

        if args.agent_briefs:
            for r in results:
                if "error" in r:
                    continue
                briefs = generate_agent_briefs(r)
                print(f"\n{'='*60}")
                print(f"  Agent Briefs: {r['ticker']} {r.get('name', '')}")
                print(f"{'='*60}")
                for role, brief in briefs.items():
                    print(f"\n--- {role} ---")
                    print(brief[:200] + "...")

        elif args.output == "json":
            print(json.dumps(results, ensure_ascii=False, indent=2, default=str))
        else:
            for r in results:
                if "error" in r:
                    continue
                print(format_full_report(r))
                print("\n---\n")

        if args.save:
            date_dir = OUTPUT_DIR / date / "agent_analysis"
            date_dir.mkdir(parents=True, exist_ok=True)
            for r in results:
                if "error" in r:
                    continue
                ticker = r["ticker"]
                stock_dir = date_dir / ticker
                stock_dir.mkdir(parents=True, exist_ok=True)
                with open(stock_dir / "full_analysis.json", "w") as f:
                    json.dump(r, f, ensure_ascii=False, indent=2, default=str)
                report = format_full_report(r)
                with open(stock_dir / "analysis_report.md", "w") as f:
                    f.write(report)
            print(f"\n✅ 已保存到 {date_dir}/")


if __name__ == "__main__":
    main()
