#!/usr/bin/env python3
"""
新闻AI推理引擎 — LLM驱动的新闻→板块→标的映射

用法:
  python3 scripts/news_ai_reasoning.py                  # 分析今日新闻，输出JSON
  python3 scripts/news_ai_reasoning.py --date 2026-06-18
"""

import argparse, json, os, sys, time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from data_loader import eastmoney_global_news, ensure_dirs


def prepare_news_prompt(news_items: list[dict]) -> str:
    """将新闻列表格式化为LLM分析提示词"""
    headlines = "\n".join(
        f"- [{n['time']}] {n['title']}" for n in news_items[:30]
    )
    return f"""你是A股新闻分析师。分析以下今日快讯，找出对A股有影响的事件。

## 规则
- 只选出有实质影响的新闻（忽略纯政治、娱乐、体育等无关内容）
- 每条影响新闻给出：利好/利空、影响板块（用东财概念板块名）、影响级别（强/中/弱）
- 板块名必须具体：如"半导体设备""水利管网""智能驾驶""证券"而非泛泛的"科技"
- 推理链条要完整：新闻→逻辑→板块→标的类型

## 今日快讯
{headlines}

## 输出格式 (严格JSON)
```json
{{
  "analyzed_at": "{datetime.now().strftime('%Y-%m-%d %H:%M')}",
  "total_news": {len(news_items)},
  "impact_events": [
    {{
      "title": "新闻标题",
      "impact": "利好/利空",
      "level": "强/中/弱",
      "reasoning": "推理链条(一句话)",
      "sectors": ["板块1", "板块2"],
      "stock_hints": ["可关注的标的类型"]
    }}
  ]
}}
```"""


def analyze_news(news_items: list[dict], output_path: str = None) -> dict:
    """
    准备新闻分析提示词并保存，供 Claude Agent 分析。
    分析结果保存为 JSON 后，pipeline 可直接读取。
    """
    prompt = prepare_news_prompt(news_items)

    result = {
        "date": datetime.now().strftime("%Y-%m-%d"),
        "news_count": len(news_items),
        "prompt": prompt,
        "status": "ready_for_agent",
        "instruction": "将此提示词发送给Claude Agent进行分析，将返回的JSON保存到同目录的 news_ai_result.json",
    }

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"提示词已保存: {output_path}")

    return result


def load_ai_result(date: str) -> dict:
    """加载已保存的AI分析结果"""
    path = ROOT / "output" / date / "news_ai_result.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


def get_ai_recommended_stocks(ai_result: dict, kline_map: dict = None) -> list[dict]:
    """
    从AI分析结果中提取推荐的标的。
    根据推理的板块→搜索标的（使用sector_classification + name search）。
    """
    if not ai_result or "impact_events" not in ai_result:
        return []

    events = ai_result.get("impact_events", [])
    recommendations = []
    for ev in events:
        if ev.get("level") not in ("强", "中"):
            continue
        recommendations.append({
            "topic": ev.get("title", "")[:40],
            "impact": ev.get("impact", ""),
            "reasoning": ev.get("reasoning", ""),
            "sectors": ev.get("sectors", []),
            "stock_hints": ev.get("stock_hints", []),
        })
    return recommendations


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="新闻AI推理引擎")
    parser.add_argument("--date", type=str, default=None, help="目标日期")
    parser.add_argument("--output", type=str, default=None, help="输出路径")
    args = parser.parse_args()

    ensure_dirs()
    date = args.date or datetime.now().strftime("%Y-%m-%d")

    # 1. 拉取新闻
    news = eastmoney_global_news(50)
    print(f"拉取 {len(news)} 条快讯")

    # 2. 生成分析提示词
    out_path = args.output or str(ROOT / "output" / date / "news_ai_prompt.json")
    result = analyze_news(news, out_path)

    # 3. 检查是否有已保存的分析结果
    existing = load_ai_result(date)
    if existing:
        recs = get_ai_recommended_stocks(existing)
        print(f"\n已有AI分析结果: {len(recs)} 条影响事件")
        for r in recs:
            print(f"  [{r['impact']}{r.get('level','')}] {r['topic']}")
            print(f"    → {r['reasoning']}")
            print(f"    → 板块: {', '.join(r.get('sectors',[]))}")
    else:
        print(f"\n提示词已保存到 {out_path}")
        print("请将提示词发送给Claude Agent分析，保存结果为 news_ai_result.json")


if __name__ == "__main__":
    main()
