#!/usr/bin/env python3
"""
真实 7-Agent 辩论引擎 — 通过 DeepSeek/Anthropic API 执行完整辩论

Phase 1: 并行数据准备 (已有数据，跳过)
Phase 2: Bull → Bear(反驳) → Risk(压力测试) → 综合研判
Phase 3-5: Research Manager → Trader → Portfolio Manager

使用方式:
  from agent_debate import debate
  result = debate(stock_data)  # → {final, verdict, entry, stop, ...}
"""

import json, os, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))


def _get_client():
    """获取 DeepSeek API 客户端"""
    import anthropic
    # 从环境或 settings.json 取 key
    api_key = os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN", "")
    base_url = os.environ.get("ANTHROPIC_BASE_URL", "")
    model = os.environ.get("ANTHROPIC_DEFAULT_SONNET_MODEL", "deepseek-v4-pro")
    if not api_key:
        try:
            settings_path = os.path.expanduser("~/.claude/settings.json")
            if os.path.exists(settings_path):
                with open(settings_path) as f:
                    s = json.load(f)
                env = s.get("env", {})
                api_key = env.get("ANTHROPIC_AUTH_TOKEN", "")
                base_url = env.get("ANTHROPIC_BASE_URL", base_url)
                model = env.get("ANTHROPIC_DEFAULT_SONNET_MODEL", model)
        except Exception:
            pass
    if not api_key:
        return None
    kwargs = {"api_key": api_key}
    if base_url:
        kwargs["base_url"] = base_url
    return anthropic.Anthropic(**kwargs), model


def debate(stock: dict) -> dict:
    """
    对单只股票执行真实 7-Agent 辩论。

    Args:
        stock: deep_analyze() 的返回 dict，含所有指标

    Returns:
        {final: BUY/HOLD/SELL, verdict, entry, stop, target, size, debate_summary}
    """
    client_info = _get_client()
    if not client_info:
        return _fallback_debate(stock)

    client, model = client_info

    name = stock.get("name", "")
    code = stock.get("code", "")
    sector = stock.get("sector", "")
    price = stock.get("close", 0)
    chg = stock.get("change_pct", 0)
    pe = stock.get("pe", 0)
    mcap = stock.get("mcap", 0)
    k_score = stock.get("candlestick_score", 0)
    v_score = stock.get("volume_score", 0)
    c_score = stock.get("chip_score", 0)
    profit_r = stock.get("profit_ratio", 0)
    reasons_bull = stock.get("reasons_bull", [])
    reasons_bear = stock.get("reasons_bear", [])
    net = stock.get("net_score", 0)
    lu = stock.get("limit_up", {}) or {}
    is_lu = lu.get("is_limit_up", False)
    lu_label = lu.get("quality_label", "")
    lu_cont = lu.get("continuation_prob", 0)

    prompt = f"""你是A股投资分析团队，对{name}({code})进行7-Agent完整辩论。日期: 2026-06-18。

## 关键数据
现价: ¥{price:.2f} | 涨跌: {chg:+.1f}% | PE: {pe:.0f} | 市值: {mcap:.0f}亿 | 板块: {sector}
K线形态: {k_score:+.0f} | 量价关系: {v_score:+.0f} | 筹码分布: {c_score:+.0f} | 获利盘: {profit_r:.0%}
涨停: {"是" if is_lu else "否"} {lu_label + ' 延续' + str(lu_cont) + '%' if is_lu else ''}
多头信号: {'; '.join(reasons_bull[:5]) if reasons_bull else '无'}
空头信号: {'; '.join(reasons_bear[:5]) if reasons_bear else '无'}
预判净分: {net:+d}

## 执行严格7-Agent辩论

**Phase 1 信号汇总** (已计算，直接引用):
- TECHNICAL: K线{k_score:+.0f} 量价{v_score:+.0f} 筹码{c_score:+.0f}
- FUNDAMENTAL: PE{pe:.0f} 市值{mcap:.0f}亿
- MACRO: 大盘温度72 TRADE

**Phase 2 串行辩论:**
- Bull Agent: 构建最强看多论证。A股中逆势走强=强庄信号。150字。
- Bear Agent: 直接反驳Bull的具体论点，必须找到技术面裂缝。150字。
- Risk Agent: 压力测试——下行风险场景、关键支撑、尾部风险。100字。

**Phase 3-5 决策:**
- Research Manager: 谁赢了？综合推荐
- Trader: 具体 ENTRY/STOP/TARGET/SIZE
- Portfolio Manager: 最终判决

## 输出格式(必须严格每行一个字段,用英文冒号):
FINAL:BUY
BULL:最强看多论点
BEAR:最强看空论点
RISK:最大风险
ENTRY:17.5
STOP:16.8
TARGET:22.0
SIZE:8
VERDICT:一句话判决"""

    try:
        msg = client.messages.create(
            model=model,
            max_tokens=600,
            messages=[{"role": "user", "content": prompt}],
            extra_body={"thinking": {"type": "disabled"}},
        )
        text = ""
        for block in msg.content:
            if hasattr(block, 'text') and block.text:
                text += block.text
            if hasattr(block, 'thinking') and block.thinking:
                text += block.thinking

        # Parse result (normalize Chinese colons)
        text = text.replace("：", ":").replace("，", ",")
        result = {"final": "HOLD", "verdict": "API解析失败", "entry": price, "stop": price*0.95, "target": price*1.1, "size": 5}

        for line in text.split('\n'):
            line = line.strip().upper()
            if line.startswith("FINAL:"):
                if "BUY" in line: result["final"] = "BUY"
                elif "SELL" in line: result["final"] = "SELL"
            elif line.startswith("VERDICT:"):
                result["verdict"] = line.split(":", 1)[1].strip()[:80]
            elif line.startswith("ENTRY:"):
                try:
                    n = float(line.split(":")[1].strip())
                    result["entry"] = n
                except: pass
            elif line.startswith("STOP:"):
                try:
                    n = float(line.split(":")[1].strip())
                    result["stop"] = n
                except: pass
            elif line.startswith("TARGET:"):
                try:
                    n = float(line.split(":")[1].strip())
                    result["target"] = n
                except: pass
            elif line.startswith("SIZE:"):
                try:
                    result["size"] = int(line.split(":")[1].strip().replace("%",""))
                except: pass
            elif line.startswith("BULL:"):
                result["bull"] = line.split(":", 1)[1].strip()[:60]
            elif line.startswith("BEAR:"):
                result["bear"] = line.split(":", 1)[1].strip()[:60]
            elif line.startswith("RISK:"):
                result["risk"] = line.split(":", 1)[1].strip()[:60]

        return result

    except Exception as e:
        print(f"  Agent辩论API失败: {e}")
        return _fallback_debate(stock)


def _fallback_debate(stock: dict) -> dict:
    """API不可用时的降级规则辩论"""
    pe = stock.get("pe", 0) or 0
    vs = stock.get("volume_score", 0)
    cs = stock.get("chip_score", 0)
    k = stock.get("candlestick_score", 0)
    net = stock.get("net_score", 0)
    price = stock.get("close", 0)
    reasons_bear = stock.get("reasons_bear", [])
    reasons_bull = stock.get("reasons_bull", [])

    fatal = 0
    if vs < -25: fatal += 1
    if pe < 0: fatal += 1
    if cs < -20: fatal += 1
    if k < -30: fatal += 1

    if fatal >= 2:
        final = "SELL"; verdict = f"多重致命风险: {'; '.join(reasons_bear[:2])}"
    elif fatal == 1 and net < 5:
        final = "HOLD"; verdict = f"有风险({reasons_bear[0] if reasons_bear else '?'})等解除"
    elif net >= 5:
        final = "BUY"; verdict = f"多头共振({'+'.join(reasons_bull[:2]) if reasons_bull else '多因子'})"
    elif net >= 2:
        final = "BUY"; verdict = "技术面偏多控仓参与"
    else:
        final = "HOLD"; verdict = "信号不够强"

    return {"final": final, "verdict": verdict, "entry": price, "stop": price*0.95, "target": price*1.1, "size": 5}


def batch_debate(stocks: list[dict], max_n: int = 10) -> list[dict]:
    """批量辩论，每只间隔1秒"""
    results = []
    for i, s in enumerate(stocks[:max_n]):
        print(f"  [{i+1}/{min(len(stocks), max_n)}] Agent辩论: {s.get('code','?')} {s.get('name','?')}...")
        r = debate(s)
        r["code"] = s.get("code", "")
        results.append(r)
        if i < len(stocks) - 1:
            time.sleep(1)
    return results


# ============================================================
# CLI 测试
# ============================================================
if __name__ == "__main__":
    test = {
        "code": "600667", "name": "太极实业", "sector": "先进封装",
        "close": 20.92, "change_pct": 9.99, "pe": 95, "mcap": 438,
        "candlestick_score": 52, "volume_score": -8, "chip_score": 13,
        "profit_ratio": 1.0, "net_score": 10,
        "reasons_bull": ["均线多头", "K线偏多(52)", "筹码锁定(获利100%)"],
        "reasons_bear": [],
        "limit_up": {"is_limit_up": True, "quality_label": "强势涨停", "continuation_prob": 70},
    }
    r = debate(test)
    print(json.dumps(r, ensure_ascii=False, indent=2))
