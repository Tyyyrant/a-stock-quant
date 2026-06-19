#!/usr/bin/env python3
"""
A股统一数据抓取 — 替代 trading-agents 的 fetch_market_data.py (yfinance)

覆盖5类数据:
  --type technical    → 腾讯K线/均线/RSI/MACD/布林带
  --type fundamentals → PE/PB/ROE/营收/利润/估值
  --type news         → 东财个股新闻+情绪关键词
  --type macro        → 北向资金/市场宽度/两融
  --type a_macro      → A股宏观 (PMI/社融/北向全貌)

用法:
  python fetch_a_share_data.py --ticker 600519 --type technical --date 2026-06-17
  python fetch_a_share_data.py --ticker 300750 --type fundamentals --date 2026-06-17
  python fetch_a_share_data.py --ticker MACRO --type a_macro --date 2026-06-17
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from data_loader import (
    UA, EM_SESSION, em_get, SESSION,
    tencent_quote, get_north_flow_history,
    ensure_dirs, get_stock_kline,
)


# ============================================================
# Technical Data
# ============================================================

def fetch_technical(ticker: str, as_of: str) -> dict:
    """
    A股技术面数据 — 基于腾讯K线+指标计算。
    替代 yfinance 的 history() + 技术指标计算。
    """
    market = 1 if ticker.startswith("6") else 0
    df = get_stock_kline(ticker, market, refresh=False)

    if df.empty:
        return {"error": f"无K线数据: {ticker}", "ticker": ticker, "type": "technical"}

    # Filter to as_of date
    df = df[df["date"] <= as_of]
    if df.empty:
        return {"error": f"无截至{as_of}的数据", "ticker": ticker, "type": "technical"}

    c = df["close"].values
    h = df["high"].values
    l = df["low"].values
    v = df["volume"].values

    latest = float(c[-1])

    # SMA
    def _sma(series, n):
        if len(series) < n:
            return None
        return round(float(np.mean(series[-n:])), 2)

    # EMA
    def _ema(series, n):
        if len(series) < n:
            return None
        alpha = 2 / (n + 1)
        ema = series[0]
        for val in series[1:]:
            ema = alpha * val + (1 - alpha) * ema
        return round(float(ema), 2)

    # RSI
    def _rsi(series, n=14):
        if len(series) < n + 1:
            return None
        deltas = np.diff(series)
        gains = np.sum(deltas[deltas > 0]) if len(deltas[deltas > 0]) else 0
        losses = -np.sum(deltas[deltas < 0]) if len(deltas[deltas < 0]) else 1
        rs = gains / max(losses, 1e-9)
        return round(float(100 - 100 / (1 + rs)), 2)

    # Bollinger
    def _bollinger(series, n=20, k=2):
        if len(series) < n:
            return None, None, None
        sma = np.mean(series[-n:])
        std = np.std(series[-n:])
        return round(float(sma + k * std), 2), round(float(sma), 2), round(float(sma - k * std), 2)

    # MACD
    ema12 = _ema(c, 12)
    ema26 = _ema(c, 26)
    dif = round(ema12 - ema26, 2) if ema12 and ema26 else None

    # ATR
    trs = []
    for i in range(1, min(len(h), 14)):
        tr = max(h[-i] - l[-i], abs(h[-i] - c[-i-1]), abs(l[-i] - c[-i-1]))
        trs.append(tr)
    atr = round(float(np.mean(trs)), 2) if trs else None

    bb_upper, bb_mid, bb_lower = _bollinger(c)

    return {
        "ticker": ticker,
        "date": as_of,
        "type": "technical",
        "price": {
            "latest": latest,
            "open": float(df["open"].iloc[-1]),
            "high": float(df["high"].iloc[-1]),
            "low": float(df["low"].iloc[-1]),
            "volume": int(df["volume"].iloc[-1]),
            "change_1d_pct": round((c[-1] - c[-2]) / c[-2] * 100, 2) if len(c) >= 2 else None,
        },
        "trend": {
            "sma20": _sma(c, 20), "sma50": _sma(c, 50), "sma200": _sma(c, 200),
            "ema10": _ema(c, 10), "ema12": ema12, "ema26": ema26,
            "price_vs_sma20": round((latest - _sma(c, 20)) / _sma(c, 20) * 100, 2) if _sma(c, 20) else None,
            "ma_alignment": "bullish" if (_sma(c, 5) or 0) > (_sma(c, 20) or 1) > (_sma(c, 50) or 0) else "mixed",
        },
        "momentum": {
            "rsi14": _rsi(c, 14),
            "macd_dif": dif,
            "atr14": atr,
            "bollinger_upper": bb_upper, "bollinger_mid": bb_mid, "bollinger_lower": bb_lower,
            "vol_ratio": round(float(v[-1]) / np.mean(v[-20:]), 2) if len(v) >= 20 else None,
        },
        "support_resistance": {
            "sma50": _sma(c, 50), "sma200": _sma(c, 200),
            "high_60d": round(float(np.max(h[-60:])), 2) if len(h) >= 60 else None,
            "low_60d": round(float(np.min(l[-60:])), 2) if len(l) >= 60 else None,
        },
    }


# ============================================================
# Fundamentals Data
# ============================================================

def fetch_fundamentals(ticker: str, as_of: str) -> dict:
    """A股基本面 — 腾讯财经 + 东财补充"""
    try:
        quotes = tencent_quote([ticker])
        q = quotes.get(ticker, {})
    except Exception:
        q = {}

    if not q:
        return {"error": f"无基本面数据: {ticker}", "ticker": ticker, "type": "fundamentals"}

    return {
        "ticker": ticker,
        "date": as_of,
        "type": "fundamentals",
        "name": q.get("name", ""),
        "price": q.get("price"),
        "valuation": {
            "pe_ttm": q.get("pe_ttm"),
            "pb": q.get("pb"),
            "mcap_yi": q.get("mcap_yi"),
            "float_mcap_yi": q.get("float_mcap_yi"),
            "limit_up": q.get("limit_up"),
            "limit_down": q.get("limit_down"),
        },
        "trading": {
            "change_pct": q.get("change_pct"),
            "turnover_pct": q.get("turnover_pct"),
            "vol_ratio": q.get("vol_ratio"),
            "amplitude_pct": q.get("amplitude_pct"),
        },
    }


# ============================================================
# News & Sentiment
# ============================================================

def fetch_news(ticker: str, as_of: str) -> dict:
    """A股新闻情绪 — 东财个股新闻"""
    url = "https://search-api-web.eastmoney.com/search/jsonp"
    params = {
        "cb": "jQuery",
        "param": json.dumps({
            "uid": "",
            "keyword": ticker,
            "type": ["cmsArticleWebOld"],
            "client": "web",
            "clientType": "web",
            "clientVersion": "curr",
            "param": {
                "cmsArticleWebOld": {
                    "searchScope": "",
                    "sort": "default",
                    "pageIndex": 1,
                    "pageSize": 20,
                    "preTag": "<em>",
                    "postTag": "</em>",
                },
            },
        }),
    }

    sentiment_keywords = {
        "positive": ["增长", "突破", "中标", "签订", "扩产", "量产", "放量", "超预期",
                     "涨停", "利好", "创新高", "回购", "增持", "利润大增", "技术领先"],
        "negative": ["下降", "亏损", "减持", "跌停", "利空", "诉讼", "处罚", "立案",
                     "爆雷", "暴雷", "退市", "预警", "下滑", "踩雷", "债务违约"],
    }

    try:
        r = em_get(url, params=params, timeout=15)
        text = r.text
        # 解析JSONP
        start = text.find("(") + 1
        end = text.rfind(")")
        if start <= 0 or end <= start:
            return {"ticker": ticker, "type": "news", "articles": [], "sentiment": "neutral"}

        data = json.loads(text[start:end])
        articles_raw = (data.get("result") or {}).get("cmsArticleWebOld", {}).get("list", [])

        articles = []
        pos_count, neg_count = 0, 0
        for art in articles_raw[:10]:
            title = art.get("title", "")
            content = art.get("content", "")
            time_str = art.get("time", "")

            # 简单情绪计数
            pos = sum(1 for kw in sentiment_keywords["positive"] if kw in title)
            neg = sum(1 for kw in sentiment_keywords["negative"] if kw in title)
            if pos > neg:
                pos_count += 1
            elif neg > pos:
                neg_count += 1

            articles.append({
                "title": title,
                "time": time_str,
                "source": art.get("source", ""),
            })

        if pos_count > neg_count:
            sentiment = "positive"
        elif neg_count > pos_count:
            sentiment = "negative"
        else:
            sentiment = "neutral"

        return {
            "ticker": ticker,
            "type": "news",
            "articles": articles,
            "sentiment": sentiment,
            "positive_count": pos_count,
            "negative_count": neg_count,
            "total_count": len(articles),
        }

    except Exception as e:
        return {"ticker": ticker, "type": "news", "error": str(e), "articles": [], "sentiment": "neutral"}


# ============================================================
# A-Share Macro
# ============================================================

def fetch_a_share_macro(ticker: str = "MACRO", as_of: str = None) -> dict:
    """A股宏观环境 — 北向资金 + 市场宽度 + 概念板块热度"""
    if as_of is None:
        as_of = datetime.now().strftime("%Y-%m-%d")

    # 1. 北向资金
    nb = get_north_flow_history(30)
    north_info = {}
    if not nb.empty and "net_total" in nb.columns:
        net_5d = nb["net_total"].tail(5).sum()
        net_20d = nb["net_total"].tail(20).sum()
        north_info = {
            "net_5d_yi": round(float(net_5d), 1),
            "net_20d_yi": round(float(net_20d), 1),
            "trend": "inflow" if net_5d > 30 else ("outflow" if net_5d < -30 else "neutral"),
            "recent_days": len(nb),
        }

    # 2. 概念板块热度
    from theme_discovery import fetch_concept_board_ranking as _fetch_boards
    concept_boards = _fetch_boards()
    top_concepts = concept_boards[:5]
    top_concept_info = [
        {"name": b["name"], "change_5d": b.get("change_pct_5d", b.get("change_pct_1d", 0))}
        for b in top_concepts
    ]

    # 3. 市场宽度
    total_up = sum(1 for b in concept_boards[:50] if float(b.get("change_pct_1d", 0) or 0) > 0)
    total_down = sum(1 for b in concept_boards[:50] if float(b.get("change_pct_1d", 0) or 0) < 0)
    breadth = round(total_up / max(total_up + total_down, 1) * 100, 1)

    return {
        "ticker": ticker,
        "date": as_of,
        "type": "a_macro",
        "northbound": north_info,
        "top_concepts": top_concept_info,
        "market_breadth": breadth,
        "environment": "risk_on" if breadth > 60 else ("risk_off" if breadth < 40 else "neutral"),
    }


# ============================================================
# CLI Dispatcher
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="A股统一数据抓取")
    parser.add_argument("--ticker", type=str, required=True, help="股票代码 或 MACRO")
    parser.add_argument("--type", type=str, required=True,
                        choices=["technical", "fundamentals", "news", "macro", "a_macro"])
    parser.add_argument("--date", type=str, default=datetime.now().strftime("%Y-%m-%d"))
    args = parser.parse_args()

    ensure_dirs()

    if args.type == "technical":
        result = fetch_technical(args.ticker, args.date)
    elif args.type == "fundamentals":
        result = fetch_fundamentals(args.ticker, args.date)
    elif args.type == "news":
        result = fetch_news(args.ticker, args.date)
    elif args.type in ("macro", "a_macro"):
        result = fetch_a_share_macro(args.ticker, args.date)
    else:
        result = {"error": f"Unknown type: {args.type}"}

    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
