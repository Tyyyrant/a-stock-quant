#!/usr/bin/env python3
"""
自上而下选股管线 — 五层过滤漏斗

Layer 1: 大盘温度 → regime (TRADE/CAUTION/SKIP) + 0-100 温度
Layer 2: 板块共振 → 跑赢指数的板块排名
Layer 3: 消息面   → 新闻催化加分
Layer 4: 技术形态 → 5种技术形态打分 + 量价辅助
Layer 5: 输出     → 排名表 + CSV

用法:
  python3 scripts/run_pipeline.py                  # 最新交易日
  python3 scripts/run_pipeline.py --date 2026-05-29
  python3 scripts/run_pipeline.py --top-n 15
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
NEWS_ROOT = ROOT.parent / "news"
sys.path.insert(0, str(ROOT / "scripts"))

from data_loader import (
    ensure_dirs, load_universe_klines,
    SECTOR_NAME_MAP, get_eastmoney_sector_map,
)
from market_diagnostic import diagnose_market, select_regime_weights
from risk_controller import apply_risk_controls

# ============================================================
# 配置
# ============================================================

# 板块共振
MIN_RESONANCE = 50.0
MIN_RESONANCE_CAUTION = 60.0
MIN_SECTOR_STOCKS = 3
MAX_SECTORS = 8

# 消息面
MIN_CATALYST = 20.0

PATTERN_NAMES = {
    "bottom_breakout": "底部突破", "key_breakout": "关键位突破",
    "ma_squeeze": "均线粘合", "macd_golden": "MACD金叉",
    "pullback": "缩量回踩",
}


# ============================================================
# Layer 1: 大盘温度 (delegated to market_diagnostic module)
# ============================================================

def load_index_kline(refresh=False):
    from data_loader import fetch_index_kline
    cache_path = ROOT / "data" / "stocks" / "INDEX_1000300.parquet"
    if refresh or not cache_path.exists():
        df = fetch_index_kline("999999", market=1, count=300)
        if df is not None and len(df) >= 60:
            df.to_parquet(cache_path, index=False)
    return pd.read_parquet(cache_path)


# ============================================================
# Layer 2: 板块共振
# ============================================================

def compute_sector_performance(kline_map, stock_sector_map, index_df, target_date,
                               fundamentals_map=None):
    """计算每板块的 1/5/10/20 日收益，与指数比较，短线加权。

    板块收益使用中位数聚合（替代均值，避免小盘股极端涨跌拉偏），
    当 fundamentals_map 提供市值数据时自动切换为市值加权均值。
    """
    # 指数同期收益
    idx = index_df.copy()
    idx = idx[idx["date"] <= target_date]
    if len(idx) < 21:
        return pd.DataFrame()
    idx_close = idx["close"].values

    def idx_ret(n):
        if len(idx_close) > n:
            return idx_close[-1] / idx_close[-n-1] - 1
        return 0.0

    idx_ret_1 = idx_ret(1)
    idx_ret_5 = idx_ret(5)
    idx_ret_10 = idx_ret(10)
    idx_ret_20 = idx_ret(20)

    # 每只股票算收益率 + 市值（如有）
    stock_rets = []
    has_mcap = fundamentals_map is not None
    for code, item in kline_map.items():
        kline = item["kline"] if isinstance(item, dict) else item
        k = kline[kline["date"] <= target_date]
        if len(k) < 21:
            continue
        close_arr = k["close"].values
        row = {
            "code": code,
            "sector": stock_sector_map.get(code, "未分类"),
            "ret_1d": close_arr[-1] / close_arr[-2] - 1 if len(close_arr) >= 2 else 0,
            "ret_5d": close_arr[-1] / close_arr[-6] - 1 if len(close_arr) >= 6 else 0,
            "ret_10d": close_arr[-1] / close_arr[-11] - 1 if len(close_arr) >= 11 else 0,
            "ret_20d": close_arr[-1] / close_arr[-21] - 1 if len(close_arr) >= 21 else 0,
        }
        if has_mcap:
            fm = fundamentals_map.get(code, {})
            mcap = fm.get("market_cap", 0) or fm.get("mcap", 0)
            row["mcap"] = mcap if mcap and mcap > 0 else None
        else:
            row["mcap"] = None
        stock_rets.append(row)

    sdf = pd.DataFrame(stock_rets)
    if sdf.empty:
        return pd.DataFrame()

    # 按板块聚合 — 使用中位数（更稳健，不受极端值影响）
    agg_funcs = {
        "n_stocks": ("code", "count"),
        "ret_1d": ("ret_1d", "median"),
        "ret_5d": ("ret_5d", "median"),
        "ret_10d": ("ret_10d", "median"),
        "ret_20d": ("ret_20d", "median"),
    }
    agg = sdf.groupby("sector").agg(**{k: v for k, v in agg_funcs.items()}).reset_index()

    # 如果有市值数据，对有效市值的股票计算加权均值作参考
    # （中位数是主力，市值加权作辅助校验）
    if has_mcap and sdf["mcap"].notna().sum() > 50:
        mcap_valid = sdf[sdf["mcap"].notna() & (sdf["mcap"] > 0)].copy()
        if not mcap_valid.empty:
            for col in ["ret_1d", "ret_5d", "ret_10d", "ret_20d"]:
                def weighted_avg(grp):
                    w = grp["mcap"] / grp["mcap"].sum()
                    return (grp[col] * w).sum()
                mcap_agg = mcap_valid.groupby("sector").apply(weighted_avg).reset_index(name=f"{col}_wavg")
                # blend: 70% median + 30% market-cap weighted (when available)
                agg = agg.merge(mcap_agg, on="sector", how="left")
                if f"{col}_wavg" in agg.columns:
                    mask = agg[f"{col}_wavg"].notna()
                    agg.loc[mask, col] = (0.7 * agg.loc[mask, col] +
                                          0.3 * agg.loc[mask, f"{col}_wavg"])
                    agg = agg.drop(columns=[f"{col}_wavg"])

    agg["idx_ret_1d"] = idx_ret_1
    agg["idx_ret_5d"] = idx_ret_5
    agg["idx_ret_10d"] = idx_ret_10
    agg["idx_ret_20d"] = idx_ret_20
    agg["excess_1d"] = agg["ret_1d"] - idx_ret_1
    agg["excess_5d"] = agg["ret_5d"] - idx_ret_5
    agg["excess_10d"] = agg["ret_10d"] - idx_ret_10
    agg["excess_20d"] = agg["ret_20d"] - idx_ret_20

    # 共振分: excess 归一化到 0-100（短线偏向当日+近期）
    for col in ["excess_1d", "excess_5d", "excess_10d", "excess_20d"]:
        vmin, vmax = agg[col].min(), agg[col].max()
        if vmax > vmin:
            agg[f"{col}_norm"] = (agg[col] - vmin) / (vmax - vmin) * 100
        else:
            agg[f"{col}_norm"] = 50.0

    agg["resonance_score"] = (
        0.30 * agg["excess_1d_norm"] +
        0.25 * agg["excess_5d_norm"] +
        0.25 * agg["excess_10d_norm"] +
        0.20 * agg["excess_20d_norm"]
    )

    return agg.sort_values("resonance_score", ascending=False)


# ============================================================
# 腾讯板块行情缓存（用于替代个股聚合计算板块收益）
# ============================================================

# 板块名 → pt 代码映射（申万行业 + 概念板块）
SECTOR_BOARD_MAP = {
    # 行业板块（申万，pt018xxxxx）
    "半导体": "pt01801081",
    "元件": "pt01801083",
    "消费电子": "pt01801085",
    "通信设备": "pt01801102",
    "软件开发": "pt01801104",
    "电池": "pt01801737",
    "自动化设备": "pt01801078",
    "通用设备": "pt01801072",
    "汽车零部件": "pt01801093",
    # 概念板块（腾讯，pt02xxxxxx）
    "机器人": "pt02003640",
    "人工智能": "pt02003800",
    "无线充电": "pt02003960",
    "芯片概念": "pt02003891",
    # PCB → 元件（PCB 无独立概念板，元件是最近的申万行业）
    "PCB": "pt01801083",
    "铜缆高速连接": "pt01801102",
    "MLCC": "pt01801083",
    "先进封装": "pt01801081",
    "玻璃基板": "pt01801083",
    "AI PC": "pt01801101",
}


def fetch_tencent_board_returns(board_pt_codes: list[str]) -> dict:
    """从腾讯行情 API 批量拉取板块涨跌幅（快，不限流）。

    返回: {pt_code: {name, chg_pct, price, last_close}}
    注意：只提供当日涨跌幅（快照），不提供历史多日收益。
    """
    if not board_pt_codes:
        return {}

    url = "https://qt.gtimg.cn/q=" + ",".join(board_pt_codes)
    try:
        import urllib.request
        req = urllib.request.Request(url)
        req.add_header("User-Agent", "Mozilla/5.0")
        resp = urllib.request.urlopen(req, timeout=10)
        data = resp.read().decode("gbk")
    except Exception:
        return {}

    results = {}
    for line in data.strip().split(";"):
        if "~" not in line or "none_match" in line:
            continue
        try:
            raw_code = line.strip().split("=")[0].replace("v_", "")  # "pt01801083"
            vals = line.split('"')[1].split("~")
            name = vals[1]
            price = float(vals[3])
            last_close = float(vals[4])
            chg = (price / last_close - 1) * 100 if last_close > 0 else 0.0
            results[raw_code] = {
                "name": name,
                "chg_pct": chg,
                "price": price,
                "last_close": last_close,
            }
        except (ValueError, IndexError):
            pass
    return results


def _enrich_sector_perf_with_board_data(sector_perf, target_date):
    """用腾讯板块真实行情 + selected_sectors.json 修正板块收益。

    优先使用腾讯板块指数（实时准确），其次使用 news 项目数据。
    """
    if sector_perf.empty:
        return

    # --- Step 1: 腾讯板块行情（工作日快照）---
    sector_to_pt = {}
    for idx, row in sector_perf.iterrows():
        sec = row["sector"]
        if sec in SECTOR_BOARD_MAP:
            sector_to_pt[sec] = SECTOR_BOARD_MAP[sec]

    if sector_to_pt:
        unique_pts = list(set(sector_to_pt.values()))
        board_data = fetch_tencent_board_returns(unique_pts)

        overridden = 0
        for idx, row in sector_perf.iterrows():
            sec = row["sector"]
            pt = sector_to_pt.get(sec)
            if pt and pt in board_data:
                bd = board_data[pt]
                new_ret = bd["chg_pct"] / 100.0
                sector_perf.at[idx, "ret_1d"] = new_ret
                sector_perf.at[idx, "excess_1d"] = new_ret - row["idx_ret_1d"]
                overridden += 1
                # Also show the board name if different from sector name
                if bd["name"] != sec:
                    sector_perf.at[idx, "sector_display"] = f"{sec}({bd['name']})"

        if overridden > 0:
            _recalc_resonance(sector_perf, overridden)

    # --- Step 2: selected_sectors.json (news 项目数据，互补) ---
    news_dir = NEWS_ROOT / "data" / "processed"
    selected_path = news_dir / target_date / "selected_sectors.json"
    if not selected_path.exists():
        return

    try:
        with open(selected_path) as f:
            sel_data = json.load(f)
    except Exception:
        return

    selected = sel_data.get("selected_sectors", [])
    if not selected:
        return

    board_chg = {}
    for sec in selected:
        name = sec.get("name", "")
        chg = sec.get("change_pct")
        if name and chg is not None:
            try:
                board_chg[name] = float(chg) / 100.0
            except (ValueError, TypeError):
                pass

    overridden2 = 0
    for idx, row in sector_perf.iterrows():
        sec = row["sector"]
        if sec in board_chg:
            sector_perf.at[idx, "ret_1d"] = board_chg[sec]
            sector_perf.at[idx, "excess_1d"] = board_chg[sec] - row["idx_ret_1d"]
            overridden2 += 1

    if overridden2 > 0:
        _recalc_resonance(sector_perf, overridden2)


def _recalc_resonance(sector_perf, overridden):
    """重新计算共振分数（在板块收益被修正后调用）"""
    col = "excess_1d"
    vmin, vmax = sector_perf[col].min(), sector_perf[col].max()
    if vmax > vmin:
        sector_perf[f"{col}_norm"] = (sector_perf[col] - vmin) / (vmax - vmin) * 100
    else:
        sector_perf[f"{col}_norm"] = 50.0

    sector_perf["resonance_score"] = (
        0.30 * sector_perf["excess_1d_norm"] +
        0.25 * sector_perf["excess_5d_norm"] +
        0.25 * sector_perf["excess_10d_norm"] +
        0.20 * sector_perf["excess_20d_norm"]
    )
    sector_perf.sort_values("resonance_score", ascending=False, inplace=True)
    sector_perf.reset_index(drop=True, inplace=True)
    print(f"  📊 用真实板块指数修正了 {overridden} 个板块的当日收益")


def select_resonant_sectors(sector_perf, temperature, signal):
    """过滤：共振分>=阈值、股票数>=3、排除宽泛代码前缀分类"""
    if sector_perf.empty:
        return sector_perf

    min_res = MIN_RESONANCE_CAUTION if signal == "CAUTION" else MIN_RESONANCE

    # 排除代码前缀大类（无投资意义的泛类别）
    GENERIC_SECTORS = {"沪市主板", "深市主板", "创业板", "科创板", "其他", "未分类"}

    filtered = sector_perf[
        (sector_perf["resonance_score"] >= min_res) &
        (sector_perf["n_stocks"] >= MIN_SECTOR_STOCKS) &
        (~sector_perf["sector"].isin(GENERIC_SECTORS))
    ].copy()

    if filtered.empty:
        # 放宽条件：保留 top3（仍排除泛类别）
        filtered = sector_perf[
            (sector_perf["n_stocks"] >= 2) &
            (~sector_perf["sector"].isin(GENERIC_SECTORS))
        ].head(3).copy()

    return filtered.head(MAX_SECTORS)


# ============================================================
# Layer 3: 消息面催化
# ============================================================

def load_news_catalyst_data(date):
    """从 news 项目加载板块级情绪数据 + 动态赛道选择"""
    news_date = date
    news_dir = NEWS_ROOT / "data" / "processed"
    candidates = sorted([d.name for d in news_dir.iterdir()
                         if d.is_dir() and d.name <= date],
                        reverse=True)
    if candidates:
        news_date = candidates[0]

    merged = {}
    base = news_dir / news_date

    # selected_sectors.json: news 项目动态选出的当日最强赛道
    selected_path = base / "selected_sectors.json"
    today_hot_sectors = {}  # {sector_name: change_pct}
    if selected_path.exists():
        with open(selected_path) as f:
            sel_data = json.load(f)
        for sec in sel_data.get("selected_sectors", []):
            name = sec.get("name", "")
            today_hot_sectors[name] = {
                "change_pct": sec.get("change_pct", 0),
                "heat_score": sec.get("heat_score", 0),
                "type": sec.get("type", ""),
                "news_count": sec.get("news_count", 0),
            }

    # news_impact.json: sector → {sentiment_score, sentiment, bullish_count, bearish_count}
    impact_path = base / "news_impact.json"
    if impact_path.exists():
        with open(impact_path) as f:
            impact = json.load(f)
        for sec in impact.get("sector_impacts", []):
            name = sec["sector"]
            entry = {
                "sentiment_score": sec.get("score", 0),
                "sentiment": sec.get("sentiment", "中性"),
                "bullish_count": sec.get("bullish_count", 0),
                "bearish_count": sec.get("bearish_count", 0),
                "total_mentions": sec.get("total_mentions", 0),
            }
            # 注入动态赛道标记
            if name in today_hot_sectors:
                entry["is_hot_sector"] = True
                entry["hot_change_pct"] = today_hot_sectors[name]["change_pct"]
                entry["hot_heat"] = today_hot_sectors[name]["heat_score"]
                entry["hot_type"] = today_hot_sectors[name]["type"]
            merged[name] = entry

    # market_data.json: 行业板块涨跌
    md_path = base / "market_data.json"
    if md_path.exists():
        with open(md_path) as f:
            md = json.load(f)
        for sec in md.get("top_industry_sectors", []) + md.get("bottom_industry_sectors", []):
            name = sec.get("name", "")
            if name not in merged:
                merged[name] = {"sentiment_score": 5, "sentiment": "中性",
                                "bullish_count": 0, "bearish_count": 0,
                                "total_mentions": 0}
            # 也注入动态赛道标记
            if name in today_hot_sectors and "is_hot_sector" not in merged[name]:
                merged[name]["is_hot_sector"] = True
                merged[name]["hot_change_pct"] = today_hot_sectors[name]["change_pct"]
                merged[name]["hot_heat"] = today_hot_sectors[name]["heat_score"]

    # 对于只在 selected_sectors 中出现但不在 news_impact 中的板块，也加入
    for name, info in today_hot_sectors.items():
        if name not in merged:
            merged[name] = {
                "sentiment_score": 5, "sentiment": "中性",
                "bullish_count": 0, "bearish_count": 0,
                "total_mentions": info["news_count"],
                "is_hot_sector": True,
                "hot_change_pct": info["change_pct"],
                "hot_heat": info["heat_score"],
                "hot_type": info["type"],
            }

    return merged


def match_sector_to_news(sector_name, news_data):
    """将板块名匹配到 news 数据中的板块名"""
    if sector_name in news_data:
        return sector_name

    # 使用 SECTOR_NAME_MAP
    candidates = SECTOR_NAME_MAP.get(sector_name, [sector_name])
    for cand in candidates:
        if cand in news_data:
            return cand

    # 模糊匹配
    for nk in news_data:
        for cand in candidates:
            if cand in nk or nk in cand:
                return nk

    return None


def score_sector_catalysts(sectors_df, news_data):
    """为通过共振的板块打分消息面催化"""
    if sectors_df.empty or not news_data:
        sectors_df = sectors_df.copy()
        sectors_df["catalyst_score"] = 50.0
        sectors_df["sentiment_label"] = "无数据"
        return sectors_df

    scores = []
    for _, row in sectors_df.iterrows():
        sector = row["sector"]
        matched = match_sector_to_news(sector, news_data)
        nd = news_data.get(matched) if matched else None

        if nd and nd.get("total_mentions", 0) > 0:
            sent_raw = nd.get("sentiment_score", 5)
            sent_score = min(sent_raw / 10 * 100, 100)
            is_hot = nd.get("is_hot_sector", False)
            hot_pct = nd.get("hot_change_pct", 0)
            scores.append({
                "sector": sector,
                "matched_sector": matched,
                "sentiment_raw": sent_raw,
                "sentiment_label": nd.get("sentiment", "中性"),
                "bullish": nd.get("bullish_count", 0),
                "bearish": nd.get("bearish_count", 0),
                "total_mentions": nd.get("total_mentions", 0),
                "_sent_score": sent_score,
                "_is_hot": is_hot,
                "_hot_pct": hot_pct,
            })
        elif nd and nd.get("is_hot_sector"):
            # 被 news 项目选为动态赛道，但无新闻覆盖也加分
            scores.append({
                "sector": sector,
                "matched_sector": matched,
                "sentiment_raw": 5,
                "sentiment_label": "动态赛道",
                "bullish": 0,
                "bearish": 0,
                "total_mentions": nd.get("total_mentions", 0),
                "_sent_score": 60,
                "_is_hot": True,
                "_hot_pct": nd.get("hot_change_pct", 0),
            })
        else:
            scores.append({
                "sector": sector,
                "matched_sector": None,
                "sentiment_raw": 5,
                "sentiment_label": "无覆盖",
                "bullish": 0,
                "bearish": 0,
                "total_mentions": 0,
                "_sent_score": 50,
                "_is_hot": False,
                "_hot_pct": 0,
            })

    sdf = pd.DataFrame(scores)

    # 热度归一化
    max_mentions = sdf["total_mentions"].max()
    if max_mentions > 0:
        sdf["_heat_score"] = sdf["total_mentions"] / max_mentions * 100
    else:
        sdf["_heat_score"] = 50.0

    # 方向分
    def direction_score(label):
        if label in ("利好", "偏利好"):
            return 100
        elif label in ("中性", "无覆盖"):
            return 50
        else:
            return 10

    sdf["_dir_score"] = sdf["sentiment_label"].apply(direction_score)

    # 动态赛道加分: news 项目选出的当日最强赛道，按涨幅给 0-30 分加成
    max_hot_pct = sdf["_hot_pct"].max()
    if max_hot_pct > 0:
        sdf["_hot_bonus"] = (sdf["_hot_pct"].clip(lower=0) / max_hot_pct) * 30
    else:
        sdf["_hot_bonus"] = 0.0

    # 加权 (含动态赛道加成)
    sdf["catalyst_score"] = (
        0.40 * sdf["_sent_score"] +
        0.25 * sdf["_heat_score"] +
        0.15 * sdf["_dir_score"] +
        0.20 * sdf["_hot_bonus"]
    )

    result = sectors_df.merge(
        sdf[["sector", "catalyst_score", "sentiment_label",
             "bullish", "bearish", "total_mentions", "_is_hot"]],
        on="sector", how="left"
    )
    result["catalyst_score"] = result["catalyst_score"].fillna(50)
    result["_is_hot"] = result["_is_hot"].fillna(False)

    return result


def filter_catalyst_sectors(sectors_with_catalyst):
    """过滤消息面不达标的板块（软过滤）"""
    if sectors_with_catalyst.empty:
        return sectors_with_catalyst

    filtered = sectors_with_catalyst[
        sectors_with_catalyst["catalyst_score"] >= MIN_CATALYST
    ].copy()

    if filtered.empty:
        # 软降级：保留 top3
        return sectors_with_catalyst.head(3).copy()

    return filtered


# ============================================================
# Layer 4: 技术形态 (delegated to factor_library + risk_controller)
# ============================================================


# ============================================================
# Layer 5: 输出
# ============================================================

def build_reasoning(row):
    """从全因子分中提取推荐理由"""
    reasons = []

    # 技术形态
    pattern_names = {
        "bottom_breakout": "底部突破", "key_breakout": "关键位突破",
        "ma_squeeze": "均线粘合", "macd_golden": "MACD金叉",
        "pullback": "缩量回踩",
    }
    for col, name in pattern_names.items():
        if col in row and row[col] > 30:
            reasons.append(f"{name}({row[col]:.0f})")

    # 新增因子信号
    if row.get("candlestick_pattern", 0) > 30:
        reasons.append(f"K线形态偏多({row['candlestick_pattern']:.0f})")
    elif row.get("candlestick_pattern", 0) < -30:
        reasons.append(f"⚠K线偏空({row['candlestick_pattern']:.0f})")

    if row.get("overheat_penalty", 0) > 30:
        reasons.append(f"⚠短期过热({row['overheat_penalty']:.0f})")

    if row.get("chip_safety", 0) < -10:
        reasons.append(f"⚠筹码压力({row['chip_safety']:.0f})")
    elif row.get("chip_safety", 0) > 15:
        reasons.append(f"筹码安全({row['chip_safety']:.0f})")

    if not reasons:
        return "多因子综合得分领先"
    return " | ".join(reasons[:4])


def render_output(ranked_df, temp_result, sector_info, output_dir, top_n):
    """终端输出 + CSV（仅推荐创业板+主板，最多5只）"""
    os.makedirs(output_dir, exist_ok=True)

    signal = temp_result["signal"]
    temp = temp_result["temperature"]
    regime = temp_result.get("recommended_weights", "normal")
    vol_regime = temp_result.get("vol_regime", "?")

    # Filter: only 创业板 + 主板
    valid_mask = ranked_df["code"].apply(
        lambda c: not (c.startswith("688") or c.startswith("8") or c.startswith("4"))
    )
    ranked_df = ranked_df[valid_mask].copy()

    # Limit to 5
    top_n = min(top_n, 5)

    # 终端输出
    print()
    print("=" * 78)
    print(f"  量化选股推荐 — 仅创业板+主板")
    print(f"  日期: {datetime.now().strftime('%Y-%m-%d')}  |  "
          f"大盘温度: {temp:.0f}/100 ({signal})  |  模式: {regime}  |  波动率: {vol_regime}")
    print("=" * 78)

    if signal == "SKIP":
        print("\n  ⚠ 大盘温度过低，建议观望不操作")
        print("  以下排名仅供参考，等待市场回暖后再参与\n")

    if ranked_df.empty:
        print("  无符合条件的股票（均已通过科创板/北交所过滤）")
        return

    # 板块来源
    print(f"\n  📊 共振板块 ({len(sector_info)}个): ", end="")
    for _, srow in sector_info.iterrows():
        hot_mark = "🔥" if srow.get("_is_hot") else ""
        print(f"[{hot_mark}{srow['sector']}] ", end="")
    print()

    # 排名表
    display = ranked_df.head(top_n).copy()
    print(f"\n  {'':>3s} {'代码':<8s} {'名称':<8s} {'板块':<10s} "
          f"{'Alpha':>6s} {'仓位':>5s} {'收盘':>7s} {'涨跌':>6s} "
          f"{'形态':>5s} {'量价':>5s} {'筹码':>5s}")
    print(f"  {'─' * 68}")

    for i, (_, row) in enumerate(display.iterrows(), 1):
        chg = row.get("change_pct", 0) or 0
        pos_pct = row.get("suggested_position_pct", 0) or 0
        sector = row.get("sector", "") or "未分类"
        cand = row.get("candlestick_pattern", 0) or 0
        vol = row.get("volume_price_quality", 0) or 0
        chip = row.get("chip_safety", 0) or 0

        def _icon(v):
            return "🟢" if v > 15 else ("🔴" if v < -15 else "⚪")

        print(f"  {i:2d}. {row['code']:<8s} {str(row.get('name','')):<8s} "
              f"{sector:<10s} "
              f"{row.get('alpha_score', 0):>6.1f} "
              f"{pos_pct*100:>4.1f}% "
              f"{row.get('close', 0):>7.2f} "
              f"{chg*100:>+5.2f}% "
              f"{_icon(cand)}{cand:>4.0f} "
              f"{_icon(vol)}{vol:>4.0f} "
              f"{_icon(chip)}{chip:>4.0f}")

        reason = build_reasoning(row)
        # Detail: add sector context
        sector_detail = ""
        for _, sr in sector_info.iterrows():
            if sr["sector"] == sector:
                reso = sr.get("resonance_score", 0)
                cat = sr.get("catalyst_score", 0)
                sent = sr.get("sentiment_label", "")
                sector_detail = f" 板块共振{reso:.0f}分 催化{cat:.0f}分({sent})"
                break
        print(f"     → {reason}{sector_detail}")

    print(f"\n  {'─' * 68}")
    print(f"  ⚠ 科创板(688)和北交所已自动过滤 | 仅推荐创业板+主板")

    # CSV
    csv_path = output_dir / "pipeline_ranking.csv"
    ranked_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"  CSV: {csv_path} ({len(ranked_df)} 只候选)\n")


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="自上而下选股管线")
    parser.add_argument("--date", type=str, default=None,
                        help="目标交易日 (YYYY-MM-DD)，默认最新")
    parser.add_argument("--top-n", type=int, default=20,
                        help="输出排名数 (默认20)")
    parser.add_argument("--agent", action="store_true",
                        help="启用 Layer 6 Agent 深度分析 (生成 Agent Briefs)")
    args = parser.parse_args()

    ensure_dirs()

    # 确定目标日期
    index_df = load_index_kline(refresh=True)
    available_dates = sorted(index_df["date"].tolist())
    if args.date:
        target_date = args.date
    else:
        target_date = available_dates[-1]
    print(f"目标日期: {target_date}")

    # 裁剪指数到目标日期
    index_df = index_df[index_df["date"] <= target_date].copy()

    # ========== Layer 1: 大盘温度 ==========
    print("\n[Layer 1] 大盘温度...")
    kline_map = load_universe_klines(watchlist_only=False, refresh=True)

    # Try to load CSI 2000 for size style diagnosis
    index_2000_df = None
    zz2000_path = ROOT / "data" / "stocks" / "INDEX_1000852.parquet"
    if zz2000_path.exists():
        index_2000_df = pd.read_parquet(zz2000_path)

    diagnosis = diagnose_market(index_df, kline_map, index_2000_df)
    temp_result = diagnosis  # keep variable name for downstream use
    print(f"  温度: {diagnosis['temperature']:.0f}/100  "
          f"状态: {diagnosis['regime']}  信号: {diagnosis['signal']}  "
          f"波动率区间: {diagnosis['vol_regime']}  "
          f"推荐权重: {diagnosis['recommended_weights']}")

    # ========== Layer 2: 板块共振 ==========
    print("\n[Layer 2] 板块共振...")
    all_codes = list(kline_map.keys())

    # Fast sector classification: use cached JSON first, eastmoney only as fallback
    # (avoids throttling delays when processing hundreds of stocks)
    cache_path = ROOT / "data" / "sector_classification.json"
    stock_sector_map = {}
    if cache_path.exists():
        with open(cache_path) as f:
            cached = json.load(f)
        covered = sum(1 for c in all_codes if c in cached)
        if covered > len(all_codes) * 0.3:
            # Use cache + code-prefix fallback for speed
            for code in all_codes:
                if code in cached:
                    stock_sector_map[code] = cached[code]
                elif code.startswith("688"):
                    stock_sector_map[code] = "科创板"
                elif code.startswith("300") or code.startswith("301"):
                    stock_sector_map[code] = "创业板"
                elif code.startswith("002") or code.startswith("000"):
                    stock_sector_map[code] = "深市主板"
                elif code.startswith("6"):
                    stock_sector_map[code] = "沪市主板"
                else:
                    stock_sector_map[code] = "其他"
            print(f"  板块分类: {len(stock_sector_map)} 只 (缓存+补全)")

    # Fallback: if cache insufficient, use code-prefix classification
    if not stock_sector_map:
        for code in all_codes:
            if code.startswith("688"):
                stock_sector_map[code] = "科创板"
            elif code.startswith("300") or code.startswith("301"):
                stock_sector_map[code] = "创业板"
            elif code.startswith("002") or code.startswith("000"):
                stock_sector_map[code] = "深市主板"
            elif code.startswith("6"):
                stock_sector_map[code] = "沪市主板"
            else:
                stock_sector_map[code] = "其他"
        print(f"  板块分类: {len(stock_sector_map)} 只 (代码前缀)")
    sector_perf = compute_sector_performance(
        kline_map, stock_sector_map, index_df, target_date,
        fundamentals_map=None,  # 市值加权可选，需先加载基本面
    )

    # 用 selected_sectors.json 的真实板块行情校验/修正 1d 收益
    # （news 项目直接从板块指数拉取，比个股聚合更准确）
    _enrich_sector_perf_with_board_data(sector_perf, target_date)

    if sector_perf.empty:
        print("  无法计算板块收益")
        return

    resonant = select_resonant_sectors(
        sector_perf, temp_result["temperature"], temp_result["signal"]
    )
    print(f"  通过板块: {len(resonant)}/{len(sector_perf)} 个")
    if not resonant.empty:
        for _, r in resonant.iterrows():
            print(f"    {r['sector']:<12s}  共振{r['resonance_score']:.0f}分  "
                  f"1d{r['excess_1d']*100:+.1f}%  5d{r['excess_5d']*100:+.1f}%  "
                  f"10d{r['excess_10d']*100:+.1f}%  20d{r['excess_20d']*100:+.1f}%  ({r['n_stocks']}只)")

    # ========== Layer 3: 消息面 ==========
    print("\n[Layer 3] 消息面催化...")
    news_data = load_news_catalyst_data(target_date)
    print(f"  加载 {len(news_data)} 个板块的新闻数据")
    sectors_with_catalyst = score_sector_catalysts(resonant, news_data)
    filtered_sectors = filter_catalyst_sectors(sectors_with_catalyst)
    print(f"  催化通过: {len(filtered_sectors)} 个板块")
    if not filtered_sectors.empty:
        for _, r in filtered_sectors.iterrows():
            sl = r.get("sentiment_label", "?")
            cs = r.get("catalyst_score", 50)
            hot = "🔥" if r.get("_is_hot") else "  "
            print(f"    {hot}{r['sector']:<12s}  催化{cs:.0f}分  "
                  f"情绪={sl}  新闻{r.get('total_mentions',0)}条")

    # ========== Layer 4: 技术形态 ==========
    print("\n[Layer 4] 技术形态...")
    # 收集候选股：过滤板块过少时自动扩大到全市场扫描
    candidate_codes = []
    for _, sec_row in filtered_sectors.iterrows():
        sec = sec_row["sector"]
        for code, s in stock_sector_map.items():
            if s == sec and code in kline_map:
                candidate_codes.append(code)

    # 候选太少时，降级：跳过板块过滤，扫描全部有数据的股票
    if len(candidate_codes) < 100:
        print(f"  ⚠ 板块候选仅{len(candidate_codes)}只，降级为全市场扫描...")
        candidate_codes = []
        for code in kline_map:
            # 排除科创板+北交所
            if not (code.startswith("688") or code.startswith("8") or code.startswith("4")):
                # 优先使用缓存中的真实板块，没有才用代码前缀兜底
                if code not in stock_sector_map:
                    if code.startswith("300") or code.startswith("301"):
                        stock_sector_map[code] = "创业板"
                    elif code.startswith("002") or code.startswith("000"):
                        stock_sector_map[code] = "深市主板"
                    elif code.startswith("6"):
                        stock_sector_map[code] = "沪市主板"
                candidate_codes.append(code)
        print(f"  全市场候选: {len(candidate_codes)} 只")

    if len(candidate_codes) == 0:
        print("  无候选股，当前市场无可操作标的")
        return

    # 加载基本面（只用于候选股）
    from data_loader import load_fundamentals_for_codes
    fundamentals_map = load_fundamentals_for_codes(candidate_codes)

    # Build sliced kline map for candidate codes
    sliced_klines = {}
    for code in candidate_codes:
        if code not in kline_map:
            continue
        item = kline_map[code]
        kline = item["kline"] if isinstance(item, dict) else item
        mask = kline["date"] <= target_date
        df = kline[mask].copy()
        if len(df) >= 60:
            # Inject sector from stock_sector_map into info
            info = dict(item.get("info", {}))
            info["sector"] = stock_sector_map.get(code, info.get("sector", "未分类"))
            sliced_klines[code] = {
                "info": info,
                "kline": df,
            }

    if not sliced_klines:
        print("  无有效K线数据的候选股")
        return

    # Run full factor pipeline
    from factor_library import compute_factors_batch, process_factor_panel, compute_composite_score

    # Determine dynamic regime
    regime = diagnosis.get("recommended_weights", "normal")

    raw_df = compute_factors_batch(
        sliced_klines, fundamentals_map,
        stock_sector_map=stock_sector_map
    )
    print(f"  原始因子: {len(raw_df)} 只 × {len([c for c in raw_df.columns if c not in ['code','name','sector']])} 因子")

    if raw_df.empty:
        print("  无候选股票通过因子计算")
        return

    # Factor processing + scoring with dynamic regime
    processed_df = process_factor_panel(raw_df)
    scored_df = compute_composite_score(processed_df, regime=regime)
    print(f"  因子处理完成 (regime={regime}), {len(scored_df)} 只")

    # Apply risk controls
    filtered_df = apply_risk_controls(
        scored_df,
        fundamentals_map=fundamentals_map,
        kline_map=sliced_klines,
        market_diagnosis=diagnosis
    )

    ranked = filtered_df  # for downstream use

    # 板块共振终筛：只保留来自共振板块的标的
    resonant_sector_names = set(filtered_sectors["sector"].tolist())
    before_sector_filter = len(ranked)
    ranked = ranked[ranked["sector"].isin(resonant_sector_names) |
                     ranked["sector"].apply(lambda s: any(rs in str(s) for rs in resonant_sector_names))]
    if len(ranked) < before_sector_filter:
        print(f"  板块共振终筛: {before_sector_filter} → {len(ranked)} 只 (仅保留{resonant_sector_names})")

    # Attach close price and daily change from sliced klines
    close_data = []
    change_data = []
    for _, row in ranked.iterrows():
        code = row["code"]
        if code in sliced_klines:
            k = sliced_klines[code]["kline"]
            close_data.append(float(k.iloc[-1]["close"]))
            change_data.append(float(k.iloc[-1]["close"] / k.iloc[-2]["close"] - 1) if len(k) >= 2 else 0)
        else:
            close_data.append(0)
            change_data.append(0)
    ranked["close"] = close_data
    ranked["change_pct"] = change_data

    # ========== Layer 5: 输出 ==========
    output_dir = ROOT / "output" / target_date
    render_output(ranked, temp_result, filtered_sectors, output_dir, args.top_n)

    # ========== Layer 6: Agent 深度分析 (可选) ==========
    if getattr(args, 'agent', False):
        print("\n[Layer 6] Agent 深度分析...")
        agent_top_n = min(args.top_n, 10)
        top_codes = ranked.head(agent_top_n)["code"].tolist()

        from multi_agent_analyzer import prepare_agent_manifest
        manifest = prepare_agent_manifest(top_codes, target_date, output_dir / "agent_analysis")

        # 输出每个标的的 Agent 分析摘要
        for stock in manifest.get("stocks", []):
            print(f"  {stock['ticker']} {stock['name']} → {stock['signal']} (net:{stock['net_score']})")
        print(f"\n  Agent 数据就绪，共 {len(manifest['stocks'])} 只标的")
        print(f"  每只标的的 Agent Briefs 保存在: {output_dir}/agent_analysis/{{ticker}}/")
        print(f"  要启动完整 7-Agent 辩论，请对每只标的运行:")
        print(f"    /trading-analysis {{ticker}}")
    else:
        # 默认输出快捷 Agent 提示
        agent_count = min(5, len(ranked))
        top_codes_hint = ranked.head(agent_count)["code"].tolist()
        print(f"\n  💡 开启深度Agent分析: python3 scripts/run_pipeline.py --date {target_date} --top-n {args.top_n} --agent")


if __name__ == "__main__":
    main()
