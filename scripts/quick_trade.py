#!/usr/bin/env python3
"""
量化短线交易流水线 v1 — 聚焦模式

流程:
  Layer 1: 大盘温度 (market_diagnostic)
  Layer 2: 选出 Top 3 共振板块
  Layer 3: 每板块挑 5 只代表股
  Layer 4: 深度分析 (K线形态+量价+筹码) → 仅输出 BUY
  Layer 5: 给出完整交易计划 (入场/止损/止盈/仓位/明日建仓建议)

用法:
  python3 scripts/quick_trade.py                          # 最新交易日
  python3 scripts/quick_trade.py --date 2026-06-17        # 指定日期
  python3 scripts/quick_trade.py --sectors 3 --per-sector 5  # 自定义数量
"""

import argparse, json, os, sys, time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from data_loader import (
    ensure_dirs, load_universe_klines, get_stock_kline,
    SECTOR_NAME_MAP, tencent_quote, load_fundamentals_for_codes,
)
from market_diagnostic import diagnose_market
from candlestick_patterns import identify_all_patterns, pattern_result_to_dict
from volume_price_analyzer import analyze_volume_price
from chip_distribution import estimate_chip_distribution
from fetch_a_share_data import fetch_technical, fetch_news, fetch_a_share_macro
from limit_up_analyzer import analyze_limit_up, analyze_sector_limit_ups

# ============================================================
# 配置
# ============================================================

TOP_SECTORS = 3
TOP_PER_SECTOR = 5
MIN_SECTOR_STOCKS = 5
MAX_SECTORS_CONSIDER = 10


# ============================================================
# Layer 1: 大盘温度
# ============================================================

def load_index_kline(refresh=False):
    from data_loader import fetch_index_kline
    cache_path = ROOT / "data" / "stocks" / "INDEX_1000300.parquet"
    if refresh or not cache_path.exists():
        # 999999有日期解析bug，用000001+腾讯fallback
        df = fetch_index_kline("000001", market=1, count=300)
        if df is not None and len(df) >= 60:
            df.to_parquet(cache_path, index=False)
    return pd.read_parquet(cache_path)


# ============================================================
# Layer 2: 板块共振 — 只选 Top 3
# ============================================================

# 板块名→腾讯 pt 代码
SECTOR_BOARD_MAP = {
    "半导体": "pt01801081", "元件": "pt01801083", "消费电子": "pt01801085",
    "通信设备": "pt01801102", "软件开发": "pt01801104", "电池": "pt01801737",
    "自动化设备": "pt01801078", "通用设备": "pt01801072", "汽车零部件": "pt01801093",
    "机器人": "pt02003640", "人工智能": "pt02003800", "芯片概念": "pt02003891",
    "PCB": "pt01801083", "铜缆高速连接": "pt01801102", "先进封装": "pt01801081",
    "AI PC": "pt01801101", "MLCC": "pt01801083", "玻璃基板": "pt01801083",
    "无线充电": "pt02003960",
}


def compute_sector_perf(kline_map, stock_sector_map, index_df, target_date):
    """计算每板块超额收益 + 共振分。使用中位数聚合。"""
    idx = index_df[index_df["date"] <= target_date]
    if len(idx) < 21:
        return pd.DataFrame()
    idx_close = idx["close"].values

    def idx_ret(n):
        return idx_close[-1] / idx_close[-n-1] - 1 if len(idx_close) > n else 0.0

    idx_ret_1, idx_ret_5 = idx_ret(1), idx_ret(5)
    idx_ret_10, idx_ret_20 = idx_ret(10), idx_ret(20)

    stock_rets = []
    for code, item in kline_map.items():
        kline = item["kline"] if isinstance(item, dict) else item
        k = kline[kline["date"] <= target_date]
        if len(k) < 21:
            continue
        c_arr = k["close"].values
        stock_rets.append({
            "code": code,
            "sector": stock_sector_map.get(code, "未分类"),
            "ret_1d": c_arr[-1] / c_arr[-2] - 1 if len(c_arr) >= 2 else 0,
            "ret_5d": c_arr[-1] / c_arr[-6] - 1 if len(c_arr) >= 6 else 0,
            "ret_10d": c_arr[-1] / c_arr[-11] - 1 if len(c_arr) >= 11 else 0,
            "ret_20d": c_arr[-1] / c_arr[-21] - 1 if len(c_arr) >= 21 else 0,
        })

    sdf = pd.DataFrame(stock_rets)
    if sdf.empty:
        return pd.DataFrame()

    agg = sdf.groupby("sector").agg(
        n_stocks=("code", "count"),
        ret_1d=("ret_1d", "median"), ret_5d=("ret_5d", "median"),
        ret_10d=("ret_10d", "median"), ret_20d=("ret_20d", "median"),
    ).reset_index()

    idx_rets = {"1d": idx_ret_1, "5d": idx_ret_5, "10d": idx_ret_10, "20d": idx_ret_20}
    for col in ["ret_1d", "ret_5d", "ret_10d", "ret_20d"]:
        period = col.split("_")[1]
        agg[f"excess_{period}"] = agg[col] - idx_rets[period]

    # 共振分
    for col_suffix in ["1d", "5d", "10d", "20d"]:
        col = f"excess_{col_suffix}"
        vmin, vmax = agg[col].min(), agg[col].max()
        agg[f"{col}_norm"] = (agg[col] - vmin) / (vmax - vmin) * 100 if vmax > vmin else 50.0

    agg["resonance_score"] = (
        0.30 * agg["excess_1d_norm"] + 0.25 * agg["excess_5d_norm"] +
        0.25 * agg["excess_10d_norm"] + 0.20 * agg["excess_20d_norm"]
    )
    return agg.sort_values("resonance_score", ascending=False)


def enrich_with_board_data(sector_perf, target_date):
    """用腾讯板块实时行情修正 1d 收益"""
    if sector_perf.empty:
        return
    sector_to_pt = {}
    for _, row in sector_perf.iterrows():
        sec = row["sector"]
        if sec in SECTOR_BOARD_MAP:
            sector_to_pt[sec] = SECTOR_BOARD_MAP[sec]

    if not sector_to_pt:
        return

    import urllib.request
    unique_pts = list(set(sector_to_pt.values()))
    url = "https://qt.gtimg.cn/q=" + ",".join(unique_pts)
    try:
        req = urllib.request.Request(url)
        req.add_header("User-Agent", "Mozilla/5.0")
        resp = urllib.request.urlopen(req, timeout=10)
        data = resp.read().decode("gbk")
    except Exception:
        return

    board_data = {}
    for line in data.strip().split(";"):
        if "~" not in line:
            continue
        try:
            raw_code = line.strip().split("=")[0].replace("v_", "")
            vals = line.split('"')[1].split("~")
            price = float(vals[3])
            last_close = float(vals[4])
            chg = (price / last_close - 1) * 100 if last_close > 0 else 0.0
            board_data[raw_code] = chg
        except (ValueError, IndexError):
            pass

    for idx, row in sector_perf.iterrows():
        sec = row["sector"]
        pt = sector_to_pt.get(sec)
        if pt and pt in board_data:
            sector_perf.at[idx, "ret_1d"] = board_data[pt] / 100.0
            sector_perf.at[idx, "excess_1d"] = board_data[pt] / 100.0 - row.get("idx_ret_1d", 0)

    # 重算共振分
    col = "excess_1d"
    vmin, vmax = sector_perf[col].min(), sector_perf[col].max()
    if vmax > vmin:
        sector_perf[f"{col}_norm"] = (sector_perf[col] - vmin) / (vmax - vmin) * 100
    sector_perf["resonance_score"] = (
        0.30 * sector_perf["excess_1d_norm"] + 0.25 * sector_perf["excess_5d_norm"] +
        0.25 * sector_perf["excess_10d_norm"] + 0.20 * sector_perf["excess_20d_norm"]
    )
    sector_perf.sort_values("resonance_score", ascending=False, inplace=True)
    sector_perf.reset_index(drop=True, inplace=True)


# ============================================================
# Layer 3: 板块内选代表股
# ============================================================

def pick_sector_leaders(sector_name, codes, kline_map, n=5):
    """
    选出真正带动板块上攻的领头羊。

    四维评分:
      1. 封板时间 (30分) — 越早封板越强，早盘>午盘>尾盘
      2. 封单力度 (25分) — 涨停日量比 + 收盘在涨停价附近
      3. 超额收益 (25分) — 个股涨幅 vs 板块涨幅中位数
      4. 板块贡献 (20分) — 市值×超额涨幅 = 对板块指数的拉动
    """
    from limit_up_analyzer import analyze_limit_up

    candidates = []
    fundamentals_map = load_fundamentals_for_codes(codes, refresh=False)

    # 先算板块中位数涨幅（排除非正常值）
    stock_chgs = []
    for code in codes:
        if code.startswith("688") or code not in kline_map:
            continue
        item = kline_map[code]
        kline = item["kline"] if isinstance(item, dict) else item
        if len(kline) < 2:
            continue
        c_arr = kline["close"].values
        chg = (c_arr[-1] / c_arr[-2] - 1) if len(c_arr) >= 2 else 0
        stock_chgs.append(chg)
    sector_median_chg = np.median(stock_chgs) if stock_chgs else 0

    for code in codes:
        if code.startswith("688"):
            continue
        if code not in kline_map:
            continue

        item = kline_map[code]
        kline = item["kline"] if isinstance(item, dict) else item
        if len(kline) < 60:
            continue

        c = kline["close"].values
        v = kline["volume"].values
        h = kline["high"].values
        o = kline["open"].values

        latest = c[-1]
        prev_close = c[-2] if len(c) >= 2 else latest
        chg_1d = (latest / prev_close - 1)
        vol_ratio = v[-1] / np.mean(v[-20:]) if len(v) >= 20 and np.mean(v[-20:]) > 0 else 1.0

        # 基本面
        fund = fundamentals_map.get(code, {})
        pe = fund.get("pe_ttm", 0) or fund.get("pe", 0) or 0
        name = fund.get("name", "") or item.get("info", {}).get("name", "")
        mcap = fund.get("mcap_yi", 0) or fund.get("mcap", 0) or 10  # 亿

        # ==== 1. 封板时间 (30分) ====
        lu = analyze_limit_up(code, kline)
        seal_score = 0
        seal_label = ""
        if lu.get("is_limit_up"):
            seal_time = lu.get("seal_time_est", "")
            if "早盘" in str(seal_time):
                seal_score = 30
                seal_label = "早盘封板"
            elif "午盘" in str(seal_time) or "上午" in str(seal_time):
                seal_score = 22
                seal_label = "午盘封板"
            elif "尾盘" in str(seal_time):
                seal_score = 10
                seal_label = "尾盘封板"
            else:
                # 收盘在涨停价附近 = 封住了但时间未知，按中档
                calc_limit = round(prev_close * 1.1, 2)
                if h[-1] >= calc_limit * 0.99:
                    seal_score = 18
                    seal_label = "涨停封住"
            # 连续板加分
            if lu.get("position_label") and "连板" in str(lu.get("position_label")):
                seal_score = min(30, seal_score + 8)
                seal_label += "(连板)"
        elif chg_1d > 0.05:
            # 未涨停但涨幅>5% = 强势未封
            seal_score = 8
            seal_label = "强势未封"
        elif chg_1d > 0.02:
            seal_score = 3
            seal_label = "温和上涨"

        # ==== 2. 封单/量能力度 (25分) ====
        # 涨停票: 封板量越大越好 / 未涨停: 量比健康为佳
        if lu.get("is_limit_up"):
            # 涨停日量比: 太小=无量空涨，太大=出货嫌疑
            if 1.5 < vol_ratio < 5:
                force_score = 25
            elif 0.8 <= vol_ratio <= 1.5:
                force_score = 18  # 缩量涨停(最强封单)
            elif vol_ratio > 5:
                force_score = 12  # 量太大，有分歧
            else:
                force_score = 10
        else:
            if 1.2 < vol_ratio < 3:
                force_score = 15  # 放量上攻
            elif vol_ratio >= 3:
                force_score = 10  # 量过大
            else:
                force_score = 5

        # ==== 3. 超额收益 (25分) ====
        excess = chg_1d - sector_median_chg
        if excess > 0.05:
            excess_score = 25
        elif excess > 0.02:
            excess_score = 18
        elif excess > 0:
            excess_score = 10
        elif excess > -0.02:
            excess_score = 3
        else:
            excess_score = 0  # 拖后腿

        # ==== 4. 板块贡献 (20分) ====
        contribution = mcap * max(excess, 0) * 100
        if contribution > 5:
            contrib_score = 20
        elif contribution > 1:
            contrib_score = 12
        elif contribution > 0.1:
            contrib_score = 6
        else:
            contrib_score = 0

        leader_score = seal_score + force_score + excess_score + contrib_score

        candidates.append({
            "code": code, "name": name, "close": latest,
            "change_pct": chg_1d, "vol_ratio": vol_ratio,
            "pe": pe, "mcap": mcap,
            "score": round(leader_score, 1),
            "excess_pct": round(excess * 100, 1),
            "seal_label": seal_label,
            "sector_median": round(sector_median_chg * 100, 1),
        })

    candidates.sort(key=lambda x: x["score"], reverse=True)
    return candidates[:n]


# 保留旧函数名作为别名
def pick_representative_stocks(sector_name, codes, kline_map, n=5):
    return pick_sector_leaders(sector_name, codes, kline_map, n)


# ============================================================
# Layer 4: 单票深度分析
# ============================================================

def deep_analyze(code, name, sector, target_date, kline_df=None, diagnosis=None):
    """
    对单只股票做深度分析:
    - K线形态
    - 量价关系
    - 筹码分布
    - 综合信号
    """
    if kline_df is None:
        market = 1 if code.startswith("6") else 0
        kline_df = get_stock_kline(code, market, refresh=False)

    if kline_df is None or kline_df.empty or len(kline_df) < 60:
        return None

    kline_df = kline_df[kline_df["date"] <= target_date].copy()

    # K线形态
    patterns = identify_all_patterns(kline_df, ticker=code)
    # 量价
    vol_price = analyze_volume_price(kline_df)
    # 筹码
    chip = estimate_chip_distribution(kline_df, ticker=code)

    # 基本面
    try:
        quotes = tencent_quote([code])
        q = quotes.get(code, {})
        pe = q.get("pe_ttm", 0)
        pb = q.get("pb", 0)
        mcap = q.get("mcap_yi", 0)
        turnover = q.get("turnover_pct", 0)
    except Exception:
        pe = pb = mcap = turnover = 0

    # 大盘温度
    temp = diagnosis.get("temperature", 50) if diagnosis else 50

    # 综合评分
    bullish = 0
    bearish = 0
    reasons_bull = []
    reasons_bear = []

    # K线
    if patterns.pattern_score > 30:
        bullish += 3
        reasons_bull.append(f"K线形态偏多({patterns.pattern_score:.0f}分)")
    elif patterns.pattern_score < -30:
        bearish += 3
        reasons_bear.append(f"K线形态偏空({patterns.pattern_score:.0f}分)")

    # 量价
    vs = vol_price.get("volume_score", 0)
    if vs > 15:
        bullish += 2
        reasons_bull.append(f"量价健康({vs:.0f}分)")
    elif vs < -15:
        bearish += 2
        reasons_bear.append(f"量价异常({vs:.0f}分)")

    # 筹码
    cs = chip.get("chip_score", 0)
    profit_ratio = chip.get("profit_ratio", 0.5)
    if cs > 15:
        bullish += 2
        reasons_bull.append(f"筹码安全({cs:.0f}分)")
    elif cs < -15:
        bearish += 2
        reasons_bear.append(f"筹码风险({cs:.0f}分)")

    # 趋势中高获利盘=锁仓
    if profit_ratio > 0.8 and patterns.pattern_score > 0:
        bullish += 2
        reasons_bull.append(f"筹码高度锁定(获利{profit_ratio:.0%})")

    # 均线多头
    try:
        tech = fetch_technical(code, target_date)
        align = tech.get("trend", {}).get("ma_alignment", "")
        if align == "bullish":
            bullish += 2
            reasons_bull.append("均线多头排列")
        rsi14 = tech.get("momentum", {}).get("rsi14")
        if rsi14 and 30 < rsi14 < 65:
            bullish += 1
            reasons_bull.append(f"RSI={rsi14:.0f}健康")
        elif rsi14 and rsi14 > 80:
            bearish += 1
            reasons_bear.append(f"RSI={rsi14:.0f}超买")
    except Exception:
        pass

    # 估值
    if pe > 0:
        if pe > 100:
            bearish += 1
            reasons_bear.append(f"PE={pe:.0f}偏高")
        elif pe < 20:
            bullish += 1
            reasons_bull.append(f"PE={pe:.0f}低估值")

    # ==== 涨停分析 (新增) ====
    lu_result = None
    try:
        lu_result = analyze_limit_up(code, kline_df)
    except Exception:
        pass

    if lu_result and lu_result["is_limit_up"]:
        lu_quality = lu_result["quality_score"]
        lu_label = lu_result["quality_label"]
        lu_cont = lu_result["continuation_prob"]

        if lu_quality >= 80:  # 龙头首板: 强力加分
            bullish += 4
            reasons_bull.append(f"🔥{lu_label} 延续{lu_cont}%")
            # 涨停龙头的量价异常不致命
            for r in list(reasons_bear):
                if "量价" in r: reasons_bear.remove(r); bearish -= 2
            if any("PE" in r for r in reasons_bear):
                reasons_bear = [r for r in reasons_bear if "PE" not in r]; bearish -= 1
        elif lu_quality >= 65:  # 强势涨停: 适度加分
            bullish += 2
            reasons_bull.append(f"涨停{lu_label} 延续{lu_cont}%")
            # 量价异常降为警告
            for r in list(reasons_bear):
                if "量价异常" in r:
                    reasons_bear.remove(r); bearish -= 2
                    reasons_bear.append(f"量价偏弱(涨停质量{lu_quality:.0f}未毙)")
        elif lu_quality >= 50:  # 跟风涨停: 不加分不扣分
            reasons_bull.append(f"涨停({lu_label}) 延续{lu_cont}%")
        else:  # 可疑涨停: 扣分
            bearish += 2
            reasons_bear.append(f"⚠涨停质量差({lu_label})")

    # 逆势加分
    if temp > 60 and bullish > bearish:
        bullish += 2
        reasons_bull.append("大盘偏暖，顺势做多")

    net = bullish - bearish
    if net >= 5: signal = "STRONG_BUY"
    elif net >= 2: signal = "BUY"
    else: signal = "PASS"

    # 硬过滤: 亏损股一律毙（A股短线不碰亏损票）
    if pe < 0:
        signal = "PASS"
        reasons_bear.append(f"⚠亏损股(PE={pe:.0f})")

    # ST股硬过滤
    if name and ('ST' in name or '*ST' in name):
        signal = "PASS"
        reasons_bear.append("⚠ST股")

    # ==== 《股是股非》九大战法过滤 (C区/出货/量价异动/洗盘/缺口) ====
    try:
        from warfare_patterns import apply_warfare_filters
        override, boost, wz_reasons = apply_warfare_filters(code, kline_df)
        if override == "PASS":
            signal = "PASS"
            for direction, reason in wz_reasons:
                reasons_bear.append(reason)
            print(f"    [战法] {code} C区/出货毙: {'; '.join(r for _,r in wz_reasons)}")
        else:
            for direction, reason in wz_reasons:
                if direction == "bull":
                    bullish += 1
                    reasons_bull.append(reason)
                else:
                    bearish += 1
                    reasons_bear.append(reason)
            if boost != 0:
                bullish += boost // 3
                print(f"    [战法] {code} 股是股非战法{boost:+d}: {'; '.join(r for _,r in wz_reasons)}")
    except Exception:
        pass

    # 重新计算net
    net = bullish - bearish
    if signal != "PASS":
        if net >= 5: signal = "STRONG_BUY"
        elif net >= 2: signal = "BUY"
        else: signal = "PASS"

    # 计算关键价位 — 短线用 MA5/MA10
    c = kline_df["close"].values
    latest_close = float(c[-1])
    day_chg_pct = float((c[-1] / c[-2] - 1) * 100) if len(c) >= 2 else 0

    # 当日大跌(>3%)不推荐
    if signal != "PASS" and day_chg_pct < -3:
        signal = "PASS"
        reasons_bear.append(f"⚠当日跌{day_chg_pct:.1f}%")
    ma5 = float(np.mean(c[-5:])) if len(c) >= 5 else latest_close * 0.97
    ma10 = float(np.mean(c[-10:])) if len(c) >= 10 else latest_close * 0.95
    ma60 = float(np.mean(c[-60:])) if len(c) >= 60 else latest_close * 0.90

    # 支撑: 筹码支撑峰 or MA5/MA10 (短线)
    chip_support = chip.get("nearest_support")
    supports = [s for s in [chip_support, ma10, ma5] if s and s < latest_close]
    entry_low = max(supports) if supports else latest_close * 0.95

    # 阻力: 筹码压力峰 or 60日高
    chip_resist = chip.get("nearest_resistance")
    high_60d = float(np.max(c[-60:])) if len(c) >= 60 else latest_close * 1.1
    resistances = [r for r in [chip_resist, high_60d] if r and r > latest_close]
    target = min(resistances) if resistances else latest_close * 1.10

    # 止损: 支撑下方 3%
    stop = entry_low * 0.97 if entry_low > 0 else latest_close * 0.93

    # 明日建仓建议
    if lu_result and lu_result["is_limit_up"]:
        tomorrow_advice = lu_result.get("next_day_signal", "观望")
    elif temp >= 60 and signal in ("BUY", "STRONG_BUY"):
        tomorrow_advice = "✅ 大盘温度适宜，明日开盘可分批建仓"
    elif temp >= 40 and signal in ("BUY", "STRONG_BUY"):
        tomorrow_advice = "⚠️ 大盘偏弱，建议半仓或等回踩入场"
    else:
        tomorrow_advice = "⛔ 大盘偏冷，观望为主"

    # 提取额外字段给 Agent
    rsi_val = 50; ma_align_flag = False; deviation_val = 0; turnover_val = 0; low_60_val = 0
    try:
        tech = fetch_technical(code, target_date)
        rsi_val = tech.get("momentum", {}).get("rsi14", 50) or 50
        ma_align_flag = tech.get("trend", {}).get("ma_alignment", "") == "bullish"
        deviation_val = tech.get("trend", {}).get("price_vs_sma20", 0) or 0
        turnover_val = turnover
        low_60_val = tech.get("support_resistance", {}).get("low_60d", latest_close*0.5) or latest_close*0.5
    except Exception:
        pass

    return {
        "code": code, "name": name, "sector": sector,
        "close": latest_close,
        "change_pct": float((c[-1] / c[-2] - 1) if len(c) >= 2 else 0),
        "signal": signal,
        "net_score": net,
        "rsi": rsi_val, "ma_align": ma_align_flag,
        "deviation": deviation_val, "turnover": turnover_val,
        "low_60": low_60_val,
        "reasons_bull": reasons_bull,
        "reasons_bear": reasons_bear,
        "pe": pe, "pb": pb, "mcap": mcap,
        "candlestick_score": patterns.pattern_score,
        "volume_score": vs,
        "chip_score": cs,
        "profit_ratio": profit_ratio,
        "entry_low": round(entry_low, 2),
        "entry_high": round(latest_close, 2),
        "stop": round(stop, 2),
        "target": round(target, 2),
        "limit_up": {
            "is_limit_up": lu_result["is_limit_up"] if lu_result else False,
            "quality_score": lu_result["quality_score"] if lu_result else 0,
            "quality_label": lu_result["quality_label"] if lu_result else "",
            "continuation_prob": lu_result["continuation_prob"] if lu_result else 0,
            "position_label": lu_result["position_label"] if lu_result else "",
            "seal_time_est": lu_result["seal_time_est"] if lu_result else "",
        } if lu_result and lu_result.get("is_limit_up") else None,
        "tomorrow_advice": tomorrow_advice,
        "market_temp": temp,
    }


# ============================================================
# 主流程
# ============================================================

def run(target_date=None, top_sectors=TOP_SECTORS, per_sector=TOP_PER_SECTOR):
    """主流水线"""
    ensure_dirs()

    if target_date is None:
        index_df = load_index_kline()
        available_dates = sorted(index_df["date"].tolist())
        target_date = available_dates[-1]
    else:
        index_df = load_index_kline()

    index_df = index_df[index_df["date"] <= target_date].copy()
    today = datetime.now().strftime("%Y-%m-%d")

    print(f"\n{'='*70}")
    print(f"  量化短线交易 — {target_date} (报告日期: {today})")
    print(f"{'='*70}")

    # ========== Layer 1: 大盘温度 ==========
    print("\n[Layer 1] 大盘温度...")
    kline_map = load_universe_klines(watchlist_only=False, refresh=False)

    index_2000_df = None
    zz2000_path = ROOT / "data" / "stocks" / "INDEX_1000852.parquet"
    if zz2000_path.exists():
        index_2000_df = pd.read_parquet(zz2000_path)

    diagnosis = diagnose_market(index_df, kline_map, index_2000_df)
    temp = diagnosis["temperature"]
    regime = diagnosis.get("recommended_weights", "normal")
    signal = diagnosis["signal"]
    vol_regime = diagnosis.get("vol_regime", "?")

    print(f"  温度: {temp:.0f}/100  状态: {regime}  信号: {signal}  波动率: {vol_regime}")

    if signal == "SKIP":
        print("\n  ⚠️ 大盘温度过低，不建议交易")
        return

    # ========== Layer 2: Top 3 共振板块 ==========
    print(f"\n[Layer 2] Top {top_sectors} 共振板块...")

    # 板块分类
    cache_path = ROOT / "data" / "sector_classification.json"
    stock_sector_map = {}
    if cache_path.exists():
        with open(cache_path) as f:
            cached = json.load(f)
        all_codes = list(kline_map.keys())
        for code in all_codes:
            if code in cached:
                stock_sector_map[code] = cached[code]
            elif code.startswith("300") or code.startswith("301"):
                stock_sector_map[code] = "创业板"
            elif code.startswith("002") or code.startswith("000"):
                stock_sector_map[code] = "深市主板"
            elif code.startswith("6"):
                stock_sector_map[code] = "沪市主板"

    sector_perf = compute_sector_perf(kline_map, stock_sector_map, index_df, target_date)
    enrich_with_board_data(sector_perf, target_date)

    if sector_perf.empty:
        print("  无法计算板块收益")
        return

    # 过滤
    GENERIC = {"沪市主板", "深市主板", "创业板", "科创板", "其他", "未分类"}
    sector_perf = sector_perf[
        (~sector_perf["sector"].isin(GENERIC)) &
        (sector_perf["n_stocks"] >= MIN_SECTOR_STOCKS)
    ]

    top_sector_list = sector_perf.head(top_sectors)

    print(f"  共振板块:")
    for _, sr in top_sector_list.iterrows():
        print(f"    {sr['sector']:<12s} 共振{sr['resonance_score']:.0f}分  "
              f"1d{sr['excess_1d']*100:+.1f}%  5d{sr['excess_5d']*100:+.1f}%  "
              f"10d{sr['excess_10d']*100:+.1f}%  ({sr['n_stocks']}只)")

    # ========== Layer 3: 每板块选代表股 ==========
    print(f"\n[Layer 3] 每板块选 {per_sector} 只代表股...")
    all_picks = []
    for _, sr in top_sector_list.iterrows():
        sec = sr["sector"]
        sec_codes = [c for c, s in stock_sector_map.items() if s == sec and c in kline_map]
        picks = pick_sector_leaders(sec, sec_codes, kline_map, n=per_sector)
        for p in picks:
            p["sector"] = sec
            p["resonance"] = sr["resonance_score"]
        all_picks.extend(picks)
        leaders_str = " | ".join(
            f"{p['name']}({p['score']:.0f}分{'🔥'+p['seal_label'] if p.get('seal_label') else ''})"
            for p in picks[:3]
        )
        print(f"  {sec}: {len(picks)}只领头羊 | {leaders_str}")

    # ========== Layer 3.5: 细分概念 + 龙头识别 ==========
    print(f"\n[Layer 3.5] 细分概念归类 + 龙头识别...")
    from data_loader import eastmoney_concept_blocks as _ecb
    from collections import Counter

    sub_sectors = {}   # {sub_tag: [codes]}
    stock_sub_tags = {}  # {code: [tags]}
    leader_board = {}    # {sub_tag: leader_code}
    cross_sector_additions = set()  # 跨板块补全候选
    meaningful_subs = {}

    for _, sr in top_sector_list.iterrows():
        sec = sr["sector"]
        sec_codes = [c for c, s in stock_sector_map.items() if s == sec and c in kline_map]
        for code in sec_codes[:50]:  # 扩大扫描以捕获细分关联
            try:
                tags = _ecb(code).get("concept_tags", [])
                stock_sub_tags[code] = tags
                for tag in tags:
                    if tag not in sub_sectors:
                        sub_sectors[tag] = []
                    sub_sectors[tag].append(code)
            except Exception:
                pass
            time.sleep(0.5)

    # 找有意义的细分标签（≥2只股票共享，且不是宽泛类别）
    GENERIC_TAGS = {"融资融券", "沪股通", "深股通", "创业板", "科创板", "机构重仓",
                    "央国企改革", "昨日涨停", "破增发价股", "创业成份"}
    meaningful_subs = {}
    for tag, codes in sub_sectors.items():
        if len(codes) >= 2 and tag not in GENERIC_TAGS and tag not in top_sector_list["sector"].values:
            meaningful_subs[tag] = codes

    # 在每个细分标签内找龙头（最早涨停+最强量比）
    for tag, codes in meaningful_subs.items():
        best_code, best_score = None, -999
        for code in codes:
            if code in kline_map:
                item = kline_map[code]
                k = item["kline"] if isinstance(item, dict) else item
                if len(k) < 20:
                    continue
                c_arr = k["close"].values
                v_arr = k["volume"].values
                chg = (c_arr[-1]/c_arr[-2]-1) if len(c_arr) >= 2 else 0
                vol_r = v_arr[-1]/max(np.mean(v_arr[-20:]), 1)
                score = chg*100 + min(vol_r, 5)*10
                if score > best_score:
                    best_score, best_code = score, code
        if best_code:
            leader_board[tag] = best_code

    # 跨板块补全：找不在共振板块但在同一细分概念的票
    cross_sector_additions = set()
    for tag, codes in meaningful_subs.items():
        if len(codes) >= 2:
            # 至少有一只已经在共振板块内
            has_resonant = any(
                stock_sector_map.get(c) in top_sector_list["sector"].values
                for c in codes
            )
            if has_resonant:
                for c in codes:
                    sec = stock_sector_map.get(c, "?")
                    if sec not in top_sector_list["sector"].values and c in kline_map:
                        if not c.startswith("688"):
                            cross_sector_additions.add((c, tag))
    if cross_sector_additions:
        print(f"  跨板块补全: {len(cross_sector_additions)} 只")

    if meaningful_subs:
        print(f"  发现 {len(meaningful_subs)} 个细分概念，{len(leader_board)} 个龙头:")
        for tag, codes in sorted(meaningful_subs.items(), key=lambda x: -len(x[1]))[:10]:
            leader = leader_board.get(tag, "?")
            print(f"    {tag}: {len(codes)}只 龙头={leader}")

    # ========== Layer 3.6: 供应链瓶颈挖掘 (全面发现引擎 v2) ==========
    print(f"\n[Layer 3.6] 供应链瓶颈挖掘 (全面发现引擎 v2)...")
    bottleneck_stocks = []
    bottleneck_candidates_map = {}  # code -> material info
    try:
        from bottleneck_discovery import run_full_discovery as run_bn, load_all_stock_names
        bn_all_stocks = load_all_stock_names()
        resonant_sec_names = [sr["sector"] for _, sr in top_sector_list.iterrows()]
        bn_result = run_bn(
            resonant_sectors=resonant_sec_names,
            target_date=target_date,
            all_stocks=bn_all_stocks,
            top_n=30,
        )
        for s in bn_result.get("verified_top", []):
            code = s["code"]
            if code not in [p["code"] for p in all_picks]:
                bottleneck_stocks.append(code)
                bottleneck_candidates_map[code] = s
        print(f"  瓶颈标的: {len(bottleneck_stocks)} 只 (全部注入候选池)")

        # 保存瓶颈数据供 report 使用
        import json as _json
        bn_output = {
            "date": target_date,
            "resonant_sectors": resonant_sec_names,
            "total_materials_covered": bn_result.get("total_materials_covered", 0),
            "total_candidates": bn_result.get("total_candidates", 0),
            "by_sector": bn_result.get("by_sector", {}),
            "materials_found": {
                k: {"layer": v["layer"], "category": v["category"],
                    "archetypes": v.get("archetypes", []), "total": v["total"],
                    "stocks": v["stocks"]}
                for k, v in bn_result.get("materials_found", {}).items()
            },
            "verified_top": bn_result.get("verified_top", []),
        }
        bn_save_path = ROOT / "output" / target_date / "bottleneck_full.json"
        bn_save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(bn_save_path, "w") as f:
            _json.dump(bn_output, f, ensure_ascii=False, indent=2)
        print(f"  瓶颈数据已保存: {bn_save_path}")
    except Exception as e:
        print(f"  瓶颈发现失败: {e}，使用旧版 map_supply_chain")
        from supply_chain_mapper import map_supply_chain
        for _, sr in top_sector_list.iterrows():
            sec = sr["sector"]
            try:
                sc = map_supply_chain(sec)
                for lk, bl in sc.get("bottleneck_layers", {}).items():
                    for bs in bl.get("stocks", [])[:3]:
                        bc = bs.get("code", "")
                        if bc and bc not in [p["code"] for p in all_picks] and bc not in bottleneck_stocks:
                            if not bc.startswith("688"):
                                bottleneck_stocks.append(bc)
            except Exception:
                pass

    # 跨板块补全标的加入候选池
    for code, tag in cross_sector_additions:
        if code not in [p["code"] for p in all_picks]:
            info = {}
            try:
                item = kline_map.get(code)
                k = item["kline"] if isinstance(item, dict) else item
                c_arr = k["close"].values; v_arr = k["volume"].values
                chg = (c_arr[-1]/c_arr[-2]-1) if len(c_arr)>=2 else 0
                vol_r = v_arr[-1]/max(np.mean(v_arr[-20:]),1) if len(v_arr)>=20 else 1
            except: chg, vol_r = 0, 1
            all_picks.append({
                "code": code, "name": info.get("name",""), "sector": f"细分:{tag}",
                "close": 0, "change_pct": chg, "vol_ratio": vol_r,
                "pe": info.get("pe_ttm",0) or 0, "bull_align": True,
                "score": chg*40 + min(vol_r,3)*10, "resonance": 0,
            })

    # 瓶颈标的加入候选池 (注入材料信息)
    for bc in bottleneck_stocks:
        if bc in kline_map:
            bn_info = bottleneck_candidates_map.get(bc, {})
            bn_materials = bn_info.get("materials", [])
            bn_layer = bn_info.get("layer", "瓶颈卡位")
            bn_cat = bn_info.get("categories", [""])[0] if bn_info.get("categories") else ""
            all_picks.append({
                "code": bc,
                "name": bn_info.get("name", ""),
                "sector": f"瓶颈:{','.join(bn_materials[:2])}",
                "bottleneck_layer": bn_layer,
                "bottleneck_material": ",".join(bn_materials[:3]),
                "close": bn_info.get("price", 0),
                "change_pct": bn_info.get("chg_pct", 0),
                "vol_ratio": 1.0,
                "pe": bn_info.get("pe", 0),
                "bull_align": True,
                "score": bn_info.get("score", 0),
                "resonance": 0,
            })

    # ========== Layer 3.7: 新闻驱动涟漪 (关键词+AI推理双源) ==========
    print(f"\n[Layer 3.7] 新闻驱动涟漪...")
    ripple_stocks = []
    # 源1: 关键词涟漪
    try:
        from news_ripple import analyze_news_ripple
        ripple_result = analyze_news_ripple(days=7, target_date=target_date)
        for s in ripple_result.get("verified_stocks", []):
            code = s["code"]
            if code in kline_map and not code.startswith("688"):
                ripple_stocks.append(code)
                all_picks.append({
                    "code": code, "name": s.get("name", ""),
                    "sector": f"新闻:{s.get('material','')}",
                    "close": s.get("price", 0), "change_pct": s.get("chg_pct", 0),
                    "vol_ratio": 1.0, "pe": 0, "bull_align": True,
                    "score": s.get("score", 0), "resonance": 0,
                })
    except Exception as e:
        print(f"  关键词涟漪跳过: {e}")

    # 源2: AI推理新闻→标的（从 news_ai_stocks.json 读取）
    ai_path = ROOT / "output" / target_date / "news_ai_stocks.json"
    if ai_path.exists():
        try:
            with open(ai_path) as f: ai_stocks = json.load(f)
            for s in ai_stocks:
                code = s["code"]
                if code in kline_map and not code.startswith("688") and code not in [p["code"] for p in all_picks]:
                    ripple_stocks.append(code)
                    all_picks.append({
                        "code": code, "name": s.get("name", ""),
                        "sector": f"新闻AI:{s.get('topic','')[:15]}",
                        "close": s.get("price", 0), "change_pct": s.get("chg_pct", 0),
                        "vol_ratio": 1.0, "pe": s.get("pe", 0), "bull_align": True,
                        "score": s.get("score", 0), "resonance": 0,
                    })
            if ai_stocks: print(f"  AI推理驱动: {len(ai_stocks)} 只")
        except Exception as e:
            print(f"  AI推理跳过: {e}")

    if ripple_stocks:
        print(f"  新闻驱动合计: {len(ripple_stocks)} 只")

    print(f"\n  共 {len(all_picks)} 只代表股待分析 (含{len(bottleneck_stocks)}只瓶颈 + {len(ripple_stocks)}只新闻驱动)")

    # ========== Layer 4: 消息面分析 ==========
    print(f"\n[Layer 4] 消息面催化...")
    news_data = {}
    news_dir = ROOT.parent / "news" / "data" / "processed"
    candidates = sorted([d.name for d in news_dir.iterdir()
                         if d.is_dir() and d.name <= target_date], reverse=True)
    if candidates:
        news_date = candidates[0]
        impact_path = news_dir / news_date / "news_impact.json"
        md_path = news_dir / news_date / "market_data.json"
        for path in [impact_path, md_path]:
            if path.exists():
                with open(path) as f:
                    data = json.load(f)
                    if "sector_impacts" in data:
                        for sec in data["sector_impacts"]:
                            news_data[sec["sector"]] = {
                                "sentiment": sec.get("sentiment", "中性"),
                                "score": sec.get("score", 5),
                                "mentions": sec.get("total_mentions", 0),
                            }
                    for name_key in ["top_industry_sectors", "bottom_industry_sectors"]:
                        for sec in data.get(name_key, []):
                            n = sec.get("name", "")
                            if n not in news_data:
                                news_data[n] = {"sentiment": "中性", "score": 5, "mentions": 0}
        print(f"  加载 {len(news_data)} 个板块的新闻数据 (日期: {news_date})")
    else:
        print(f"  无新闻数据，跳过")

    # ========== Layer 5: 深度分析 (K线+量价+筹码+消息) ==========
    print(f"\n[Layer 5] 深度分析 (K线+量价+筹码+消息)...")
    results = []
    for i, pick in enumerate(all_picks):
        code = pick["code"]
        name = pick["name"]
        sec = pick["sector"]
        print(f"  [{i+1}/{len(all_picks)}] {code} {name} ({sec})...")
        try:
            item = kline_map.get(code)
            kline = item["kline"] if isinstance(item, dict) else item
            result = deep_analyze(code, name, sec, target_date, kline_df=kline, diagnosis=diagnosis)

            # 注入领头羊评分 (从 Layer 3 pick 透传)
            if result and pick.get("score"):
                result["leader_score"] = pick["score"]
                result["excess_pct"] = pick.get("excess_pct", 0)
                result["seal_label"] = pick.get("seal_label", "")

            # 注入消息面
            if result and sec in news_data:
                nd = news_data[sec]
                result["news_sentiment"] = nd["sentiment"]
                result["news_score"] = nd["score"]
                result["news_mentions"] = nd["mentions"]
                if nd["sentiment"] in ("利好", "偏利好"):
                    result["net_score"] += 2
                    result["reasons_bull"].append(f"板块消息面{nd['sentiment']}({nd['mentions']}条新闻)")
                elif nd["sentiment"] in ("利空", "偏利空"):
                    result["net_score"] -= 2
                    result["reasons_bear"].append(f"板块消息面{nd['sentiment']}")
                result["signal"] = ("STRONG_BUY" if result["net_score"] >= 5 else
                                    "BUY" if result["net_score"] >= 2 else "PASS")
            if result:
                results.append(result)
        except Exception as e:
            print(f"    [ERR] {e}")
        if i < len(all_picks) - 1:
            time.sleep(0.3)

    # 只保留 BUY
    buy_results = [r for r in results if r and r["signal"] in ("BUY", "STRONG_BUY")]
    buy_results.sort(key=lambda x: x["net_score"], reverse=True)

    # ========== Layer 6: Agent 辩论 (对 Top BUY 标的) ==========
    agent_verdicts = {}
    if buy_results:
        agent_top_n = min(len(buy_results), 8)  # 真实辩论只跑Top8
        print(f"\n[Layer 6] 真实7-Agent辩论 (DeepSeek API, Top {agent_top_n})...")
        try:
            from agent_debate import debate
            for i, r in enumerate(buy_results[:agent_top_n]):
                code = r["code"]
                name = r["name"]
                print(f"  [{i+1}/{agent_top_n}] {code} {name}...")
                try:
                    agent_result = debate(r)
                    if agent_result:
                        agent_verdicts[code] = agent_result
                        agent_final = agent_result.get("final", "")
                        if agent_final == "SELL":
                            r["signal"] = "PASS"
                            r["agent_note"] = f"❌ 7-Agent判SELL: {agent_result.get('verdict','')[:50]}"
                        elif agent_final == "HOLD":
                            r["signal"] = "PASS"
                            r["agent_note"] = f"⚠️ 7-Agent判HOLD: {agent_result.get('verdict','')[:50]}"
                        else:
                            r["agent_note"] = f"✅ BUY | {agent_result.get('bull','')[:30]}"
                            r["entry_low"] = agent_result.get("entry", r["entry_low"])
                            r["stop"] = agent_result.get("stop", r["stop"])
                            r["target"] = agent_result.get("target", r["target"])
                    time.sleep(1)  # API限流
                except Exception as e:
                    print(f"    [ERR] {e}")
        except Exception as e:
            print(f"  Agent辩论跳过: {e}，使用规则降级")
            # Fallback: use simplified debate for remaining
            for i, r in enumerate(buy_results[:agent_top_n]):
                if r["code"] not in agent_verdicts:
                    ar = run_full_agent_debate(r, diagnosis)
                    if ar and ar.get("final") == "SELL":
                        r["signal"] = "PASS"

        # 过滤
        buy_results = [r for r in buy_results if r["signal"] != "PASS"]
        buy_results.sort(key=lambda x: x["net_score"], reverse=True)

    # ========== Layer 7: 输出 ==========
    print(f"\n{'='*70}")
    print(f"  最终推荐 — 仅 BUY ({len(buy_results)} 只)")
    print(f"{'='*70}")
    print(f"  大盘温度: {temp:.0f}/100 ({signal})  模式: {regime}  波动率: {vol_regime}")
    print(f"  共振板块: {', '.join(sr['sector'] for _, sr in top_sector_list.iterrows())}")
    print()

    if not buy_results:
        print("  ⚠️ 当前无符合 BUY 条件的标的")
        return buy_results

    for i, r in enumerate(buy_results, 1):
        print(f"  {'─'*60}")
        print(f"  #{i} {r['code']} {r['name']} | {r['sector']} | ¥{r['close']:.2f} | {r['change_pct']*100:+.2f}%")
        print(f"     信号: {r['signal']} (net:{r['net_score']:+d}) | PE:{r['pe']:.0f} | 市值:{r['mcap']:.0f}亿")
        print(f"     K线:{r['candlestick_score']:.0f} 量价:{r['volume_score']:.0f} 筹码:{r['chip_score']:.0f} 获利盘:{r['profit_ratio']:.0%}")
        ns = r.get('news_sentiment', '无')
        # 细分概念+龙头
        sub_tags = stock_sub_tags.get(r['code'], [])
        leader_of = [t for t, l in leader_board.items() if l == r['code']]
        follower_of = [t for t, l in leader_board.items() if r['code'] in meaningful_subs.get(t, []) and l != r['code']]
        if leader_of:
            print(f"     👑 细分龙头: {', '.join(leader_of[:3])}")
        elif follower_of:
            print(f"     🔗 跟随龙头: {leader_board.get(follower_of[0],'?')} ({follower_of[0]})")
        if sub_tags:
            relevant = [t for t in sub_tags if t in meaningful_subs][:5]
            if relevant:
                print(f"     🏷 细分标签: {', '.join(relevant)}")
        leader_score = r.get('leader_score', 0)
        excess_pct = r.get('excess_pct', 0)
        seal = r.get('seal_label', '')
        if leader_score > 0:
            leader_info = f"领头羊{leader_score:.0f}分 超额{excess_pct:+.1f}%"
            if seal:
                leader_info += f" {seal}"
            print(f"     消息: {ns} ({r.get('news_mentions',0)}条) | {leader_info}")
        else:
            print(f"     消息: {ns} ({r.get('news_mentions',0)}条)")
        print(f"     看多: {'; '.join(r['reasons_bull'])}")
        if r['reasons_bear']:
            print(f"     看空: {'; '.join(r['reasons_bear'])}")
        if r.get('agent_note'):
            print(f"     🤖 {r['agent_note']}")
        print(f"     入场: ¥{r['entry_low']:.2f} ~ ¥{r['entry_high']:.2f}")
        print(f"     止损: ¥{r['stop']:.2f} | 止盈: ¥{r['target']:.2f}")
        print(f"     明日: {r['tomorrow_advice']}")
    print(f"  {'─'*60}")

    # 保存
    output_dir = ROOT / "output" / target_date
    output_dir.mkdir(parents=True, exist_ok=True)
    df_out = pd.DataFrame(buy_results)
    df_out.to_csv(output_dir / "trade_recommendations.csv", index=False, encoding="utf-8-sig")
    print(f"\n  CSV: {output_dir}/trade_recommendations.csv")

    return buy_results


# ============================================================
# Agent 快速分析 (替代完整7-Agent以节省时间)
# ============================================================

def run_full_agent_debate(scored, diagnosis):
    """
    完整7-Agent辩论逻辑 — 替代快速版。
    Bull/Bear/Risk 三方对抗 + PM 最终决策。
    """
    pe = scored.get("pe", 0) or 0
    vs = scored.get("volume_score", 0)
    cs = scored.get("chip_score", 0)
    k_score = scored.get("candlestick_score", 0)
    profit_r = scored.get("profit_ratio", 0)
    turnover = scored.get("turnover", 0) or 0
    deviation = scored.get("deviation", 0) or 0
    change = scored.get("change_pct", 0) or 0
    rsi = scored.get("rsi", 0) or 50
    ma_align = scored.get("ma_align", False)
    reasons_bull = scored.get("reasons_bull", [])
    reasons_bear = scored.get("reasons_bear", [])
    low_60 = scored.get("low_60", 0) or 1
    price = scored.get("close", 0)
    support = scored.get("entry_low", price*0.9)
    resist = scored.get("target", price*1.1)

    # === Bull ===
    bull_pts = 0
    if ma_align: bull_pts += 2
    if k_score > 40: bull_pts += 2
    if profit_r > 0.7 and k_score > 0: bull_pts += 2
    if 40 < rsi < 65: bull_pts += 1
    if change > 3: bull_pts += 1
    bull_label = 'HIGH' if bull_pts >= 6 else ('MEDIUM' if bull_pts >= 3 else 'LOW')

    # === Bear ===
    bear_pts = 0
    fatal = []
    if vs < -25: bear_pts += 3; fatal.append(f"量价严重异常({vs})→对倒出货")
    elif vs < -15: bear_pts += 1
    if pe < 0: bear_pts += 3; fatal.append(f"亏损股无安全边际")
    elif pe > 200: bear_pts += 2
    if turnover > 25: bear_pts += 3; fatal.append(f"换手{turnover:.0f}%随时踩踏")
    elif turnover > 15: bear_pts += 1
    if deviation > 35: bear_pts += 2
    if cs < -20: bear_pts += 2; fatal.append(f"筹码恶劣({cs})")
    bear_label = 'HIGH' if bear_pts >= 5 else ('MEDIUM' if bear_pts >= 2 else 'LOW')

    # === Risk ===
    risk_pts = 0
    if price / max(low_60, 0.01) > 2: risk_pts += 3
    if pe < 0: risk_pts += 2
    if profit_r > 0.95: risk_pts += 1
    if not support or support < price * 0.7: risk_pts += 2
    risk_label = 'HIGH' if risk_pts >= 5 else ('MEDIUM' if risk_pts >= 2 else 'LOW')

    # === PM 决策 ===
    if len(fatal) >= 2:
        final = 'SELL'
        verdict = f'多重致命风险: {"; ".join(fatal[:2])}'
    elif len(fatal) == 1:
        final = 'HOLD' if bull_pts < 6 else 'BUY'
        verdict = f'有风险({fatal[0]})，等解除再评估' if final == 'HOLD' else '风险可控，顺势做多'
    elif bear_pts > bull_pts + 2:
        final = 'HOLD'
        verdict = '空头占优，风险收益不对称'
    elif bull_pts >= 5:
        final = 'BUY'
        verdict = '多头共振+风险可控'
    elif bull_pts >= 3:
        final = 'BUY'
        verdict = '技术面偏多，控仓参与'
    else:
        final = 'HOLD'
        verdict = '信号不够强，等待更明确买点'

    # 入场/止损
    entry_low = max(support, price*0.93) if support else price*0.93
    stop = entry_low * 0.95
    if k_score > 80 and vs < -20:
        stop = price * 0.95
    target = min(resist, price*1.15) if resist else price*1.15
    size = max(3, min(10, 12 - risk_pts))

    return {
        "final": final, "verdict": verdict,
        "entry": round(entry_low, 2), "stop": round(stop, 2), "target": round(target, 2),
        "size": size,
        "bull_label": bull_label, "bear_label": bear_label, "risk_label": risk_label,
        "fatal": fatal,
    }


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="量化短线交易流水线 — 聚焦模式")
    parser.add_argument("--date", type=str, default=None, help="目标日期 YYYY-MM-DD")
    parser.add_argument("--sectors", type=int, default=3, help="Top N 共振板块")
    parser.add_argument("--per-sector", type=int, default=5, help="每板块选几只")
    args = parser.parse_args()

    run(target_date=args.date,
        top_sectors=args.sectors,
        per_sector=args.per_sector)


if __name__ == "__main__":
    main()
