#!/usr/bin/env python3
"""
因子库 — 计算所有 Barra + Alpha + 技术形态因子

因子标准化流程：
  1. 原始因子值
  2. 去极值（Winsorize, 3σ）
  3. Z-score 标准化
  4. 行业中性化（对行业哑变量回归取残差）
  5. 方向调整（direction=-1 的取负）
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from indicator_engine import compute_all as compute_indicators


def load_factor_config() -> dict:
    with open(ROOT / "config" / "factors.yaml") as f:
        return yaml.safe_load(f)


def load_weights_config() -> dict:
    with open(ROOT / "config" / "factor_weights.yaml") as f:
        return yaml.safe_load(f)


# ============================================================
# 核心因子计算
# ============================================================

def compute_factors(kline: pd.DataFrame, fundamentals: dict = None,
                    sector_info: dict = None,
                    margin_info: dict = None,
                    dragon_tiger_info: dict = None) -> dict:
    """
    对单只股票计算全部因子值（最新一期）

    Args:
        kline: OHLCV DataFrame
        fundamentals: {"pe": ..., "pb": ..., "roe": ..., "market_cap": ..., ...}
        sector_info: {"name": ..., "sector_pct_20d": ..., "north_flow": ...}

    Returns:
        {"factor_name": raw_value, ...}
    """
    df = kline
    if df.empty or len(df) < 60:
        return {}

    ind = compute_indicators(df)
    c = df["close"].values
    v = df["volume"].values
    idx = -1  # 最新一期

    factors = {}

    # ---- Barra: Size ----
    if fundamentals and fundamentals.get("market_cap"):
        factors["size"] = np.log(float(fundamentals["market_cap"]))

    # ---- Barra: Volatility ----
    vol_60d = ind["returns"]["volatility_60d"][idx]
    if vol_60d and not np.isnan(vol_60d):
        factors["volatility"] = float(vol_60d)

    # ---- Barra: Momentum ----
    factors["momentum_1m"] = _safe_ret(ind, "ret_20d", idx)  # 每月约20交易日
    factors["momentum_6m"] = _safe_ret(ind, "ret_60d", idx)

    # ---- Barra: Value ----
    if fundamentals:
        if fundamentals.get("pe") and fundamentals["pe"] > 0:
            factors["value_ep"] = 1.0 / fundamentals["pe"]
        if fundamentals.get("pb") and fundamentals["pb"] > 0:
            factors["value_bp"] = 1.0 / fundamentals["pb"]

    # ---- Barra: Quality ----
    if fundamentals:
        if fundamentals.get("roe"):
            factors["quality_roe"] = fundamentals["roe"]
        if fundamentals.get("debt_to_equity"):
            factors["quality_lev"] = fundamentals["debt_to_equity"]

    # ---- Barra: Growth ----
    if fundamentals and fundamentals.get("revenue_yoy"):
        factors["growth_revenue"] = fundamentals["revenue_yoy"]

    # ---- Barra: Liquidity ----
    avg_amount = np.mean(v[-20:])
    if avg_amount > 0:
        factors["liquidity"] = np.log(avg_amount)

    # ---- Alpha: Reversal ----
    factors["reversal_5d"] = -(_safe_ret(ind, "ret_5d", idx) or 0)
    factors["reversal_1d"] = -(_safe_ret(ind, "daily", idx) or 0)

    # ---- Alpha: North Flow ----
    if sector_info and sector_info.get("north_flow") is not None:
        factors["north_flow"] = sector_info["north_flow"]
    # 退而求其次：用市场数据中的北向净流入
    elif sector_info and sector_info.get("net_flow") is not None:
        factors["north_flow"] = sector_info["net_flow"]

    # ---- Alpha: Money Flow (简化：用连续量价同步作为代理) ----
    sync = ind["volume"]["price_vol_sync"]
    if sync is not None:
        factors["money_flow"] = float(np.mean(sync[-5:]))  # 近5日量价同步比例

    # ---- Alpha: Analyst ----
    if fundamentals and fundamentals.get("analyst_count"):
        factors["analyst_coverage"] = fundamentals["analyst_count"]

    # ---- Alpha: Sector Momentum ----
    # 优先用 news 项目的板块数据
    if sector_info and sector_info.get("momentum") is not None:
        factors["sector_momentum"] = float(sector_info["momentum"])
    elif sector_info and sector_info.get("sector_pct_20d") is not None:
        factors["sector_momentum"] = float(sector_info["sector_pct_20d"])
    # 否则暂不填，交给 compute_factors_batch 做板块聚合
    else:
        ret_20d = _safe_ret(ind, "ret_20d", idx)
        if ret_20d is not None:
            factors["_ret_20d"] = ret_20d  # 临时存储，后续聚合用

    # ---- Alpha: Sentiment (板块情绪) ----
    if sector_info and sector_info.get("sentiment_score") is not None:
        factors["sentiment"] = float(sector_info["sentiment_score"])

    # ---- Technical ----
    factors["bottom_breakout"] = _score_bottom_breakout(df, ind)
    factors["key_breakout"] = _score_key_breakout(df, ind)
    factors["ma_squeeze"] = _score_ma_squeeze(df, ind)
    factors["macd_golden"] = _score_macd_golden(df, ind)
    factors["pullback"] = _score_pullback(df, ind)

    # ---- NEW: 过热惩罚 ----
    factors["overheat_penalty"] = _score_overheat_penalty(df, ind)

    # ---- NEW: K线形态 / 量价 / 筹码 (来自新分析模块) ----
    try:
        from candlestick_patterns import identify_all_patterns
        patterns = identify_all_patterns(df, lookback_days=60)
        factors["candlestick_pattern"] = patterns.pattern_score  # -100~100
    except Exception:
        factors["candlestick_pattern"] = 0.0

    try:
        from volume_price_analyzer import analyze_volume_price
        vp = analyze_volume_price(df)
        factors["volume_price_quality"] = vp.get("volume_score", 0)
    except Exception:
        factors["volume_price_quality"] = 0.0

    try:
        from chip_distribution import estimate_chip_distribution
        chip = estimate_chip_distribution(df, lookback=120)
        factors["chip_safety"] = chip.get("chip_score", 0)
    except Exception:
        factors["chip_safety"] = 0.0

    # ---- Volume ----
    vr = ind["volume"]["volume_ratio"][idx]
    if vr and not np.isnan(vr):
        factors["volume_ratio"] = float(vr)
    av = ind["volume"]["abnormal_volume"][idx]
    if av and not np.isnan(av):
        factors["abnormal_volume"] = float(av)

    # ---- Alpha: Margin ----
    if margin_info:
        if margin_info.get("margin_change_5d_pct") is not None:
            factors["margin_trend"] = margin_info["margin_change_5d_pct"]
        if margin_info.get("rzye") and fundamentals and fundamentals.get("market_cap"):
            factors["margin_ratio"] = margin_info["rzye"] / fundamentals["market_cap"]

    # ---- Alpha: Dragon Tiger ----
    if dragon_tiger_info:
        if dragon_tiger_info.get("institution_net_buy_wan") is not None:
            factors["dragon_tiger_inst"] = dragon_tiger_info["institution_net_buy_wan"]

    # cleanup NaN
    return {k: v for k, v in factors.items() if v is not None and not (isinstance(v, float) and np.isnan(v))}


def compute_factors_batch(kline_map: dict, fundamentals_map: dict = None,
                          sector_map: dict = None,
                          stock_sector_map: dict = None) -> pd.DataFrame:
    """
    批量计算股票池的全部因子。

    Args:
        kline_map: {code: {"info": {...}, "kline": DataFrame}}
        fundamentals_map: {code: {pe, pb, market_cap, roe, ...}}
        sector_map: {sector_name: {momentum, sentiment_score, sentiment}}
        stock_sector_map: {code: {sector_momentum, sentiment_score, ...}}
    """
    records = []
    # 批量获取融资融券和龙虎榜数据（仅当需要时）
    margin_map = {}
    dragon_tiger_map = {}
    # 仅当代码列表不太大时才获取（避免过长的东财限流等待）
    if len(kline_map) <= 50:
        try:
            from data_loader import get_margin_batch, get_dragon_tiger_batch
            margin_map = get_margin_batch(list(kline_map.keys())[:30])  # 限制30只避免太久
            dragon_tiger_map = get_dragon_tiger_batch(list(kline_map.keys())[:30])
        except Exception:
            pass

    for code, item in kline_map.items():
        kline = item.get("kline", item) if isinstance(item, dict) else item
        fund = (fundamentals_map or {}).get(code, {})
        info = item.get("info", {}) if isinstance(item, dict) else {}
        sector_name = info.get("sector", "")

        # 合并板块数据
        sector_info = {}
        if sector_name and sector_map:
            sector_info.update((sector_map or {}).get(sector_name, {}))
        if code and stock_sector_map:
            val = stock_sector_map.get(code, {})
            if isinstance(val, str):
                sector_info["sector_name"] = val
            elif isinstance(val, dict):
                sector_info.update(val)

        factors = compute_factors(kline, fundamentals=fund, sector_info=sector_info,
                                 margin_info=margin_map.get(code),
                                 dragon_tiger_info=dragon_tiger_map.get(code))
        factors["code"] = code
        factors["name"] = info.get("name", fund.get("name", ""))
        factors["sector"] = sector_name
        records.append(factors)

    df = pd.DataFrame(records)

    # 板块动量聚合：用同板块股票的 _ret_20d 平均作为 sector_momentum
    if "_ret_20d" in df.columns and "sector" in df.columns:
        sector_avg = df.groupby("sector")["_ret_20d"].mean()
        for i, row in df.iterrows():
            sec = row["sector"]
            if sec in sector_avg.index and pd.isna(row.get("sector_momentum")):
                df.at[i, "sector_momentum"] = sector_avg[sec]
        df = df.drop(columns=["_ret_20d"])

    return df


# ============================================================
# 技术形态评分函数
# ============================================================

def _score_bottom_breakout(df: pd.DataFrame, ind: dict) -> float:
    """底部倍量突破评分 0-100"""
    c = df["close"].values
    v = df["volume"].values
    idx = -1
    # D1: oversold degree (weight 0.25)
    ret_20d = _safe_ret(ind, "ret_20d", idx) or 0
    if ret_20d <= -0.30: d1 = 100
    elif ret_20d <= -0.20: d1 = 80
    elif ret_20d <= -0.10: d1 = 50
    elif ret_20d > 0: d1 = 0
    else: d1 = 20

    # D2: base quality — amplitude compression (weight 0.25)
    window = min(20, len(c))
    recent_c = c[-window:]
    amplitude = (recent_c.max() - recent_c.min()) / recent_c.mean()
    if amplitude < 0.10: d2 = 100
    elif amplitude < 0.15: d2 = 70
    elif amplitude < 0.20: d2 = 40
    else: d2 = 10

    # D3: breakout power — volume ratio + daily change + candle quality (weight 0.30)
    vol_ratio = _safe_val(ind["volume"]["volume_ratio"], idx) or 1.0
    daily_change = _safe_ret(ind, "daily", idx) or 0
    open_p = df["open"].values[idx]
    close_p = c[idx]
    high_p = df["high"].values[idx]

    vol_sub = 100 if vol_ratio >= 3 else (80 if vol_ratio >= 2 else (50 if vol_ratio >= 1.5 else (20 if vol_ratio >= 1.0 else 0)))
    chg_sub = 100 if daily_change >= 0.07 else (80 if daily_change >= 0.05 else (50 if daily_change >= 0.03 else (20 if daily_change >= 0.01 else 0)))
    if close_p > open_p:
        body = close_p - open_p
        upper_shadow = (high_p - close_p) / max(body, 0.001)
        candle_sub = 100 if upper_shadow < 0.3 else (70 if upper_shadow < 0.5 else 30)
    else:
        candle_sub = 0
    d3 = (vol_sub + chg_sub + candle_sub) / 3

    # D4: position relative to MA20 (weight 0.20)
    above_ma20 = _safe_val(ind["ma"]["above_ma20"], idx)
    d4 = 100 if above_ma20 else 0

    return round(d1 * 0.25 + d2 * 0.25 + d3 * 0.30 + d4 * 0.20, 1)


def _score_key_breakout(df: pd.DataFrame, ind: dict) -> float:
    """关键位突破评分 0-100"""
    idx = -1
    score = 0.0
    c = df["close"].values

    breakout_ma60 = _safe_val(ind["pattern"]["breakout_ma60"], idx) or False
    breakout_ma120 = _safe_val(ind["pattern"]["breakout_ma120"], idx) or False
    vol_ratio = _safe_val(ind["volume"]["volume_ratio"], idx) or 1.0

    # 突破了什么级别？
    if breakout_ma120:
        score += 35
    elif breakout_ma60:
        score += 25
    else:
        # 检查是否在关键均线上方
        above_ma120 = _safe_val(ind["ma"]["above_ma120"], idx)
        above_ma60 = _safe_val(ind["ma"]["above_ma60"], idx)
        if above_ma120:
            score += 20
        elif above_ma60:
            score += 10

    # 量能确认
    if vol_ratio >= 2.5:
        score += 30
    elif vol_ratio >= 2.0:
        score += 24
    elif vol_ratio >= 1.5:
        score += 15
    else:
        score += 5

    # K线确认
    open_p = df["open"].values[idx]
    close_p = c[idx]
    high_p = df["high"].values[idx]
    if close_p > open_p:
        upper_shadow = (high_p - close_p) / max(close_p - open_p, 0.001)
        if upper_shadow < 0.3:
            score += 20
        elif upper_shadow < 0.6:
            score += 12
        else:
            score += 5
    else:
        score += 0

    # 站稳确认（收盘价在突破位上方）
    if breakout_ma60 or breakout_ma120:
        score += 10  # 当日突破本身就是最强的

    return min(100, score)


def _score_ma_squeeze(df: pd.DataFrame, ind: dict) -> float:
    """均线粘合发散评分 0-100"""
    idx = -1
    squeeze = _safe_val(ind["pattern"]["ma_squeeze_pct"], idx)
    if squeeze is None or np.isnan(squeeze):
        return 0.0

    score = 0.0

    # 粘合度
    if squeeze < 1.0:
        score += 35
    elif squeeze < 2.0:
        score += 28
    elif squeeze < 3.0:
        score += 17
    else:
        return 0.0  # 不够粘合，不考虑

    # 发散方向
    bull_align = _safe_val(ind["pattern"]["bull_align"], idx) or False
    bull_start = _safe_val(ind["pattern"]["bull_start"], idx) or False
    if bull_start:
        score += 30  # 刚形成多头，最佳时机
    elif bull_align:
        score += 20
    else:
        score += 0

    # 量能确认
    vol_ratio = _safe_val(ind["volume"]["volume_ratio"], idx) or 1.0
    if vol_ratio >= 1.5:
        score += 20
    elif vol_ratio >= 1.2:
        score += 12
    else:
        score += 5

    return min(100, score)


def _score_macd_golden(df: pd.DataFrame, ind: dict) -> float:
    """MACD 零轴金叉评分 0-100"""
    idx = -1
    cross = _safe_val(ind["macd"]["cross"], idx) or 0
    zero_dist = _safe_val(ind["macd"]["zero_dist_pct"], idx)

    if cross != 1:  # 非金叉日
        return 0.0

    score = 0.0

    # 位置
    if zero_dist is not None:
        if zero_dist < 0.01:
            score += 40  # 零轴附近
        elif zero_dist < 0.03:
            score += 30
        elif zero_dist < 0.05:
            score += 20
        else:
            score += 10

    # 红柱是否放大
    macd_bar = _safe_val(ind["macd"]["macd"], idx) or 0
    macd_prev = _safe_val(ind["macd"]["macd"], idx - 1) or 0
    if macd_bar > macd_prev and macd_bar > 0:
        score += 25
    elif macd_bar > macd_prev:
        score += 15

    # DIF加速
    dif_slope = _safe_val(ind["macd"]["dif_slope"], idx) or 0
    dif_slope_prev = _safe_val(ind["macd"]["dif_slope"], idx - 1) or 0
    if dif_slope > 0 and dif_slope > dif_slope_prev:
        score += 20
    elif dif_slope > 0:
        score += 10

    return min(100, score)


def _score_pullback(df: pd.DataFrame, ind: dict) -> float:
    """缩量回踩企稳评分 0-100"""
    idx = -1
    c = df["close"].values
    v = df["volume"].values

    # 前提：之前有突破信号
    breakout_recent = False
    for i in range(max(0, idx - 20), idx):
        if (_safe_val(ind["pattern"]["breakout_ma60"], i)
                or _safe_val(ind["pattern"]["breakout_ma120"], i)
                or _safe_val(ind["pattern"]["ma_golden_cross"], i)):
            breakout_recent = True
            break

    if not breakout_recent:
        return 0.0

    score = 0.0

    # 缩量程度
    vol_ratio = _safe_val(ind["volume"]["volume_ratio"], idx) or 1.0
    if vol_ratio < 0.5:
        score += 35
    elif vol_ratio < 0.7:
        score += 24
    elif vol_ratio < 1.0:
        score += 12
    else:
        score += 0  # 放量回踩不是好信号

    # 支撑确认
    above_ma20 = _safe_val(ind["ma"]["above_ma20"], idx) or False
    if above_ma20:
        score += 30
    else:
        # 破了关键支撑
        score -= 20

    # 企稳信号
    daily_ret = _safe_ret(ind, "daily", idx) or 0
    if daily_ret > 0:
        score += 20  # 收阳
    elif daily_ret > -0.01:
        score += 10  # 微跌
    else:
        score += 0   # 还在跌

    return max(0, min(100, score))


def _score_overheat_penalty(df: pd.DataFrame, ind: dict) -> float:
    """
    过热惩罚因子: 多维度判断是否短期过热，返回 0（正常）~ 100（严重过热）。

    扣分逻辑:
      - 近5日涨幅 > 15%: 20分
      - 近5日涨幅 > 10%: 10分
      - 量比 > 5: 15分
      - 连续3阳+每日涨>3%: 15分
      - RSI > 80: 15分
      - KDJ J值 > 100: 10分
      - 换手率极高: 10分
      - 偏离MA20 > 15%: 15分
    """
    idx = -1
    c = df["close"].values
    v = df["volume"].values
    n = len(c)
    penalty = 0.0

    # 近5日涨幅
    if n >= 6:
        ret_5d = (c[idx] - c[idx - 5]) / c[idx - 5]
        if ret_5d > 0.15:
            penalty += 20
        elif ret_5d > 0.10:
            penalty += 10
        elif ret_5d > 0.07:
            penalty += 5

    # 量比
    if n >= 20:
        vol_ratio = v[idx] / np.mean(v[-20:]) if np.mean(v[-20:]) > 0 else 1.0
        if vol_ratio > 5:
            penalty += 15
        elif vol_ratio > 3:
            penalty += 8

    # 连续阳线
    if n >= 4:
        consec_up = 0
        for i in range(idx, max(idx - 5, 0), -1):
            chg = (c[i] - c[i - 1]) / c[i - 1] if i > 0 else 0
            if chg > 0.03:
                consec_up += 1
            else:
                break
        if consec_up >= 3:
            penalty += 15
        elif consec_up >= 2:
            penalty += 5

    # RSI
    rsi14 = _safe_val(ind.get("rsi", {}).get("rsi14"), idx)
    if rsi14 and rsi14 > 80:
        penalty += 15
    elif rsi14 and rsi14 > 70:
        penalty += 5

    # KDJ J值
    j_val = _safe_val(ind.get("kdj", {}).get("j"), idx)
    if j_val and j_val > 100:
        penalty += 10
    elif j_val and j_val > 90:
        penalty += 5

    # 偏离MA20
    ma20 = np.mean(c[-20:]) if n >= 20 else c[idx]
    if ma20 > 0:
        deviation = (c[idx] - ma20) / ma20
        if deviation > 0.15:
            penalty += 15
        elif deviation > 0.10:
            penalty += 8
        elif deviation > 0.05:
            penalty += 3

    return min(penalty, 100)


# ============================================================
# 标准化与中性化
# ============================================================

def winsorize(series: pd.Series, n_mad: float = 5.0) -> pd.Series:
    """
    去极值（MAD方法，对异常值稳健）。
    5 MAD ≈ 3.3 sigma for normal distribution.
    """
    median = series.median()
    mad = (series - median).abs().median()
    if mad == 0:
        return series
    upper = median + n_mad * mad * 1.4826
    lower = median - n_mad * mad * 1.4826
    return series.clip(lower, upper)


def zscore(series: pd.Series) -> pd.Series:
    """Z-score 标准化"""
    mu, std = series.mean(), series.std()
    if std == 0:
        return pd.Series(0, index=series.index)
    return (series - mu) / std


def industry_neutralize(factor_df: pd.DataFrame, factor_col: str,
                        sector_col: str = "sector") -> pd.Series:
    """
    行业中性化：用因子值对行业哑变量做回归，取残差
    残差 = 剔除行业影响后的纯因子暴露
    """
    df = factor_df.dropna(subset=[factor_col, sector_col]).copy()
    if df.empty or df[sector_col].nunique() < 2:
        return df[factor_col] - df[factor_col].mean()

    # 行业哑变量
    sector_dummies = pd.get_dummies(df[sector_col], drop_first=True)
    if sector_dummies.shape[1] == 0:
        return df[factor_col] - df[factor_col].mean()

    X = sector_dummies.values
    y = df[factor_col].values
    # OLS: β = (X'X)^-1 X'y
    try:
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        y_pred = X @ beta
        residuals = y - y_pred
    except np.linalg.LinAlgError:
        residuals = y - y.mean()

    return pd.Series(residuals, index=df.index)


def process_factor_panel(raw_df: pd.DataFrame,
                         factor_config: dict = None) -> pd.DataFrame:
    """
    完整的因子处理管线：
      raw_df: 行=股票, 列包含各因子原始值 + code/name/sector

    Returns: 处理后的因子面板（标准化+中性化+方向调整）
    """
    if factor_config is None:
        factor_config = load_factor_config()

    df = raw_df.copy()
    factor_cols = []

    for fname, fcfg in factor_config.get("factors", {}).items():
        if fname not in df.columns:
            continue
        factor_cols.append(fname)

        # 1. 去极值
        series = df[fname].astype(float)
        series = winsorize(series)

        # 2. Z-score
        series = zscore(series)

        # 3. 行业中性化（如果配置了）
        neutralization = fcfg.get("neutralization", [])
        if "industry" in neutralization and "sector" in df.columns:
            series = industry_neutralize(pd.DataFrame({
                fname: series, "sector": df["sector"]
            }), fname)

        # 4. 方向调整
        direction = fcfg.get("direction", 1)
        if direction == -1:
            series = -series

        df[f"{fname}_adj"] = series

    return df


def compute_composite_score(df: pd.DataFrame, regime: str = "normal") -> pd.DataFrame:
    """
    计算综合 Alpha 得分
    regime: normal / defensive / offensive
    """
    weights_config = load_weights_config()
    regime_weights = weights_config.get("regimes", {}).get(regime, {})

    df = df.copy()
    score_cols = []

    # 技术形态因子内部组合
    tech_breakdown = weights_config.get("technical_breakdown", {})
    tech_cols = []
    for tname, tw in tech_breakdown.items():
        col = f"{tname}_adj"
        if col in df.columns:
            tech_cols.append((col, tw))
    if tech_cols:
        df["technical_adj"] = sum(
            df[c].fillna(0) * w / sum(w for _, w in tech_cols)
            for c, w in tech_cols
        )
    else:
        df["technical_adj"] = 0
    score_cols.append(("technical_adj", regime_weights.get("technical", 0.25)))

    # 反转因子
    rev_cols = [c for c in [f"{n}_adj" for n in ["reversal_5d", "reversal_1d"]] if c in df.columns]
    if rev_cols:
        df["reversal_adj"] = df[rev_cols].mean(axis=1)
    else:
        df["reversal_adj"] = 0
    score_cols.append(("reversal_adj", regime_weights.get("reversal", 0.15)))

    # 低波因子
    if "volatility_adj" in df.columns:
        score_cols.append(("volatility_adj", regime_weights.get("lowvol", 0.12)))

    # 北向资金
    if "north_flow_adj" in df.columns:
        score_cols.append(("north_flow_adj", regime_weights.get("north", 0.12)))

    # 质量
    qual_cols = [c for c in [f"{n}_adj" for n in ["quality_roe", "quality_lev"]] if c in df.columns]
    if qual_cols:
        df["quality_adj"] = df[qual_cols].mean(axis=1)
    else:
        df["quality_adj"] = 0
    score_cols.append(("quality_adj", regime_weights.get("quality", 0.10)))

    # 资金流
    if "money_flow_adj" in df.columns:
        score_cols.append(("money_flow_adj", regime_weights.get("money_flow", 0.08)))

    # 动量
    mom_cols = [c for c in [f"{n}_adj" for n in ["momentum_1m", "momentum_6m"]] if c in df.columns]
    if mom_cols:
        df["momentum_adj"] = df[mom_cols].mean(axis=1)
    else:
        df["momentum_adj"] = 0
    score_cols.append(("momentum_adj", regime_weights.get("momentum", 0.08)))

    # 价值
    val_cols = [c for c in [f"{n}_adj" for n in ["value_ep", "value_bp"]] if c in df.columns]
    if val_cols:
        df["value_adj"] = df[val_cols].mean(axis=1)
    else:
        df["value_adj"] = 0
    score_cols.append(("value_adj", regime_weights.get("value", 0.05)))

    # 分析师
    if "analyst_coverage_adj" in df.columns:
        score_cols.append(("analyst_coverage_adj", regime_weights.get("analyst", 0.05)))

    # 板块情绪 (from news project)
    if "sentiment_adj" in df.columns:
        score_cols.append(("sentiment_adj", regime_weights.get("sentiment", 0.06)))

    # 行业动量
    if "sector_momentum_adj" in df.columns:
        score_cols.append(("sector_momentum_adj", regime_weights.get("sector_mom", 0.03)))

    # 成长
    if "growth_revenue_adj" in df.columns:
        score_cols.append(("growth_revenue_adj", regime_weights.get("growth", 0.0)))

    # ---- NEW: 深度分析因子 (v2) ----
    # K线形态
    if "candlestick_pattern_adj" in df.columns:
        score_cols.append(("candlestick_pattern_adj", regime_weights.get("candlestick", 0.06)))
    # 量价质量
    if "volume_price_quality_adj" in df.columns:
        score_cols.append(("volume_price_quality_adj", regime_weights.get("volume_quality", 0.06)))
    # 筹码安全
    if "chip_safety_adj" in df.columns:
        score_cols.append(("chip_safety_adj", regime_weights.get("chip_safety", 0.06)))

    # 加权求和
    total_weight = sum(w for _, w in score_cols)
    df["alpha_score"] = sum(
        df[c].fillna(0) * w / total_weight for c, w in score_cols
    )

    # ---- NEW: 过热惩罚 (从 alpha_score 中直接扣减) ----
    if "overheat_penalty_adj" in df.columns:
        # 过热惩罚归一化到 0-1，最多扣 15% 的 alpha 分
        oh_max = df["overheat_penalty_adj"].max()
        if oh_max > 0:
            oh_factor = df["overheat_penalty_adj"].clip(lower=0) / max(oh_max, 1)
            df["alpha_score"] = df["alpha_score"] * (1 - oh_factor * 0.15)

    # 共振加成
    resonance = weights_config.get("resonance_bonus", {})
    if resonance.get("enabled"):
        threshold = resonance.get("threshold", 0.5)
        min_dims = resonance.get("min_dimensions", 3)
        bonus = resonance.get("bonus_pct", 0.10)

        dim_cols = [c for c, _ in score_cols if c in df.columns]
        above = (df[dim_cols] > threshold).sum(axis=1)
        df["resonance_bonus"] = (above >= min_dims).astype(float) * bonus
        df["alpha_score"] = df["alpha_score"] * (1 + df["resonance_bonus"])

    # 最终标准化到 0-100
    if df["alpha_score"].std() > 0:
        df["alpha_score"] = (df["alpha_score"] - df["alpha_score"].min()) / \
                            (df["alpha_score"].max() - df["alpha_score"].min() + 1e-10) * 100

    return df.sort_values("alpha_score", ascending=False)


# ============================================================
# 工具函数
# ============================================================

def _safe_val(arr, idx):
    """安全取数组值"""
    if arr is None:
        return None
    arr_list = arr
    if hasattr(arr, "values"):
        arr_list = arr.values
    elif hasattr(arr, "iloc"):
        arr_list = arr.values
    if isinstance(arr_list, (np.ndarray, list)):
        if abs(idx) >= len(arr_list) or len(arr_list) == 0:
            return None
        val = arr_list[idx]
        if isinstance(val, (np.floating,)):
            return float(val)
        if isinstance(val, (np.bool_,)):
            return bool(val)
        if isinstance(val, (np.integer,)):
            return int(val)
        return val
    return None


def _safe_ret(ind, key, idx):
    val = _safe_val(ind["returns"][key], idx)
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return None
    return val


# ============================================================
# CLI 测试
# ============================================================

if __name__ == "__main__":
    sys.path.insert(0, str(ROOT / "scripts"))
    from data_loader import (ensure_dirs, load_universe_klines,
                              load_fundamentals_for_codes,
                              load_news_sector_data,
                              map_stocks_to_sectors,
                              build_stock_universe)

    ensure_dirs()

    print("因子计算测试")
    print("=" * 50)

    # 1. 加载 K 线数据
    kline_map = load_universe_klines(watchlist_only=True, refresh=False)
    if not kline_map:
        print("无数据，请先运行 data_loader.py 获取K线")
        sys.exit(1)

    # 2. 加载基本面数据
    codes = list(kline_map.keys())
    fundamentals_map = load_fundamentals_for_codes(codes)

    # 3. 加载 news 项目板块数据
    date = "2026-05-29"  # 默认日期
    import os
    if len(sys.argv) > 1:
        date = sys.argv[1]
    elif os.path.exists(str(NEWS_ROOT := ROOT.parent / "news" / "data" / "processed")):
        # 自动找最新日期
        processed_dirs = sorted((ROOT.parent / "news" / "data" / "processed").glob("2026-*"))
        if processed_dirs:
            date = processed_dirs[-1].name

    print(f"使用 news 数据日期: {date}")
    sector_map = load_news_sector_data(date)
    universe = build_stock_universe(watchlist_only=True)
    stock_sector_map = map_stocks_to_sectors(universe, sector_map)

    print(f"基本面: {len(fundamentals_map)} 只")
    has_roe = sum(1 for v in fundamentals_map.values() if v.get("roe"))
    print(f"  有ROE数据: {has_roe} 只")
    print(f"板块数据: {len(sector_map)} 个板块")
    print(f"个股板块映射: {len(stock_sector_map)} 只")

    # 4. 计算原始因子
    raw_df = compute_factors_batch(kline_map, fundamentals_map,
                                   sector_map, stock_sector_map)
    print(f"\n原始因子矩阵: {raw_df.shape[0]} 只股票 × {raw_df.shape[1]} 列")
    all_factor_cols = [c for c in raw_df.columns if c not in ['code','name','sector']]
    print(f"因子列 ({len(all_factor_cols)} 个): {all_factor_cols}")

    # 5. 处理因子
    processed_df = process_factor_panel(raw_df)
    adj_cols = [c for c in processed_df.columns if c.endswith("_adj")]
    print(f"处理后因子列: {adj_cols}")

    # 6. 计算综合得分
    scored_df = compute_composite_score(processed_df, regime="normal")
    print("\n综合得分 TOP 15:")
    display_cols = ["code", "name", "sector", "alpha_score"] + \
                   [c for c in ["technical_adj", "reversal_adj", "volatility_adj",
                                "quality_adj", "momentum_adj", "value_adj",
                                "north_flow_adj", "money_flow_adj"]
                    if c in scored_df.columns]
    pd.set_option("display.max_columns", 20)
    pd.set_option("display.width", 200)
    print(scored_df[display_cols].head(15).to_string(index=False))

    # 7. 保存
    out_dir = ROOT / "data" / "signals"
    out_dir.mkdir(parents=True, exist_ok=True)
    scored_df.to_csv(out_dir / "latest_scores.csv", index=False)
    print(f"\n结果已保存至 {out_dir / 'latest_scores.csv'}")
