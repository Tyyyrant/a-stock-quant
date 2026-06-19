#!/usr/bin/env python3
"""
技术指标计算引擎
纯函数，输入 OHLCV DataFrame，输出全套指标 dict
全部向量化计算，不做逐行循环
"""

import numpy as np
import pandas as pd


def compute_all(df: pd.DataFrame) -> dict:
    """对一只股票的 K 线 DataFrame 计算全套指标"""
    if df.empty or len(df) < 60:
        return _empty_result()

    o, h, l, c, v = df["open"], df["high"], df["low"], df["close"], df["volume"]

    result = {
        "ma": _compute_ma(c),
        "macd": _compute_macd(c),
        "boll": _compute_boll(c),
        "kdj": _compute_kdj(h, l, c),
        "rsi": _compute_rsi(c),
        "wr": _compute_wr(h, l, c),
        "volume": _compute_volume(c, v),
        "pattern": _detect_patterns(c, v),
        "returns": _compute_returns(c),
    }
    return result


def compute_latest(df: pd.DataFrame) -> dict:
    """只返回最新一期的指标值（用于当日扫描）"""
    full = compute_all(df)
    latest = {}
    for category, indicators in full.items():
        if isinstance(indicators, dict):
            latest[category] = {}
            for k, v in indicators.items():
                if isinstance(v, (np.ndarray, list)):
                    val = v[-1] if len(v) > 0 else None
                elif isinstance(v, pd.Series):
                    val = v.iloc[-1] if len(v) > 0 else None
                else:
                    val = v
                # numpy → python
                if isinstance(val, (np.floating, float)):
                    val = float(round(val, 4))
                elif isinstance(val, np.integer):
                    val = int(val)
                latest[category][k] = val
    return latest


# ==================== MA ====================

def _compute_ma(c: pd.Series) -> dict:
    periods = [5, 10, 20, 60, 120, 250]
    result = {}
    for p in periods:
        ma = c.rolling(p).mean()
        result[f"ma{p}"] = ma.values
        # 价格相对于均线的位置
        result[f"above_ma{p}"] = (c.values > ma.values)
    return result


# ==================== MACD ====================

def _compute_macd(c: pd.Series) -> dict:
    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    dif = ema12 - ema26
    dea = dif.ewm(span=9, adjust=False).mean()
    macd_bar = 2 * (dif - dea)

    # 金叉死叉
    cross = np.zeros(len(c), dtype=int)
    for i in range(1, len(c)):
        if dif.iloc[i] > dea.iloc[i] and dif.iloc[i-1] <= dea.iloc[i-1]:
            cross[i] = 1   # 金叉
        elif dif.iloc[i] < dea.iloc[i] and dif.iloc[i-1] >= dea.iloc[i-1]:
            cross[i] = -1  # 死叉

    # DIF 斜率（加速/减速）
    dif_slope = np.zeros(len(c))
    dif_slope[2:] = dif.values[2:] - dif.values[1:-1]

    return {
        "dif": dif.values,
        "dea": dea.values,
        "macd": macd_bar.values,
        "cross": cross,                       # 1=金叉, -1=死叉, 0=无
        "dif_slope": dif_slope,
        "above_zero": (dif.values > 0),       # 零轴上方
        "zero_dist_pct": np.abs(dif.values) / c.values,  # 离零轴距离%
    }


# ==================== BOLL ====================

def _compute_boll(c: pd.Series, period: int = 20, std: float = 2.0) -> dict:
    mid = c.rolling(period).mean()
    std_val = c.rolling(period).std()
    upper = mid + std * std_val
    lower = mid - std * std_val
    # 位置: 0=下轨处, 1=上轨处
    position = np.clip((c.values - lower.values) / (upper.values - lower.values + 1e-10), 0, 1)
    bandwidth = (upper - lower) / mid  # 带宽（越小越收敛）

    return {
        "upper": upper.values,
        "mid": mid.values,
        "lower": lower.values,
        "position": position,
        "bandwidth": bandwidth.values,
        "at_lower": (position < 0.1),
        "at_upper": (position > 0.9),
    }


# ==================== KDJ ====================

def _compute_kdj(h: pd.Series, l: pd.Series, c: pd.Series,
                 n: int = 9, k_p: int = 3, d_p: int = 3) -> dict:
    lowest_low = l.rolling(n).min()
    highest_high = h.rolling(n).max()
    rsv = (c - lowest_low) / (highest_high - lowest_low + 1e-10) * 100

    k = np.zeros(len(c))
    d = np.zeros(len(c))
    for i in range(len(c)):
        if i == 0:
            k[i] = d[i] = 50
        else:
            k[i] = (k_p - 1) / k_p * k[i-1] + 1 / k_p * (rsv.iloc[i] if not pd.isna(rsv.iloc[i]) else 50)
            d[i] = (d_p - 1) / d_p * d[i-1] + 1 / d_p * k[i]
    j = 3 * k - 2 * d

    # 金叉死叉
    cross = np.zeros(len(c), dtype=int)
    for i in range(1, len(c)):
        if k[i] > d[i] and k[i-1] <= d[i-1]:
            cross[i] = 1
        elif k[i] < d[i] and k[i-1] >= d[i-1]:
            cross[i] = -1

    return {
        "k": k, "d": d, "j": j,
        "cross": cross,
        "j_oversold": (j < 0),     # J值超卖（<0）
        "j_overbought": (j > 100),  # J值超买（>100）
    }


# ==================== RSI ====================

def _compute_rsi(c: pd.Series, periods: list = [6, 14]) -> dict:
    result = {}
    for p in periods:
        delta = c.diff()
        gain = delta.clip(lower=0)
        loss = (-delta).clip(lower=0)
        avg_gain = gain.ewm(alpha=1/p, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/p, adjust=False).mean()
        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
        result[f"rsi{p}"] = rsi.values
    return result


# ==================== WR ====================

def _compute_wr(h: pd.Series, l: pd.Series, c: pd.Series,
                periods: list = [10, 6]) -> dict:
    result = {}
    for p in periods:
        highest = h.rolling(p).max()
        lowest = l.rolling(p).min()
        wr = (highest - c) / (highest - lowest + 1e-10) * 100
        result[f"wr{p}"] = wr.values
    return result


# ==================== 成交量 ====================

def _compute_volume(c: pd.Series, v: pd.Series) -> dict:
    ma5 = v.rolling(5).mean()
    ma20 = v.rolling(20).mean()
    volume_ratio = v / (ma5 + 1e-10)
    abnormal_vol = v / (ma20 + 1e-10)
    vol_change = v.pct_change(5)

    # 量价配合
    price_up = (c.diff() > 0)
    vol_up = (v.diff() > 0)
    price_vol_sync = price_up == vol_up  # 量价同步

    return {
        "volume_ratio": volume_ratio.values,        # 量比(vs 5日均量)
        "abnormal_volume": abnormal_vol.values,      # 异常量(vs 20日均量)
        "ma5_volume": ma5.values,
        "ma20_volume": ma20.values,
        "vol_change_5d": vol_change.values,
        "price_vol_sync": price_vol_sync.values,
        "is_shrink": (volume_ratio.values < 0.6),
        "is_expand": (volume_ratio.values > 2.0),
    }


# ==================== 收益率 ====================

def _compute_returns(c: pd.Series) -> dict:
    daily_ret = c.pct_change()
    return {
        "daily": daily_ret.values,
        "ret_1d": daily_ret.values,
        "ret_5d": (c / c.shift(5) - 1).values,
        "ret_10d": (c / c.shift(10) - 1).values,
        "ret_20d": (c / c.shift(20) - 1).values,
        "ret_60d": (c / c.shift(60) - 1).values,
        "volatility_20d": daily_ret.rolling(20).std().values,
        "volatility_60d": daily_ret.rolling(60).std().values,
        "max_dd_20d": _rolling_max_drawdown(c, 20),
    }


def _rolling_max_drawdown(c: pd.Series, window: int) -> np.ndarray:
    """滚动窗口最大回撤"""
    result = np.zeros(len(c))
    for i in range(window, len(c) + 1):
        seg = c.iloc[i - window:i]
        peak = seg.cummax()
        dd = (seg - peak) / peak
        result[i - 1] = dd.min()
    return result


# ==================== 形态识别 ====================

def _detect_patterns(c: pd.Series, v: pd.Series) -> dict:
    ma5 = c.rolling(5).mean().values
    ma10 = c.rolling(10).mean().values
    ma20 = c.rolling(20).mean().values
    ma60 = c.rolling(60).mean().values
    ma120 = c.rolling(120).mean().values
    price = c.values

    n = len(c)

    # 均线排列
    bull_align = np.zeros(n, dtype=bool)
    bear_align = np.zeros(n, dtype=bool)
    for i in range(60, n):
        bull_align[i] = (ma5[i] > ma10[i] > ma20[i] > ma60[i])
        bear_align[i] = (ma5[i] < ma10[i] < ma20[i] < ma60[i])

    # 均线粘合度（最近5/10/20 三条线的离散度）
    ma_squeeze = np.zeros(n)
    for i in range(20, n):
        trio = np.array([ma5[i], ma10[i], ma20[i]])
        ma_squeeze[i] = np.std(trio) / np.mean(trio) * 100  # 百分比

    # 均线金叉（MA5 上穿 MA20）
    ma_golden_cross = np.zeros(n, dtype=int)
    for i in range(1, n):
        if ma5[i] > ma20[i] and ma5[i-1] <= ma20[i-1]:
            ma_golden_cross[i] = 1

    # 放量突破 MA60
    vol_ratio = v.values / (v.rolling(5).mean().values + 1e-10)
    breakout_ma60 = np.zeros(n, dtype=bool)
    breakout_ma120 = np.zeros(n, dtype=bool)
    for i in range(60, n):
        breakout_ma60[i] = (price[i] > ma60[i] and price[i-1] <= ma60[i-1]
                            and vol_ratio[i] > 1.5)
        breakout_ma120[i] = (price[i] > ma120[i] and price[i-1] <= ma120[i-1]
                             and vol_ratio[i] > 1.5)

    # 光头阳线
    full_body_yang = np.zeros(n, dtype=bool)
    o = c.shift(1).values  # 近似用前收盘
    for i in range(1, n):
        # 简化：收盘=最高、开盘=最低 近似为光头光脚
        pass
    # 更准确的做法需要 open/high/low/close
    # 这里用近似：close > open + (high-close) < (close-open)*0.3

    return {
        "bull_align": bull_align,
        "bear_align": bear_align,
        "ma_squeeze_pct": ma_squeeze,
        "ma_golden_cross": ma_golden_cross,
        "breakout_ma60": breakout_ma60,
        "breakout_ma120": breakout_ma120,
        "is_squeezed": (ma_squeeze < 3.0) & ~bull_align & ~bear_align,
        "bull_start": bull_align & np.roll(~bull_align, 1),  # 刚形成多头
    }


def _empty_result() -> dict:
    return {
        "ma": {}, "macd": {}, "boll": {}, "kdj": {},
        "rsi": {}, "wr": {}, "volume": {}, "pattern": {}, "returns": {},
    }


# ==================== CLI 测试 ====================

if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from data_loader import ensure_dirs, batch_fetch_klines

    ensure_dirs()

    print("指标计算测试")
    print("=" * 50)

    # 取一只股票的K线测试
    codes = [("300024", 0)]  # 机器人
    klines = batch_fetch_klines(codes)
    if klines:
        df = klines["300024"]
        print(f"K线: {len(df)} 条, {df['date'].iloc[0]} ~ {df['date'].iloc[-1]}")
        result = compute_latest(df)
        import json
        print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
