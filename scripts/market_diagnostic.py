#!/usr/bin/env python3
"""
市场环境诊断模块 — 独立于选股管线
- 大盘温度 (4维度: MA60位置/波动率/量能健康/市场宽度)
- 波动率区间 (低波/正常/高波)
- 风格诊断 (大小盘/价值成长)
- 动态因子权重选择

用法:
  from market_diagnostic import diagnose_market, select_regime_weights
  diagnosis = diagnose_market(index_df, kline_map)
  regime = select_regime_weights(diagnosis)
"""

import numpy as np
import pandas as pd

# 大盘温度阈值
TEMP_TRADE = 60       # >= 60 积极交易
TEMP_CAUTION = 30     # >= 30 谨慎，< 30 跳过

# 波动率阈值 (年化)
VOL_LOW = 0.15        # < 15% 低波区间
VOL_HIGH = 0.25       # > 25% 高波区间


# ============================================================
# 大盘温度计算
# ============================================================

def compute_market_temperature(index_df, kline_map=None) -> dict:
    """
    四维度计算大盘温度 0-100。

    维度：
      1. MA60 位置 (35%): 指数相对 MA60 的位置
      2. 波动率 (25%): 近20日年化波动率（低波=安全）
      3. 量能健康 (20%): 当日量 vs 20日均量
      4. 市场宽度 (20%): 抽样 %股票站在 MA20 以上

    Args:
        index_df: DataFrame with columns [date, open, close, high, low, volume]
        kline_map: {code: {"kline": DataFrame}} 用于计算宽度

    Returns:
        {temperature, regime, signal, details: {...}}
    """
    df = index_df.copy()
    if len(df) < 60:
        return {"temperature": 50, "regime": "choppy", "signal": "CAUTION",
                "details": {}}

    close = df["close"].values
    volume = df["volume"].values
    latest_close = close[-1]

    # 1. MA60 位置 (35%)
    ma60 = pd.Series(close).rolling(60).mean().values[-1]
    ma60_pct = (latest_close - ma60) / ma60 * 100 if ma60 > 0 else 0
    ma60_score = _score_ma60_position(ma60_pct)

    # 2. 波动率 (25%)
    daily_ret = np.diff(close[-21:]) / close[-21:-1]
    vol_20d = np.std(daily_ret) * np.sqrt(252) if len(daily_ret) > 0 else 0.25
    vol_score = _score_volatility(vol_20d)

    # 3. 量能健康 (20%)
    ma20_vol = pd.Series(volume).rolling(20).mean().values[-1]
    vol_ratio = volume[-1] / ma20_vol if ma20_vol > 0 else 1.0
    volume_health_score = _score_volume_health(vol_ratio)

    # 4. 市场宽度 (20%) — 抽样 %站在MA20以上
    breadth_score = _compute_breadth(kline_map) if kline_map else 50

    # FIX: 原来 run_pipeline.py 中 0.20 * vol_score 是 bug，
    # 应该用 0.20 * volume_health_score
    temp = (0.35 * ma60_score + 0.25 * vol_score +
            0.20 * volume_health_score + 0.20 * breadth_score)

    if temp >= TEMP_TRADE:
        regime, signal = "bull", "TRADE"
    elif temp >= TEMP_CAUTION:
        regime, signal = "choppy", "CAUTION"
    else:
        regime, signal = "bear", "SKIP"

    return {
        "temperature": round(temp, 1),
        "regime": regime,
        "signal": signal,
        "details": {
            "ma60_position": round(ma60_pct, 2),
            "ma60_score": ma60_score,
            "volatility": round(vol_20d, 3),
            "vol_score": vol_score,
            "volume_ratio": round(vol_ratio, 2),
            "volume_health_score": volume_health_score,
            "breadth": breadth_score,
        }
    }


# ============================================================
# 波动率区间分类
# ============================================================

def classify_volatility_regime(index_df) -> str:
    """
    将当前波动率分为：low / normal / high
    """
    if len(index_df) < 21:
        return "normal"
    close = index_df["close"].values
    daily_ret = np.diff(close[-21:]) / close[-21:-1]
    vol_20d = np.std(daily_ret) * np.sqrt(252) if len(daily_ret) > 0 else 0.25

    if vol_20d < VOL_LOW:
        return "low"
    elif vol_20d > VOL_HIGH:
        return "high"
    else:
        return "normal"


# ============================================================
# 风格诊断
# ============================================================

def classify_style_regime(index_df, index_2000_df=None) -> dict:
    """
    大小盘 + 价值/成长风格诊断。

    Args:
        index_df: 沪深300 日线 DataFrame
        index_2000_df: 中证2000 日线 DataFrame（可选，用于大小盘判断）

    Returns:
        {size_style: "large"|"small", value_style: "value"|"growth",
         size_score, value_score}
    """
    result = {"size_style": "large", "value_style": "growth",
              "size_score": 0.0, "value_score": 0.0}

    # 大小盘风格
    if index_df is not None and len(index_df) >= 21:
        idx300_ret = _calc_nday_return(index_df, 20)

        if index_2000_df is not None and len(index_2000_df) >= 21:
            idx2000_ret = _calc_nday_return(index_2000_df, 20)
            result["size_score"] = idx2000_ret - idx300_ret
            result["size_style"] = "small" if result["size_score"] > 0 else "large"
        else:
            result["size_score"] = idx300_ret
            result["size_style"] = "large" if idx300_ret > 0 else "small"

    # 价值/成长 — 用波动率做代理（高波时价值通常优于成长）
    if index_df is not None and len(index_df) >= 21:
        close = index_df["close"].values
        daily_ret = np.diff(close[-21:]) / close[-21:-1]
        vol_20d = np.std(daily_ret) * np.sqrt(252) if len(daily_ret) > 0 else 0.25
        result["value_style"] = "value" if vol_20d > 0.20 else "growth"
        result["value_score"] = vol_20d

    return result


# ============================================================
# 因子权重选择
# ============================================================

def select_regime_weights(diagnosis: dict) -> str:
    """
    根据市场诊断结果选择因子权重方案。
    Returns: "normal" | "defensive" | "offensive"

    逻辑:
      - temp < 40 或 vol_regime == "high" → "defensive"（防御模式）
      - temp > 70 → "offensive"（进攻模式）
      - 其他 → "normal"（正常模式）
    """
    temp = diagnosis.get("temperature", 50)
    details = diagnosis.get("details", {})
    vol_regime = diagnosis.get("vol_regime", "normal")

    if temp < 40 or vol_regime == "high":
        return "defensive"
    elif temp > 70:
        return "offensive"
    else:
        return "normal"


# ============================================================
# 完整市场诊断（对外主接口）
# ============================================================

def diagnose_market(index_df, kline_map=None, index_2000_df=None) -> dict:
    """
    完整市场诊断 — 温度 + 波动率区间 + 风格 + 推荐权重。

    Args:
        index_df: 沪深300/上证指数 日K线 DataFrame
        kline_map: {code: {"kline": DataFrame}} 用于宽度计算
        index_2000_df: 中证2000 日K线（可选，用于大小盘风格）

    Returns:
        {temperature, regime, signal, vol_regime, style_regime,
         recommended_weights, details}
    """
    temp_result = compute_market_temperature(index_df, kline_map)
    vol_regime = classify_volatility_regime(index_df)
    style_regime = classify_style_regime(index_df, index_2000_df)
    rec_weights = select_regime_weights({
        **temp_result,
        "vol_regime": vol_regime,
        "style_regime": style_regime
    })

    return {
        **temp_result,
        "vol_regime": vol_regime,
        "style_regime": style_regime,
        "recommended_weights": rec_weights,
    }


# ============================================================
# 辅助函数
# ============================================================

def classify_regimes(index_df) -> dict[str, str]:
    """
    用 MA60 趋势将市场分为三个状态:
      - bull (上涨市): close > MA60 且 MA60 斜率向上
      - bear (下跌市): close < MA60 且 MA60 斜率向下
      - choppy (震荡市): 其余情况

    Returns: {date: regime} 字典，用于回测
    """
    df = index_df.copy()
    df["ma60"] = df["close"].rolling(60).mean()
    df["ma60_slope"] = df["ma60"].diff(20)
    df["regime"] = "choppy"

    bull_mask = (df["close"] > df["ma60"]) & (df["ma60_slope"] > 0)
    bear_mask = (df["close"] < df["ma60"]) & (df["ma60_slope"] < 0)

    df.loc[bull_mask, "regime"] = "bull"
    df.loc[bear_mask, "regime"] = "bear"

    return dict(zip(df["date"].astype(str), df["regime"]))


def _score_ma60_position(ma60_pct: float) -> float:
    if ma60_pct > 5: return 95
    elif ma60_pct > 2: return 75
    elif ma60_pct > 0: return 60
    elif ma60_pct > -3: return 40
    elif ma60_pct > -5: return 25
    else: return 10


def _score_volatility(vol_20d: float) -> float:
    if vol_20d < 0.15: return 95
    elif vol_20d < 0.20: return 75
    elif vol_20d < 0.25: return 55
    elif vol_20d < 0.35: return 35
    else: return 10


def _score_volume_health(vol_ratio: float) -> float:
    if 0.7 < vol_ratio < 1.5: return 90
    elif 0.5 < vol_ratio < 2.0: return 65
    elif 0.3 < vol_ratio < 3.0: return 40
    else: return 15


def _compute_breadth(kline_map, sample_n: int = 100) -> float:
    """抽样计算 %股票站在MA20以上"""
    if not kline_map:
        return 50
    codes = list(kline_map.keys())
    if len(codes) > sample_n:
        rng = np.random.RandomState(42)
        codes = rng.choice(codes, sample_n, replace=False).tolist()

    above = 0
    total = 0
    for code in codes:
        item = kline_map[code]
        kline = item["kline"] if isinstance(item, dict) else item
        if len(kline) < 22:
            continue
        close_arr = kline["close"].values
        ma20 = np.mean(close_arr[-21:-1])
        if close_arr[-1] > ma20:
            above += 1
        total += 1

    if total == 0:
        return 50
    pct_above = above / total * 100

    if 40 <= pct_above <= 60:
        return 95
    elif 30 <= pct_above <= 70:
        return 70
    elif 20 <= pct_above <= 80:
        return 45
    else:
        return 20


def _calc_nday_return(df: pd.DataFrame, n: int) -> float:
    """计算近N日收益率"""
    if len(df) <= n:
        return 0.0
    return float(df["close"].iloc[-1] / df["close"].iloc[-n-1] - 1)


# ============================================================
# CLI 测试
# ============================================================

if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from data_loader import ensure_dirs, load_universe_klines

    ensure_dirs()
    ROOT = Path(__file__).resolve().parent.parent

    print("=" * 60)
    print("市场环境诊断测试")
    print("=" * 60)

    # 加载指数数据
    index_path = ROOT / "data" / "stocks" / "INDEX_1000300.parquet"
    if not index_path.exists():
        print("无沪深300指数数据，请先运行 fetch_history.py")
        sys.exit(1)

    index_df = pd.read_parquet(index_path)
    print(f"指数数据: {len(index_df)} 天 ({index_df['date'].iloc[0]} ~ {index_df['date'].iloc[-1]})")

    # 加载股票K线用于宽度计算
    kline_map = load_universe_klines(watchlist_only=True, refresh=False)

    # 运行诊断
    diagnosis = diagnose_market(index_df, kline_map)

    print(f"\n大盘温度: {diagnosis['temperature']:.0f}/100")
    print(f"市场状态: {diagnosis['regime']} ({diagnosis['signal']})")
    print(f"波动率区间: {diagnosis['vol_regime']}")
    print(f"风格诊断: {diagnosis['style_regime']}")
    print(f"推荐权重: {diagnosis['recommended_weights']}")

    details = diagnosis["details"]
    print(f"\n细分:")
    print(f"  MA60位置: {details['ma60_position']:+.1f}% (分{details['ma60_score']})")
    print(f"  波动率: {details['volatility']*100:.0f}% (分{details['vol_score']})")
    print(f"  量比: {details['volume_ratio']:.2f} (分{details['volume_health_score']})")
    print(f"  宽度: {details['breadth']:.0f}")

    # 测试 classify_regimes
    regime_map = classify_regimes(index_df)
    bull_days = sum(1 for v in regime_map.values() if v == "bull")
    bear_days = sum(1 for v in regime_map.values() if v == "bear")
    choppy_days = sum(1 for v in regime_map.values() if v == "choppy")
    print(f"\n历史市况分布（MA60趋势法）:")
    print(f"  上涨市: {bull_days} 天")
    print(f"  震荡市: {choppy_days} 天")
    print(f"  下跌市: {bear_days} 天")
