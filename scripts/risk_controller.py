#!/usr/bin/env python3
"""
风控层 — 选股后的最后一道过滤
- 黑名单过滤 (ST/新股/低流动/跌停/高质押/立案)
- Barra因子暴露约束
- 行业集中度控制
- 动态仓位管理
- 移动止损计算

配置: config/portfolio.yaml

用法:
  from risk_controller import apply_risk_controls
  filtered_df = apply_risk_controls(scored_df, fundamentals_map, kline_map, diagnosis)
"""

import numpy as np
import pandas as pd
import yaml
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parent.parent


# ============================================================
# 配置加载
# ============================================================

def load_portfolio_config() -> dict:
    with open(ROOT / "config" / "portfolio.yaml") as f:
        return yaml.safe_load(f)


# ============================================================
# 黑名单检查
# ============================================================

def check_blacklist(code: str, fundamentals: dict, kline,
                    config: dict = None) -> tuple:
    """
    检查单个股票是否应被黑名单过滤。

    Returns:
        (is_blacklisted: bool, reason: str)
    """
    if config is None:
        config = load_portfolio_config()
    bl = config.get("blacklist", {})

    # 1. ST / *ST
    if bl.get("exclude_st"):
        name = str(fundamentals.get("name", ""))
        if name and ("ST" in name or "*ST" in name):
            return True, "ST股"
        # Also check by code prefix for *ST that may not be in name
        if name and ("退市" in name):
            return True, "退市风险股"

    # 1b. 科创板 / 北交所 (只推荐创业板+主板)
    if code.startswith("688"):
        return True, "科创板(仅推荐创业板+主板)"
    if code.startswith("8") or code.startswith("4"):
        return True, "北交所(仅推荐创业板+主板)"

    # 2. 新股（上市不足N天）
    min_days = bl.get("exclude_new_stocks_days", 60)
    list_date_str = fundamentals.get("list_date", "")
    if list_date_str:
        try:
            if len(str(list_date_str)) == 8:
                list_date = datetime.strptime(str(list_date_str), "%Y%m%d")
                if (datetime.now() - list_date).days < min_days:
                    return True, f"上市不足{min_days}天"
        except Exception:
            pass

    # 3. 最小市值
    min_mcap = bl.get("min_market_cap", 20)  # 亿
    mcap = fundamentals.get("market_cap") or 0
    if mcap > 0 and mcap / 1e8 < min_mcap:
        return True, f"市值不足{min_mcap}亿"

    # 4. 最小日均成交额
    min_amount = bl.get("min_avg_amount", 3000) * 10000  # 万元→元
    if kline is not None and len(kline) >= 20:
        avg_amount = kline["amount"].tail(20).mean()
        if pd.notna(avg_amount) and avg_amount < min_amount:
            return True, f"日均成交不足{bl.get('min_avg_amount', 3000)}万"

    # 5. 近5日有过跌停
    if kline is not None and len(kline) >= 5:
        limit_down = fundamentals.get("limit_down")
        if limit_down and limit_down > 0:
            recent_lows = kline["low"].tail(5)
            if (recent_lows <= limit_down * 1.005).any():
                return True, "近5日有过跌停"

    # 6. 近20日涨幅过大（不追高）
    if kline is not None and len(kline) >= 21:
        ret_20d = (kline["close"].iloc[-1] / kline["close"].iloc[-21] - 1)
        max_pct_20d = bl.get("max_pct_20d", 50) / 100
        if ret_20d > max_pct_20d:
            return True, f"近20日涨幅超{bl.get('max_pct_20d', 50)}%"

    # 7. 近5日涨幅过大
    if kline is not None and len(kline) >= 6:
        ret_5d = (kline["close"].iloc[-1] / kline["close"].iloc[-6] - 1)
        max_pct_5d = bl.get("max_pct_5d", 20) / 100
        if ret_5d > max_pct_5d:
            return True, f"近5日涨幅超{bl.get('max_pct_5d', 20)}%"

    # 8. 高质押比例
    pledge_ratio = fundamentals.get("pledge_ratio")
    if pledge_ratio and float(pledge_ratio) > 0.70:
        return True, f"大股东质押{float(pledge_ratio)*100:.0f}%>70%"

    # 9. 立案调查
    if fundamentals.get("under_investigation"):
        return True, "有未了结立案调查"

    return False, ""


# ============================================================
# 仓位计算
# ============================================================

def compute_position_size(volatility: float, temperature: float,
                         base_pct: float = 0.15, config: dict = None) -> float:
    """
    根据市场状态动态调整仓位。

    Formula: base_pct × temp_factor × vol_factor

    temp_factor: temperature/100, clamped to [0.3, 1.2]
    vol_factor: 0.15/volatility, capped at 1.5

    Args:
        volatility: 年化波动率（如 0.20）
        temperature: 大盘温度（0-100）
        base_pct: 基础仓位比例
    """
    if config is None:
        config = load_portfolio_config()
    max_single = config.get("position", {}).get("max_single_stock", 0.20)

    temp_factor = max(0.3, min(1.2, temperature / 100))
    vol_factor = min(1.5, 0.15 / max(volatility, 0.05))

    adjusted = base_pct * temp_factor * vol_factor
    return round(min(adjusted, max_single), 4)


# ============================================================
# Barra 因子暴露约束
# ============================================================

def apply_barra_limits(df: pd.DataFrame, config: dict = None) -> pd.DataFrame:
    """
    约束 Barra 因子暴露范围。

    - SIZE: 除非风格明确指示，限制在 ±0.5 标准差内
    - VOL: 限制 ≤ 1.0（不持过高波动）
    """
    if config is None:
        config = load_portfolio_config()
    barra = config.get("barra_limits", {})

    df = df.copy()

    size_limits = barra.get("size_exposure", [-0.5, 0.5])
    if "size_adj" in df.columns:
        df = df[(df["size_adj"] >= size_limits[0]) & (df["size_adj"] <= size_limits[1])]

    vol_limits = barra.get("volatility_exposure", [-1.5, 1.0])
    if "volatility_adj" in df.columns:
        df = df[(df["volatility_adj"] >= vol_limits[0]) &
                (df["volatility_adj"] <= vol_limits[1])]

    return df


# ============================================================
# 行业集中度控制
# ============================================================

def apply_industry_concentration(df: pd.DataFrame, top_n: int = 20,
                                 config: dict = None) -> pd.DataFrame:
    """
    限制单一行业股票占比不超过 max_industry_weight（默认30%）。

    保留高分段，移除低分段中超出行业限制的股票。
    """
    if config is None:
        config = load_portfolio_config()
    max_pct = config.get("barra_limits", {}).get("max_industry_weight", 0.30)

    if "sector" not in df.columns:
        return df

    max_per_sector = max(int(top_n * max_pct), 2)
    sector_counts = {}
    keep_mask = []

    for _, row in df.iterrows():
        sec = row.get("sector", "未分类")
        cnt = sector_counts.get(sec, 0)
        if cnt < max_per_sector:
            keep_mask.append(True)
            sector_counts[sec] = cnt + 1
        else:
            keep_mask.append(False)

    return df[pd.Series(keep_mask, index=df.index)]


# ============================================================
# 流动性检查
# ============================================================

def apply_liquidity_filter(df: pd.DataFrame, kline_map: dict = None,
                           config: dict = None) -> pd.DataFrame:
    """
    检查持仓规模不超过日均成交的1%（保证能一天卖完）。
    """
    if config is None:
        config = load_portfolio_config()
    liq_pct = config.get("liquidity", {}).get("max_position_pct_of_volume", 0.01)

    if kline_map is None or "code" not in df.columns:
        return df

    df = df.copy()
    df["max_position_by_liquidity"] = 0.0

    for i, row in df.iterrows():
        code = row["code"]
        item = kline_map.get(code)
        if item is None:
            continue
        kline = item.get("kline") if isinstance(item, dict) else item
        if kline is not None and len(kline) >= 20:
            avg_amount = kline["amount"].tail(20).mean()
            if pd.notna(avg_amount) and avg_amount > 0:
                df.at[i, "max_position_by_liquidity"] = avg_amount * liq_pct

    return df


# ============================================================
# 止损/止盈/移动止损
# ============================================================

def get_stop_loss_take_profit(config: dict = None) -> dict:
    """
    获取止损止盈参数。
    """
    if config is None:
        config = load_portfolio_config()
    risk = config.get("risk", {})
    return {
        "stop_loss": risk.get("stop_loss", -0.05),
        "trailing_stop": risk.get("trailing_stop", -0.08),
        "take_profit": risk.get("take_profit", 0.15),
        "max_hold_days": risk.get("max_hold_days", 20),
    }


def compute_trailing_stop(entry_price: float, highest_price: float,
                          config: dict = None) -> float:
    """
    计算移动止损价格。
    trailing_stop_price = highest_price × (1 - trailing_stop_pct)
    """
    if config is None:
        config = load_portfolio_config()
    trail_pct = abs(config.get("risk", {}).get("trailing_stop", -0.08))
    return highest_price * (1 - trail_pct)


def compute_stop_loss_price(entry_price: float, config: dict = None) -> float:
    """固定止损价格"""
    if config is None:
        config = load_portfolio_config()
    stop_pct = abs(config.get("risk", {}).get("stop_loss", -0.05))
    return entry_price * (1 - stop_pct)


def compute_take_profit_price(entry_price: float, config: dict = None) -> float:
    """止盈价格"""
    if config is None:
        config = load_portfolio_config()
    tp_pct = config.get("risk", {}).get("take_profit", 0.15)
    return entry_price * (1 + tp_pct)


# ============================================================
# 完整风控管道（对外主接口）
# ============================================================

def apply_risk_controls(scored_df: pd.DataFrame,
                        fundamentals_map: dict = None,
                        kline_map: dict = None,
                        market_diagnosis: dict = None,
                        config: dict = None) -> pd.DataFrame:
    """
    对多因子打分结果执行完整风控流程。

    流程：
      1. 黑名单过滤（ST/新股/低流动/跌停/高质押）
      2. Barra 因子暴露约束
      3. 行业集中度控制
      4. 动态仓位计算
      5. 流动性检查

    Args:
        scored_df: compute_composite_score() 的输出，含 [code, alpha_score, sector, ...]
        fundamentals_map: {code: {pe, pb, market_cap, limit_up, limit_down, ...}}
        kline_map: {code: {"kline": DataFrame}}
        market_diagnosis: diagnose_market() 的输出
        config: 覆盖 portfolio.yaml 配置

    Returns:
        过滤+增强后的 DataFrame，新增列:
        - suggested_position_pct: 建议仓位百分比
        - max_position_by_liquidity: 流动性允许的最大持仓额
        - risk_flags: 风险标记
    """
    if config is None:
        config = load_portfolio_config()

    if fundamentals_map is None:
        fundamentals_map = {}
    if kline_map is None:
        kline_map = {}

    df = scored_df.copy()

    # ── 1. 黑名单过滤 ──
    blacklist_reasons = {}
    pass_mask = []
    for _, row in df.iterrows():
        code = row["code"]
        fund = fundamentals_map.get(code, {})
        kline = None
        if code in kline_map:
            item = kline_map[code]
            kline = item.get("kline") if isinstance(item, dict) else item
        is_bl, reason = check_blacklist(code, fund, kline, config)
        if is_bl:
            blacklist_reasons[code] = reason
        pass_mask.append(not is_bl)

    n_blacklisted = sum(1 for m in pass_mask if not m)
    if n_blacklisted > 0:
        bl_codes = [c for c, m in zip(df["code"], pass_mask) if not m]
        print(f"  风控: 黑名单过滤 {n_blacklisted} 只 ({', '.join(bl_codes[:5])}...)")

    df = df[pd.Series(pass_mask, index=df.index)].copy()

    # ── 2. Barra 因子暴露约束 ──
    n_before = len(df)
    df = apply_barra_limits(df, config)
    if len(df) < n_before:
        print(f"  风控: Barra约束过滤 {n_before - len(df)} 只")

    # ── 3. 行业集中度控制 ──
    n_before = len(df)
    df = apply_industry_concentration(df, top_n=20, config=config)
    if len(df) < n_before:
        print(f"  风控: 行业集中度过滤 {n_before - len(df)} 只")

    # ── 4. 动态仓位计算 ──
    if market_diagnosis:
        vol = market_diagnosis.get("details", {}).get("volatility", 0.20)
        temp = market_diagnosis.get("temperature", 50)
    else:
        vol, temp = 0.20, 50

    base_pct = config.get("position", {}).get("max_single_stock", 0.15)
    df["suggested_position_pct"] = df.apply(
        lambda _: compute_position_size(vol, temp, base_pct, config), axis=1
    )

    # ── 5. 流动性检查 ──
    df = apply_liquidity_filter(df, kline_map, config)

    # ── 6. 标记超流动性限制的股票 ──
    df["liquidity_safe"] = True
    for i, row in df.iterrows():
        max_by_liq = row.get("max_position_by_liquidity", 0)
        if max_by_liq > 0:
            suggested_amount = row.get("suggested_position_pct", 0.15) * 1_000_000  # 假设100万本金
            if suggested_amount > max_by_liq:
                df.at[i, "liquidity_safe"] = False
                df.at[i, "suggested_position_pct"] = max(
                    max_by_liq / 1_000_000, 0.01  # 最低1%
                )

    return df.sort_values("alpha_score", ascending=False)


# ============================================================
# CLI 测试
# ============================================================

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from data_loader import ensure_dirs

    ensure_dirs()

    print("=" * 60)
    print("风控模块测试")
    print("=" * 60)

    # 测试止损/止盈计算
    print("\n止损止盈:")
    entry = 100.0
    st = compute_stop_loss_price(entry)
    tp = compute_take_profit_price(entry)
    ts = compute_trailing_stop(entry, 120.0)
    print(f"  买入价={entry}, 止损={st:.2f}, 止盈={tp:.2f}")
    print(f"  最高价=120, 移动止损={ts:.2f}")

    # 测试仓位计算
    print("\n仓位计算:")
    for temp, vol in [(80, 0.12), (50, 0.20), (25, 0.35)]:
        size = compute_position_size(vol, temp)
        print(f"  温度={temp} 波动率={vol*100:.0f}% → 仓位={size*100:.1f}%")

    # 测试黑名单
    print("\n黑名单检查:")
    tests = [
        ({"name": "ST大控"}, None, "ST股"),
        ({"name": "正常公司", "market_cap": 50e8, "limit_down": 18.0}, None, "正常"),
        ({"name": "小市值", "market_cap": 5e8}, None, "市值不足"),
    ]
    for fund, kline, desc in tests:
        is_bl, reason = check_blacklist("000001", fund, kline)
        print(f"  {desc}: blocked={is_bl} reason={reason}")

    print("\n风控模块加载成功 ✓")
