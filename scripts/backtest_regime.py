#!/usr/bin/env python3
"""
按市场状态分段的回测分析 — DEPRECATED
功能已合并至 backtest.py，本文件保留向后兼容。

请改用:
  from backtest import Backtest, classify_regimes
  bt = Backtest(..., regime_map=regime_map)

或新模块:
  from market_diagnostic import classify_regimes
"""

import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from backtest import Backtest
from market_diagnostic import classify_regimes

# 向后兼容：重新导出 BACKTEST_CONFIG
from backtest import BACKTEST_CONFIG  # noqa: F401

# 向后兼容：RegimeBacktest 现在等同于 Backtest + regime_map
RegimeBacktest = Backtest  # noqa: F401


# 向后兼容的独立 CLI 入口
def main():
    warnings.warn(
        "backtest_regime.py 已废弃，请使用 backtest.py 配合 regime_map 参数",
        DeprecationWarning
    )
    from data_loader import (
        ensure_dirs, load_universe_klines, load_fundamentals_for_codes,
        load_news_sector_data, map_stocks_to_sectors, build_stock_universe,
    )
    import pandas as pd
    from factor_library import compute_factors_batch, process_factor_panel, compute_composite_score

    ensure_dirs()

    print("=" * 60)
    print("  多因子选股模型 — 分市况回测 (兼容模式)")
    print("  提示: 请使用 backtest.py --regime-aware")
    print("=" * 60)

    # 1. 加载指数数据，划分市况
    print("\n[1/5] 加载沪深300指数，划分市场状态...")
    index_df = pd.read_parquet(ROOT / "data" / "stocks" / "INDEX_1000300.parquet")
    regime_map = classify_regimes(index_df)
    print(f"  指数数据: {len(index_df)} 天")

    # 2. 加载数据
    print("[2/5] 加载股票数据...")
    kline_map = load_universe_klines(watchlist_only=True, refresh=False)
    codes = list(kline_map.keys())
    fundamentals_map = load_fundamentals_for_codes(codes)

    print("[3/5] 加载板块情绪数据...")
    news_dir = ROOT.parent / "news" / "data" / "processed"
    news_dates = sorted([d.name for d in news_dir.glob("2026-*") if d.is_dir()])
    sector_map = {}
    universe = build_stock_universe(watchlist_only=True)
    if news_dates:
        latest_news_date = news_dates[-1]
        sector_map = load_news_sector_data(latest_news_date)
        print(f"  使用 news 数据: {latest_news_date}, {len(sector_map)} 个板块")
    stock_sector_map = map_stocks_to_sectors(universe, sector_map)

    # 4. 运行回测
    print("[4/5] 运行回测...")
    bt = Backtest(kline_map, fundamentals_map, stock_sector_map,
                  regime_map=regime_map)
    stats = bt.run()
    print(f"\n  全周期收益: {stats.get('total_return', 0)*100:+.2f}%")
    print(f"  夏普比率: {stats.get('sharpe_ratio', 0):.2f}")


if __name__ == "__main__":
    main()
