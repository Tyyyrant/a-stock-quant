#!/usr/bin/env python3
"""
信息系数 (IC) 分析 — 评估多因子模型的选股排序能力

核心问题：因子评分高的股票，后续涨幅是否确实高于评分低的？

指标：
  - Rank IC (Spearman): 因子得分与未来收益的秩相关系数，>0.05 有意义，>0.10 优秀
  - IC_IR: IC均值/IC标准差，衡量因子稳定性，>0.5 可用
  - 分位数收益：top20% 股票 vs bottom20% 的收益差

不跑回测策略，仅评估排序质量。
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from data_loader import (
    ensure_dirs, load_universe_klines, load_fundamentals_for_codes,
    load_news_sector_data, map_stocks_to_sectors, build_stock_universe,
)
from factor_library import (
    compute_factors_batch, process_factor_panel, compute_composite_score,
)

FORWARD_PERIODS = [1, 3, 5, 10, 20]  # 未来收益窗口（交易日）


def compute_forward_returns(kline_map, date, period):
    """计算从 date 起持有 period 天的未来收益"""
    returns = {}
    for code, item in kline_map.items():
        kline = item["kline"] if isinstance(item, dict) else item
        idx = kline["date"].searchsorted(date)
        start_idx = idx
        end_idx = min(idx + period, len(kline) - 1)
        if start_idx < len(kline) and end_idx > start_idx:
            start_price = float(kline.iloc[start_idx]["close"])
            end_price = float(kline.iloc[end_idx]["close"])
            returns[code] = end_price / start_price - 1
    return returns


def run_ic_analysis(kline_map, fundamentals_map, stock_sector_map,
                    trading_days, sample_dates):
    """
    对一系列历史日期计算因子得分和未来收益的 IC。
    sample_dates: 采样日期列表（每隔 N 天取一个，避免过密）
    """
    records = []
    total = len(sample_dates)

    for i, date in enumerate(sample_dates):
        if i % 20 == 0:
            print(f"  IC进度: {date} ({i+1}/{total})")

        # 切片K线到该日期
        sliced = {}
        for code in kline_map:
            item = kline_map[code]
            kline = item["kline"] if isinstance(item, dict) else item
            mask = kline["date"] <= date
            df = kline[mask].copy()
            if len(df) >= 60:
                sliced[code] = {
                    "info": item.get("info", {}),
                    "kline": df,
                }

        if len(sliced) < 30:
            continue

        # 计算因子
        raw_df = compute_factors_batch(
            sliced, fundamentals_map,
            stock_sector_map=stock_sector_map
        )
        if raw_df.empty or len(raw_df) < 30:
            continue

        processed_df = process_factor_panel(raw_df)
        scored_df = compute_composite_score(processed_df, regime="normal")

        # 对每个 forward period 计算 forward return
        for period in FORWARD_PERIODS:
            fwd_rets = compute_forward_returns(kline_map, date, period)

            # 对齐数据
            merged = []
            for _, row in scored_df.iterrows():
                code = row["code"]
                score = row.get("alpha_score")
                fwd = fwd_rets.get(code)
                if score is not None and fwd is not None and not pd.isna(score):
                    merged.append({
                        "code": code,
                        "score": score,
                        "fwd_return": fwd,
                    })

            if len(merged) < 30:
                continue

            df_m = pd.DataFrame(merged)
            # Rank IC (Spearman)
            ic, ic_pval = stats.spearmanr(df_m["score"], df_m["fwd_return"])
            # Pearson IC 也看看
            pearson_ic = df_m["score"].corr(df_m["fwd_return"])

            records.append({
                "date": date,
                "period": period,
                "n_stocks": len(df_m),
                "rank_ic": ic,
                "rank_ic_pval": ic_pval,
                "pearson_ic": pearson_ic,
            })

    return pd.DataFrame(records)


def print_ic_report(ic_df):
    """打印IC分析报告"""
    if ic_df.empty:
        print("无IC数据")
        return

    print("\n" + "=" * 65)
    print("  信息系数 (IC) 分析报告")
    print("=" * 65)
    print(f"  样本日期: {ic_df['date'].nunique()} 天")
    print(f"  平均股票数/日: {ic_df['n_stocks'].mean():.0f} 只")

    print(f"\n  {'周期':<8s} {'均值IC':>8s} {'IC_IR':>8s} {'|IC|>0.05':>10s} {'IC>0':>8s}  {'解读'}")
    print(f"  {'─' * 55}")

    for period in FORWARD_PERIODS:
        sub = ic_df[ic_df["period"] == period]
        if sub.empty:
            continue
        mean_ic = sub["rank_ic"].mean()
        std_ic = sub["rank_ic"].std()
        ic_ir = mean_ic / std_ic if std_ic > 0 else 0
        hit_rate = (sub["rank_ic"] > 0).mean()  # IC为正的比例
        strong_signal = (sub["rank_ic"].abs() > 0.05).mean()  # |IC|>0.05的比例

        if mean_ic > 0.08 and ic_ir > 0.5:
            interp = "优秀 ✓"
        elif mean_ic > 0.03 and ic_ir > 0.3:
            interp = "有效 △"
        elif mean_ic > 0:
            interp = "偏弱 ~"
        else:
            interp = "无效 ✗"

        print(f"  {f'{period}日持有':<8s} {mean_ic:>+8.4f} {ic_ir:>8.2f} "
              f"{strong_signal:>9.0%} {hit_rate:>8.0%}  {interp}")

    print(f"\n  解读:")
    print(f"    均值IC > 0: 因子评分与未来收益正相关")
    print(f"    IC_IR > 0.5: 因子稳定性好，信号可靠")
    print(f"    IC>0 比例: 因子在多少天数里方向正确")
    print(f"    |IC|>0.05: 因子在多少天数里有统计意义")


def compute_quantile_returns(kline_map, fundamentals_map, stock_sector_map,
                             trading_days, sample_dates):
    """计算分位数收益：top20% vs bottom20%"""
    quantile_groups = {p: {"top": [], "bottom": [], "spread": []} for p in FORWARD_PERIODS}

    for i, date in enumerate(sample_dates):
        sliced = {}
        for code in kline_map:
            item = kline_map[code]
            kline = item["kline"] if isinstance(item, dict) else item
            mask = kline["date"] <= date
            df = kline[mask].copy()
            if len(df) >= 60:
                sliced[code] = {"info": item.get("info", {}), "kline": df}

        if len(sliced) < 50:
            continue

        raw_df = compute_factors_batch(sliced, fundamentals_map,
                                       stock_sector_map=stock_sector_map)
        if raw_df.empty or len(raw_df) < 50:
            continue

        processed_df = process_factor_panel(raw_df)
        scored_df = compute_composite_score(processed_df, regime="normal")

        # 按 alpha_score 分5组
        scored_df["quantile"] = pd.qcut(
            scored_df["alpha_score"], q=5,
            labels=["Q1_bottom", "Q2", "Q3", "Q4", "Q5_top"],
            duplicates="drop"
        )

        for period in FORWARD_PERIODS:
            fwd = compute_forward_returns(kline_map, date, period)
            scored_df["fwd"] = scored_df["code"].map(fwd)
            valid = scored_df.dropna(subset=["fwd"])

            top = valid[valid["quantile"] == "Q5_top"]["fwd"].mean()
            bottom = valid[valid["quantile"] == "Q1_bottom"]["fwd"].mean()

            if pd.notna(top):
                quantile_groups[period]["top"].append(top)
            if pd.notna(bottom):
                quantile_groups[period]["bottom"].append(bottom)
            if pd.notna(top) and pd.notna(bottom):
                quantile_groups[period]["spread"].append(top - bottom)

    print("\n" + "=" * 65)
    print("  分位数收益分析 (Top20% vs Bottom20%)")
    print("=" * 65)
    print(f"  {'周期':<8s} {'Top20%':>10s} {'Bottom20%':>10s} {'多空spread':>12s}")
    print(f"  {'─' * 45}")

    for period in FORWARD_PERIODS:
        g = quantile_groups[period]
        if g["top"] and g["bottom"]:
            t = np.mean(g["top"]) * 100
            b = np.mean(g["bottom"]) * 100
            s = np.mean(g["spread"]) * 100
            print(f"  {f'{period}日持有':<8s} {t:>+9.2f}% {b:>+9.2f}% {s:>+11.2f}%")

    return quantile_groups


def output_top_picks(scored_df, date, top_n=20):
    """输出最近交易日的TOP选股"""
    print(f"\n  ── {date} TOP{top_n} 选股 ──")
    # 只选存在的列
    avail_cols = ["code", "name", "sector", "alpha_score"]
    show_cols = [c for c in avail_cols if c in scored_df.columns]
    top = scored_df.head(top_n)[show_cols]
    for _, row in top.iterrows():
        parts = f"  {row['code']} {row.get('name',''):<8s} "
        if 'sector' in show_cols:
            parts += f"{row.get('sector',''):<10s} "
        parts += f"alpha={row['alpha_score']:.0f}"
        print(parts)


# ============================================================
# CLI
# ============================================================

def main():
    ensure_dirs()

    print("=" * 60)
    print("  多因子模型 — IC分析 (排序质量检验)")
    print("=" * 60)

    # 1. 加载数据
    print("\n[1/4] 加载股票池K线...")
    kline_map = load_universe_klines(watchlist_only=False, refresh=False)
    codes = list(kline_map.keys())
    print(f"  共 {len(codes)} 只有效K线")

    if len(codes) < 30:
        print("  股票太少，请先运行 download_broad.py 下载沪深300+中证500")
        return

    fundamentals_map = load_fundamentals_for_codes(codes)

    # 2. 板块情绪
    print("[2/4] 加载板块情绪...")
    news_dir = ROOT.parent / "news" / "data" / "processed"
    news_dates = sorted([d.name for d in news_dir.glob("2026-*") if d.is_dir()])
    sector_map = {}
    universe = build_stock_universe(watchlist_only=False)
    if news_dates:
        latest_news_date = news_dates[-1]
        sector_map = load_news_sector_data(latest_news_date)
    stock_sector_map = map_stocks_to_sectors(universe, sector_map)

    # 3. 使用指数交易日作为参考，生成采样日期
    print("[3/4] 计算IC...")
    index_df = pd.read_parquet(ROOT / "data" / "stocks" / "INDEX_1000300.parquet")
    trading_days = sorted(index_df["date"].tolist())
    trading_days = [d for d in trading_days if d >= "2025-06-01"]

    # 每10个交易日采样一次
    sample_dates = trading_days[::10]
    print(f"  回测区间: {trading_days[0]} ~ {trading_days[-1]} ({len(trading_days)} 天)")
    print(f"  采样日期: {len(sample_dates)} 个 ({sample_dates[0]} ~ {sample_dates[-1]})")

    # 4. 运行 IC 分析
    ic_df = run_ic_analysis(kline_map, fundamentals_map, stock_sector_map,
                            trading_days, sample_dates)

    if ic_df.empty:
        print("IC分析无结果")
        return

    print_ic_report(ic_df)

    # 5. 分位数收益
    quantiles = compute_quantile_returns(kline_map, fundamentals_map,
                                         stock_sector_map, trading_days,
                                         sample_dates)

    # 6. 最近交易日TOP选股
    print("\n[4/4] 最近交易日TOP选股...")
    latest_date = trading_days[-1]
    sliced = {}
    for code in kline_map:
        item = kline_map[code]
        kline = item["kline"] if isinstance(item, dict) else item
        mask = kline["date"] <= latest_date
        df = kline[mask].copy()
        if len(df) >= 60:
            sliced[code] = {"info": item.get("info", {}), "kline": df}

    raw_df = compute_factors_batch(sliced, fundamentals_map,
                                   stock_sector_map=stock_sector_map)
    if not raw_df.empty:
        processed_df = process_factor_panel(raw_df)
        scored_df = compute_composite_score(processed_df, regime="normal")
        top = scored_df.sort_values("alpha_score", ascending=False)
        output_top_picks(top, latest_date, top_n=20)

    # 保存
    ic_out = ROOT / "data" / "backtest" / "ic_analysis.csv"
    ic_df.to_csv(ic_out, index=False)
    print(f"\nIC数据: {ic_out}")


if __name__ == "__main__":
    main()
