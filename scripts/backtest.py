#!/usr/bin/env python3
"""
历史回测引擎

对多因子选股模型做历史回测，验证因子有效性。
关键约束：无 look-ahead bias —— 每个历史日期只用该日期之前的数据。

回测流程：
  每个交易日 t:
    1. slice K线到 t 日
    2. 计算因子（只看 t 之前的数据）
    3. 排名选股
    4. 次日开盘买入 top N
    5. 跟踪持仓，按止损/止盈/持仓天数退出

输出：收益曲线、胜率、盈亏比、夏普比率、最大回撤
"""

import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from data_loader import (
    ensure_dirs, load_universe_klines, load_fundamentals_for_codes,
    load_news_sector_data, map_stocks_to_sectors, build_stock_universe,
    load_news_data
)
from factor_library import (
    compute_factors, compute_factors_batch,
    process_factor_panel, compute_composite_score,
    load_factor_config, load_weights_config
)
from market_diagnostic import classify_regimes, diagnose_market, select_regime_weights

# ============================================================
# 回测配置
# ============================================================

BACKTEST_CONFIG = {
    "start_date": "2025-06-01",    # 回测开始（需要足够历史数据计算指标）
    "end_date": "2026-05-29",      # 回测结束
    "top_n": 5,                    # 每日选股数量
    "capital": 1_000_000,          # 初始资金
    "stop_loss": -0.05,            # 止损线
    "take_profit": 0.15,           # 止盈线
    "max_hold_days": 20,           # 最长持仓天数
    "max_positions": 8,            # 最大同时持仓数
    "single_position_pct": 0.15,   # 单票仓位上限
    "commission": 0.0003,          # 手续费（万3）
    "slippage": 0.001,             # 滑点
}

# ============================================================
# 核心回测逻辑
# ============================================================

class Backtest:
    def __init__(self, kline_map, fundamentals_map, stock_sector_map,
                 config=None, regime_map=None, split_date=None):
        self.kline_map = kline_map
        self.fundamentals_map = fundamentals_map
        self.stock_sector_map = stock_sector_map
        self.cfg = {**BACKTEST_CONFIG, **(config or {})}
        self.regime_map = regime_map
        self.split_date = split_date

        # 找到共同的交易日
        self.trading_days = self._find_common_days()
        print(f"共同交易日: {len(self.trading_days)} 天 "
              f"({self.trading_days[0]} ~ {self.trading_days[-1]})")

        # 过滤回测区间
        self.trading_days = [d for d in self.trading_days
                             if self.cfg["start_date"] <= d <= self.cfg["end_date"]]
        print(f"回测区间: {len(self.trading_days)} 天")

        # 状态
        self.positions = []          # 当前持仓
        self.closed_trades = []      # 已平仓
        self.equity_curve = []       # 权益曲线
        self.cash = self.cfg["capital"]
        self.total_value = self.cash

    def _find_common_days(self):
        """找到所有股票共有的交易日"""
        all_dates = None
        for code, item in self.kline_map.items():
            kline = item["kline"] if isinstance(item, dict) else item
            dates = set(kline["date"].tolist())
            if all_dates is None:
                all_dates = dates
            else:
                all_dates &= dates
        return sorted(all_dates)

    def _get_kline_upto(self, code, date):
        """获取某只股票在指定日期之前的K线"""
        item = self.kline_map[code]
        kline = item["kline"] if isinstance(item, dict) else item
        mask = kline["date"] <= date
        return kline[mask].copy()

    def _compute_rankings_for_date(self, date):
        """对给定日期运行因子管线，返回排序结果"""
        # 为每只股票切片K线
        sliced_klines = {}
        for code in self.kline_map:
            df = self._get_kline_upto(code, date)
            if len(df) >= 60:  # 至少需要60天计算指标
                sliced_klines[code] = {
                    "info": self.kline_map[code].get("info", {}),
                    "kline": df,
                }

        if len(sliced_klines) < 3:
            return None

        # 计算因子
        raw_df = compute_factors_batch(
            sliced_klines,
            self.fundamentals_map,
            stock_sector_map=self.stock_sector_map
        )

        if raw_df.empty or len(raw_df) < 3:
            return None

        # 处理因子 + 打分
        processed_df = process_factor_panel(raw_df)
        # Determine regime for this date
        regime = self._get_regime_for_date(date)
        scored_df = compute_composite_score(processed_df, regime=regime)

        return scored_df.sort_values("alpha_score", ascending=False)

    def _get_regime_for_date(self, date):
        """Determine which factor weight regime to use for a given date."""
        # If we have a pre-computed regime map, use it
        if self.regime_map:
            regime_label = self.regime_map.get(date)
            # Map bull/choppy/bear to normal/defensive/offensive weight schemes
            # Only use defensive in bear markets
            if regime_label == "bear":
                return "defensive"
            else:
                return "normal"

        # Fallback: always use normal
        return "normal"

    def run(self):
        """执行回测"""
        print(f"\n开始回测...")
        print(f"  初始资金: {self.cfg['capital']:,.0f}")
        print(f"  选股数量: top {self.cfg['top_n']}")
        print(f"  止损: {self.cfg['stop_loss']*100:.0f}%  止盈: {self.cfg['take_profit']*100:.0f}%")
        print(f"  最大持仓: {self.cfg['max_positions']} 只")
        print()

        min_days = 60  # 需要足够历史数据
        start_idx = max(0, min_days)

        for i, date in enumerate(self.trading_days[start_idx:]):
            if i % 40 == 0:
                print(f"  进度: {date} ({i+1}/{len(self.trading_days)-start_idx})")

            # 1. 更新现有持仓（按当日收盘价）
            self._mark_to_market(date)

            # 2. 检查退出条件（止损/止盈/到期）
            self._check_exits(date)

            # 3. 选股
            rankings = self._compute_rankings_for_date(date)
            if rankings is None:
                self._record_equity(date)
                continue

            # 4. 买入（次日开盘价，当天选股次日买）
            if i < len(self.trading_days[start_idx:]) - 1:
                next_date = self.trading_days[start_idx + i + 1]
                self._enter_positions(rankings, date, next_date)

            # 5. 记录权益
            self._record_equity(date)

        # 最后一天清仓
        last_date = self.trading_days[-1]
        self._close_all(last_date)
        self._record_equity(last_date)

        return self._compute_stats()

    def _mark_to_market(self, date):
        """按当日收盘价更新持仓市值"""
        total = self.cash
        for pos in self.positions:
            item = self.kline_map.get(pos["code"])
            if not item:
                continue
            kline = item["kline"] if isinstance(item, dict) else item
            row = kline[kline["date"] == date]
            if row.empty:
                continue
            pos["current_price"] = float(row["close"].iloc[0])
            pos["current_value"] = pos["shares"] * pos["current_price"]
            pos["return_pct"] = (pos["current_price"] / pos["entry_price"] - 1)
            total += pos["current_value"]
        self.total_value = total

    def _check_exits(self, date):
        """检查止损/止盈/到期"""
        surviving = []
        for pos in self.positions:
            # Check if stock is at limit-down (cannot sell)
            item = self.kline_map.get(pos["code"])
            if item:
                kline = item["kline"] if isinstance(item, dict) else item
                fund = self.fundamentals_map.get(pos["code"], {})
                limit_down = fund.get("limit_down")
                current_price = pos.get("current_price", 0)
                if limit_down and limit_down > 0 and current_price > 0 and current_price <= limit_down * 1.005:
                    surviving.append(pos)
                    pos["hold_days"] = pos.get("hold_days", 0) + 1
                    continue  # 跌停卖不掉

            ret = pos.get("return_pct", 0.0)
            hold_days = pos.get("hold_days", 0)
            exit_reason = None

            if ret <= self.cfg["stop_loss"]:
                exit_reason = "止损"
            elif ret >= self.cfg["take_profit"]:
                exit_reason = "止盈"
            elif hold_days >= self.cfg["max_hold_days"]:
                exit_reason = "到期"

            if exit_reason:
                # 以次日开盘价或触发日收盘价平仓
                exit_price = pos.get("current_price", pos["entry_price"])
                exit_value = pos["shares"] * exit_price
                cost = exit_value * self.cfg["commission"]
                self.cash += exit_value - cost

                self.closed_trades.append({
                    "code": pos["code"],
                    "name": pos.get("name", ""),
                    "entry_date": pos["entry_date"],
                    "exit_date": date,
                    "entry_price": pos["entry_price"],
                    "exit_price": exit_price,
                    "return_pct": (exit_price / pos["entry_price"] - 1),
                    "exit_reason": exit_reason,
                    "hold_days": hold_days,
                    "entry_regime": pos.get("entry_regime", "unknown"),
                })
            else:
                pos["hold_days"] = hold_days + 1
                surviving.append(pos)

        self.positions = surviving

    def _enter_positions(self, rankings, signal_date, entry_date):
        """根据排名买入（次日开盘价成交）"""
        # 按排名取 top N，过滤已在持仓中的
        held_codes = {p["code"] for p in self.positions}
        candidates = rankings[~rankings["code"].isin(held_codes)]

        # 仓位计算
        slots_available = self.cfg["max_positions"] - len(self.positions)
        to_buy = min(self.cfg["top_n"], slots_available, len(candidates))

        if to_buy <= 0:
            return

        position_size = self.total_value * self.cfg["single_position_pct"]

        for _, row in candidates.head(to_buy).iterrows():
            code = row["code"]
            item = self.kline_map.get(code)
            if not item:
                continue

            kline = item["kline"] if isinstance(item, dict) else item
            row_k = kline[kline["date"] == entry_date]
            if row_k.empty:
                continue

            buy_price = float(row_k["open"].iloc[0])

            # Check limit-up: cannot buy at limit-up
            fund = self.fundamentals_map.get(code, {})
            limit_up = fund.get("limit_up")
            if limit_up and limit_up > 0 and buy_price >= limit_up * 0.995:
                continue  # 涨停买不到

            # 滑点
            buy_price *= (1 + self.cfg["slippage"])

            # 计算买入股数（整百）
            amount = min(position_size, self.cash * 0.95)
            shares = int(amount / buy_price / 100) * 100
            if shares < 100:
                continue

            cost = shares * buy_price * (1 + self.cfg["commission"])
            if cost > self.cash:
                shares = int(self.cash * 0.95 / buy_price / 100) * 100
                cost = shares * buy_price * (1 + self.cfg["commission"])

            if shares < 100:
                continue

            self.cash -= cost
            # Get regime for entry date
            entry_regime = self.regime_map.get(entry_date, "choppy") if self.regime_map else "unknown"
            self.positions.append({
                "code": code,
                "name": row.get("name", ""),
                "sector": row.get("sector", ""),
                "entry_date": entry_date,
                "entry_price": buy_price,
                "shares": shares,
                "current_price": buy_price,
                "current_value": shares * buy_price,
                "return_pct": 0.0,
                "hold_days": 0,
                "signal_score": row.get("alpha_score", 0),
                "entry_regime": entry_regime,
            })

    def _close_all(self, date):
        """清空所有持仓"""
        for pos in list(self.positions):
            exit_price = pos.get("current_price", pos["entry_price"])
            exit_value = pos["shares"] * exit_price
            cost = exit_value * self.cfg["commission"]
            self.cash += exit_value - cost
            self.closed_trades.append({
                "code": pos["code"],
                "name": pos.get("name", ""),
                "entry_date": pos["entry_date"],
                "exit_date": date,
                "entry_price": pos["entry_price"],
                "exit_price": exit_price,
                "return_pct": (exit_price / pos["entry_price"] - 1),
                "exit_reason": "回测结束",
                "hold_days": pos.get("hold_days", 0),
                "entry_regime": pos.get("entry_regime", "unknown"),
            })
        self.positions = []

    def _record_equity(self, date):
        total = self.cash + sum(
            p.get("current_value", p["shares"] * p["entry_price"])
            for p in self.positions
        )
        self.total_value = total
        regime = self.regime_map.get(date, "unknown") if self.regime_map else "unknown"
        self.equity_curve.append({
            "date": date,
            "equity": total,
            "cash": self.cash,
            "positions": len(self.positions),
            "regime": regime,
        })

    def _compute_stats(self):
        """计算回测统计"""
        trades = self.closed_trades
        equity = pd.DataFrame(self.equity_curve)

        if equity.empty:
            return {"error": "无数据"}

        # 基本统计
        total_return = (equity["equity"].iloc[-1] / self.cfg["capital"] - 1)
        n_trades = len(trades)
        winning = [t for t in trades if t["return_pct"] > 0]
        n_win = len(winning)
        win_pct = n_win / n_trades if n_trades > 0 else 0

        avg_win = np.mean([t["return_pct"] for t in winning]) if winning else 0
        losing = [t for t in trades if t["return_pct"] <= 0]
        avg_loss = np.mean([t["return_pct"] for t in losing]) if losing else 0
        profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else float("inf")

        # 日收益率
        equity["daily_return"] = equity["equity"].pct_change()
        daily_ret = equity["daily_return"].dropna()

        # 夏普比率（年化）
        risk_free = 0.02 / 252
        sharpe = (daily_ret.mean() - risk_free) / daily_ret.std() * np.sqrt(252) if daily_ret.std() > 0 else 0

        # 最大回撤
        cummax = equity["equity"].cummax()
        drawdown = (equity["equity"] - cummax) / cummax
        max_dd = drawdown.min()

        # 年化收益
        days = len(equity)
        annual_return = (1 + total_return) ** (252 / days) - 1 if days > 0 else 0

        # 最大连续亏损次数
        max_consecutive_loss = 0
        consecutive = 0
        for t in trades:
            if t["return_pct"] <= 0:
                consecutive += 1
                max_consecutive_loss = max(max_consecutive_loss, consecutive)
            else:
                consecutive = 0

        # 按退出原因统计
        exit_counts = {}
        for t in trades:
            reason = t.get("exit_reason", "未知")
            exit_counts[reason] = exit_counts.get(reason, 0) + 1

        # 基准比较（等权持有全部股票）
        bench_return = self._benchmark_return()

        # 分市况统计（如果有 regime_map）
        regime_stats = {}
        if self.regime_map:
            for regime_name in ["bull", "choppy", "bear"]:
                regime_trades = [t for t in trades if t.get("entry_regime") == regime_name]
                if regime_trades:
                    regime_win = [t for t in regime_trades if t["return_pct"] > 0]
                    regime_n_win = len(regime_win)
                    regime_win_pct = regime_n_win / len(regime_trades)
                    regime_avg_ret = np.mean([t["return_pct"] for t in regime_trades])
                    regime_stats[regime_name] = {
                        "n_trades": len(regime_trades),
                        "win_rate": regime_win_pct,
                        "avg_return": regime_avg_ret,
                    }

        stats = {
            "period": f"{self.trading_days[0]} ~ {self.trading_days[-1]}",
            "total_return": total_return,
            "annual_return": annual_return,
            "benchmark_return": bench_return,
            "alpha": total_return - bench_return,
            "total_trades": n_trades,
            "win_rate": win_pct,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "profit_loss_ratio": profit_factor,
            "max_drawdown": max_dd,
            "sharpe_ratio": sharpe,
            "max_consecutive_loss": max_consecutive_loss,
            "exit_distribution": exit_counts,
            "final_equity": equity["equity"].iloc[-1],
            "n_days": len(equity),
            "by_regime": regime_stats,
        }
        return stats

    def _benchmark_return(self):
        """等权持有所有股票的基准收益"""
        first_day = self.trading_days[0]
        last_day = self.trading_days[-1]
        returns = []
        for code, item in self.kline_map.items():
            kline = item["kline"] if isinstance(item, dict) else item
            start_row = kline[kline["date"] == first_day]
            end_row = kline[kline["date"] == last_day]
            if not start_row.empty and not end_row.empty:
                ret = float(end_row["close"].iloc[0]) / float(start_row["open"].iloc[0]) - 1
                returns.append(ret)
        return np.mean(returns) if returns else 0

    def run_with_split(self):
        """Run backtest with in-sample/out-of-sample split."""
        if not self.split_date:
            print("No split_date set, running full period")
            return self.run()

        in_sample_days = [d for d in self.trading_days if d < self.split_date]
        out_sample_days = [d for d in self.trading_days if d >= self.split_date]

        print(f"\n样本内: {in_sample_days[0]} ~ {in_sample_days[-1]} ({len(in_sample_days)} 天)")
        print(f"样本外: {out_sample_days[0]} ~ {out_sample_days[-1]} ({len(out_sample_days)} 天)")

        # In-sample
        original_days = self.trading_days
        self.trading_days = in_sample_days
        is_stats = self._run_period("样本内")

        # Reset state for out-of-sample
        self.cash = self.cfg["capital"]
        self.total_value = self.cash
        self.positions = []
        self.closed_trades = []
        self.equity_curve = []

        self.trading_days = out_sample_days
        oos_stats = self._run_period("样本外")

        self.trading_days = original_days
        return {"in_sample": is_stats, "out_sample": oos_stats}

    def _run_period(self, label):
        """Run backtest for a specific period."""
        print(f"\n[{label}] 开始回测...")
        min_days = 60
        start_idx = max(0, min_days)

        for i, date in enumerate(self.trading_days[start_idx:]):
            if i % 40 == 0:
                print(f"  [{label}] 进度: {date} ({i+1}/{len(self.trading_days)-start_idx})")

            self._mark_to_market(date)
            self._check_exits(date)

            rankings = self._compute_rankings_for_date(date)
            if rankings is None:
                self._record_equity(date)
                continue

            if i < len(self.trading_days[start_idx:]) - 1:
                next_date = self.trading_days[start_idx + i + 1]
                self._enter_positions(rankings, date, next_date)

            self._record_equity(date)

        last_date = self.trading_days[-1]
        self._close_all(last_date)
        self._record_equity(last_date)

        stats = self._compute_stats()
        stats["label"] = label
        return stats


# ============================================================
# CLI
# ============================================================

def main():
    ensure_dirs()

    print("=" * 60)
    print("  多因子选股模型 — 历史回测")
    print("=" * 60)

    # 1. 加载数据
    print("\n[1/4] 加载数据...")
    kline_map = load_universe_klines(watchlist_only=True, refresh=False)
    codes = list(kline_map.keys())
    fundamentals_map = load_fundamentals_for_codes(codes)

    # 2. 加载 news 板块数据
    print("[2/4] 加载板块情绪数据...")
    news_dir = ROOT.parent / "news" / "data" / "processed"
    # 找到所有可用的news数据日期
    news_dates = sorted([d.name for d in news_dir.glob("2026-*") if d.is_dir()])

    sector_map = {}
    universe = build_stock_universe(watchlist_only=True)
    if news_dates:
        # 使用最近一个交易日的情绪数据（假设情绪短期稳定）
        latest_news_date = news_dates[-1]
        sector_map = load_news_sector_data(latest_news_date)
        print(f"  使用 news 数据: {latest_news_date}, {len(sector_map)} 个板块")
    stock_sector_map = map_stocks_to_sectors(universe, sector_map)

    # Load index data for regime classification
    index_path = ROOT / "data" / "stocks" / "INDEX_1000300.parquet"
    regime_map = None
    if index_path.exists():
        index_df = pd.read_parquet(index_path)
        regime_map = classify_regimes(index_df)
        print(f"  指数数据: {len(index_df)} 天, 划分 {len(set(regime_map.values()))} 种市况")

    # 3. 运行回测
    print("[3/4] 运行回测...")
    bt = Backtest(kline_map, fundamentals_map, stock_sector_map,
                  config=BACKTEST_CONFIG, regime_map=regime_map)
    stats = bt.run()

    # 4. 输出结果
    print("\n[4/4] 回测结果")
    print("=" * 60)
    print(f"""
  回测区间: {stats.get('period', 'N/A')}

  收益指标:
    总收益:      {stats['total_return']*100:+.2f}%
    年化收益:    {stats['annual_return']*100:+.2f}%
    基准收益:    {stats['benchmark_return']*100:+.2f}% (等权持有)
    超额Alpha:   {stats['alpha']*100:+.2f}%

  风险指标:
    夏普比率:    {stats['sharpe_ratio']:.2f}
    最大回撤:    {stats['max_drawdown']*100:.2f}%
    最大连亏:    {stats['max_consecutive_loss']} 笔

  交易统计:
    总交易:      {stats['total_trades']} 笔
    胜率:        {stats['win_rate']*100:.1f}%
    平均盈利:    {stats['avg_win']*100:+.2f}%
    平均亏损:    {stats['avg_loss']*100:+.2f}%
    盈亏比:      {stats['profit_loss_ratio']:.2f}
""")

    print("  退出分布:")
    for reason, count in stats.get("exit_distribution", {}).items():
        pct = count / max(stats['total_trades'], 1) * 100
        print(f"    {reason}: {count} 笔 ({pct:.0f}%)")

    if stats.get("by_regime"):
        print("\n  分市况表现:")
        regime_names = {"bull": "上涨市", "choppy": "震荡市", "bear": "下跌市"}
        for rname, rlabel in regime_names.items():
            rs = stats["by_regime"].get(rname)
            if rs:
                print(f"    {rlabel}: {rs['n_trades']}笔 胜率{rs['win_rate']*100:.0f}% 均值{rs['avg_return']*100:+.2f}%")

    # 保存交易记录
    trades_df = pd.DataFrame(bt.closed_trades)
    out_dir = ROOT / "data" / "backtest"
    out_dir.mkdir(parents=True, exist_ok=True)
    trades_df.to_csv(out_dir / "trades.csv", index=False)
    pd.DataFrame(bt.equity_curve).to_csv(out_dir / "equity.csv", index=False)

    print(f"\n交易记录: {out_dir / 'trades.csv'}")
    print(f"权益曲线: {out_dir / 'equity.csv'}")

    # 近期交易展示
    if not trades_df.empty:
        print("\n最近 10 笔交易:")
        recent = trades_df.tail(10)[["code", "name", "entry_date", "exit_date",
                                      "return_pct", "exit_reason", "hold_days"]]
        recent["return_pct"] = recent["return_pct"].apply(lambda x: f"{x*100:+.2f}%")
        print(recent.to_string(index=False))


if __name__ == "__main__":
    main()
