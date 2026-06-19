#!/usr/bin/env python3
"""
批量下载沪深300+中证500成分股历史K线数据。
增量下载——已有 parquet 缓存的跳过。
"""

import json
import time
from pathlib import Path

import pandas as pd
from mootdx.quotes import Quotes

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "stocks"
DATA_DIR.mkdir(parents=True, exist_ok=True)

TDX = Quotes.factory(market="standard", timeout=15)


def fetch_kline(code, market, count=300):
    """下载个股日K线"""
    try:
        df = TDX.bars(symbol=code, frequency=9, start=0, offset=count)
        if df is None or df.empty:
            return None
        df = df.rename(columns={
            "datetime": "date", "open": "open", "close": "close",
            "high": "high", "low": "low", "volume": "volume",
        })
        df["date"] = df["date"].astype(str).str[:10]
        for col in ["open", "close", "high", "low", "volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        if "amount" in df.columns:
            df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
        else:
            df["amount"] = 0.0
        df = df.dropna(subset=["open", "close"])
        df = df.sort_values("date").reset_index(drop=True)
        if len(df) < 60:
            return None
        return df[["date", "open", "close", "high", "low", "volume", "amount"]]
    except Exception as e:
        return None


def main():
    with open(ROOT / "data" / "index_stocks.json") as f:
        codes = json.load(f)

    print(f"总计 {len(codes)} 只待下载")

    new_count = 0
    skip_count = 0
    fail_count = 0

    for i, code in enumerate(codes):
        cache_path = DATA_DIR / f"{code}.parquet"
        if cache_path.exists():
            skip_count += 1
            continue

        market = 1 if code.startswith("6") else 0
        df = fetch_kline(code, market, count=300)

        if df is not None and len(df) >= 60:
            df.to_parquet(cache_path, index=False)
            new_count += 1
        else:
            fail_count += 1

        if (i + 1) % 20 == 0:
            print(f"  进度: {i+1}/{len(codes)}  新增{new_count}  跳过{skip_count}  失败{fail_count}")

        time.sleep(0.12)  # 限速

    print(f"\n完成: 新增{new_count}  已有{skip_count}  失败{fail_count}")


if __name__ == "__main__":
    main()
