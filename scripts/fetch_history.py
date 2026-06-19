#!/usr/bin/env python3
"""
历史数据批量获取脚本
下载 CSI 300 + 中证 500 成分股的历史 K 线 + 指数数据
用于回测
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from mootdx.quotes import Quotes

ROOT = Path(__file__).resolve().parent.parent
STOCK_DIR = ROOT / "data" / "stocks"
STOCK_DIR.mkdir(parents=True, exist_ok=True)

SESSION = requests.Session()
SESSION.headers.update({
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
    "Accept-Language": "zh-CN,zh;q=0.9",
})

TDX = Quotes.factory(market="standard", timeout=15)


def fetch_kline(code: str, market: int = 0, count: int = 500) -> pd.DataFrame:
    """获取日K线"""
    try:
        df = TDX.bars(symbol=code, frequency=9, start=0, offset=count)
        if df is None or df.empty:
            return pd.DataFrame()
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
        return df[["date", "open", "close", "high", "low", "volume", "amount"]]
    except Exception as e:
        print(f"  [ERR] {code}: {e}")
        return pd.DataFrame()


def get_csi300_stocks() -> list[dict]:
    """获取沪深300成分股列表（通过东方财富）"""
    stocks = []
    try:
        url = "https://push2.eastmoney.com/api/qt/clist/get"
        params = {
            "pn": 1, "pz": 500,
            "fs": "b:IF300",  # 沪深300
            "fields": "f2,f12,f14",
            "np": 1,
        }
        resp = SESSION.get(url, params=params, timeout=15)
        data = resp.json()
        if data.get("data") and data["data"].get("diff"):
            for item in data["data"]["diff"]:
                code = item.get("f12", "")
                market = 1 if code.startswith("6") else 0
                stocks.append({
                    "code": code,
                    "name": item.get("f14", ""),
                    "market": market,
                    "index": "CSI300",
                })
    except Exception as e:
        print(f"  [ERR] 沪深300列表: {e}")
    return stocks


def get_csi500_stocks() -> list[dict]:
    """获取中证500成分股"""
    stocks = []
    try:
        url = "https://push2.eastmoney.com/api/qt/clist/get"
        params = {
            "pn": 1, "pz": 600,
            "fs": "b:IF500",
            "fields": "f2,f12,f14",
            "np": 1,
        }
        resp = SESSION.get(url, params=params, timeout=15)
        data = resp.json()
        if data.get("data") and data["data"].get("diff"):
            for item in data["data"]["diff"]:
                code = item.get("f12", "")
                market = 1 if code.startswith("6") else 0
                stocks.append({
                    "code": code,
                    "name": item.get("f14", ""),
                    "market": market,
                    "index": "CSI500",
                })
    except Exception as e:
        print(f"  [ERR] 中证500列表: {e}")
    return stocks


def fetch_index_history():
    """下载主要指数历史 K 线（东方财富 API）"""
    index_map = {
        "1.000001": "上证指数",
        "0.399001": "深证成指",
        "0.399006": "创业板指",
        "1.000688": "科创50",
        "1.000300": "沪深300",
        "0.399905": "中证500",
        "1.000852": "中证1000",
    }
    print("\n===== 指数历史数据 =====")
    for secid, name in index_map.items():
        path = STOCK_DIR / f"INDEX_{secid.replace('.', '')}.parquet"
        if path.exists():
            df = pd.read_parquet(path)
            print(f"  {name}: 已有 {len(df)} 条")
            continue
        try:
            url = "https://push2his.eastmoney.com/api/qt/stock/kline/get"
            params = {
                "secid": secid,
                "fields1": "f1,f2,f3,f4,f5,f6",
                "fields2": "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61",
                "klt": 101,  # 日线
                "fqt": 1,    # 前复权
                "end": "20500101",
                "lmt": 1000,
            }
            resp = SESSION.get(url, params=params, timeout=30)
            data = resp.json()
            if not data.get("data") or not data["data"].get("klines"):
                print(f"  [SKIP] {name}: 无数据")
                continue

            rows = []
            for line in data["data"]["klines"]:
                parts = line.split(",")
                rows.append({
                    "date": parts[0],
                    "open": float(parts[1]),
                    "close": float(parts[2]),
                    "high": float(parts[3]),
                    "low": float(parts[4]),
                    "volume": float(parts[5]),
                    "amount": float(parts[6]),
                })
            df = pd.DataFrame(rows)
            df.to_parquet(path, index=False)
            print(f"  {name}: {len(df)} 条, {df['date'].iloc[0]} ~ {df['date'].iloc[-1]}")
        except Exception as e:
            print(f"  [ERR] {name}: {e}")
        time.sleep(0.5)


def fetch_stock_history(stocks: list[dict], batch_name: str, count: int = 500):
    """批量下载个股历史K线"""
    print(f"\n===== {batch_name} ({len(stocks)} 只) =====")
    success, skip, fail = 0, 0, 0
    for i, s in enumerate(stocks):
        path = STOCK_DIR / f"{s['code']}.parquet"
        if path.exists() and not _should_refresh(path):
            existing = pd.read_parquet(path)
            if len(existing) > 200:
                skip += 1
                continue
        time.sleep(0.12)
        df = fetch_kline(s["code"], s.get("market", 0), count)
        if not df.empty:
            df.to_parquet(path, index=False)
            success += 1
            if success % 50 == 0:
                print(f"  进度: {success + skip}/{len(stocks)}")
        else:
            fail += 1
    print(f"  完成: 新增{success}, 已有{skip}, 失败{fail}")


def _should_refresh(path: Path) -> bool:
    """超过3天的缓存刷新"""
    import os
    age = time.time() - os.path.getmtime(str(path))
    return age > 86400 * 3


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"

    print("历史数据获取")
    print("=" * 50)

    # 1. 指数
    if mode in ("all", "index"):
        fetch_index_history()

    # 2. 沪深300
    if mode in ("all", "csi300"):
        stocks_300 = get_csi300_stocks()
        fetch_stock_history(stocks_300, "沪深300", count=500)

    # 3. 中证500
    if mode in ("all", "csi500"):
        stocks_500 = get_csi500_stocks()
        fetch_stock_history(stocks_500, "中证500", count=500)

    # 统计
    parquets = list(STOCK_DIR.glob("*.parquet"))
    non_index = [p for p in parquets if not p.stem.startswith("INDEX_")]
    print(f"\n总计: {len(parquets)} 个parquet文件 ({len(non_index)} 个股, {len(parquets) - len(non_index)} 指数)")
