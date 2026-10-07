#!/usr/bin/env python3
"""
用新浪日K补齐指数(不复权) data/stocks/INDEX_*.parquet，修正历史遗留的 market 标错。
背景：mootdx 空 / 东财 K线封IP / 腾讯 WAF 封；新浪 quotes.sina.cn 稳定。
      原 INDEX_1000300.parquet 被写成平安银行(000001深市股)，应为沪深300(1.000300)。

映射(东财 secid -> 新浪 symbol)：
  INDEX_1000001 上证指数   sh000001
  INDEX_1000300 沪深300    sh000300
  INDEX_1000852 中证800    sh000852
  INDEX_1000688 科创50     sh000688
  INDEX_0399001 深证成指   sz399001
  INDEX_0399006 创业板指   sz399006
  INDEX_0399905 中证500    sz399905
"""
import json, subprocess, time, sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
STOCK_DIR = ROOT / "data" / "stocks"

SYM = {
    'INDEX_1000001': 'sh000001',   # 上证指数
    'INDEX_1000300': 'sh000300',   # 沪深300
    'INDEX_1000852': 'sh000852',   # 中证800
    'INDEX_1000688': 'sh000688',   # 科创50
    'INDEX_0399001': 'sz399001',   # 深证成指
    'INDEX_0399006': 'sz399006',   # 创业板指
    'INDEX_0399905': 'sz399905',   # 中证500
}


def fetch(sym, datalen=1000):
    url = (f'https://quotes.sina.cn/cn/api/json_v2.php/CN_MarketDataService.getKLineData'
           f'?symbol={sym}&scale=240&ma=no&datalen={datalen}')
    out = subprocess.run(['curl', '-s', '--max-time', '12',
                          '-H', 'User-Agent: Mozilla/5.0', url],
                         capture_output=True, text=True, timeout=15).stdout
    return json.loads(out)


def main():
    for fname, sym in SYM.items():
        path = STOCK_DIR / f'{fname}.parquet'
        days = fetch(sym)
        if not isinstance(days, list) or not days:
            print(f'{fname} <- {sym}  失败/空')
            continue
        rows = []
        for r in days:
            try:
                v = float(r['volume'])
                c = float(r['close'])
                rows.append((r['day'], float(r['open']), c, float(r['high']),
                             float(r['low']), v, round(v * c, 2)))
            except (KeyError, ValueError):
                continue
        df = pd.DataFrame(rows, columns=['date', 'open', 'close', 'high', 'low', 'volume', 'amount'])
        df = df.sort_values('date').reset_index(drop=True)
        df.to_parquet(path, index=False)
        print(f'{fname} <- {sym}  {df["date"].min()} ~ {df["date"].max()}  close={df["close"].iloc[-1]}')
        time.sleep(0.2)


if __name__ == '__main__':
    main()
