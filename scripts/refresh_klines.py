#!/usr/bin/env python3
"""
增量刷新K线缓存：仅追加缺失的最新交易日，不全量重拉。
使用 data_loader.fetch_kline_tdx 保证与主流程一致的解析逻辑。
"""
import os, sys, time
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
STOCK_DIR = ROOT / "data" / "stocks"

from data_loader import fetch_kline_tdx

def main():
    target_date = sys.argv[1] if len(sys.argv) > 1 else "2026-06-22"

    files = sorted([f for f in os.listdir(STOCK_DIR) if f.endswith('.parquet') and not f.startswith('INDEX_')])
    updated = 0
    skipped = 0
    errors = 0
    total = len(files)

    print(f"增量刷新K线: {total} 个缓存文件 → 目标日期 {target_date}")

    t_start = time.time()
    for i, fname in enumerate(files):
        code = fname.replace('.parquet', '')
        market = 1 if code.startswith('6') else 0
        cache_path = STOCK_DIR / fname

        try:
            cached = pd.read_parquet(cache_path)
            max_date = str(cached['date'].max())

            if max_date >= target_date:
                skipped += 1
                if i % 1000 == 0 and i > 0:
                    elapsed = time.time() - t_start
                    print(f"  {i}/{total} | 更新{updated} 跳过{skipped} 错误{errors} | {elapsed:.0f}s")
                continue

            time.sleep(0.03)
            fresh = fetch_kline_tdx(code, market, count=5)

            if fresh.empty:
                errors += 1
                continue

            combined = pd.concat([cached, fresh], ignore_index=True)
            combined = combined.drop_duplicates(subset=['date'], keep='last')
            combined = combined.sort_values('date').reset_index(drop=True)
            combined.to_parquet(cache_path, index=False)
            updated += 1

        except Exception:
            errors += 1

        if i % 1000 == 0 and i > 0:
            elapsed = time.time() - t_start
            rate = i / elapsed
            remaining = (total - i) / rate
            print(f"  {i}/{total} | 更新{updated} 跳过{skipped} 错误{errors} | {elapsed:.0f}s | 预计剩余{remaining:.0f}s")

    elapsed = time.time() - t_start
    print(f"\n完成: 更新 {updated} 只, 跳过(已最新) {skipped} 只, 错误 {errors} 只 | 耗时 {elapsed:.0f}s")

if __name__ == "__main__":
    main()
