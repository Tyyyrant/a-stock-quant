#!/usr/bin/env python3
"""
数据加载层 — V3 (a-stock-data refactored)
- 行情: mootdx (K线, TCP, 不封IP) + 腾讯财经 (PE/PB/市值/换手率, HTTP, 不封IP)
- 板块分类: 东财 slist (概念/行业/地域归属)
- 北向资金: 同花顺 hsgtApi + 本地CSV自缓存
- 融资融券/龙虎榜/大宗交易: 东财 datacenter (已内置 em_get 限流)
- 基本面: 腾讯财经(优先) → 东财 push2(fallback)
- 财务: mootdx finance (ROE/EPS等季报数据)
- 新闻/板块情绪: 来自 news 项目 JSON

数据源优先级: mootdx > 腾讯 > 东财(仅独有数据, 通过 em_get 限流)
"""

import json
import os
import sys
import time
import random
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import requests
import urllib.request
import yaml
from mootdx.quotes import Quotes

ROOT = Path(__file__).resolve().parent.parent
NEWS_ROOT = ROOT.parent / "news"
DATA_DIR = ROOT / "data"
STOCK_CACHE_DIR = DATA_DIR / "stocks"

# ============================================================
# HTTP Sessions
# ============================================================

SESSION = requests.Session()
SESSION.headers.update({
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
    "Accept-Language": "zh-CN,zh;q=0.9",
})

TDX = Quotes.factory(market="standard", timeout=15)

# ============================================================
# 东财防封基础设施 (来自 a-stock-data §共用 helper)
# ============================================================

UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"

EM_SESSION = requests.Session()
EM_SESSION.headers.update({
    "User-Agent": UA,
    "Connection": "close",      # 禁用 Keep-Alive，避免 eastmoney 断开连接
})
EM_MIN_INTERVAL = 1.5          # 两次东财请求最小间隔(秒)；遇到断开时自动加大
_em_last_call = [0.0]          # 模块级上次请求时间戳

DATACENTER_URL = "https://datacenter-web.eastmoney.com/api/data/v1/get"


def em_get(url: str, params: Optional[dict] = None, headers: Optional[dict] = None,
           timeout: int = 20, max_retries: int = 2, **kwargs):
    """东财统一请求入口：自动节流 + 复用 session + 重试 + 默认 UA。
    所有 eastmoney.com 接口都应通过它请求，避免高频被封 IP。"""
    wait = EM_MIN_INTERVAL - (time.time() - _em_last_call[0])
    if wait > 0:
        time.sleep(wait + random.uniform(0.1, 0.5))

    merged_headers = {"User-Agent": UA, "Connection": "close"}
    if headers:
        merged_headers.update(headers)

    last_error = None
    for attempt in range(max_retries + 1):
        try:
            resp = EM_SESSION.get(url, params=params, headers=merged_headers,
                                  timeout=timeout, **kwargs)
            _em_last_call[0] = time.time()
            return resp
        except Exception as e:
            last_error = e
            if attempt < max_retries:
                backoff = (attempt + 1) * 2
                print(f"    [RETRY] {url[:60]}... ({attempt+1}/{max_retries}) 等待{backoff}s: {e}")
                time.sleep(backoff)
                # 重建 session
                try:
                    EM_SESSION.close()
                except Exception:
                    pass
                EM_SESSION.headers.update({"User-Agent": UA, "Connection": "close"})

    _em_last_call[0] = time.time()
    raise last_error


def eastmoney_datacenter(report_name: str, columns: str = "ALL",
                          filter_str: str = "", page_size: int = 50,
                          sort_columns: str = "", sort_types: str = "-1") -> list[dict]:
    """东财数据中心统一查询 — 龙虎榜/解禁/融资融券/大宗交易/股东户数/分红 共用（已内置限流）"""
    params = {
        "reportName": report_name, "columns": columns,
        "filter": filter_str, "pageNumber": "1", "pageSize": str(page_size),
        "sortColumns": sort_columns, "sortTypes": sort_types,
        "source": "WEB", "client": "WEB",
    }
    r = em_get(DATACENTER_URL, params=params, timeout=15)
    d = r.json()
    if d.get("result") and d["result"].get("data"):
        return d["result"]["data"]
    return []


# ============================================================
# 目录初始化
# ============================================================

def ensure_dirs():
    STOCK_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    (DATA_DIR / "factor_cache").mkdir(parents=True, exist_ok=True)
    (DATA_DIR / "signals").mkdir(parents=True, exist_ok=True)
    (DATA_DIR / "backtest").mkdir(parents=True, exist_ok=True)
    # 北向资金缓存目录
    (Path.home() / ".tradingagents" / "cache").mkdir(parents=True, exist_ok=True)


# ============================================================
# news 项目数据
# ============================================================

def load_news_data(date: str) -> dict:
    """加载 news 项目的所有 processed 数据"""
    base = NEWS_ROOT / "data" / "processed" / date
    result = {}
    for fname in ["market_data.json", "sector_stocks.json",
                   "news_impact.json", "analysis.json"]:
        fp = base / fname
        if fp.exists():
            with open(fp) as f:
                result[fname.replace(".json", "")] = json.load(f)
    return result


def load_watchlist() -> dict:
    with open(NEWS_ROOT / "config" / "watchlist.yaml") as f:
        return yaml.safe_load(f)


# ============================================================
# K 线数据 (mootdx TCP, 不封IP)
# ============================================================

def fetch_kline_tdx(code: str, market: int, count: int = 250) -> pd.DataFrame:
    """mootdx 获取个股日 K 线，返回 DataFrame"""
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
        print(f"  [WARN] K线 {code}: {e}")
        return pd.DataFrame()


def get_stock_kline(code: str, market: int = 0,
                    refresh: bool = False) -> pd.DataFrame:
    """获取个股日线（缓存优先）"""
    cache_path = STOCK_CACHE_DIR / f"{code}.parquet"

    if not refresh and cache_path.exists():
        cached = pd.read_parquet(cache_path)
        if not cached.empty:
            return cached

    df = fetch_kline_tdx(code, market, count=300)
    if not df.empty:
        df.to_parquet(cache_path, index=False)
    return df


def batch_fetch_klines(codes: list[tuple[str, int]],
                       refresh: bool = False) -> dict[str, pd.DataFrame]:
    """批量获取 K 线，返回 {code: DataFrame}"""
    results = {}
    for i, (code, market) in enumerate(codes):
        if i > 0:
            time.sleep(0.15)  # 限速（mootdx TCP 不需要太严格）
        df = get_stock_kline(code, market, refresh=refresh)
        if not df.empty:
            results[code] = df
        else:
            print(f"  [SKIP] {code}: 无K线数据")
    return results


# ============================================================
# 腾讯财经 — PE/PB/市值/换手率/涨跌停 (HTTP, 不封IP) — a-stock-data §1.2
# ============================================================

def tencent_quote(codes: list[str]) -> dict[str, dict]:
    """
    批量拉取腾讯财经实时行情。
    支持个股、指数、ETF。
    返回: {code: {name, price, pe_ttm, pb, mcap_yi, float_mcap_yi,
                  turnover_pct, limit_up, limit_down, change_pct, vol_ratio, ...}}
    """
    prefixed = []
    for c in codes:
        if c.startswith(("6", "9")):
            prefixed.append(f"sh{c}")
        elif c.startswith("8"):
            prefixed.append(f"bj{c}")
        else:
            prefixed.append(f"sz{c}")

    # 分批，每批最多 50 个（避免 URL 过长）
    result = {}
    batch_size = 50
    for i in range(0, len(prefixed), batch_size):
        batch = prefixed[i:i + batch_size]
        url = "https://qt.gtimg.cn/q=" + ",".join(batch)
        try:
            req = urllib.request.Request(url)
            req.add_header("User-Agent", "Mozilla/5.0")
            resp = urllib.request.urlopen(req, timeout=10)
            data = resp.read().decode("gbk")

            for line in data.strip().split(";"):
                if not line.strip() or "=" not in line or '"' not in line:
                    continue
                key = line.split("=")[0].split("_")[-1]
                vals = line.split('"')[1].split("~")
                if len(vals) < 53:
                    continue
                code = key[2:]
                result[code] = {
                    "name":         vals[1],
                    "price":        float(vals[3]) if vals[3] else 0,
                    "last_close":   float(vals[4]) if vals[4] else 0,
                    "open":         float(vals[5]) if vals[5] else 0,
                    "change_amt":   float(vals[31]) if vals[31] else 0,
                    "change_pct":   float(vals[32]) if vals[32] else 0,
                    "high":         float(vals[33]) if vals[33] else 0,
                    "low":          float(vals[34]) if vals[34] else 0,
                    "amount_wan":   float(vals[37]) if vals[37] else 0,
                    "turnover_pct": float(vals[38]) if vals[38] else 0,
                    "pe_ttm":       float(vals[39]) if vals[39] else 0,
                    "amplitude_pct":float(vals[43]) if vals[43] else 0,
                    "mcap_yi":      float(vals[44]) if vals[44] else 0,  # 总市值(亿)
                    "float_mcap_yi":float(vals[45]) if vals[45] else 0,  # 流通市值(亿)
                    "pb":           float(vals[46]) if vals[46] else 0,
                    "limit_up":     float(vals[47]) if vals[47] else 0,
                    "limit_down":   float(vals[48]) if vals[48] else 0,
                    "vol_ratio":    float(vals[49]) if vals[49] else 0,
                    "pe_static":    float(vals[52]) if vals[52] else 0,
                }
        except Exception as e:
            print(f"  [WARN] 腾讯行情批次: {e}")
        if i + batch_size < len(prefixed):
            time.sleep(0.1)

    return result


def tencent_fundamentals(codes: list[str]) -> dict[str, dict]:
    """
    通过腾讯财经获取基本面数据（PE/PB/市值/换手率/涨跌停）。
    不封IP，优先于东财 push2。
    """
    quotes = tencent_quote(codes)
    result = {}
    for code, q in quotes.items():
        result[code] = {
            "name": q.get("name", ""),
            "price": q.get("price"),
            "change_pct": q.get("change_pct"),
            "pe": q.get("pe_ttm") if q.get("pe_ttm") and q["pe_ttm"] > 0 else None,
            "pb": q.get("pb") if q.get("pb") and q["pb"] > 0 else None,
            "market_cap": q.get("mcap_yi", 0) * 1e8 if q.get("mcap_yi") else None,  # 亿→元
            "circ_market_cap": q.get("float_mcap_yi", 0) * 1e8 if q.get("float_mcap_yi") else None,
            "turnover_pct": q.get("turnover_pct"),
            "limit_up": q.get("limit_up"),
            "limit_down": q.get("limit_down"),
            "vol_ratio": q.get("vol_ratio"),
            "source": "tencent",
        }
    return result


# ============================================================
# 东财 push2 基本面 (fallback) — 用于腾讯拿不到的字段 (ROE/营收同比等)
# ============================================================

def fetch_fundamentals_batch_eastmoney(codes: list[str]) -> dict[str, dict]:
    """
    通过东财 push2 获取补充基本面数据（ROE/营收同比/利润同比/资产负债率）。
    仅在腾讯财经缺数据时作为 fallback。
    """
    results = {}
    if not codes:
        return results

    for i in range(0, len(codes), 50):
        batch = codes[i:i+50]
        code_str = ",".join(
            f"1.{c}" if c.startswith("6") else f"0.{c}" for c in batch
        )
        try:
            url = "https://push2.eastmoney.com/api/qt/ulist.np/get"
            params = {
                "fltt": 2,
                "fields": "f2,f3,f9,f12,f14,f15,f16,f17,f18,f20,f21,f23,f37,f38,f39,f40,f41,f42,f43,f44,f45,f46",
                "secids": code_str,
            }
            r = em_get(url, params=params,
                       headers={"Referer": "https://quote.eastmoney.com/"}, timeout=15)
            data = r.json()
            if data.get("data") and data["data"].get("diff"):
                for item in data["data"]["diff"]:
                    code = item.get("f12", "")
                    pe_val = item.get("f9")
                    results[code] = {
                        "name": item.get("f14", ""),
                        "price": item.get("f2"),
                        "change_pct": item.get("f3"),
                        "pe": float(pe_val) if pe_val and str(pe_val) != "-" else None,
                        "pb": _safe_float(item.get("f23")),
                        "market_cap": _safe_float(item.get("f20")),
                        "circ_market_cap": _safe_float(item.get("f21")),
                        "roe": _safe_float(item.get("f37")),
                        "revenue_yoy": _safe_float(item.get("f44")),
                        "profit_yoy": _safe_float(item.get("f45")),
                        "debt_to_equity": _safe_float(item.get("f42")),
                        "source": "eastmoney",
                    }
        except Exception as e:
            print(f"  [WARN] 东财基本面批次: {e}")

    return results


def _safe_float(val):
    """安全转 float"""
    if val is None:
        return None
    try:
        v = float(val)
        return v if not np.isnan(v) else None
    except (ValueError, TypeError):
        return None


def load_fundamentals_for_codes(codes: list[str],
                                refresh: bool = False) -> dict[str, dict]:
    """
    为指定股票代码加载基本面（带缓存）。
    策略: 腾讯财经优先（不封IP）→ 东财 push2 补充 ROE/营收同比等
    """
    cache_path = DATA_DIR / "fundamentals_cache.json"
    cache = {}
    if not refresh and cache_path.exists():
        with open(cache_path) as f:
            cache = json.load(f)

    today = datetime.now().strftime("%Y-%m-%d")
    need_fetch = []
    for c in codes:
        entry = cache.get(c, {})
        if entry.get("date") != today or not entry.get("pe"):
            need_fetch.append(c)

    if need_fetch:
        print(f"  获取 {len(need_fetch)} 只股票的基本面（腾讯优先）...")

        # Step 1: 腾讯财经（PE/PB/市值/涨跌停）
        tencent_data = tencent_fundamentals(need_fetch)

        # Step 2: 找出腾讯缺 ROE 的股票，用东财补充
        missing_roe = [c for c in need_fetch
                       if c in tencent_data and tencent_data[c].get("market_cap") is not None]
        eastmoney_data = {}
        if missing_roe:
            eastmoney_data = fetch_fundamentals_batch_eastmoney(missing_roe)

        # 合并
        for c in need_fetch:
            t = tencent_data.get(c, {})
            e = eastmoney_data.get(c, {})
            merged = {"date": today, "source": "tencent+eastmoney"}

            # 优先用腾讯的数据
            merged.update({
                "name": t.get("name") or e.get("name", ""),
                "price": t.get("price"),
                "change_pct": t.get("change_pct"),
                "pe": t.get("pe") or e.get("pe"),
                "pb": t.get("pb") or e.get("pb"),
                "market_cap": t.get("market_cap") or e.get("market_cap"),
                "circ_market_cap": t.get("circ_market_cap") or e.get("circ_market_cap"),
                "turnover_pct": t.get("turnover_pct"),
                "limit_up": t.get("limit_up"),
                "limit_down": t.get("limit_down"),
            })

            # 东财独有字段
            merged.update({
                "roe": e.get("roe"),
                "revenue_yoy": e.get("revenue_yoy"),
                "profit_yoy": e.get("profit_yoy"),
                "debt_to_equity": e.get("debt_to_equity"),
            })

            cache[c] = merged

        with open(cache_path, "w") as f:
            json.dump(cache, f, ensure_ascii=False)

    return {c: cache.get(c, {}) for c in codes}


# ============================================================
# 东财 slist — 概念/行业/地域板块归属 — a-stock-data §3.3
# ============================================================

def eastmoney_concept_blocks(code: str) -> dict:
    """
    个股所属板块/概念归属（东财 slist，一次请求拿全，已内置限流）。
    boards 混合 行业/概念/地域，板块名自解释。
    """
    market_code = 1 if code.startswith("6") else 0
    params = {
        "fltt": "2", "invt": "2",
        "secid": f"{market_code}.{code}",
        "spt": "3", "pi": "0", "pz": "200", "po": "1",
        "fields": "f12,f14,f3,f128",
    }
    headers = {"User-Agent": UA, "Referer": "https://quote.eastmoney.com/"}
    try:
        r = em_get("https://push2.eastmoney.com/api/qt/slist/get",
                   params=params, headers=headers, timeout=15)
        d = r.json()
    except Exception as e:
        print(f"  [WARN] 东财板块归属 {code}: {e}")
        return {"total": 0, "boards": [], "concept_tags": []}

    diff = (d.get("data") or {}).get("diff") or {}
    items = diff.values() if isinstance(diff, dict) else diff
    boards = []
    for it in items:
        boards.append({
            "name": it.get("f14", ""),
            "code": it.get("f12", ""),
            "change_pct": it.get("f3", ""),
            "lead_stock": it.get("f128", ""),
        })
    return {
        "total": len(boards),
        "boards": boards,
        "concept_tags": [b["name"] for b in boards],
    }


def get_eastmoney_sector_map(codes: list[str]) -> dict[str, dict]:
    """
    批量获取股票板块归属。
    返回: {code: {sectors: [str], primary_sector: str, all_tags: [str]}}
    结果缓存到 sector_classification_eastmoney.json
    """
    cache_path = DATA_DIR / "sector_classification_eastmoney.json"
    cache = {}
    if cache_path.exists():
        with open(cache_path) as f:
            cache = json.load(f)

    # 检查哪些需要刷新（7天过期）
    today = datetime.now().strftime("%Y-%m-%d")
    need_fetch = []
    for c in codes:
        entry = cache.get(c, {})
        if entry.get("date") != today or not entry.get("primary_sector"):
            need_fetch.append(c)

    if need_fetch:
        print(f"  获取 {len(need_fetch)} 只股票的板块分类（东财 slist）...")
        for i, code in enumerate(need_fetch):
            if i > 0 and i % 10 == 0:
                print(f"    板块进度: {i}/{len(need_fetch)}")
            blocks = eastmoney_concept_blocks(code)
            tags = blocks.get("concept_tags", [])
            primary = tags[0] if tags else "未分类"
            cache[code] = {
                "date": today,
                "primary_sector": primary,
                "sectors": tags,
                "all_tags": tags,
                "boards": blocks.get("boards", []),
            }
        with open(cache_path, "w") as f:
            json.dump(cache, f, ensure_ascii=False)
        print(f"  板块分类完成: {len(cache)} 只")

    return {c: cache.get(c, {"primary_sector": "未分类", "sectors": [], "all_tags": []})
            for c in codes}


# ============================================================
# 同花顺北向资金 — hsgtApi + 本地CSV缓存 — a-stock-data §3.2
# ============================================================

HSGT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "Chrome/117.0.0.0 Safari/537.36"
    ),
    "Host": "data.hexin.cn",
    "Referer": "https://data.hexin.cn/",
}


def hsgt_realtime() -> pd.DataFrame:
    """
    沪深股通当日实时分钟流向。
    返回: {time, hgt_yi(沪股通累计净买入), sgt_yi(深股通累计净买入)} 单位: 亿元
    """
    url = "https://data.hexin.cn/market/hsgtApi/method/dayChart/"
    try:
        r = requests.get(url, headers=HSGT_HEADERS, timeout=10)
        d = r.json()
        times = d.get("time", [])
        hgt = d.get("hgt", [])
        sgt = d.get("sgt", [])

        n = len(times)
        return pd.DataFrame({
            "time": times,
            "hgt_yi": hgt[:n] + [None] * (n - len(hgt)),
            "sgt_yi": sgt[:n] + [None] * (n - len(sgt)),
        })
    except Exception as e:
        print(f"  [WARN] 北向实时: {e}")
        return pd.DataFrame()


def _northbound_cache_path() -> Path:
    p = Path.home() / ".tradingagents" / "cache" / "northbound_daily.csv"
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def _save_northbound_snapshot(date: str, hgt: float, sgt: float):
    path = _northbound_cache_path()
    rows = {}
    if path.exists():
        for line in path.read_text().strip().split("\n")[1:]:
            parts = line.split(",")
            if len(parts) == 3:
                rows[parts[0]] = line
    rows[date] = f"{date},{hgt},{sgt}"
    with open(path, "w") as f:
        f.write("date,hgt_yi,sgt_yi\n")
        for d in sorted(rows.keys()):
            f.write(rows[d] + "\n")


def _load_northbound_history(n: int = 60) -> pd.DataFrame:
    path = _northbound_cache_path()
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    return df.tail(n)


def get_north_flow_history(n_days: int = 60) -> pd.DataFrame:
    """
    获取北向资金历史（先拉实时，再读缓存）。
    返回 DataFrame: date, hgt_yi, sgt_yi, net_total(合计)
    """
    # 尝试拉实时数据并缓存
    try:
        df_rt = hsgt_realtime()
        if not df_rt.empty:
            last = df_rt.dropna().iloc[-1] if len(df_rt.dropna()) > 0 else None
            if last is not None:
                today_str = datetime.now().strftime("%Y-%m-%d")
                _save_northbound_snapshot(today_str, last["hgt_yi"], last["sgt_yi"])
    except Exception:
        pass

    df = _load_northbound_history(n_days)
    if not df.empty and "net_total" not in df.columns:
        df["net_total"] = df["hgt_yi"].fillna(0) + df["sgt_yi"].fillna(0)
    return df


def get_north_flow_for_codes(codes: list[str], lookback_days: int = 20) -> dict[str, float]:
    """
    将北向资金映射到个股（当前返回市场级数据）。
    后续可扩展为个股级北向持仓变化。
    Returns: {code: net_flow_change_pct, ...}
    """
    df = get_north_flow_history(lookback_days + 5)
    if df.empty or len(df) < 2:
        return {c: 0.0 for c in codes}


def ths_hot_reason(date: str = None) -> pd.DataFrame:
    """
    同花顺当日强势股+题材归因。
    来源: 同花顺 zx.10jqka.com.cn（零鉴权）
    date: 'YYYY-MM-DD'，None=今天
    返回: DataFrame with columns [代码, 名称, 涨幅%, 题材归因, 换手率%, 成交额]
    """
    from datetime import date as _date
    if date is None:
        date = _date.today().strftime("%Y-%m-%d")
    url = (
        f"http://zx.10jqka.com.cn/event/api/getharden/"
        f"date/{date}/orderby/date/orderway/desc/charset/GBK/"
    )
    headers = {"User-Agent": UA}
    try:
        r = requests.get(url, headers=headers, timeout=10)
        data = r.json()
        if data.get("errocode", 0) != 0:
            return pd.DataFrame()
        rows = data.get("data") or []
        df = pd.DataFrame(rows)
        if df.empty:
            return df
        df = df.rename(columns={
            "name": "名称", "code": "代码", "reason": "题材归因",
            "zhangfu": "涨幅%", "huanshou": "换手率%",
            "chengjiaoe": "成交额", "chengjiaoliang": "成交量",
            "market": "市场",
        })
        return df
    except Exception:
        return pd.DataFrame()

    recent = df["net_total"].tail(lookback_days)
    if len(recent) < 2:
        return {c: 0.0 for c in codes}

    # 近N日北向变化趋势
    change = recent.iloc[-1] - recent.iloc[0] if len(recent) >= lookback_days else 0
    # 简单归一化
    north_signal = 1.0 if change > 10 else (-1.0 if change < -10 else change / 10)

    return {c: north_signal for c in codes}


# ============================================================
# 融资融券 — 东财 datacenter — a-stock-data §4.1
# ============================================================

def get_margin_data(code: str, lookback_days: int = 30) -> dict:
    """
    获取单只股票融资融券明细。
    返回: {rzye(融资余额), rzmre_5d(近5日融资买入均值),
           margin_change_5d_pct, rqye(融券余额), margin_dates: [str]}
    """
    data = eastmoney_datacenter(
        "RPTA_WEB_RZRQ_GGMX",
        filter_str=f'(SCODE="{code}")',
        page_size=lookback_days,
        sort_columns="DATE", sort_types="-1",
    )
    if not data:
        return {}

    rows = []
    for row in data:
        rows.append({
            "date": str(row.get("DATE", ""))[:10],
            "rzye": row.get("RZYE", 0),        # 融资余额(元)
            "rzmre": row.get("RZMRE", 0),       # 融资买入额
            "rzche": row.get("RZCHE", 0),       # 融资偿还额
            "rqye": row.get("RQYE", 0),         # 融券余额(元)
            "rzrqye": row.get("RZRQYE", 0),     # 融资融券余额合计
        })

    if not rows:
        return {}

    latest = rows[0]
    recent_5 = rows[:min(5, len(rows))]
    margin_change_5d = 0.0
    if len(rows) >= 6:
        margin_change_5d = (latest["rzye"] - rows[5]["rzye"]) / max(abs(rows[5]["rzye"]), 1)

    return {
        "rzye": latest["rzye"],
        "rzmre_5d_avg": sum(r["rzmre"] for r in recent_5) / len(recent_5),
        "margin_change_5d_pct": round(margin_change_5d * 100, 2),
        "rqye": latest["rqye"],
        "rzrqye": latest["rzrqye"],
        "margin_dates": [r["date"] for r in rows[:lookback_days]],
    }


def get_margin_batch(codes: list[str], lookback_days: int = 30) -> dict[str, dict]:
    """
    批量获取融资融券数据。
    通过 em_get 串行限流（每只股票间隔约 1.5s）。
    """
    results = {}
    print(f"  获取 {len(codes)} 只股票的融资融券数据...")
    for i, code in enumerate(codes):
        if i > 0 and i % 20 == 0:
            print(f"    融资融券进度: {i}/{len(codes)}")
        data = get_margin_data(code, lookback_days)
        if data:
            results[code] = data

    print(f"  融资融券: {len(results)} 只有数据")
    return results


# ============================================================
# 龙虎榜 — 东财 datacenter — a-stock-data §3.5
# ============================================================

def get_dragon_tiger_data(code: str, trade_date: str = None,
                           lookback_days: int = 30) -> dict:
    """
    获取单只股票龙虎榜数据。
    Returns: {recent_count, latest_net_buy_wan, institution_net_buy_wan, has_institution}
    """
    if trade_date is None:
        trade_date = datetime.now().strftime("%Y-%m-%d")

    start_date = (datetime.strptime(trade_date, "%Y-%m-%d") -
                  timedelta(days=lookback_days)).strftime("%Y-%m-%d")

    # 上榜记录
    data = eastmoney_datacenter(
        "RPT_DAILYBILLBOARD_DETAILSNEW",
        filter_str=f"(TRADE_DATE>='{start_date}')(TRADE_DATE<='{trade_date}')(SECURITY_CODE=\"{code}\")",
        page_size=30,
        sort_columns="TRADE_DATE", sort_types="-1",
    )

    if not data:
        return {"recent_count": 0, "latest_net_buy_wan": 0,
                "institution_net_buy_wan": 0, "has_institution": False}

    latest_net = (data[0].get("BILLBOARD_NET_AMT") or 0) / 10000  # → 万元

    # 查找机构席位参与
    inst_net = 0.0
    has_inst = False
    if data:
        latest_date = str(data[0].get("TRADE_DATE", ""))[:10]
        buy_data = eastmoney_datacenter(
            "RPT_BILLBOARD_DAILYDETAILSBUY",
            filter_str=f"(TRADE_DATE='{latest_date}')(SECURITY_CODE=\"{code}\")",
            page_size=10,
            sort_columns="BUY", sort_types="-1",
        )
        sell_data = eastmoney_datacenter(
            "RPT_BILLBOARD_DAILYDETAILSSELL",
            filter_str=f"(TRADE_DATE='{latest_date}')(SECURITY_CODE=\"{code}\")",
            page_size=10,
            sort_columns="SELL", sort_types="-1",
        )
        for row in buy_data:
            if str(row.get("OPERATEDEPT_CODE", "")) == "0":
                inst_net += (row.get("BUY") or 0)
                has_inst = True
        for row in sell_data:
            if str(row.get("OPERATEDEPT_CODE", "")) == "0":
                inst_net -= (row.get("SELL") or 0)

    return {
        "recent_count": len(data),
        "latest_net_buy_wan": round(latest_net, 1),
        "institution_net_buy_wan": round(inst_net / 10000, 1) if inst_net else 0,
        "has_institution": has_inst,
    }


def get_dragon_tiger_batch(codes: list[str], trade_date: str = None,
                            lookback_days: int = 30) -> dict[str, dict]:
    """
    批量获取龙虎榜数据。
    通过 em_get 串行限流。
    """
    results = {}
    print(f"  获取 {len(codes)} 只股票的龙虎榜数据...")
    for i, code in enumerate(codes):
        if i > 0 and i % 20 == 0:
            print(f"    龙虎榜进度: {i}/{len(codes)}")
        data = get_dragon_tiger_data(code, trade_date, lookback_days)
        if data and data.get("recent_count", 0) > 0:
            results[code] = data

    print(f"  龙虎榜: {len(results)} 只近期上榜")
    return results


# ============================================================
# news 项目数据接口
# ============================================================

def load_news_sector_data(date: str) -> dict[str, dict]:
    """
    从 news 项目加载板块级别数据。
    返回 {sector_name: {"momentum": ..., "sentiment_score": ..., "sentiment": ...}}
    （唯一版本，合并了 market_data.json + news_impact.json）
    """
    result = {}
    base = NEWS_ROOT / "data" / "processed" / date

    # 1. 板块涨跌（market_data.json）
    md_path = base / "market_data.json"
    if md_path.exists():
        with open(md_path) as f:
            md = json.load(f)
        for sec in md.get("top_industry_sectors", []):
            name = sec.get("name", "")
            result.setdefault(name, {})["momentum"] = sec.get("change_pct", 0)
        for sec in md.get("bottom_industry_sectors", []):
            name = sec.get("name", "")
            result.setdefault(name, {})["momentum"] = sec.get("change_pct", 0)
        for sec in md.get("top_concept_sectors", []):
            name = sec.get("name", "")
            result.setdefault(name, {})["momentum"] = sec.get("change_pct", 0)

    # 2. 板块情绪（news_impact.json）
    impact_path = base / "news_impact.json"
    if impact_path.exists():
        with open(impact_path) as f:
            impact = json.load(f)
        for sec in impact.get("sector_impacts", []):
            name = sec["sector"]
            result.setdefault(name, {}).update({
                "sentiment_score": sec.get("score", 0),
                "sentiment": sec.get("sentiment", "中性"),
                "bullish_count": sec.get("bullish_count", 0),
                "bearish_count": sec.get("bearish_count", 0),
            })

    return result


# watchlist 板块 → news 项目板块的映射
SECTOR_NAME_MAP = {
    "机器人/人形机器人": ["机器人/人形机器人", "人工智能", "机器人"],
    "PCB": ["半导体", "消费电子", "PCB"],
    "MLCC": ["半导体", "消费电子", "MLCC"],
    "玻璃基板/先进封装": ["半导体", "玻璃基板/先进封装", "先进封装"],
    "先进封装材料": ["半导体", "先进封装材料", "先进封装"],
}


def map_stocks_to_sectors(universe: list[dict],
                          news_sectors: dict[str, dict]) -> dict[str, dict]:
    """
    将 watchlist 中的 sector 名映射到 news 项目中的 sector 名，
    用于获取板块动量和情绪数据。
    """
    result = {}
    for stock in universe:
        sector = stock.get("sector", "")
        code = stock["code"]
        mapped = {"sector_momentum": None, "sentiment_score": 0, "sentiment": "中性"}

        candidates = SECTOR_NAME_MAP.get(sector, [sector])

        for cand in candidates:
            if cand in news_sectors:
                ns = news_sectors[cand]
                mapped["sector_momentum"] = ns.get("momentum")
                mapped["sentiment_score"] = ns.get("sentiment_score", 0)
                mapped["sentiment"] = ns.get("sentiment", "中性")
                break

        if mapped["sector_momentum"] is None:
            for ns_name, ns_data in news_sectors.items():
                for cand in candidates:
                    if cand in ns_name or ns_name in cand:
                        mapped["sector_momentum"] = ns_data.get("momentum")
                        mapped["sentiment_score"] = ns_data.get("sentiment_score", 0)
                        mapped["sentiment"] = ns_data.get("sentiment", "中性")
                        break
                if mapped["sector_momentum"] is not None:
                    break

        result[code] = mapped
    return result


# ============================================================
# 股票池构建
# ============================================================

def build_stock_universe(watchlist_only: bool = True) -> list[dict]:
    """构建待选股票池"""
    wl = load_watchlist()

    if watchlist_only:
        stocks = []
        for sector in wl.get("focus_sectors", []):
            for s in sector.get("representative_stocks", []):
                stocks.append({
                    "code": s["code"],
                    "name": s["name"],
                    "market": s.get("market", 0),
                    "sector": sector["name"],
                })
        seen = set()
        uniq = []
        for s in stocks:
            if s["code"] not in seen:
                seen.add(s["code"])
                uniq.append(s)
        return uniq

    # 全量模式
    stocks = []
    for fp in sorted(STOCK_CACHE_DIR.glob("*.parquet")):
        code = fp.stem
        if code.startswith("INDEX_"):
            continue
        sector_name = "未分类"
        for sec in wl.get("focus_sectors", []):
            for s in sec.get("representative_stocks", []):
                if s["code"] == code:
                    sector_name = sec["name"]
                    break
        market = 1 if code.startswith("6") else 0
        stocks.append({
            "code": code,
            "name": "",
            "market": market,
            "sector": sector_name,
        })
    return stocks


def load_universe_klines(watchlist_only: bool = True,
                         refresh: bool = False) -> dict:
    """加载股票池的所有历史K线，返回 {code: {"info": {...}, "kline": DataFrame}}"""
    universe = build_stock_universe(watchlist_only)

    if watchlist_only or refresh:
        codes = [(s["code"], s["market"]) for s in universe]
        print(f"加载 {len(codes)} 只股票的历史K线...")
        klines = batch_fetch_klines(codes, refresh=refresh)
    else:
        print(f"从本地缓存加载股票K线...")
        klines = {}
        for s in universe:
            path = STOCK_CACHE_DIR / f"{s['code']}.parquet"
            if path.exists():
                df = pd.read_parquet(path)
                if not df.empty:
                    klines[s["code"]] = df

    result = {}
    for stock in universe:
        code = stock["code"]
        if code in klines:
            result[code] = {
                "info": stock,
                "kline": klines[code],
            }

    print(f"成功加载 {len(result)} 只股票的K线数据")
    return result


def load_index_klines() -> dict[str, pd.DataFrame]:
    """加载所有指数K线"""
    indices = {}
    for fp in sorted(STOCK_CACHE_DIR.glob("INDEX_*.parquet")):
        name = fp.stem.replace("INDEX_", "")
        df = pd.read_parquet(fp)
        if not df.empty:
            indices[name] = df
    return indices


# ============================================================
# 全 A 股股票池 + 批量K线下载 (V4 — a-stock-data 增强)
# ============================================================

def fetch_all_a_stock_codes() -> list[dict]:
    """
    从东财 push2 拉取全 A 股股票列表（~5000只）。
    返回: [{code, name, market, industry}, ...]
    """
    all_stocks = []
    # 沪深两市分页拉取
    for mkt_code, mkt_name in [("m:0+t:6,m:0+t:13,m:0+t:80,m:1+t:2,m:0+t:7", ""),
                                 ("m:1+t:2,m:1+t:23", "")]:
        for page in range(1, 30):
            url = "https://push2.eastmoney.com/api/qt/clist/get"
            params = {
                "pn": str(page), "pz": "200",
                "po": "1", "np": "1",
                "fltt": "2", "invt": "2",
                "fs": "m:0+t:6,m:0+t:13,m:0+t:80,m:1+t:2,m:0+t:7",
                "fields": "f2,f12,f14,f127",
            }
            headers = {"User-Agent": UA, "Referer": "https://quote.eastmoney.com/"}
            try:
                r = em_get(url, params=params, headers=headers, timeout=20)
                d = r.json()
                items = (d.get("data") or {}).get("diff") or []
                if not items:
                    break
                for it in items:
                    code = it.get("f12", "")
                    name = it.get("f14", "")
                    if not code or "退市" in name or "ST" in name:
                        continue
                    market = 1 if code.startswith("6") else 0
                    all_stocks.append({
                        "code": code, "name": name, "market": market,
                        "industry": it.get("f127", ""),
                    })
            except Exception:
                break
    return all_stocks


def download_full_market_klines(refresh: bool = False) -> int:
    """
    增量下载全 A 股 K 线到 data/stocks/（已有缓存跳过）。
    返回: 新下载的数量
    """
    stocks = fetch_all_a_stock_codes()
    new_count = 0
    print(f"全 A 股共 {len(stocks)} 只，开始增量下载K线...")
    for i, s in enumerate(stocks):
        code, market = s["code"], s["market"]
        cache_path = STOCK_CACHE_DIR / f"{code}.parquet"
        if not refresh and cache_path.exists():
            continue
        try:
            df = fetch_kline_tdx(code, market)
            if df is not None and len(df) >= 60:
                df.to_parquet(cache_path, index=False)
                new_count += 1
        except Exception:
            pass
        if i % 100 == 0:
            print(f"  进度: {i}/{len(stocks)} (新增{new_count})")
    print(f"完成: 新增{new_count} 只")
    return new_count


def stock_fund_flow_120d(code: str) -> list[dict]:
    """
    个股资金流（日级，最近120个交易日）。
    来源: 东财 push2his
    返回: [{date, main_net, small_net, mid_net, large_net, super_net}]
    单位: 元
    """
    market_code = 1 if code.startswith("6") else 0
    url = "https://push2his.eastmoney.com/api/qt/stock/fflow/daykline/get"
    params = {
        "secid": f"{market_code}.{code}",
        "fields1": "f1,f2,f3,f7",
        "fields2": "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61,f62,f63,f64,f65",
        "lmt": "120",
    }
    headers = {
        "User-Agent": UA,
        "Referer": "https://quote.eastmoney.com/",
        "Origin": "https://quote.eastmoney.com",
    }
    try:
        r = em_get(url, params=params, headers=headers, timeout=15)
        d = r.json()
    except Exception:
        return []
    klines = d.get("data", {}).get("klines", [])
    rows = []
    for line in klines:
        parts = line.split(",")
        if len(parts) >= 7:
            rows.append({
                "date": parts[0],
                "main_net": float(parts[1]) if parts[1] != "-" else 0,
                "small_net": float(parts[2]) if parts[2] != "-" else 0,
                "mid_net": float(parts[3]) if parts[3] != "-" else 0,
                "large_net": float(parts[4]) if parts[4] != "-" else 0,
                "super_net": float(parts[5]) if parts[5] != "-" else 0,
            })
    return rows


def holder_num_change(code: str, page_size: int = 4) -> list[dict]:
    """
    股东户数变化（季度级）。
    返回: [{date, holder_num, change_ratio, avg_shares}]
    """
    data = eastmoney_datacenter(
        "RPT_HOLDERNUMLATEST",
        filter_str=f'(SECURITY_CODE="{code}")',
        page_size=page_size,
        sort_columns="END_DATE", sort_types="-1",
    )
    rows = []
    for row in data:
        rows.append({
            "date": str(row.get("END_DATE", ""))[:10],
            "holder_num": row.get("HOLDER_NUM", 0),
            "change_ratio": row.get("HOLDER_NUM_RATIO", 0),
            "avg_shares": row.get("AVG_FREE_SHARES", 0),
        })
    return rows


# ============================================================
# 新闻数据 (a-stock-data 7×24 全球资讯)
# ============================================================

def eastmoney_global_news(page_size: int = 100) -> list[dict]:
    """
    东财全球财经资讯（7×24 滚动快讯）。
    返回: [{title, summary, time, source}]
    """
    import uuid
    url = "https://np-weblist.eastmoney.com/comm/web/getFastNewsList"
    params = {
        "client": "web", "biz": "web_724",
        "fastColumn": "102", "sortEnd": "",
        "pageSize": str(page_size),
        "req_trace": str(uuid.uuid4()),
    }
    headers = {"User-Agent": UA, "Referer": "https://kuaixun.eastmoney.com/"}
    try:
        r = em_get(url, params=params, headers=headers, timeout=10)
        d = r.json()
    except Exception:
        return []
    rows = []
    for item in d.get("data", {}).get("fastNewsList", []):
        rows.append({
            "title": item.get("title", ""),
            "summary": item.get("summary", "")[:300],
            "time": item.get("showTime", ""),
            "source": "东财全球",
        })
    return rows


def eastmoney_stock_news(code: str, page_size: int = 20) -> list[dict]:
    """东财个股新闻（JSONP 接口）"""
    import re
    cb = "jQuery_news"
    url = "https://search-api-web.eastmoney.com/search/jsonp"
    inner = json.dumps({
        "uid": "", "keyword": code,
        "type": ["cmsArticleWebOld"], "client": "web",
        "clientType": "web", "clientVersion": "curr",
        "param": {"cmsArticleWebOld": {"searchScope": "default", "sort": "default",
                  "pageIndex": 1, "pageSize": page_size, "preTag": "", "postTag": ""}},
    }, separators=(',', ':'))
    params = {"cb": cb, "param": inner}
    headers = {"User-Agent": UA, "Referer": "https://so.eastmoney.com/"}
    try:
        r = em_get(url, params=params, headers=headers, timeout=15)
        text = r.text
        json_str = text[text.index("(") + 1 : text.rindex(")")]
        d = json.loads(json_str)
        articles = d.get("result", {}).get("cmsArticleWebOld", []) or []
        rows = []
        for a in articles:
            rows.append({
                "title": re.sub(r'<[^>]+>', '', a.get("title", "")),
                "content": re.sub(r'<[^>]+>', '', a.get("content", ""))[:200],
                "time": a.get("date", ""),
                "source": a.get("mediaName", ""),
                "url": a.get("url", ""),
            })
        return rows
    except Exception:
        return []


# ============================================================
# 指数数据
# ============================================================

def fetch_index_kline(index_code: str, market: int = 1,
                      count: int = 250) -> pd.DataFrame:
    """获取指数 K 线 — mootdx 优先，腾讯财经 fallback"""
    df = fetch_kline_tdx(index_code, market, count)
    if not df.empty and len(df) >= 30:
        return df

    # Fallback to Tencent for indices (mootdx sometimes fails on SZ/CSI codes)
    prefix = f"sh{index_code}" if market == 1 else f"sz{index_code}"
    try:
        import json
        url = (f"https://web.ifzq.gtimg.cn/appstock/app/fqkline/get"
               f"?param={prefix},day,,,{count},qfq")
        req = urllib.request.Request(url)
        req.add_header("User-Agent", UA)
        resp = urllib.request.urlopen(req, timeout=10)
        data = json.loads(resp.read().decode("utf-8"))
        rows = data.get("data", {}).get(prefix, {}).get("day", []) or \
               data.get("data", {}).get(prefix, {}).get("qfqday", [])
        if rows:
            records = [{"date": r[0], "open": float(r[1]), "close": float(r[2]),
                        "high": float(r[3]), "low": float(r[4]),
                        "volume": float(r[5]), "amount": 0.0} for r in rows]
            df = pd.DataFrame(records)
            return df.sort_values("date").reset_index(drop=True)
    except Exception as e:
        print(f"  [WARN] Tencent fallback {index_code}: {e}")

    return df


# ============================================================
# CLI 测试
# ============================================================

if __name__ == "__main__":
    ensure_dirs()

    print("=" * 60)
    print("数据加载测试 (a-stock-data refactored)")
    print("=" * 60)

    # 1. 测试腾讯财经
    print("\n[1] 腾讯财经基本面...")
    tencent = tencent_fundamentals(["600519", "000858", "300750", "688017"])
    for code, q in tencent.items():
        print(f"  {q.get('name','')}({code}): PE={q.get('pe')} PB={q.get('pb')} "
              f"市值={q.get('market_cap',0)/1e8:.0f}亿 涨停={q.get('limit_up')}")

    # 2. 测试板块分类
    print("\n[2] 东财板块分类...")
    sector_map = get_eastmoney_sector_map(["600519", "000858", "300750"])
    for code, info in sector_map.items():
        tags = info.get("all_tags", [])[:8]
        print(f"  {code}: {info['primary_sector']} → {', '.join(tags)}")

    # 3. 测试北向资金
    print("\n[3] 北向资金历史...")
    nb = get_north_flow_history(10)
    if not nb.empty:
        print(f"  {len(nb)} 天数据")
        print(nb.tail(5).to_string(index=False))

    # 4. 测试 K 线加载
    print("\n[4] 股票K线...")
    data = load_universe_klines(watchlist_only=True, refresh=False)
    print(f"  共 {len(data)} 只")

    # 5. 测试基本面
    print("\n[5] 基本面（腾讯+东财）...")
    codes = list(data.keys())[:5]
    fund = load_fundamentals_for_codes(codes, refresh=True)
    for code, f in fund.items():
        print(f"  {f.get('name', code)}: PE={f.get('pe')} PB={f.get('pb')} "
              f"ROE={f.get('roe')} 市值={f.get('market_cap',0)/1e8 if f.get('market_cap') else 0:.0f}亿")
