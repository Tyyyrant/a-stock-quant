#!/usr/bin/env python3
"""
供应链瓶颈全面发现引擎 v2

彻底替代旧 map_supply_chain() 的局限性:
- 旧: 依赖 supply_chain.yaml 预建知识库 (只有5个主题)
- 新: 38种瓶颈材料 × 5000+股票名称关键词搜索 → 全覆盖

策略:
  1. 定义瓶颈材料→搜索关键词映射 (覆盖半导体/PCB/先进封装/元件全材料链)
  2. 加载全A 5800只股票名称
  3. 名称关键词搜索 → 初步匹配
  4. 概念标签验证 (可选,需API)
  5. 技术面评分排序
  6. 输出 Top N 瓶颈标的

用法:
  python3 scripts/bottleneck_discovery.py --date 2026-06-19
  python3 scripts/bottleneck_discovery.py --date 2026-06-19 --top 20
"""

import argparse, json, os, sys, time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

# ============================================================
# 瓶颈材料全景图谱 (38种 → 关键词 → A股名称特征)
# ============================================================

BOTTLENECK_MATERIALS = {
    # ═══ 半导体材料 ═══
    "硅材料/硅片": {
        "category": "半导体材料",
        "layer": "L5上游材料",
        "keywords": ["硅", "硅片", "硅材料", "单晶硅", "多晶硅", "硅棒", "硅晶圆", "大硅片"],
        "exclude_keywords": ["硅胶", "硅橡胶", "有机硅", "硅油"],  # 排除非半导体硅
        "archetypes": ["国产替代锁定", "产能瓶颈"],
    },
    "光刻胶": {
        "category": "半导体材料",
        "layer": "L5上游材料",
        "keywords": ["光刻胶", "光阻", "光刻", "感光", "光固化"],
        "archetypes": ["单源供应商", "国产替代锁定"],
    },
    "电子特气": {
        "category": "半导体材料",
        "layer": "L5上游材料",
        "keywords": ["电子气", "特气", "高纯气", "工业气体", "气体", "氖气", "氩气", "氦气",
                    "氟化", "硅烷", "磷烷", "硼烷", "六氟", "四氟"],
        "exclude_keywords": ["天然气", "石油气", "液化气", "燃气"],
        "archetypes": ["认证壁垒", "国产替代锁定"],
    },
    "溅射靶材": {
        "category": "半导体材料",
        "layer": "L5上游材料",
        "keywords": ["靶材", "溅射靶", "靶", "镀膜靶", "PVD靶"],
        "exclude_keywords": ["靶场", "打靶"],
        "archetypes": ["认证壁垒", "国产替代锁定"],
    },
    "湿电子化学品": {
        "category": "半导体材料",
        "layer": "L5上游材料",
        "keywords": ["电子化学", "湿化学", "超纯", "高纯试剂", "电子级", "化学试剂",
                    "显影液", "蚀刻液", "剥离液", "清洗液", "电镀液"],
        "exclude_keywords": ["化肥", "农药", "涂料"],
        "archetypes": ["认证壁垒", "国产替代锁定"],
    },
    "CMP抛光材料": {
        "category": "半导体材料",
        "layer": "L5上游材料",
        "keywords": ["抛光液", "抛光垫", "CMP", "研磨液", "抛光电", "半导体抛光"],
        "exclude_keywords": ["抛光机", "研磨机", "研磨设备", "精研科技", "抛光设备"],
        "archetypes": ["耗材绑定", "工艺壁垒"],
    },
    "高纯石英": {
        "category": "半导体材料",
        "layer": "L5上游材料",
        "keywords": ["石英", "高纯石英", "石英砂", "石英坩埚"],
        "archetypes": ["单源供应商", "工艺壁垒"],
    },
    "光掩模/掩膜版": {
        "category": "半导体材料",
        "layer": "L5上游材料",
        "keywords": ["掩模", "掩膜", "光掩模", "光罩", "photomask"],
        "archetypes": ["单源供应商", "工艺壁垒"],
    },
    "封装基板/载板": {
        "category": "封装材料",
        "layer": "L5上游材料",
        "keywords": ["载板", "基板", "IC基板", "ABF", "BT基板", "封装基板", "IC载板"],
        "exclude_keywords": ["铝基板", "铝基"], # 排除LED用
        "archetypes": ["产能瓶颈", "认证壁垒"],
    },
    "环氧塑封料/EMC": {
        "category": "封装材料",
        "layer": "L5上游材料",
        "keywords": ["塑封料", "EMC", "环氧塑封", "封装料", "模塑料"],
        "archetypes": ["耗材绑定", "认证壁垒"],
    },
    "底部填充胶/Underfill": {
        "category": "封装材料",
        "layer": "L5上游材料",
        "keywords": ["底部填充", "underfill", "填充胶", "封装胶", "电子胶"],
        "exclude_keywords": ["建筑胶", "结构胶"],
        "archetypes": ["耗材绑定"],
    },
    "键合丝": {
        "category": "封装材料",
        "layer": "L5上游材料",
        "keywords": ["键合丝", "键合线", "bonding", "铜丝", "金丝"],
        "archetypes": ["国产替代锁定"],
    },
    "先进封装": {
        "category": "封装",
        "layer": "L2中游系统",
        "keywords": ["封装", "先进封装", "封测", "晶圆级", "chiplet", "芯粒"],
        "exclude_keywords": ["封装设备"],
        "archetypes": ["产能瓶颈", "工艺壁垒"],
    },
    "HBM": {
        "category": "封装",
        "layer": "L3中游器件",
        "keywords": ["HBM", "高带宽", "堆叠", "TSV", "硅通孔"],
        "archetypes": ["工艺壁垒", "国产替代锁定"],
    },
    # ═══ PCB/元件材料 ═══
    "铜箔": {
        "category": "PCB材料",
        "layer": "L5上游材料",
        "keywords": ["铜箔", "电子铜箔", "电解铜箔", "压延铜箔", "锂电铜箔"],
        "archetypes": ["产能瓶颈", "规模效应"],
    },
    "覆铜板/CCL": {
        "category": "PCB材料",
        "layer": "L4上游设备",
        "keywords": ["覆铜板", "CCL", "铜板", "基材"],
        "exclude_keywords": ["铜箔"],  # 区分铜箔和覆铜板
        "archetypes": ["规模效应", "认证壁垒"],
    },
    "钨材料": {
        "category": "半导体材料",
        "layer": "L5上游材料",
        "keywords": ["钨业", "钨高新", "钨矿", "硬质合金", "钨材", "钨制品"],
        "archetypes": ["单源供应商", "国产替代锁定"],
    },
    "高频高速材料": {
        "category": "PCB材料",
        "layer": "L5上游材料",
        "keywords": ["高频高速", "高频板", "高速板", "PTFE", "碳氢树脂", "LCP膜", "MPI"],
        "archetypes": ["工艺壁垒", "国产替代锁定"],
    },
    "PCB油墨": {
        "category": "PCB材料",
        "layer": "L5上游材料",
        "keywords": ["油墨", "阻焊", "PCB油墨", "感光油墨"],
        "archetypes": ["耗材绑定"],
    },
    "PCB制造": {
        "category": "PCB",
        "layer": "L3中游器件",
        "keywords": ["电路板", "PCB", "印制板", "HDI", "线路板", "印刷电路"],
        "archetypes": ["规模效应", "认证壁垒"],
    },
    # ═══ 散热/热管理 ═══
    "散热材料": {
        "category": "热管理",
        "layer": "L5上游材料",
        "keywords": ["散热", "导热", "热界面", "TIM", "导热硅脂", "导热垫",
                    "石墨片", "散热膜", "散热器", "液冷板", "热管", "均温板",
                    "石墨烯导热", "金刚石散热", "钻石散热"],
        "archetypes": ["工艺壁垒", "设计锁定"],
    },
    "液冷系统": {
        "category": "热管理",
        "layer": "L4上游设备",
        "keywords": ["液冷", "水冷", "冷却", "冷板", "冷却液", "散热"],
        "archetypes": ["设计锁定", "认证壁垒"],
    },
    # ═══ 半导体设备零部件 ═══
    "半导体设备": {
        "category": "设备",
        "layer": "L4上游设备",
        "keywords": ["半导体设备", "晶圆", "刻蚀", "薄膜沉积", "清洗机", "CVD", "PVD",
                    "ALD", "离子注入", "检测设备", "测试机", "分选机", "划片机"],
        "archetypes": ["单源供应商", "国产替代锁定"],
    },
    "半导体零部件": {
        "category": "设备零部件",
        "layer": "L3中游器件",
        "keywords": ["真空泵", "阀门", "密封件", "射频电源", "运动台",
                    "静电卡盘", "加热器", "流量计", "压力计", "半导体零部件"],
        "archetypes": ["单源供应商", "认证壁垒"],
    },
    "光刻机零部件": {
        "category": "设备零部件",
        "layer": "L3中游器件",
        "keywords": ["光刻", "光源", "镜头", "物镜", "掩模台", "光刻机"],
        "archetypes": ["单源供应商", "国产替代锁定"],
    },
    # ═══ 功率半导体 ═══
    "IGBT/SiC": {
        "category": "功率半导体",
        "layer": "L3中游器件",
        "keywords": ["IGBT", "SiC", "碳化硅", "氮化镓", "GaN", "功率器件",
                    "MOSFET", "功率半导体", "功率模块"],
        "archetypes": ["产能瓶颈", "国产替代锁定"],
    },
    # ═══ 被动元件 ═══
    "MLCC": {
        "category": "被动元件",
        "layer": "L3中游器件",
        "keywords": ["MLCC", "电容", "陶瓷电容", "钽电容", "薄膜电容"],
        "exclude_keywords": ["电解电容"],  # 低端
        "archetypes": ["产能瓶颈", "规模效应"],
    },
    "电感/磁材": {
        "category": "被动元件",
        "layer": "L3中游器件",
        "keywords": ["电感", "磁材", "磁性材料", "铁氧体", "磁芯", "磁粉",
                    "电感器", "变压器", "线圈"],
        "archetypes": ["工艺壁垒"],
    },
    "晶振/谐振器": {
        "category": "被动元件",
        "layer": "L3中游器件",
        "keywords": ["晶振", "谐振器", "振荡器", "TCXO", "OCXO", "晶体"],
        "archetypes": ["认证壁垒", "国产替代锁定"],
    },
    # ═══ 连接器/铜连接 ═══
    "高速连接器": {
        "category": "连接器",
        "layer": "L3中游器件",
        "keywords": ["连接器", "铜连接", "高速背板", "高速连接", "铜缆"],
        "archetypes": ["设计锁定", "认证壁垒"],
    },
    # ═══ 光学 ═══
    "光学镜头": {
        "category": "光学",
        "layer": "L3中游器件",
        "keywords": ["光学", "镜头", "透镜", "棱镜", "滤光片", "摄像头"],
        "exclude_keywords": ["眼镜"],
        "archetypes": ["工艺壁垒"],
    },
    # ═══ EDA/工业软件 ═══
    "EDA/IP": {
        "category": "工业软件",
        "layer": "L5上游材料",
        "keywords": ["EDA", "IP核", "芯片设计", "仿真", "验证"],
        "archetypes": ["单源供应商", "国产替代锁定"],
    },
}

# 瓶颈材料→共振板块的关联 (用于解释为什么这个材料是瓶颈)
BOTTLENECK_TO_SECTOR = {
    "半导体": ["硅材料/硅片", "光刻胶", "电子特气", "溅射靶材", "湿电子化学品",
              "CMP抛光材料", "高纯石英", "光掩模/掩膜版", "半导体设备", "半导体零部件",
              "光刻机零部件", "IGBT/SiC", "EDA/IP", "钨材料"],
    "PCB": ["铜箔", "覆铜板/CCL", "高频高速材料", "PCB油墨", "PCB制造", "散热材料"],
    "元件": ["MLCC", "电感/磁材", "晶振/谐振器", "高速连接器", "PCB制造"],
    "先进封装": ["封装基板/载板", "环氧塑封料/EMC", "底部填充胶/Underfill",
               "键合丝", "先进封装", "HBM", "散热材料"],
    "消费电子": ["MLCC", "光学镜头", "高速连接器", "散热材料", "PCB制造"],
    "通信设备": ["高频高速材料", "高速连接器", "PCB制造", "散热材料"],
    "自动化设备": ["IGBT/SiC", "半导体设备", "电感/磁材"],
    "机器人": ["IGBT/SiC", "减速器", "电感/磁材", "散热材料"],
    "电池": ["铜箔", "散热材料"],
    "通用设备": ["IGBT/SiC", "电感/磁材", "半导体零部件"],
    "汽车零部件": ["IGBT/SiC", "MLCC", "连接器", "散热材料"],
}


def load_all_stock_names() -> list[dict]:
    """加载全A股票名称列表"""
    path = ROOT / "data" / "all_stocks.json"
    if path.exists():
        with open(path) as f:
            raw = json.load(f)
        # 过滤有效股票代码 (6位数字，沪深北)
        valid = []
        for s in raw:
            code = s.get("code", "")
            if len(code) == 6 and code.isdigit():
                if code.startswith(("6", "0", "3", "8")):
                    valid.append({"code": code, "name": s.get("name", "").strip("\x00").strip()})
        return valid
    return []


def discover_bottleneck_candidates(
    resonant_sectors: list[str] = None,
    all_stocks: list[dict] = None,
    top_per_material: int = 10,
) -> dict:
    """
    主函数: 全面发现瓶颈候选标的。

    Args:
        resonant_sectors: 共振板块列表 (如 ["半导体", "PCB", "先进封装"])
        all_stocks: 全A股票列表
        top_per_material: 每种材料最多保留几只

    Returns:
        {
            "materials_found": {material_name: [stocks]},
            "global_top": [stocks sorted by score],
            "by_sector": {sector: [material_names]},
            "total_candidates": int,
        }
    """
    if all_stocks is None:
        all_stocks = load_all_stock_names()

    if not all_stocks:
        print("  [ERROR] 无法加载全A股票名称")
        return {"materials_found": {}, "global_top": [], "by_sector": {}, "total_candidates": 0}

    # 如果没有指定共振板块，搜索所有材料
    if resonant_sectors is None:
        materials_to_search = set(BOTTLENECK_MATERIALS.keys())
    else:
        materials_to_search = set()
        for sec in resonant_sectors:
            mats = BOTTLENECK_TO_SECTOR.get(sec, [])
            materials_to_search.update(mats)
        # 如果共振板块没匹配到，fallback到全部
        if not materials_to_search:
            print(f"  [WARN] 共振板块 {resonant_sectors} 无材料映射，搜索全部")
            materials_to_search = set(BOTTLENECK_MATERIALS.keys())

    print(f"  共振板块: {resonant_sectors}")
    print(f"  瓶颈材料: {len(materials_to_search)} 种")

    # 建立名称→代码索引 (去除非A股指数)
    name_to_codes = defaultdict(list)
    for s in all_stocks:
        name = s["name"]
        code = s["code"]
        if not name or name in ("主板Ａ股", "主板Ｂ股", "创业板", "科创板"):
            continue
        # 过滤指数/ETF/B股等
        if code.startswith("39") or code.startswith("9"):
            continue
        if len(code) != 6 or not code.isdigit():
            continue
        name_to_codes[name].append(code)

    materials_found = {}
    all_candidates = {}  # code -> best match info

    for mat_name in sorted(materials_to_search):
        mat_info = BOTTLENECK_MATERIALS[mat_name]
        keywords = mat_info.get("keywords", [])
        excludes = set(mat_info.get("exclude_keywords", []))

        matched = []  # [(code, name, match_keyword)]

        for s in all_stocks:
            name = s["name"]
            code = s["code"]
            if not name or len(code) != 6:
                continue
            if not code.startswith(("6", "0", "3", "8")):
                continue

            # 检查排除词
            if any(ex in name for ex in excludes):
                continue

            # 检查包含关键词
            matched_keyword = None
            for kw in keywords:
                if kw in name:
                    matched_keyword = kw
                    break

            if matched_keyword:
                matched.append((code, name, matched_keyword))

        if matched:
            # 去重 (同一code可能匹配多个keyword)
            seen = {}
            for code, name, kw in matched:
                if code not in seen:
                    seen[code] = {"code": code, "name": name, "match_keyword": kw}
            materials_found[mat_name] = {
                "layer": mat_info["layer"],
                "category": mat_info["category"],
                "archetypes": mat_info.get("archetypes", []),
                "stocks": list(seen.values()),
                "total": len(seen),
            }

            # 合并到全局候选
            for code, info in seen.items():
                if code not in all_candidates:
                    all_candidates[code] = {
                        "code": code,
                        "name": info["name"],
                        "materials": [],
                        "layers": [],
                        "categories": [],
                        "archetypes": [],
                    }
                all_candidates[code]["materials"].append(mat_name)
                all_candidates[code]["layers"].append(mat_info["layer"])
                all_candidates[code]["categories"].append(mat_info["category"])
                for a in mat_info.get("archetypes", []):
                    if a not in all_candidates[code]["archetypes"]:
                        all_candidates[code]["archetypes"].append(a)

    # 统计
    total_materials = len(materials_found)
    total_candidates = len(all_candidates)

    print(f"  材料覆盖: {total_materials}/{len(materials_to_search)}")
    for mat_name, data in sorted(materials_found.items(), key=lambda x: -x[1]["total"]):
        top3 = [f"{s['name']}({s['code']})" for s in data["stocks"][:3]]
        print(f"    {mat_name}: {data['total']}只 {top3}")

    # 构建 sector → materials 映射
    by_sector = {}
    if resonant_sectors:
        for sec in resonant_sectors:
            mats = BOTTLENECK_TO_SECTOR.get(sec, [])
            found_mats = [m for m in mats if m in materials_found]
            by_sector[sec] = found_mats

    result = {
        "materials_found": {
            k: v for k, v in sorted(materials_found.items(), key=lambda x: -x[1]["total"])
        },
        "all_candidates": all_candidates,
        "total_candidates": total_candidates,
        "total_materials_covered": total_materials,
        "by_sector": by_sector,
        "resonant_sectors": resonant_sectors,
    }

    return result


def supplement_known_producers(materials_found: dict) -> dict:
    """
    补充已知的瓶颈材料生产商 (从 news_ripple MATERIAL_GRAPH 手工映射)。
    名称关键词搜索无法覆盖所有公司 (如 昊华科技 不含"气体"关键词但确是氖气龙头)。

    Returns: 更新后的 materials_found
    """
    from news_ripple import MATERIAL_GRAPH as MG

    # material_graph 材料名 → 瓶颈材料名 映射
    MG_TO_BOTTLENECK = {
        "六氟化钨": "电子特气", "钨靶材": "溅射靶材", "钨粉": "钨材料", "钨丝": "钨材料",
        "氟化氢": "电子特气", "氟化氩": "电子特气", "氟聚酰亚胺": "电子特气",
        "氖气": "电子特气",
        "高纯石英": "高纯石英", "高纯多晶硅": "硅材料/硅片",
        "溅射靶材": "溅射靶材",
        "封装基板": "封装基板/载板",
        "chiplet": "先进封装", "HBM": "HBM",
        "先进封装": "先进封装",
        "PCB": "PCB制造",
        "EDA软件": "EDA/IP",
    }

    added_count = 0
    for mg_material, mg_data in MG.items():
        target_mat = MG_TO_BOTTLENECK.get(mg_material)
        if not target_mat:
            # Fallback: check if material name directly maps
            if mg_material in materials_found:
                target_mat = mg_material
            else:
                continue

        if target_mat not in materials_found:
            materials_found[target_mat] = {
                "layer": "L5上游材料",
                "category": "补充",
                "archetypes": [],
                "stocks": [],
                "total": 0,
            }

        existing_codes = {s["code"] for s in materials_found[target_mat]["stocks"]}

        for producer in mg_data.get("domestic_producers", []):
            code = producer["code"]
            name = producer["name"]
            if code not in existing_codes and not code.startswith("688"):
                materials_found[target_mat]["stocks"].append({
                    "code": code,
                    "name": name,
                    "match_keyword": f"图谱:{mg_material}",
                })
                existing_codes.add(code)
                added_count += 1

    if added_count:
        print(f"  补充已知生产商: {added_count} 只")
    return materials_found


def tech_verify_candidates(
    discovery_result: dict,
    target_date: str,
    top_n: int = 30
) -> list[dict]:
    """
    对瓶颈候选做技术面验证评分。

    Returns: sorted list of verified stocks
    """
    from data_loader import get_stock_kline
    from candlestick_patterns import identify_all_patterns
    from volume_price_analyzer import analyze_volume_price
    from chip_distribution import estimate_chip_distribution

    all_candidates = discovery_result.get("all_candidates", {})
    if not all_candidates:
        return []

    print(f"\n  技术面验证 ({len(all_candidates)} 只候选)...")
    verified = []

    for i, (code, info) in enumerate(all_candidates.items()):
        # 过滤科创板(688) - A股短线不碰
        if code.startswith("688"):
            continue
        try:
            market = 1 if code.startswith("6") else 0
            dk = get_stock_kline(code, market, refresh=False)
            if dk is None or len(dk) < 60:
                continue
            dk = dk[dk["date"] <= target_date].copy()
            if len(dk) < 20:
                continue

            p = identify_all_patterns(dk, ticker=code)
            v = analyze_volume_price(dk)
            c = estimate_chip_distribution(dk)

            price = float(dk["close"].values[-1])
            chg = (price / float(dk["close"].values[-2]) - 1) if len(dk) >= 2 else 0
            chg_pct = round(chg * 100, 2)

            ks = round(p.pattern_score, 1)
            vs_val = v.get("volume_score", 0)
            cs_val = c.get("chip_score", 0)
            profit_ratio = c.get("profit_ratio", 0)

            # 瓶颈评分: 材料数越多越重要 + 技术面
            n_materials = len(info["materials"])
            mat_bonus = min(n_materials * 5, 20)  # 跨材料叠加最多+20

            tech_score = ks * 0.35 + max(vs_val, -50) * 0.3 + cs_val * 0.25 + mat_bonus

            # 亏损股扣分但不毙 (瓶颈标的稀缺，PE高也值得看)
            try:
                from data_loader import tencent_quote
                q = tencent_quote([code])
                pe_val = q.get(code, {}).get("pe_ttm", 0) or 0
            except:
                pe_val = 0

            verified.append({
                "code": code,
                "name": info["name"],
                "materials": info["materials"],
                "n_materials": n_materials,
                "categories": info["categories"],
                "archetypes": info["archetypes"],
                "layer": info["layers"][0] if info["layers"] else "?",
                "price": price,
                "chg_pct": chg_pct,
                "k_score": ks,
                "v_score": vs_val,
                "c_score": cs_val,
                "profit_ratio": round(profit_ratio, 2),
                "pe": round(pe_val, 0),
                "score": round(tech_score, 1),
            })
        except Exception as e:
            pass

    # 按 score 排序
    verified.sort(key=lambda x: -x["score"])
    verified = verified[:top_n]

    print(f"  技术验证通过: {len(verified)} 只 (Top {top_n})")
    for i, s in enumerate(verified[:10], 1):
        mats = ",".join(s["materials"][:3])
        print(f"    #{i} {s['code']} {s['name']:<8s} {s['score']:+.0f} "
              f"K{s['k_score']:.0f} V{s['v_score']:.0f} C{s['c_score']:.0f} "
              f"PE{s['pe']:.0f} ←{mats}")

    return verified


def run_full_discovery(
    resonant_sectors: list[str],
    target_date: str,
    all_stocks: list[dict] = None,
    top_n: int = 30,
) -> dict:
    """完整瓶颈发现 + 技术验证流程"""
    if all_stocks is None:
        all_stocks = load_all_stock_names()

    print(f"\n{'='*60}")
    print(f"  供应链瓶颈全面发现 — {target_date}")
    print(f"{'='*60}")

    # Phase 1: 名称关键词搜索
    print("\n[Phase 1] 名称关键词搜索...")
    discovery = discover_bottleneck_candidates(
        resonant_sectors=resonant_sectors,
        all_stocks=all_stocks,
    )

    # Phase 1.5: 补充已知生产商 (解决名称不含关键词的公司如昊华科技)
    print("\n[Phase 1.5] 补充已知生产商...")
    discovery["materials_found"] = supplement_known_producers(discovery["materials_found"])

    # 重建 all_candidates
    new_all = {}
    for mat_name, mat_data in discovery["materials_found"].items():
        for s in mat_data["stocks"]:
            code = s["code"]
            if code not in new_all:
                new_all[code] = {
                    "code": code, "name": s.get("name", ""),
                    "materials": [], "layers": [], "categories": [], "archetypes": [],
                }
            if mat_name not in new_all[code]["materials"]:
                new_all[code]["materials"].append(mat_name)
            layer = mat_data.get("layer", "?")
            if layer not in new_all[code]["layers"]:
                new_all[code]["layers"].append(layer)
            cat = mat_data.get("category", "?")
            if cat not in new_all[code]["categories"]:
                new_all[code]["categories"].append(cat)
            for a in mat_data.get("archetypes", []):
                if a not in new_all[code]["archetypes"]:
                    new_all[code]["archetypes"].append(a)
    discovery["all_candidates"] = new_all
    discovery["total_candidates"] = len(new_all)

    # Phase 2: 技术面验证
    print("\n[Phase 2] 技术面验证评分...")
    verified = tech_verify_candidates(
        discovery,
        target_date=target_date,
        top_n=top_n,
    )

    discovery["verified_top"] = verified
    return discovery


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="供应链瓶颈全面发现引擎 v2")
    parser.add_argument("--date", type=str, required=True, help="目标日期")
    parser.add_argument("--sectors", type=str, default=None, help="共振板块(逗号分隔)")
    parser.add_argument("--top", type=int, default=30, help="保留Top N")
    parser.add_argument("--output", type=str, default=None, help="输出JSON路径")
    args = parser.parse_args()

    if args.sectors:
        sectors = [s.strip() for s in args.sectors.split(",")]
    else:
        # 默认覆盖主要板块
        sectors = ["半导体", "PCB", "先进封装", "元件"]

    all_stocks = load_all_stock_names()
    print(f"全A股票: {len(all_stocks)} 只")

    result = run_full_discovery(
        resonant_sectors=sectors,
        target_date=args.date,
        all_stocks=all_stocks,
        top_n=args.top,
    )

    # 输出
    output_path = args.output
    if not output_path:
        output_path = str(ROOT / "output" / args.date / "bottleneck_full.json")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # 精简输出 (去掉 all_candidates 中的冗余信息)
    output = {
        "date": args.date,
        "resonant_sectors": sectors,
        "total_materials_covered": result["total_materials_covered"],
        "total_candidates": result["total_candidates"],
        "by_sector": result["by_sector"],
        "materials_found": {
            k: {
                "layer": v["layer"],
                "category": v["category"],
                "archetypes": v["archetypes"],
                "total": v["total"],
                "stocks": v["stocks"],
            }
            for k, v in result["materials_found"].items()
        },
        "verified_top": result["verified_top"],
    }

    with open(output_path, "w") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\n✅ 已保存: {output_path}")

    # 打印Top 10摘要
    print(f"\n{'='*60}")
    print(f"  🏆 瓶颈标的 Top 10")
    print(f"{'='*60}")
    for i, s in enumerate(result["verified_top"][:10], 1):
        mats = ", ".join(s["materials"][:3])
        print(f"  #{i} {s['code']} {s['name']:<8s} {s['score']:+.0f}分 "
              f"¥{s['price']:.2f} {s['chg_pct']:+.1f}% "
              f"K{s['k_score']:.0f} V{s['v_score']:.0f} PE{s['pe']:.0f} ←{mats}")


if __name__ == "__main__":
    main()
