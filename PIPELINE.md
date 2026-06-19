# 量化短线交易完整流水线

## 架构总览

```
                    全A 5000+ 股票
                         │
        ┌────────────────┼────────────────┐
        ▼                ▼                ▼
   Layer 1           Layer 2           Layer 3
   大盘温度          板块共振          代表股选取
                        │
        ┌───────────────┼───────────────┬──────────────┐
        ▼               ▼               ▼              ▼
   Layer 3.5        Layer 3.6        Layer 3.7       Layer 4
   细分概念+龙头    供应链瓶颈发现    新闻涟漪        消息面催化
                        │
                        ▼
                   Layer 5
                   深度分析 (每只候选股)
                   K线形态 + 量价 + 筹码 + 涨停 + 战法 + 基本面
                   + 《股是股非》C区过滤 + 出货检测
                        │
                        ▼
                   Layer 6
                   7-Agent 辩论 (DeepSeek API)
                        │
                        ▼
                   Layer 7
                   输出 CSV + 生成 HTML/PNG 报告
```

---

## Layer 1: 大盘温度

**模块**: `market_diagnostic.py`

```
输入: 上证指数K线 + 全A 5000+ K线
输出: {temperature, signal, regime, vol_regime}
```

- 温度 0-100: 综合涨跌比、量能、波动率
- 信号: TRADE(交易) / CAUTION(谨慎) / SKIP(观望)
- 模式: offensive(进攻) / normal(观察) / defensive(防御)
- SKIP 时直接终止，不推荐任何标的

---

## Layer 2: 板块共振 Top 3

**模块**: `quick_trade.py::compute_sector_perf()`

```
输入: 5000+ K线 + 板块分类 + 腾讯板块实时行情
输出: Top 3 共振板块 (按 resonance_score 排序)
```

**算法**:
1. 每只股票按 `sector_classification.json` 归类板块
2. 计算每个板块的 1d/5d/10d/20d 中位数收益
3. 减去同期上证指数收益 → 超额收益
4. 归一化后加权求和 → 共振分 (30%×1d + 25%×5d + 25%×10d + 20%×20d)
5. 腾讯板块实时行情修正 1d 收益
6. 过滤通用板块 (沪市主板/创业板等) + 至少5只成分股
7. 取共振分 Top 3

**示例输出**:
```
PCB      共振90分  1d+2.8%  5d+22.2%  10d+15.5%  (174只)
半导体    共振85分  1d+3.8%  5d+19.6%  10d+14.3%  (154只)
先进封装  共振82分  1d+3.8%  5d+15.4%  10d+14.3%  (45只)
```

---

## Layer 3: 代表股初选 (每板块5只)

**策略**: 排除科创板(688)、按当日涨幅×40 + 量比×10 + 均线多头×20 综合评分

```
输入: 板块股票列表 + K线 + 基本面
输出: 每板块 Top 5 代表股
```

每条记录包含: `code, name, close, change_pct, vol_ratio, pe, bull_align, score`

---

## Layer 3.5: 细分概念归类 + 龙头识别

**模块**: `data_loader::eastmoney_concept_blocks()`

```
对每个共振板块的前50只成分股:
  1. 调用东财API查询每只股所属概念板块
  2. 过滤通用标签 (融资融券/沪股通/富时罗素等)
  3. ≥2只共享的细分概念 = 有意义细分
  4. 每个细分内找龙头 (最早涨停+最强量比)
  5. 跨板块补全: 同细分但不在共振板块的标的
```

**输出**: `leader_board (细分→龙头code)`, `meaningful_subs (细分→成分股)`, `cross_sector_additions`

---

## Layer 3.6: 供应链瓶颈发现

**模块**: `bottleneck_discovery.py` (全面发现引擎 v2)

```
输入: Top 3 共振板块名称
流程:
  Phase 1:   38种瓶颈材料 × 5800+股票名称关键词搜索
  Phase 1.5: 从 MATERIAL_GRAPH 补充已知生产商 (解决名称不含关键词的问题)
  Phase 2:   技术面验证评分 (K线×0.35 + 量价×0.3 + 筹码×0.25 + 材料叠加加分)
输出: Top 30 瓶颈标的 + 材料映射 + 瓶颈数据JSON
```

**覆盖材料** (按共振板块自动匹配):
| 共振板块 | 关联瓶颈材料 |
|---------|-------------|
| 半导体 | 硅材料/硅片、光刻胶、电子特气、溅射靶材、湿电子化学品、CMP抛光、高纯石英、光掩模、半导体设备/零部件、IGBT/SiC、EDA/IP、钨材料(电子特气上游) |
| PCB | 铜箔、覆铜板/CCL、高频高速、PCB油墨、PCB制造、散热材料 |
| 先进封装 | 封装基板/载板、EMC、Underfill、键合丝、HBM、TSV |
| 元件 | MLCC、电感/磁材、晶振、高速连接器 |

**注入候选池**: 所有瓶颈标的全部注入 (不只5只)，sector 格式为 `瓶颈:材料名1,材料名2`

---

## Layer 3.7: 新闻驱动涟漪 (双源)

### 源1: 关键词涟漪 (`news_ripple.py`)

```
输入: 东财全球资讯 + 28种材料知识图谱
流程:
  1. 提取关键词 (国家/动作/产品)
  2. 在 MATERIAL_GRAPH 中匹配
  3. 涟漪传播: L1直接标的 → L2关联标的
  4. 技术面验证 (上涨≤2天优先加分)
输出: 通过验证的涟漪标的
```

### 源2: AI新闻推理 (`news_ai_reasoning.py`)

```
输入: 东财全球资讯 → DeepSeek API
流程:
  1. LLM 分析新闻→推理影响事件
  2. 输出: 事件+影响板块+标的类型
  3. 板块→概念→成分股映射
输出: news_ai_stocks.json
```

新闻标的 sector 格式: `新闻AI:特朗普官宣苹果与英特尔合作造芯`

---

## Layer 4: 消息面催化

**数据源**: `news/data/processed/{date}/`

```
对每个板块加载:
  - sector_impacts (sentiment, score, mentions)
  - top_industry_sectors / bottom_industry_sectors
```

在后续深度分析中注入消息面，利好板块+2分、利空-2分。

---

## Layer 5: 深度分析 (每只候选股)

**模块**: `quick_trade.py::deep_analyze()`

### 5.1 K线形态评分
**模块**: `candlestick_patterns.py`
- 识别: 三白兵、启明星、锤子线、吞没形态、十字星等
- 输出: `pattern_score` (-100 ~ +100)

### 5.2 量价关系
**模块**: `volume_price_analyzer.py`
- 量价配合/背离/放量滞涨/缩量上涨
- 输出: `volume_score` (-50 ~ +50)

### 5.3 筹码分布
**模块**: `chip_distribution.py`
- 估算获利盘比例、筹码集中度、支撑/压力峰
- 输出: `chip_score`, `profit_ratio`, `nearest_support/resistance`

### 5.4 涨停分析
**模块**: `limit_up_analyzer.py`
- 6维评分: 封板时间、封单强度、位置、量价配合、板块领导力、K线形态
- 质量标签: 龙头首板(≥80) / 强势涨停(≥65) / 跟风涨停(≥50) / 可疑涨停(<50)
- 延续概率: `continuation_prob`
- **涨停质量覆盖**: 质量≥80 → 量价异常降级不毙; 质量≥65 → 量价异常降为警告

### 5.5 均线+RSI
**模块**: `fetch_a_share_data.py::fetch_technical()`
- 均线多头 +2分; RSI 30-65健康 +1分; RSI>80超买 -1分

### 5.6 估值过滤
- PE>100 偏高 -1分; PE<20 低估值 +1分

### 5.7 战法匹配
**模块**: `warfare_patterns.py`
- 逼空星线: MACD A/B区 + 缩量星 + 在MA5/10上方
- 拉高抢筹: 量比>2 + 涨幅>4.5% + 阳体>3% + 收盘近高>98%
- A区起涨: 价格A + 量A + 增量
- 猎取B区: MA10上穿MA30首次 + MA10上升

### 5.8 《股是股非》策略过滤 (NEW)
**模块**: `zhangting_strategies.py`

| 检测项 | 动作 |
|--------|------|
| **C区风险** (高位倒灌/均线死叉/深度破位) | **硬毙** `signal=PASS` |
| **出货信号** (高位倒灌/阳奉阴违/放量滞涨) | 扣2分; 高位倒灌直接毙 |
| **量价异动** (底部放量突破/地量后倍量等) | 加2-3分 |
| **均线归位** (散乱→多头排列) | 加3分 |
| **洗盘反包** (大阴次日阳包阴+量确认) | 加4分 |

### 5.9 硬过滤 (不可豁免)
1. **亏损股**: PE<0 → 毙
2. **ST股**: 名称含ST → 毙
3. **当日大跌**: 跌>3% → 毙 (已涨停除外)
4. **C区风险** (NEW) → 毙

### 5.10 综合评分
```
net_score = bullish - bearish
signal:
  net >= 5  → STRONG_BUY
  net >= 2  → BUY
  net < 2   → PASS (不推荐)
```

### 5.11 价位计算 (短线用MA5/MA10)
- 支撑: max(筹码支撑峰, MA10, MA5) 下方
- 阻力: min(筹码压力峰, 60日高)
- 止损: 支撑 × 0.97
- 入场区间: [支撑, 现价]

---

## Layer 6: 7-Agent 辩论

**模块**: `agent_debate.py` (DeepSeek API)

```
Phase 1: 数据汇总 (已完成)
Phase 2: Bull → Bear(反驳) → Risk(压力测试)
Phase 3-5: Research Manager → Trader → Portfolio Manager

输出格式:
  FINAL: BUY/SELL/HOLD
  BULL: 最强看多
  BEAR: 最强看空
  RISK: 最大风险
  ENTRY/STOP/TARGET/SIZE
  VERDICT: 一句话判决
```

- SELL/HOLD → 降级为 PASS
- BUY → 采用 Agent 的入场/止损/止盈价位
- API 不可用时降级为规则辩论 (基于致命风险计数)

---

## Layer 7: 输出

### 7.1 CSV 保存
`output/{date}/trade_recommendations.csv`
包含所有 BUY 信号的完整字段 (30+列)

### 7.2 HTML/PNG 报告
**模块**: `generate_report.py` (Playwright 2x retina)

**报告布局**:
```
┌──────────────────────────────────────────────┐
│  深色Header: 标题 · 温度 · 信号 · 模式 · 板块 │
├─────────────────┬────────────────────────────┤
│  共 板块共振    │  战 战法信号                │
│  5行 × 6列     │  5行 × 7列(含战法标签)     │
│  (共振板块Top5) │  (战法总分排序)            │
├─────────────────┼────────────────────────────┤
│  链 供应链瓶颈  │  闻 新闻涟漪                │
│  N行+材料标签  │  5行+新闻主题              │
│  (bottleneck   │  (AI推理+关键词)           │
│   _full.json)  │                            │
├─────────────────┴────────────────────────────┤
│  🥇 Top1  │  🥈 Top2  │  🥉 Top3             │
│  (net_score前三, 三列等宽奖牌卡)             │
├──────────────────────────────────────────────┤
│  回测验证: 前日推荐→今日实际表现              │
└──────────────────────────────────────────────┘
```

---

## 数据源总览

| 层级 | 模块 | 数据源 | 说明 |
|------|------|--------|------|
| 行情 | data_loader | mootdx TCP + 腾讯HTTP | K线/实时价/PE/PB/市值 |
| 板块 | sector_classification.json + 东财slist | 股票→板块映射 |
| 概念 | eastmoney_concept_blocks | 每只股所属概念标签 |
| 基本面 | tencent_quote | PE/PB/市值/换手率 |
| 新闻 | eastmoney_global_news | 7×24快讯 |
| AI | DeepSeek V4 Pro API | Agent辩论 + 新闻推理 |
| 瓶颈 | all_stocks.json (5800只) | 名称关键词搜索 + MATERIAL_GRAPH补充 |
| 回测 | K线比较 | 前日推荐 vs 次日实际涨跌 |

---

## 涉及模块清单

```
scripts/
├── quick_trade.py              # 主流水线 (7层)
├── market_diagnostic.py        # Layer 1: 大盘温度
├── data_loader.py              # 数据加载 (K线/基本面/新闻/概念)
├── candlestick_patterns.py     # K线形态识别
├── volume_price_analyzer.py    # 量价关系分析
├── chip_distribution.py        # 筹码分布估算
├── limit_up_analyzer.py        # 涨停质量+延续概率
├── warfare_patterns.py         # 四大战法 (逼空星线/拉高抢筹/A区起涨/猎取B区)
├── zhangting_strategies.py     # NEW 《股是股非》6大策略
├── bottleneck_discovery.py     # NEW 供应链瓶颈全面发现
├── news_ripple.py              # 新闻涟漪 (关键词)
├── news_ai_reasoning.py        # 新闻AI推理 (LLM)
├── agent_debate.py             # 7-Agent辩论 (API)
├── supply_chain_mapper.py      # 旧版供应链映射 (备用)
├── fetch_a_share_data.py       # 技术面/消息面/宏观
├── generate_report.py          # HTML/PNG报告生成
config/
├── factor_weights.yaml         # 因子权重
├── factors.yaml                # 因子注册
├── supply_chain.yaml           # 产业链知识库
├── sector_classification.json  # 板块分类缓存
data/
├── all_stocks.json             # 全A 5800只股票名称
├── stocks/*.parquet            # K线缓存
output/{date}/
├── trade_recommendations.csv   # 推荐清单
├── bottleneck_full.json        # 瓶颈发现结果
├── news_ai_stocks.json         # AI推理标的
├── quant_report.html           # HTML报告
└── quant_report.png            # PNG截图 (2x retina)
```
