# 新闻驱动 × 供应链涟漪 — 开发文档

## 1. 痛点与目标

### 当前盲区

我们的板块共振选的是**已经涨完的板块**（PCB/半导体/先进封装都是涨了5天10天才被识别）。新闻驱动要做的是**提前发现还没涨的板块**。

### 核心逻辑（以钨为例）

```
新闻: "日本拟限制六氟化钨对华出口"
  │
  ├─ 第一层·直接冲击
  │   受影响产品: 六氟化钨(WF6)
  │   国产替代标的: 钨业相关 → 厦门钨业、中钨高新、章源钨业
  │   ↑ 这是任何普通交易者都能想到的
  │
  └─ 第二层·供应链涟漪 ← 我们的超额收益来源
      日本→中国高端钨产品的其他品类:
        ├─ 钨靶材(半导体溅射)   → 江丰电子、阿石创
        ├─ 钨丝(光伏切割)       → 中钨高新、厦门钨业
        ├─ 钨粉/碳化钨(刀具)    → 章源钨业、翔鹭钨业
        └─ 钨电极(焊接)         → 安泰科技
      ↑ 聪明钱在炒第一层时，我们已埋伏第二层
```

### 三层涟漪模型

```
Layer 0: 新闻事件
  识别: 供应中断/出口限制/技术封锁/事故停产/政策变化

Layer 1: 直接冲击 (新闻→产品→国内替代)
  六氟化钨断供 → 谁在国内生产六氟化钨? → 钨业股票

Layer 2: 供应链涟漪 (产品→关联产品→关联标的)
  日本还出口什么钨产品? → 钨靶材/钨丝/钨粉 → 对应标的

Layer 3: 瓶颈扩散 (涟漪产品→其供应链瓶颈层)
  钨靶材的上游是什么? → 高纯钨粉 → 谁做高纯钨粉?
```

---

## 2. 系统架构

```
┌─────────────────────────────────────────────────────────────┐
│                    News Ingestion Layer                      │
│                                                             │
│  东财全球资讯(eastmoney_global_news)    ← 7×24快讯           │
│  东财个股新闻(eastmoney_stock_news)     ← 个股相关            │
│  news项目(selected_sectors.json)        ← 板块情绪            │
│                                                             │
│  关键词提取: 产品名/国家名/限制类型/产业链节点                  │
│  例: "日本" + "限制" + "六氟化钨" + "出口"                    │
│      → {国家:日本, 动作:限制, 产品:六氟化钨, 类型:出口管制}    │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                  Material Knowledge Graph                     │
│                     (新增: config/material_graph.yaml)        │
│                                                             │
│  六氟化钨(WF6):                                              │
│    category: 电子特气                                         │
│    source_countries: [日本, 中国]                             │
│    domestic_producers: [厦门钨业, 中钨高新, ...]               │
│    related_products:                                          │
│      - 钨靶材 → 溅射靶材 → 半导体薄膜沉积                     │
│      - 钨丝   → 光伏切割耗材                                  │
│      - 钨粉   → 硬质合金刀具                                  │
│      - 钨电极 → 氩弧焊耗材                                    │
│    supply_chain_role: L5_材料                                 │
│    bottleneck_prototype: 单源供应商                            │
│                                                             │
│  钨靶材:                                                      │
│    category: 溅射靶材                                         │
│    upstream: [高纯钨粉]                                       │
│    domestic_producers: [江丰电子, 阿石创, 隆华科技]            │
│    related_concepts: [半导体设备, 先进封装]                    │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                    Ripple Engine                              │
│                                                             │
│  Step 1: 新闻 → 受影响产品 (关键词匹配 material_graph)         │
│  Step 2: 受影响产品 → 直接受益标的 (domestic_producers)        │
│  Step 3: 受影响产品 → 关联产品 (related_products)              │
│  Step 4: 关联产品 → 涟漪受益标的 (their domestic_producers)    │
│  Step 5: 对每层标的做 K线+量价+筹码 技术验证                   │
│                                                             │
│  Ripple Score =                                              │
│    新闻冲击力(0-1) × 国产替代紧急性 × 技术面确认 × 层数衰减     │
│    Layer1权重 1.0  /  Layer2权重 0.7  /  Layer3权重 0.4       │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│              现有选股管线的增强                                 │
│                                                             │
│  quick_trade.py 新增选股来源:                                  │
│    - "板块共振" (已有)                                         │
│    - "供应链瓶颈" (已有)                                       │
│    - "新闻驱动·直接冲击" (新增) ← ripple_layer_1               │
│    - "新闻驱动·涟漪效应" (新增) ← ripple_layer_2               │
│                                                             │
│  报告新增板块:                                                 │
│    📰 新闻驱动标的 (单独卡片, 标注涟漪层数和新闻来源)             │
└─────────────────────────────────────────────────────────────┘
```

---

## 3. 新增文件清单

| 文件 | 作用 |
|------|------|
| `config/material_graph.yaml` | **新材料知识图谱** — 产品→国内替代标的→关联产品→供应链角色 |
| `scripts/news_ripple.py` | **涟漪引擎** — 新闻关键词提取 + 材料匹配 + 涟漪传播 + 标的发现 |
| `scripts/news_ingestion.py` | 新闻采集增强 — 汇聚东财全球资讯+news项目，做关键词结构化提取 |
| `DESIGN_NEWS_RIPPLE.md` | 本文档 |

### 修改文件

| 文件 | 改动 |
|------|------|
| `scripts/quick_trade.py` | 新增选股来源: "新闻驱动" → 调用 ripple 引擎 |
| `config/supply_chain.yaml` | 补充钨/稀土/电子特气等材料类条目 |
| `output/.../quant_report.html` | 新增"📰 新闻驱动"卡片，样式匹配现有报告 |

---

## 4. 核心实现

### 4.1 `material_graph.yaml` 结构

```yaml
materials:
  六氟化钨:
    aliases: [WF6, 六氟]
    category: 电子特气
    source_countries: [日本, 中国]
    restriction_risk: high     # 被限制的概率
    domestic_producers:         # Layer1 直接受益
      - code: "600549"         # 厦门钨业
        name: 厦门钨业
        relevance: 0.9
      - code: "000657"         # 中钨高新
        name: 中钨高新
        relevance: 0.9
    related_products:           # Layer2 涟漪
      - product: 钨靶材
        relation: "日本同源出口管制风险"
        ripple_probability: 0.8
      - product: 钨丝
        relation: "同材质光伏耗材"
        ripple_probability: 0.5
      - product: 钨粉
        relation: "上游粉体材料"
        ripple_probability: 0.6
    supply_chain:
      layer: L5
      role: 半导体制造材料

  钨靶材:
    aliases: [钨溅射靶, W靶材]
    category: 溅射靶材
    domestic_producers:
      - code: "300666"         # 江丰电子
        name: 江丰电子
        relevance: 1.0
      - code: "300706"         # 阿石创
        name: 阿石创
        relevance: 0.8
    related_concepts: [半导体设备, 先进封装]

  # 可扩展更多材料...
```

### 4.2 `news_ripple.py` 核心逻辑

```python
def extract_keywords_from_news(news_text: str) -> list[dict]:
    """从新闻中提取结构化关键词"""
    # 匹配: 国家名 + 限制动词 + 产品名
    countries = ["日本", "美国", "韩国", "荷兰", "德国", "台湾"]
    actions = ["限制", "断供", "制裁", "禁止", "出口管制", "封锁", "停产"]
    # → 返回 [{country, action, product, confidence}]


def match_material(product_name: str) -> dict:
    """在 material_graph 中匹配受影响材料"""
    # 精确匹配 + 别名匹配 + 模糊匹配


def ripple_propagate(material: dict, layers: int = 2) -> dict:
    """
    涟漪传播。
    Layer1: material.domestic_producers → 直接标的
    Layer2: material.related_products[].domestic_producers → 涟漪标的
    Returns: {layer1: [...], layer2: [...]}
    """


def score_ripple_stocks(ripple_stocks: list, kline_map: dict) -> list:
    """
    对涟漪标的做技术面验证。
    - 形态: 是否有底部突破/均线粘合
    - 量价: 是否有放量异动
    - 筹码: 是否有主力吸筹
    - 大盘位置: 是否还没涨（还没被市场发现）
    → 优先推荐"还没涨"的 Layer2 标的（预期差最大）
    """
```

### 4.3 关键词提取策略

```python
# 预定义的产品名→材料映射（高频词直接命中）
PRODUCT_KEYWORDS = {
    "六氟化钨": "六氟化钨",
    "WF6": "六氟化钨",
    "光刻胶": "光刻胶",
    "氟化氢": "氟化氢",
    "氟聚酰亚胺": "氟聚酰亚胺",
    "高纯氖气": "氖气",
    "EDA": "EDA软件",
    "离子注入": "离子注入机",
    # ...可扩展至100+关键词
}

# 国家→限制动作→产业的推理链
RESTRICTION_PATTERNS = [
    (r"(日本|美国|荷兰).{0,10}(限制|断供|禁止|出口管制).{0,20}(出口|供应)"),
    (r"(限制|断供|禁止).{0,10}(对华|对中国).{0,20}(出口|供应)"),
    (r"(停产|事故|爆炸).{0,10}(工厂|产线|供应)"),
]
```

---

## 5. 与现有管线集成

### `quick_trade.py` 增强

```python
# 在 Layer 3.6 (供应链瓶颈) 之后新增:

# ========== Layer 3.7: 新闻驱动涟漪 ==========
print("\n[Layer 3.7] 新闻驱动涟漪分析...")
from news_ripple import fetch_recent_news, extract_ripple_stocks

news_items = fetch_recent_news(days=7)  # 近7天新闻
ripple_results = extract_ripple_stocks(news_items, kline_map)

if ripple_results:
    for item in ripple_results:
        print(f"  📰 {item['news_title'][:40]}...")
        print(f"     Layer1直接: {len(item['layer1'])}只")
        print(f"     Layer2涟漪: {len(item['layer2'])}只")
        # 涟漪标的加入候选池
        for s in item['layer1'] + item['layer2']:
            all_picks.append({...})
```

### 报告新增卡片

在现有报告的基础上，如果当天有新闻驱动标的，新增一个卡片：

```
┌────────────────────────────────────────┐
│ 📰 新闻驱动标的                          │
│                                        │
│ 新闻: "日本拟限制六氟化钨对华出口" (06-15) │
│                                        │
│ Layer1·直接冲击                         │
│   厦门钨业  +8.2%  K线+45  量价+12      │
│   中钨高新  +6.1%  K线+38  量价+8       │
│                                        │
│ Layer2·供应链涟漪 ← 预期差最大            │
│   江丰电子(钨靶材)  +3.2%  K线+22  ← 还没涨│
│   阿石创(钨靶材)    +1.8%  K线+18  ← 埋伏 │
│                                        │
│ 涟漪逻辑: 日本还出口钨靶材→同源管制风险    │
└────────────────────────────────────────┘
```

---

## 6. 与供应链瓶颈 skill 的协同

两个 skill 天然互补：

| 维度 | 供应链瓶颈 | 新闻驱动涟漪 |
|------|-----------|------------|
| 触发方式 | 主线发现→拆链 | 新闻关键词→材料图谱 |
| 标的来源 | 瓶颈层知识库 | 国内替代+关联产品 |
| 时间维度 | 主线已经热了 | 新闻刚出/还没涨 |
| 预期差 | 中等（追趋势） | **大（埋伏）** |

**协同逻辑**：新闻找到的材料 → 查询其供应链角色 → 如果是瓶颈材料 → 涟漪加倍（因为既有新闻催化又有瓶颈稀缺性）。

```
例子: 六氟化钨
  新闻识别 → 材料图谱匹配 → 查询 supply_chain.yaml →
  发现它是 "半导体 L5 材料 · 耗材绑定 · 认证壁垒" →
  涟漪标的既是新闻驱动又是瓶颈卡位 → 双逻辑共振 → 最高优先级推荐
```

---

## 7. 实施计划

### Phase 1: 数据底座 (0.5天)

- [ ] 创建 `config/material_graph.yaml`，覆盖 20+ 关键材料（钨/稀土/光刻胶/电子特气/高纯材料）
- [ ] 每个材料录入: 别名、国产替代标的、关联产品、供应链角色
- [ ] 创建 `scripts/news_ingestion.py`，聚合东财全球资讯+news项目 JSON

### Phase 2: 涟漪引擎 (0.5天)

- [ ] 创建 `scripts/news_ripple.py`
- [ ] 关键词提取 + 材料匹配 + 涟漪传播 + 技术面验证
- [ ] CLI 测试: `python3 scripts/news_ripple.py --days 7`

### Phase 3: 管线集成 (0.5天)

- [ ] `quick_trade.py` 新增 Layer 3.7
- [ ] 涟漪标的注入候选池，标注 `source="新闻驱动"`
- [ ] 报告新增 "📰 新闻驱动" 卡片

### Phase 4: 持续扩充 (持续)

- [ ] 每次遇到新的供应链新闻，在 `material_graph.yaml` 中追加
- [ ] 扩充关键词库和限制模式匹配
- [ ] 回测: 历史新闻→涟漪标的→实际涨幅验证

---

## 8. 关键设计决策

1. **material_graph 是核心资产**：这是人工整理的知识库，不是算法自动发现的。AI 助手可以帮助扩充，但需要人工验证。

2. **涟漪 Layer2 才是超额收益来源**：Layer1 所有人都能想到，Layer2 需要产业链知识。我们的优势在于有供应链拆链能力。

3. **新闻驱动不替代板块共振**：两者是互补的——板块共振做已热的、新闻驱动做还没热的。

4. **技术面验证必不可少**：不是所有涟漪标的都值得买。必须过 K线+量价+筹码 验证，只推荐"新闻催化 + 技术面启动"的。

5. **与 supply_chain_mapper 深度绑定**：涟漪材料如果在瓶颈层，优先级最高（双逻辑共振）。
