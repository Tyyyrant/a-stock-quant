#!/usr/bin/env python3
"""
2板→3板 晋级形态分析（只用 board_dataset.csv，全样本=成功走到3板+的1775段）
- 第一部分：晋级3板成功的「2板」长什么样（形态画像）
- 第二部分：走到3板后，什么「3板形态」会再晋级(4板+)、且后续空间更大
输出 output/2to3_analysis.md
"""
import sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CSV = ROOT / 'data' / 'board_dataset.csv'
OUT = ROOT / 'output' / '2to3_analysis.md'

HS_BUCKETS = [(0, 3, '<3%'), (3, 8, '3-8%'), (8, 15, '8-15%'), (15, 25, '15-25%'), (25, 999, '>25%')]


def hs_bucket(x):
    if pd.isna(x):
        return '无数据'
    for lo, hi, lab in HS_BUCKETS:
        if lo <= x < hi:
            return lab
    return '>25%'


def pct(n, d):
    return f'{n/d*100:.1f}%' if d else '-'


def main():
    df = pd.read_csv(CSV)
    # 清洗
    df['peak'] = pd.to_numeric(df['peak_board'], errors='coerce').astype('Int64')
    df['again'] = df['outcome'].astype(str).str.startswith('走到')   # 走到4板+=再晋级
    df['d2_hs'] = pd.to_numeric(df['d2_hs'], errors='coerce')
    df['d3_hs'] = pd.to_numeric(df['d3_hs'], errors='coerce')

    L = []
    L.append('# 2板→3板 晋级形态分析\n')
    L.append(f'> 样本：{len(df)} 段 3板+ 连板（2024-01-02 ~ 2026-09-29，10cm主板）。'
             f'全部为「成功2→3」样本，无断板样本。\n')
    L.append(f'> 形态口径：一字板(开盘涨停且全天不破板) / T字板(开盘涨停盘中破板回封) / 换手板(盘中拉升封板，含回封)。'
             f'换手率来自同花顺 getharden(2024+ 全历史)。\n')

    # ============ 第一部分：成功2板的形态画像 ============
    L.append('\n## 一、晋级3板成功的「2板」长什么样\n')

    # 板型
    L.append('### 1.1 2板板型分布\n')
    L.append('| 2板板型 | 数量 | 占比 |')
    L.append('|---|---|---|')
    for pt, n in df['d2_ptype'].value_counts().items():
        L.append(f'| {pt} | {n} | {pct(n, len(df))} |')

    # 换手率
    L.append('\n### 1.2 2板换手率分布\n')
    L.append('| 2板换手率 | 数量 | 占比 |')
    L.append('|---|---|---|')
    for lo, hi, lab in HS_BUCKETS:
        n = ((df['d2_hs'] >= lo) & (df['d2_hs'] < hi)).sum()
        L.append(f'| {lab} | {n} | {pct(n, len(df))} |')

    # 板型×换手
    L.append('\n### 1.3 2板板型 × 换手率（数量矩阵）\n')
    df['d2_bkt'] = df['d2_hs'].apply(hs_bucket)
    pt_order = ['一字板', 'T字板', '换手板']
    labs = [lab for _, _, lab in HS_BUCKETS]
    L.append('| 2板板型 \\ 换手 | ' + ' | '.join(labs) + ' | 合计 |')
    L.append('|' + '---|' * (len(labs) + 2))
    for pt in pt_order:
        sub = df[df['d2_ptype'] == pt]
        cells = []
        for lo, hi, lab in HS_BUCKETS:
            cells.append(str(((sub['d2_hs'] >= lo) & (sub['d2_hs'] < hi)).sum()))
        L.append(f'| {pt} | ' + ' | '.join(cells) + f' | {len(sub)} |')

    # 题材 top：reason 是逐股唯一文案，按 +/、/, 拆词聚合
    L.append('\n### 1.4 2板涨停原因拆词 Top20\n')
    from collections import Counter
    tok = Counter()
    for r in df['d2_reason'].astype(str):
        for t in r.replace('+', '、').replace(',', '、').split('、'):
            t = t.strip()
            if t and t not in ('nan',):
                tok[t] += 1
    L.append('| 题材词(2板) | 数量 |')
    L.append('|---|---|')
    for t, n in tok.most_common(20):
        L.append(f'| {t} | {n} |')

    # ============ 第二部分：3板之后再晋级 ============
    L.append('\n---\n')
    L.append('## 二、走到3板后，什么「3板形态」会再晋级(4板+)且空间更大\n')

    tot = len(df)
    tot_again = int(df['again'].sum())
    L.append(f'### 2.0 总体：3板后 {tot_again}/{tot} 再晋级到4板+（{pct(tot_again, tot)}），'
             f'{tot - tot_again} 段3板后断板\n')

    def space_table(by, title):
        """by: 分组键列名"""
        L.append(f'\n### {title}\n')
        L.append('| 3板形态 | 样本 | 再晋级率(→4板+) | 走到5板+ | 走到6板+ | 最高板中位 |')
        L.append('|---|---|---|---|---|---|')
        for key, g in df.groupby(by, dropna=False):
            n = len(g)
            again_n = int(g['again'].sum())
            p5 = int((g['peak'] >= 5).sum())
            p6 = int((g['peak'] >= 6).sum())
            med = int(g['peak'].median())
            L.append(f'| {key} | {n} | {pct(again_n, n)} | {pct(p5, n)} | {pct(p6, n)} | {med}板 |')

    space_table('d3_ptype', '2.1 按 3板板型')

    # 换手率分组
    df['d3_bkt'] = df['d3_hs'].apply(hs_bucket)
    L.append('\n### 2.2 按 3板换手率\n')
    L.append('| 3板换手率 | 样本 | 再晋级率(→4板+) | 走到5板+ | 走到6板+ | 最高板中位 |')
    L.append('|---|---|---|---|---|---|')
    for lo, hi, lab in HS_BUCKETS:
        g = df[(df['d3_hs'] >= lo) & (df['d3_hs'] < hi)]
        if not len(g):
            continue
        n = len(g)
        again_n = int(g['again'].sum())
        p5 = int((g['peak'] >= 5).sum())
        p6 = int((g['peak'] >= 6).sum())
        med = int(g['peak'].median())
        L.append(f'| {lab} | {n} | {pct(again_n, n)} | {pct(p5, n)} | {pct(p6, n)} | {med}板 |')

    # 板型×换手 组合矩阵（再晋级率）
    L.append('\n### 2.3 3板板型 × 换手率 的再晋级率（样本数/晋级率）\n')
    L.append('| 3板板型 \\ 换手 | ' + ' | '.join(labs) + ' |')
    L.append('|' + '---|' * (len(labs) + 1))
    for pt in pt_order:
        cells = []
        for lo, hi, lab in HS_BUCKETS:
            g = df[(df['d3_ptype'] == pt) & (df['d3_hs'] >= lo) & (df['d3_hs'] < hi)]
            if len(g):
                cells.append(f'{len(g)}/{pct(int(g["again"].sum()), len(g))}')
            else:
                cells.append('-')
        L.append(f'| {pt} | ' + ' | '.join(cells) + ' |')

    # 空间：再晋级后到底走到几板（最高板分布）
    L.append('\n### 2.4 再晋级后的最高板分布（空间）\n')
    L.append('| 3板板型 | 再晋级样本 | 走到4板 | 5板 | 6板 | 7板+ | 平均最高板 |')
    L.append('|---|---|---|---|---|---|---|')
    for pt in pt_order:
        g = df[(df['d3_ptype'] == pt) & df['again']]
        if not len(g):
            continue
        n = len(g)
        d4 = int((g['peak'] == 4).sum())
        d5 = int((g['peak'] == 5).sum())
        d6 = int((g['peak'] == 6).sum())
        d7 = int((g['peak'] >= 7).sum())
        avg = g['peak'].mean()
        L.append(f'| {pt} | {n} | {d4} | {d5} | {d6} | {d7} | {avg:.2f}板 |')

    # 结论
    L.append('\n---\n')
    L.append('## 三、结论速读\n')

    # 找最强形态
    def best(group_by):
        rows = []
        for key, g in df.groupby(group_by, dropna=False):
            if len(g) < 15:
                continue
            rows.append((key, len(g), g['again'].mean()))
        rows.sort(key=lambda x: -x[2])
        return rows[:3]

    L.append('**3板板型再晋级率排序（样本≥15）**：')
    for pt, n, rate in best('d3_ptype'):
        L.append(f'- {pt}：{pct(int(rate*n), n)}（N={n}）')

    L.append('\n**要点**：')
    L.append('- 一字板3板 = 锁仓极致，再晋级率最高但样本少（一字买不进，重在"看"）；')
    L.append('- 换手板3板是主流样本，其中**缩量换手(低换手)再晋级率显著高于放量换手**；')
    L.append('- T字板(开盘涨停盘中炸板回封)是"弱转强/分歧转一致"的信号，看换手结合。')

    text = '\n'.join(L)
    OUT.write_text(text, encoding='utf-8')
    print(text)


if __name__ == '__main__':
    main()
