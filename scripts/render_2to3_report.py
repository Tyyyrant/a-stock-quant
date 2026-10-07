#!/usr/bin/env python3
"""
2进3 打板分析 —— 什么形态的「2板」后续空间更大
独立分析，只读 data/board_dataset.csv（3板+连板样本）。
特征一律取 2板；结果看 后续最高板(空间)。
输出 output/2to3_report.png
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CSV = ROOT / 'data' / 'board_dataset.csv'
OUT = ROOT / 'output' / '2to3_report.png'

plt.rcParams['font.sans-serif'] = ['Hiragino Sans GB', 'STHeiti', 'Arial Unicode MS', 'PingFang HK']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['axes.edgecolor'] = '#dddddd'

C_ONE = '#e5484d'   # 一字
C_T = '#f5a623'     # T字
C_H = '#4f7cd6'     # 换手
C_LOCK = '#e5484d'  # 缩量锁仓
C_MID = '#4f7cd6'
C_BOOM = '#9aa0a6'  # 爆量
C_TXT = '#1a1d21'


def main():
    df = pd.read_csv(CSV, converters={'code': lambda x: str(x).zfill(6)})
    df['d2_hs'] = pd.to_numeric(df['d2_hs'], errors='coerce')
    df['peak'] = pd.to_numeric(df['peak_board'], errors='coerce')
    N = len(df)

    def stat(g):
        return {'n': len(g), 'avg': g['peak'].mean(),
                'p4': (g['peak'] >= 4).mean() * 100,
                'p5': (g['peak'] >= 5).mean() * 100,
                'p6': (g['peak'] >= 6).mean() * 100}

    # 2板板型
    pts = ['一字板', 'T字板', '换手板']
    pt_stat = {p: stat(df[df['d2_ptype'] == p]) for p in pts}
    # 2板换手
    hs = [('<3%', 0, 3), ('3-8%', 3, 8), ('8-15%', 8, 15), ('15-25%', 15, 25), ('>25%', 25, 999)]
    hs_stat = {lab: stat(df[(df['d2_hs'] >= lo) & (df['d2_hs'] < hi)]) for lab, lo, hi in hs}

    # ---- 画布 ----
    fig = plt.figure(figsize=(14, 17.5), dpi=110, facecolor='white')
    gs = GridSpec(5, 2, figure=fig, height_ratios=[0.62, 1.5, 1.6, 1.9, 0.62],
                  hspace=0.5, wspace=0.22, left=0.07, right=0.95, top=0.945, bottom=0.035)

    fig.text(0.07, 0.975, '2进3 打板分析：什么形态的「2板」后续空间更大', fontsize=24, fontweight='bold', color=C_TXT)
    fig.text(0.07, 0.955,
             f'样本 {N} 段成功2→3（已走到3板+，10cm主板，2024-01~2026-09） ｜  特征=2板，结果=后续最高板(空间) ｜  一字板2板=开盘即封死，买不进',
             fontsize=11.5, color='#555b60')

    # ===== 图1：2板板型 → 平均最高板 =====
    ax1 = fig.add_subplot(gs[1, 0])
    vals = [pt_stat[p]['avg'] for p in pts]
    colors = [C_ONE, C_T, C_H]
    bars = ax1.bar(pts, vals, color=colors, width=0.58)
    for b, p in zip(bars, pts):
        ax1.text(b.get_x() + b.get_width()/2, b.get_height() + 0.06,
                 f'{pt_stat[p]["avg"]:.2f}板', ha='center', fontsize=15, fontweight='bold', color=C_TXT)
        ax1.text(b.get_x() + b.get_width()/2, b.get_height() + 0.03,
                 f'到4板+ {pt_stat[p]["p4"]:.0f}% ｜ N={pt_stat[p]["n"]}', ha='center', fontsize=9, color='#777')
    ax1.annotate('买不进', xy=(0, vals[0]), xytext=(0.42, 4.42), fontsize=11.5, color=C_ONE,
                 fontweight='bold', ha='center', arrowprops=dict(arrowstyle='->', color=C_ONE, lw=1.4))
    ax1.set_title('2板板型 → 平均最高板（空间）', fontsize=13.5, fontweight='bold', loc='left')
    ax1.set_ylim(3.4, 4.7)
    ax1.set_ylabel('平均最高板')
    ax1.spines[['top', 'right']].set_visible(False)
    ax1.grid(axis='y', alpha=0.25)

    # ===== 图2：2板换手率 → 到4板+率 =====
    ax2 = fig.add_subplot(gs[1, 1])
    labs = [h[0] for h in hs]
    vals = [hs_stat[l]['p4'] for l in labs]
    colors = [C_LOCK, C_MID, C_MID, C_MID, C_BOOM]
    bars = ax2.bar(labs, vals, color=colors, width=0.6)
    for b, v in zip(bars, vals):
        ax2.text(b.get_x() + b.get_width()/2, b.get_height() + 1,
                 f'{v:.0f}%', ha='center', fontsize=13, fontweight='bold', color=C_TXT)
    ax2.annotate('缩量锁仓', xy=(0, vals[0]), xytext=(0.1, vals[0] + 8), fontsize=10, color=C_LOCK,
                 fontweight='bold', ha='center')
    ax2.annotate('爆量分歧', xy=(4, vals[4]), xytext=(3.7, vals[4] + 10), fontsize=10, color=C_BOOM, ha='center')
    ax2.set_title('2板换手率 → 走到4板+的概率', fontsize=13.5, fontweight='bold', loc='left')
    ax2.set_ylim(0, 72)
    ax2.set_ylabel('到4板+ %')
    ax2.spines[['top', 'right']].set_visible(False)
    ax2.grid(axis='y', alpha=0.25)

    # ===== 图3：可打的2板 T字 vs 换手 → 空间分布 =====
    ax3 = fig.add_subplot(gs[2, :])
    x = np.arange(3)
    w = 0.36
    labels = ['走到4板+', '走到5板+', '走到6板+']
    t_vals = [pt_stat['T字板']['p4'], pt_stat['T字板']['p5'], pt_stat['T字板']['p6']]
    h_vals = [pt_stat['换手板']['p4'], pt_stat['换手板']['p5'], pt_stat['换手板']['p6']]
    b1 = ax3.bar(x - w/2, t_vals, w, color=C_T, label='T字板(炸板回封)')
    b2 = ax3.bar(x + w/2, h_vals, w, color=C_H, label='换手板(盘中封板)')
    for bars in (b1, b2):
        for b in bars:
            ax3.text(b.get_x() + b.get_width()/2, b.get_height() + 1.0,
                     f'{b.get_height():.0f}%', ha='center', fontsize=12, fontweight='bold', color=C_TXT)
    ax3.set_title('可打的2板：T字板 vs 换手板 → 后续空间（走到几板的概率）', fontsize=14, fontweight='bold', loc='left')
    ax3.set_xticks(x)
    ax3.set_xticklabels(labels)
    ax3.set_ylim(0, 62)
    ax3.set_ylabel('概率 %')
    ax3.legend(fontsize=11, frameon=False, loc='upper right')
    ax3.spines[['top', 'right']].set_visible(False)
    ax3.grid(axis='y', alpha=0.25)

    # ===== 图4：数据表 =====
    ax4 = fig.add_subplot(gs[3, :])
    ax4.axis('off')
    rows = [
        ['2板形态', '样本', '平均最高板', '走到4板+', '走到5板+', '走到6板+', '能否打板'],
        ['一字板(开盘涨停不破)', f"{pt_stat['一字板']['n']}", f"{pt_stat['一字板']['avg']:.2f}板",
         f"{pt_stat['一字板']['p4']:.1f}%", f"{pt_stat['一字板']['p5']:.1f}%", f"{pt_stat['一字板']['p6']:.1f}%", '买不进'],
        ['T字板(炸板回封)', f"{pt_stat['T字板']['n']}", f"{pt_stat['T字板']['avg']:.2f}板",
         f"{pt_stat['T字板']['p4']:.1f}%", f"{pt_stat['T字板']['p5']:.1f}%", f"{pt_stat['T字板']['p6']:.1f}%", '可打'],
        ['换手板(盘中拉升)', f"{pt_stat['换手板']['n']}", f"{pt_stat['换手板']['avg']:.2f}板",
         f"{pt_stat['换手板']['p4']:.1f}%", f"{pt_stat['换手板']['p5']:.1f}%", f"{pt_stat['换手板']['p6']:.1f}%", '可打'],
        ['', '', '', '', '', '', ''],
        ['缩量<3%(锁仓)', f"{hs_stat['<3%']['n']}", f"{hs_stat['<3%']['avg']:.2f}板",
         f"{hs_stat['<3%']['p4']:.1f}%", f"{hs_stat['<3%']['p5']:.1f}%", f"{hs_stat['<3%']['p6']:.1f}%", '最强换手'],
        ['爆量>25%(分歧)', f"{hs_stat['>25%']['n']}", f"{hs_stat['>25%']['avg']:.2f}板",
         f"{hs_stat['>25%']['p4']:.1f}%", f"{hs_stat['>25%']['p5']:.1f}%", f"{hs_stat['>25%']['p6']:.1f}%", '最弱'],
    ]
    tbl = ax4.table(cellText=rows, loc='center', cellLoc='center', bbox=[0.02, 0.1, 0.96, 0.85])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10.5)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor('#e0e0e0')
        if r == 0:
            cell.set_facecolor('#f0f2f5')
            cell.set_text_props(fontweight='bold')
        if c == 0 and r in (1, 2, 3, 5, 6):
            cell.set_text_props(fontweight='bold')
        if c == 1 and r in (1, 2, 3):
            cell.set_text_props(fontweight='bold')
        cell.set_height(0.11)
    ax4.set_title('数据总表', fontsize=13.5, fontweight='bold', loc='left')

    # ===== 结论 =====
    concl = ('结论：① 2板一字板后续空间最大(平均4.19板、55%到4板+)但开盘即封死、买不进，只能当"看"的辨识信号；'
             '② 可打的2板里 T字板(平均4.02板、47.5%到4板+) > 换手板(3.78板、41.7%)；'
             '③ 换手维度：2板缩量锁仓(<3%)最强(51%到4板+)，爆量(>25%)最弱(38%)。'
             '打法：2板优先打 T字板/换手板里缩量(<3%)的，避开爆量分歧的。')
    fig.text(0.07, 0.02, concl, fontsize=11.5, color='#333', va='bottom', ha='left',
             bbox=dict(boxstyle='round,pad=0.6', facecolor='#fff4f4', edgecolor=C_ONE, alpha=0.55))

    fig.savefig(OUT, facecolor='white', bbox_inches='tight')
    print(f'已生成 {OUT}')


if __name__ == '__main__':
    main()
