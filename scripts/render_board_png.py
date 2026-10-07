#!/usr/bin/env python3
"""把情绪盘面 HTML 渲染成整页 PNG，用于分享复盘结果。

依赖: pip install playwright && python -m playwright install chromium
用法: python3 scripts/render_board_png.py [输出.png]
      (缺省输出 output/emotion/emotion_board.png)
"""
import sys
from pathlib import Path
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parent.parent
HTML = ROOT / 'output' / 'emotion' / 'emotion_board.html'
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / 'output' / 'emotion' / 'emotion_board.png'


def main():
    with sync_playwright() as p:
        browser = p.chromium.launch()
        # 宽=盘面 max-width(1180)+边距，scale=2 出高清图
        page = browser.new_page(viewport={'width': 1240, 'height': 900},
                                device_scale_factor=2)
        page.goto(HTML.as_uri())
        page.wait_for_timeout(600)  # 等内联 JS 完成渲染
        page.screenshot(path=str(OUT), full_page=True)
        browser.close()
    print(f'已生成 {OUT}')


if __name__ == '__main__':
    main()
