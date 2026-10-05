from __future__ import annotations

PAGE_TYPES = [
    ("cover",        "封面",     "#C62828"),  # 红
    ("title_page",   "扉页",     "#8E24AA"),  # 紫
    ("color_illus",  "彩色插图", "#D81B60"),  # 洋红
    ("blank",        "空白页",   "#546E7A"),  # 蓝灰
    ("toc_page",     "目录",     "#0277BD"),  # 蓝
    ("illustration", "插图",     "#00796B"),  # 青绿
    ("paragraph",    "正文",     "#1A237E"),  # 深靛蓝
    ("afterword",    "后记",     "#B84E00"),  # 橙褐
    ("colophon",     "版权页",   "#6D4C41"),  # 棕
    ("unknown",      "未分类",   "#827717"),  # 橄榄
]

TYPE_LABEL = {t: l for t, l, _ in PAGE_TYPES}
TYPE_COLOR = {t: c for t, l, c in PAGE_TYPES}
