#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Novel Formatter Studio — 图形界面 v2
按 Page Manager / OCR 适配器 / Formatter Engine / EPUB Builder 四个工作区设计，
每个工作区的视觉结构参照产品原型（缩略图网格 + 类型侧栏 / 适配器列表 + JSON预览 /
Pipeline侧栏 + 前后对比 / 书籍结构树 + 源码预览）。

用法: python3 gui.py
依赖(可选): pip3 install pillow      — 真实缩略图预览
           pip3 install pymupdf     — PDF 输入支持
"""

from __future__ import annotations
import os, sys, json, threading, subprocess, re
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).parent))

try:
    from PIL import Image, ImageTk
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

# ── 配色 & 常量 ────────────────────────────────────────────────────────────────

BG      = "#F3F2EE"
CARD    = "#FFFFFF"
INK     = "#1D1D1B"
MUTED   = "#84837C"
BORDER  = "#E3E1DA"
ACC     = "#4A63D3"
ACC_BG  = "#EEF0FB"

PAGE_TYPES = [
    ("cover",        "封面",     "#C0542F"),
    ("color_illus",  "彩色插图", "#B5417C"),
    ("blank",        "空白页",   "#8C8B84"),
    ("toc_page",     "目录",     "#2C6FB5"),
    ("illustration", "插图",     "#127A56"),
    ("paragraph",    "正文",     "#4A3FA3"),
    ("afterword",    "后记",     "#93650D"),
    ("colophon",     "版权页",   "#5B5A54"),
    ("unknown",      "未分类",   "#AFAEA7"),
]
TYPE_LABEL = {t: l for t, l, _ in PAGE_TYPES}
TYPE_COLOR = {t: c for t, c, _ in PAGE_TYPES}
IMAGE_EXTS = {'.png', '.jpg', '.jpeg', '.heic', '.tif', '.tiff', '.bmp', '.gif'}

OCR_ADAPTERS = [
    ("apple_vision", "Apple Vision OCR", "macOS", "#4A3FA3",
     "Apple Live Text / Vision 框架，竖排识别优先", True),
    ("pdf_craft",    "pdf-craft",        "跨平台", "#127A56",
     "pdf-craft 开源工具输出，支持版面分析", False),
    ("paddle_ocr",   "PaddleOCR",        "跨平台", "#C0542F",
     "百度 PaddleOCR，坐标为像素值数组", False),
    ("google_vision","Google Vision API","云端",   "#93650D",
     "Google Cloud Vision JSON 响应", False),
]

FORMATTER_STEPS = [
    ("clean_metadata",    "清理模块",   "自动",     "删除页码、页眉、出版信息"),
    ("merge_sentences",   "断句修复",   "规则+AI",  "合并OCR错误换行，恢复连续段落"),
    ("remove_duplicates", "重复删除",   "自动",     "删除OCR扫描产生的重复段落和对白"),
    ("dialogue_restore",  "对白恢复",   "规则",     "识别对白行，恢复单独换行排版"),
    ("detect_chapters",   "章节识别",   "规则+AI",  "自动识别章节标题，生成TOC"),
    ("normalize_punct",   "标点规范",   "自动",     "统一省略号、破折号、括号格式"),
]
BADGE_COLOR = {"自动": "#127A56", "规则": "#2C6FB5", "规则+AI": "#4A3FA3"}


def blend(hex_color: str, alpha: float = 0.15, bg: str = BG) -> str:
    def p(h):
        h = h.lstrip("#")
        if len(h) == 3:
            h = h[0]*2 + h[1]*2 + h[2]*2
        return int(h[:2],16), int(h[2:4],16), int(h[4:6],16)
    fr,fg,fb = p(hex_color); br,bg2,bb = p(bg)
    return "#{:02X}{:02X}{:02X}".format(
        int(fr*alpha+br*(1-alpha)), int(fg*alpha+bg2*(1-alpha)), int(fb*alpha+bb*(1-alpha)))


def pill(parent, text, fg="#333", bg=CARD, border=BORDER, **kw):
    f = tk.Frame(parent, bg=bg, highlightbackground=border, highlightthickness=1)
    tk.Label(f, text=text, bg=bg, fg=fg, font=("Helvetica", 11), padx=8, pady=3).pack()
    return f


# ══════════════════════════════════════════════════════════════════════════════
#  主应用
# ══════════════════════════════════════════════════════════════════════════════

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Novel Formatter Studio")
        self.geometry("1320x840")
        self.minsize(1040, 660)
        self.configure(bg=BG)

        self.image_folder   = tk.StringVar()
        self.output_epub    = tk.StringVar()
        self.book_title     = tk.StringVar(value="未命名书籍")
        self.book_author    = tk.StringVar()
        self.book_publisher = tk.StringVar()
        self.book_volume    = tk.StringVar()
        self.shortcut_name  = tk.StringVar(value="ExtractText")
        self.css_template   = tk.StringVar(value="denki")
        self.writing_mode   = tk.StringVar(value="vertical")

        self.page_images    = []
        self.page_overrides = {}
        self.page_auto_type = {}
        self.selected_pages = set()
        self.thumb_cache    = {}
        self.step_vars      = {sid: tk.BooleanVar(value=True) for sid,_,_,_ in FORMATTER_STEPS}
        self.active_adapter = tk.StringVar(value="apple_vision")

        self.ocr_result_doc = None
        self.fmt_result_doc = None

        self._style()
        self._build()

    def _style(self):
        s = ttk.Style(self)
        s.theme_use("clam")
        s.configure(".", background=BG, foreground=INK, font=("Helvetica", 12))
        s.configure("TFrame", background=BG)
        s.configure("TLabel", background=BG, foreground=INK)
        s.configure("TEntry", fieldbackground=CARD, relief="flat", borderwidth=1, padding=5)
        s.configure("TButton", relief="flat", padding=(10,5))
        s.configure("Accent.TButton", background=ACC, foreground="white", relief="flat", padding=(12,6))
        s.map("Accent.TButton", background=[("active","#3853BE")])
        s.configure("TCheckbutton", background=BG, foreground=INK)
        s.configure("TNotebook", background=BG, tabmargins=[0,0,0,0])
        s.configure("TNotebook.Tab", background="#E7E5DF", foreground=MUTED, padding=[16,9], font=("Helvetica",12))
        s.map("TNotebook.Tab", background=[("selected", CARD)], foreground=[("selected", INK)])
        s.configure("Horizontal.TProgressbar", troughcolor="#E4E2DB", background=ACC, thickness=5)
        s.configure("TCombobox", fieldbackground=CARD, relief="flat")
        s.configure("Treeview", background=CARD, fieldbackground=CARD, rowheight=25, font=("Helvetica",11))
        s.configure("Treeview.Heading", background="#EFEEE9", font=("Helvetica",11,"bold"))

    def _build(self):
        nb = ttk.Notebook(self)
        nb.pack(fill="both", expand=True)
        self._nb = nb

        self._tab_pages = PageManagerTab(nb, self)
        self._tab_ocr   = OCRTab(nb, self)
        self._tab_fmt   = FormatterTab(nb, self)
        self._tab_epub  = EPUBTab(nb, self)

        nb.add(self._tab_pages, text="  📑  页面管理  ")
        nb.add(self._tab_ocr,   text="  🔍  OCR 适配器  ")
        nb.add(self._tab_fmt,   text="  ✨  Formatter  ")
        nb.add(self._tab_epub,  text="  📚  EPUB Builder  ")

    def set_status(self, msg, progress=-1):
        pass


# ══════════════════════════════════════════════════════════════════════════════
#  Tab 1 — Page Manager  (参照图4: 左侧类型列表 + 顶部工具栏 + 缩略图网格)
# ══════════════════════════════════════════════════════════════════════════════

class PageManagerTab(tk.Frame):
    def __init__(self, parent, app: App):
        super().__init__(parent, bg=BG)
        self.app = app
        self._build()

    def _build(self):
        root = tk.Frame(self, bg=CARD, highlightbackground=BORDER, highlightthickness=1)
        root.pack(fill="both", expand=True, padx=10, pady=10)

        # ── 顶部工具栏 ────────────────────────────────────────────────────────
        top = tk.Frame(root, bg=CARD, height=48)
        top.pack(fill="x"); top.pack_propagate(False)

        tk.Label(top, text="📄", bg=CARD, font=("Helvetica",14)).pack(side="left", padx=(14,4))
        self._file_lbl = tk.Label(top, text="未打开文件", bg=CARD, fg=INK, font=("Helvetica",12,"bold"))
        self._file_lbl.pack(side="left")
        self._count_lbl = tk.Label(top, text="", bg=CARD, fg=MUTED, font=("Helvetica",11))
        self._count_lbl.pack(side="left", padx=8)

        ttk.Button(top, text="📂 打开", command=self._open).pack(side="right", padx=10, pady=8)
        ttk.Button(top, text="≡ 列表", command=lambda: None).pack(side="right", padx=2)
        ttk.Button(top, text="⊞ 网格", command=lambda: None).pack(side="right", padx=2)

        tk.Frame(root, bg=BORDER, height=1).pack(fill="x")

        # 进度条（OCR 状态占位，页面加载后可扩展）
        self._prog = ttk.Progressbar(root, mode="determinate", style="Horizontal.TProgressbar")
        self._prog.pack(fill="x")

        body = tk.Frame(root, bg=CARD)
        body.pack(fill="both", expand=True)

        # ── 左侧类型列表 ──────────────────────────────────────────────────────
        side = tk.Frame(body, bg=CARD, width=170)
        side.pack(side="left", fill="y"); side.pack_propagate(False)

        tk.Label(side, text="页面类型", bg=CARD, fg=MUTED,
                 font=("Helvetica",10)).pack(anchor="w", padx=14, pady=(12,4))
        tk.Label(side, text="点击筛选页面", bg=CARD, fg="#B7B6AF",
                 font=("Helvetica",9)).pack(anchor="w", padx=14, pady=(0,8))

        self._filter_var = "all"
        self._filter_rows = {}
        list_area = tk.Frame(side, bg=CARD)
        list_area.pack(fill="x")

        for ttype, label, color in [("all","全部页面","#333")] + [(t,l,c) for t,l,c in PAGE_TYPES]:
            row = tk.Frame(list_area, bg=CARD, cursor="hand2")
            row.pack(fill="x", padx=6, pady=1)
            dot = tk.Frame(row, bg=color, width=8, height=8)
            dot.pack(side="left", padx=(8,6), pady=8)
            lbl = tk.Label(row, text=label, bg=CARD, fg="#3A3A38", font=("Helvetica",11), anchor="w")
            lbl.pack(side="left", fill="x", expand=True)
            cnt = tk.Label(row, text="0", bg=CARD, fg=MUTED, font=("Helvetica",10))
            cnt.pack(side="right", padx=8)
            for w in (row, dot, lbl, cnt):
                w.bind("<Button-1>", lambda e, t=ttype: self._set_filter(t))
            self._filter_rows[ttype] = (row, lbl, cnt)

        tk.Frame(side, bg=BORDER, height=1).pack(fill="x", pady=8)
        ttk.Button(side, text="🪄 自动分类", command=self._auto_classify).pack(fill="x", padx=10, pady=2)
        ttk.Button(side, text="▶ 开始 OCR", style="Accent.TButton",
                   command=self._go_ocr).pack(fill="x", padx=10, pady=(2,10))

        tk.Frame(body, bg=BORDER, width=1).pack(side="left", fill="y")

        # ── 批量标签栏 ────────────────────────────────────────────────────────
        main = tk.Frame(body, bg=CARD)
        main.pack(side="left", fill="both", expand=True)

        tagbar = tk.Frame(main, bg="#FAFAF7", height=40)
        tagbar.pack(fill="x"); tagbar.pack_propagate(False)
        self._sel_lbl = tk.Label(tagbar, text="选中 0 页 · 标记为：", bg="#FAFAF7", fg=MUTED, font=("Helvetica",11))
        self._sel_lbl.pack(side="left", padx=10)
        for ttype, label, color in PAGE_TYPES:
            tk.Button(tagbar, text=label, bg=blend(color,0.15,"#FAFAF7"), fg=color,
                      activebackground=blend(color,0.32,"#FAFAF7"), activeforeground=color,
                      relief="flat", bd=0, padx=8, pady=3, font=("Helvetica",10),
                      cursor="hand2", command=lambda t=ttype: self._batch_tag(t)
                      ).pack(side="left", padx=2, pady=6)
        ttk.Button(tagbar, text="取消选择", command=self._clear_sel).pack(side="right", padx=8)
        ttk.Button(tagbar, text="全选", command=self._select_all).pack(side="right", padx=2)

        tk.Frame(main, bg=BORDER, height=1).pack(fill="x")

        # 缩略图网格
        canvas_wrap = tk.Frame(main, bg=CARD)
        canvas_wrap.pack(fill="both", expand=True)
        self._canvas = tk.Canvas(canvas_wrap, bg=CARD, highlightthickness=0)
        vsb = ttk.Scrollbar(canvas_wrap, orient="vertical", command=self._canvas.yview)
        self._canvas.configure(yscrollcommand=vsb.set)
        vsb.pack(side="right", fill="y")
        self._canvas.pack(fill="both", expand=True)
        self._inner = tk.Frame(self._canvas, bg=CARD)
        self._win = self._canvas.create_window((0,0), window=self._inner, anchor="nw")
        self._inner.bind("<Configure>", lambda e: self._canvas.configure(scrollregion=self._canvas.bbox("all")))
        self._canvas.bind("<Configure>", self._on_resize)
        self._canvas.bind("<MouseWheel>", lambda e: self._canvas.yview_scroll(-1*(e.delta//120),"units"))
        self._canvas.bind("<Button-4>", lambda e: self._canvas.yview_scroll(-1,"units"))
        self._canvas.bind("<Button-5>", lambda e: self._canvas.yview_scroll(1,"units"))

        self._empty = tk.Label(self._inner,
            text="打开图片文件夹 / PDF / 单张图片开始\n\n支持 PNG · JPG · HEIC · TIFF · PDF",
            bg=CARD, fg=MUTED, font=("Helvetica",13), justify="center")
        self._empty.pack(pady=100)

        self._set_filter("all")

    # ── 打开文件（文件夹 / PDF / 图片，支持多选）──────────────────────────────

    def _open(self):
        menu = tk.Menu(self, tearoff=0)
        menu.add_command(label="打开文件夹...", command=self._open_folder)
        menu.add_command(label="打开图片/PDF文件...", command=self._open_files)
        try:
            x = self.winfo_pointerx(); y = self.winfo_pointery()
            menu.tk_popup(x, y)
        finally:
            menu.grab_release()

    def _open_folder(self):
        folder = filedialog.askdirectory(title="选择图片文件夹")
        if not folder:
            return
        self._load_inputs([folder], display_name=Path(folder).name)

    def _open_files(self):
        paths = filedialog.askopenfilenames(
            title="选择图片或 PDF 文件（可多选）",
            filetypes=[
                ("图片和PDF", "*.png *.jpg *.jpeg *.heic *.tif *.tiff *.bmp *.gif *.pdf"),
                ("PDF 文件", "*.pdf"),
                ("图片文件", "*.png *.jpg *.jpeg *.heic *.tif *.tiff *.bmp *.gif"),
                ("所有文件", "*.*"),
            ])
        if not paths:
            return
        name = Path(paths[0]).name if len(paths) == 1 else f"{len(paths)} 个文件"
        self._load_inputs(list(paths), display_name=name)

    def _load_inputs(self, raw_paths: list[str], display_name: str):
        """展开输入（文件夹/PDF/图片混合）为最终图片路径列表"""
        self._file_lbl.config(text=display_name)
        self._empty.config(text="正在处理输入...")
        self.app.image_folder.set(raw_paths[0] if Path(raw_paths[0]).is_dir() else str(Path(raw_paths[0]).parent))

        def _bg():
            try:
                from adapters.pdf_input import expand_inputs
                work_dir = raw_paths[0] if Path(raw_paths[0]).is_dir() else str(Path(raw_paths[0]).parent)
                images = expand_inputs(raw_paths, work_dir=work_dir)
                images = sorted(set(images), key=lambda p: Path(p).name)
                self.after(0, lambda: self._finish_load(images))
            except Exception as e:
                self.after(0, lambda: messagebox.showerror("处理失败", str(e)))
                self.after(0, lambda: self._empty.config(
                    text="打开图片文件夹 / PDF / 单张图片开始\n\n支持 PNG · JPG · HEIC · TIFF · PDF"))

        threading.Thread(target=_bg, daemon=True).start()

    def _finish_load(self, images: list[str]):
        if not images:
            messagebox.showwarning("无图片", "没有找到可用的图片（PDF转换或图片扫描均为空）")
            return
        self.app.page_images    = [Path(p) for p in images]
        self.app.page_overrides = {}
        self.app.page_auto_type = {}
        self.app.selected_pages = set()
        self.app.thumb_cache    = {}
        self._count_lbl.config(text=f"{len(images)} 页")
        if not self.app.output_epub.get():
            base = Path(images[0]).parent
            self.app.output_epub.set(str(base.parent / (base.name + ".epub")))
        self._auto_classify(silent=True)
        self._render()
        if not HAS_PIL:
            messagebox.showinfo(
                "未安装 Pillow",
                "当前只能看到彩色类型标签，无法预览真实缩略图。\n\n"
                "在终端运行以下命令后重启程序即可看到真实图片:\n"
                "  pip3 install pillow"
            )

    # ── 自动分类（基于文件名规则）─────────────────────────────────────────────

    def _auto_classify(self, silent=False):
        if not self.app.page_images:
            if not silent:
                messagebox.showinfo("提示", "请先打开图片")
            return
        for i, p in enumerate(self.app.page_images):
            page_no = i + 1
            if page_no in self.app.page_overrides:
                continue
            name = p.stem.lower()
            if i == 0:
                t = "cover"
            elif any(k in name for k in ("blank","white","空白")):
                t = "blank"
            elif any(k in name for k in ("color","colour","彩")):
                t = "color_illus"
            elif any(k in name for k in ("toc","目次","contents","mokuji")):
                t = "toc_page"
            elif any(k in name for k in ("illus","insert","挿絵","口絵")):
                t = "illustration"
            elif any(k in name for k in ("after","postscript","あとがき","后记")):
                t = "afterword"
            elif any(k in name for k in ("colophon","奥付")):
                t = "colophon"
            else:
                t = "paragraph"
            self.app.page_auto_type[page_no] = t
        self._render()
        if not silent:
            messagebox.showinfo("完成", "已根据文件名自动分类，请检查并手动修正")

    def _go_ocr(self):
        if not self.app.page_images:
            messagebox.showinfo("提示", "请先打开图片")
            return
        self.app._nb.select(1)

    # ── 渲染 ──────────────────────────────────────────────────────────────────

    def _ptype(self, page_no):
        return self.app.page_overrides.get(page_no) or self.app.page_auto_type.get(page_no, "unknown")

    def _render(self):
        for w in self._inner.winfo_children():
            w.destroy()

        pages = self.app.page_images
        filt = getattr(self, "_current_filter", "all")
        if filt != "all":
            pages = [p for i,p in enumerate(pages) if self._ptype(i+1) == filt]

        if not pages:
            tk.Label(self._inner, text="（此类型暂无页面）", bg=CARD, fg=MUTED,
                     font=("Helvetica",12)).pack(pady=60)
            self._update_counts()
            return

        W, H = 108, 150
        self._canvas.update_idletasks()
        canvas_w = self._canvas.winfo_width()
        if canvas_w <= 1:  # 画布尚未完成初次布局，退回一个合理默认值
            canvas_w = 640
        cols = max(4, (canvas_w - 16) // (W + 10))

        for idx, path in enumerate(pages):
            i_orig = self.app.page_images.index(path)
            page_no = i_orig + 1
            ptype = self._ptype(page_no)
            color = TYPE_COLOR.get(ptype,"#AAAAAA")
            label = TYPE_LABEL.get(ptype,"?")
            sel = page_no in self.app.selected_pages

            col, row = idx % cols, idx // cols
            cell = tk.Frame(self._inner, bg=CARD, padx=5, pady=5)
            cell.grid(row=row, column=col, sticky="n")

            border = tk.Frame(cell, bg=ACC if sel else BORDER, padx=2 if sel else 1, pady=2 if sel else 1)
            border.pack()

            img_area = tk.Frame(border, bg="#EAEAE4", width=W, height=H-22)
            img_area.pack(); img_area.pack_propagate(False)

            if HAS_PIL:
                if path in self.app.thumb_cache:
                    tk.Label(img_area, image=self.app.thumb_cache[path], bg="#EAEAE4").place(relx=.5,rely=.5,anchor="center")
                else:
                    threading.Thread(target=self._load_thumb, args=(path,img_area,W,H-22), daemon=True).start()
            else:
                tk.Label(img_area, text=f"第 {page_no} 页", bg="#EAEAE4", fg="#666",
                         font=("Helvetica",11,"bold")).place(relx=.5,rely=.5,anchor="center")

            tagf = tk.Frame(border, bg=color, height=22, width=W)
            tagf.pack(fill="x"); tagf.pack_propagate(False)
            tk.Label(tagf, text=label, bg=color, fg="white", font=("Helvetica",10)).place(relx=.5,rely=.5,anchor="center")

            tk.Label(cell, text=f"第 {page_no} 页", bg=CARD, fg=MUTED, font=("Helvetica",9)).pack()

            for w in (cell, border, img_area, tagf):
                w.bind("<Button-1>", lambda e,p=page_no: self._click(p))
                w.bind("<Button-2>", lambda e,p=page_no: self._context(e,p))
                w.bind("<Button-3>", lambda e,p=page_no: self._context(e,p))

        self._update_counts()
        self._sel_lbl.config(text=f"选中 {len(self.app.selected_pages)} 页 · 标记为：")

    def _load_thumb(self, path, frame, w, h):
        try:
            img = Image.open(path); img.thumbnail((w,h), Image.LANCZOS)
            photo = ImageTk.PhotoImage(img)
            self.app.thumb_cache[path] = photo
            self.after(0, lambda: self._place(frame, photo))
        except Exception as e:
            print(f"[缩略图加载失败] {path}: {e}")

    def _place(self, frame, photo):
        try:
            l = tk.Label(frame, image=photo, bg="#EAEAE4"); l.image = photo
            l.place(relx=.5, rely=.5, anchor="center")
        except Exception:
            pass  # frame 可能因为筛选/刷新已被销毁，属于正常情况，无需提示

    def _on_resize(self, e):
        self._canvas.itemconfig(self._win, width=e.width)
        last_w = getattr(self, "_last_render_width", 0)
        if abs(e.width - last_w) < 20:
            return  # 宽度变化很小（如启动时的多次Configure），跳过，避免打断缩略图加载
        self._last_render_width = e.width
        self.after(150, self._render)

    def _click(self, page_no):
        if page_no in self.app.selected_pages: self.app.selected_pages.discard(page_no)
        else: self.app.selected_pages.add(page_no)
        self._render()

    def _context(self, event, page_no):
        if page_no not in self.app.selected_pages:
            self.app.selected_pages = {page_no}
        m = tk.Menu(self, tearoff=0)
        m.add_command(label=f"第 {page_no} 页 — 设置类型", state="disabled")
        m.add_separator()
        for ttype,label,color in PAGE_TYPES:
            m.add_command(label=f"  {label}", command=lambda t=ttype: self._batch_tag(t))
        m.add_separator()
        m.add_command(label="取消选择", command=self._clear_sel)
        m.post(event.x_root, event.y_root)

    def _batch_tag(self, ttype):
        if not self.app.selected_pages:
            messagebox.showinfo("提示","请先点选页面（可多选）"); return
        for p in self.app.selected_pages:
            self.app.page_overrides[p] = ttype
        self.app.selected_pages.clear()
        self._render()

    def _select_all(self):
        self.app.selected_pages = set(range(1, len(self.app.page_images)+1))
        self._render()

    def _clear_sel(self):
        self.app.selected_pages.clear()
        self._render()

    def _set_filter(self, ttype):
        self._current_filter = ttype
        for t,(row,lbl,cnt) in self._filter_rows.items():
            if t == ttype:
                row.config(bg=ACC_BG); lbl.config(bg=ACC_BG, fg=ACC, font=("Helvetica",11,"bold")); cnt.config(bg=ACC_BG, fg=ACC)
            else:
                row.config(bg=CARD); lbl.config(bg=CARD, fg="#3A3A38", font=("Helvetica",11)); cnt.config(bg=CARD, fg=MUTED)
        self._render()

    def _update_counts(self):
        counts = Counter(self._ptype(i+1) for i in range(len(self.app.page_images)))
        total = len(self.app.page_images)
        for t,(row,lbl,cnt) in self._filter_rows.items():
            cnt.config(text=str(total if t=="all" else counts.get(t,0)))


# ══════════════════════════════════════════════════════════════════════════════
#  Tab 2 — OCR 适配器  (参照图3: 左侧适配器列表 + 右侧输入/结果预览)
# ══════════════════════════════════════════════════════════════════════════════

class OCRTab(tk.Frame):
    def __init__(self, parent, app: App):
        super().__init__(parent, bg=BG)
        self.app = app
        self._build()

    def _build(self):
        root = tk.Frame(self, bg=CARD, highlightbackground=BORDER, highlightthickness=1)
        root.pack(fill="both", expand=True, padx=10, pady=10)

        body = tk.Frame(root, bg=CARD)
        body.pack(fill="both", expand=True)

        # ── 左侧：OCR 适配器列表 ──────────────────────────────────────────────
        left = tk.Frame(body, bg=CARD, width=250)
        left.pack(side="left", fill="y"); left.pack_propagate(False)
        tk.Label(left, text="OCR 适配器", bg=CARD, fg=INK,
                 font=("Helvetica",13,"bold")).pack(anchor="w", padx=14, pady=(14,2))
        tk.Label(left, text="选择识别引擎", bg=CARD, fg=MUTED,
                 font=("Helvetica",10)).pack(anchor="w", padx=14, pady=(0,10))

        self._adapter_cards = {}
        for aid, name, badge, color, desc, enabled in OCR_ADAPTERS:
            card = tk.Frame(left, bg=CARD, highlightthickness=1,
                            highlightbackground=BORDER, cursor="hand2")
            card.pack(fill="x", padx=10, pady=4)
            top = tk.Frame(card, bg=CARD); top.pack(fill="x", padx=10, pady=(8,2))
            tk.Frame(top, bg=color, width=8, height=8).pack(side="left", pady=4)
            tk.Label(top, text=name, bg=CARD, fg=INK, font=("Helvetica",12,"bold")
                     ).pack(side="left", padx=6)
            bcolor = "#8C8B84" if not enabled else color
            bl = tk.Label(top, text=badge, bg=blend(bcolor,0.15,CARD), fg=bcolor,
                          font=("Helvetica",9), padx=6, pady=1)
            bl.pack(side="right")
            tk.Label(card, text=desc, bg=CARD, fg=MUTED, font=("Helvetica",10),
                     wraplength=210, justify="left").pack(anchor="w", padx=10, pady=(0,8))
            if not enabled:
                overlay = tk.Label(card, text="即将支持", bg=CARD, fg="#B7B6AF",
                                   font=("Helvetica",9,"italic"))
                overlay.pack(anchor="e", padx=10, pady=(0,6))
            for w in (card, top):
                w.bind("<Button-1>", lambda e, a=aid, en=enabled: self._select_adapter(a, en))
            self._adapter_cards[aid] = card

        tk.Frame(left, bg=BORDER, height=1).pack(fill="x", pady=8)

        # 快捷指令 + 输入选择
        tk.Label(left, text="快捷指令名称", bg=CARD, fg=MUTED,
                 font=("Helvetica",10)).pack(anchor="w", padx=14, pady=(6,2))
        row = tk.Frame(left, bg=CARD); row.pack(fill="x", padx=10)
        ttk.Entry(row, textvariable=self.app.shortcut_name).pack(side="left", fill="x", expand=True)
        ttk.Button(row, text="验证", width=4, command=self._test_shortcut).pack(side="right", padx=(4,0))

        tk.Label(left, text="输入", bg=CARD, fg=MUTED,
                 font=("Helvetica",10)).pack(anchor="w", padx=14, pady=(10,2))
        ibar = tk.Frame(left, bg=CARD); ibar.pack(fill="x", padx=10, pady=(0,4))
        ttk.Button(ibar, text="📂 文件夹", command=self._pick_folder).pack(side="left", fill="x", expand=True, padx=(0,2))
        ttk.Button(ibar, text="🖼 单图/PDF", command=self._pick_files).pack(side="left", fill="x", expand=True, padx=(2,0))
        self._input_lbl = tk.Label(left, text="（尚未选择输入）", bg=CARD, fg=MUTED,
                                   font=("Helvetica",10), wraplength=220, justify="left")
        self._input_lbl.pack(anchor="w", padx=14, pady=(2,10))

        ttk.Button(left, text="▶  开始 OCR", style="Accent.TButton",
                   command=self._run_ocr).pack(fill="x", padx=10, pady=(0,10))

        tk.Frame(body, bg=BORDER, width=1).pack(side="left", fill="y")

        # ── 右侧：结果预览（标签切换：日志 / 结果摘要）───────────────────────
        right = tk.Frame(body, bg=CARD)
        right.pack(side="left", fill="both", expand=True)

        tabbar = tk.Frame(right, bg=CARD, height=40)
        tabbar.pack(fill="x"); tabbar.pack_propagate(False)
        self._view = tk.StringVar(value="log")
        for val, label in [("log","OCR 日志"),("result","识别结果")]:
            b = tk.Radiobutton(tabbar, text=label, value=val, variable=self._view,
                               command=self._switch_view, bg=CARD, indicatoron=False,
                               relief="flat", padx=14, pady=6, font=("Helvetica",11),
                               selectcolor=ACC_BG)
            b.pack(side="left", padx=(10 if val=="log" else 4, 4), pady=6)

        tk.Frame(right, bg=BORDER, height=1).pack(fill="x")

        self._log_box = scrolledtext.ScrolledText(
            right, bg="#1B1B1D", fg="#CFCFC6", font=("Menlo",11),
            relief="flat", bd=0, padx=12, pady=10, state="disabled")
        self._log_box.pack(fill="both", expand=True)

        self._result_box = tk.Text(
            right, bg=CARD, fg=INK, font=("Helvetica",12),
            relief="flat", bd=0, padx=14, pady=12, state="disabled", wrap="word")

    def _switch_view(self):
        if self._view.get() == "log":
            self._result_box.pack_forget()
            self._log_box.pack(fill="both", expand=True)
        else:
            self._log_box.pack_forget()
            self._result_box.pack(fill="both", expand=True)

    def _select_adapter(self, aid, enabled):
        if not enabled:
            messagebox.showinfo("即将支持", f"「{dict((a,n) for a,n,*_ in OCR_ADAPTERS)[aid]}」适配器正在开发中，\n目前请使用 Apple Vision OCR。")
            return
        self.app.active_adapter.set(aid)
        for a, card in self._adapter_cards.items():
            card.config(highlightbackground=ACC if a==aid else BORDER,
                       highlightthickness=2 if a==aid else 1)

    def _log(self, msg):
        self._log_box.config(state="normal")
        self._log_box.insert("end", msg+"\n")
        self._log_box.see("end")
        self._log_box.config(state="disabled")

    def _pick_folder(self):
        folder = filedialog.askdirectory(title="选择图片文件夹")
        if folder:
            self.app.image_folder.set(folder)
            self._pending_inputs = [folder]
            self._input_lbl.config(text=f"文件夹: {Path(folder).name}")

    def _pick_files(self):
        paths = filedialog.askopenfilenames(
            title="选择图片或 PDF（可多选）",
            filetypes=[("图片和PDF","*.png *.jpg *.jpeg *.heic *.tif *.tiff *.bmp *.gif *.pdf"),
                      ("所有文件","*.*")])
        if paths:
            self._pending_inputs = list(paths)
            name = Path(paths[0]).name if len(paths)==1 else f"{len(paths)} 个文件"
            self._input_lbl.config(text=f"文件: {name}")

    def _test_shortcut(self):
        name = self.app.shortcut_name.get().strip()
        self._view.set("log"); self._switch_view()
        self._log(f"验证快捷指令「{name}」...")
        def _check():
            try:
                r = subprocess.run(["shortcuts","list"], capture_output=True, text=True, timeout=10)
                if name in r.stdout:
                    self.after(0, lambda: self._log(f"✅ 找到「{name}」，配置正确"))
                else:
                    self.after(0, lambda: self._log(f"⚠️  未找到「{name}」，请在「快捷指令」App 新建同名快捷指令"))
            except FileNotFoundError:
                self.after(0, lambda: self._log("❌ 找不到 shortcuts 命令，请确认 macOS ≥ Monterey"))
            except Exception as e:
                self.after(0, lambda: self._log(f"❌ {e}"))
        threading.Thread(target=_check, daemon=True).start()

    def _run_ocr(self):
        inputs = getattr(self, "_pending_inputs", None)
        if not inputs and self.app.page_images:
            # 使用页面管理器已加载的图片
            inputs = [str(p) for p in self.app.page_images]
        if not inputs:
            messagebox.showerror("错误", "请先选择文件夹或图片/PDF文件\n（或先在「页面管理」打开文件）")
            return
        self._view.set("log"); self._switch_view()
        threading.Thread(target=self._ocr_bg, args=(inputs,), daemon=True).start()

    def _ocr_bg(self, inputs):
        self.after(0, lambda: self._log("="*44))
        self.after(0, lambda: self._log(f"开始 OCR（{self.app.active_adapter.get()}）"))
        self.after(0, lambda: self._log("="*44))
        try:
            from adapters.apple_vision_adapter import run as ocr_run
            import io, contextlib
            buf = io.StringIO()
            overrides = {k:v for k,v in self.app.page_overrides.items()}
            with contextlib.redirect_stdout(buf):
                doc = ocr_run(
                    input_paths=inputs,
                    page_overrides=overrides,
                    shortcut_name=self.app.shortcut_name.get(),
                    verbose=True,
                )
            for line in buf.getvalue().splitlines():
                if line.strip():
                    l = line
                    self.after(0, lambda m=l: self._log(m))

            self.app.ocr_result_doc = doc

            json_path = None
            try:
                folder = inputs[0] if Path(inputs[0]).is_dir() else str(Path(inputs[0]).parent)
                json_path = Path(folder) / "ocr_result.json"
                with open(json_path,"w",encoding="utf-8") as f:
                    f.write(doc.to_json())
            except Exception:
                pass

            summary = self._make_summary(doc)
            self.after(0, lambda: self._show_summary(summary))
            self.after(0, lambda: self._log(f"\n✅ OCR 完成: {len(doc.blocks)} 块，{len(doc.toc)} 章节"))
            if json_path:
                self.after(0, lambda: self._log(f"💾 JSON 已保存: {json_path}"))
        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            self.after(0, lambda: self._log(f"\n❌ 错误:\n{tb}"))
            self.after(0, lambda: messagebox.showerror("OCR 失败", str(e)))

    def _make_summary(self, doc) -> str:
        lines = [f"书名: {doc.metadata.title or '(未填写)'}",
                 f"OCR 引擎: {doc.metadata.source_engine}",
                 f"总页数: {len(doc.pages)}  ·  总块数: {len(doc.blocks)}", "",
                 "页面类型统计:"]
        c = Counter(p.page_type.value for p in doc.pages)
        for t,n in c.most_common():
            lines.append(f"  {TYPE_LABEL.get(t,t):8s} {n} 页")
        lines += ["", "章节目录:"]
        lines += [f"  {e.chapter_index}. {e.title}" for e in doc.toc] or ["  （未识别到章节）"]
        return "\n".join(lines)

    def _show_summary(self, text):
        self._view.set("result"); self._switch_view()
        self._result_box.config(state="normal")
        self._result_box.delete("1.0","end")
        self._result_box.insert("1.0", text)
        self._result_box.config(state="disabled")


# ══════════════════════════════════════════════════════════════════════════════
#  Tab 3 — Formatter Engine  (参照图2: 顶部步骤条 + 左侧步骤卡片 + 前后对比)
# ══════════════════════════════════════════════════════════════════════════════

class FormatterTab(tk.Frame):
    def __init__(self, parent, app: App):
        super().__init__(parent, bg=BG)
        self.app = app
        self._active_step = FORMATTER_STEPS[0][0]
        self._build()

    def _build(self):
        root = tk.Frame(self, bg=CARD, highlightbackground=BORDER, highlightthickness=1)
        root.pack(fill="both", expand=True, padx=10, pady=10)

        # 顶部
        top = tk.Frame(root, bg=CARD, height=48)
        top.pack(fill="x"); top.pack_propagate(False)
        tk.Label(top, text="Formatter Engine", bg=CARD, fg=INK,
                 font=("Helvetica",13,"bold")).pack(side="left", padx=14)
        ttk.Button(top, text="▶  全部运行", style="Accent.TButton",
                   command=self._run_all).pack(side="right", padx=10, pady=8)
        ttk.Button(top, text="↩ 撤销上一步", command=self._undo).pack(side="right", padx=2)
        ttk.Button(top, text="📂 载入JSON", command=self._load_json).pack(side="right", padx=2)
        ttk.Button(top, text="💾 保存结果", command=self._save_json).pack(side="right", padx=2)
        self._version_lbl = tk.Label(top, text="", bg=CARD, fg=MUTED, font=("Helvetica",10))
        self._version_lbl.pack(side="right", padx=10)

        tk.Frame(root, bg=BORDER, height=1).pack(fill="x")

        body = tk.Frame(root, bg=CARD)
        body.pack(fill="both", expand=True)

        # 左侧步骤卡片
        left = tk.Frame(body, bg=CARD, width=230)
        left.pack(side="left", fill="y"); left.pack_propagate(False)

        self._step_cards = {}
        for sid, label, badge, desc in FORMATTER_STEPS:
            card = tk.Frame(left, bg=CARD, cursor="hand2",
                            highlightthickness=1, highlightbackground=BORDER)
            card.pack(fill="x", padx=10, pady=3)
            hdr = tk.Frame(card, bg=CARD); hdr.pack(fill="x", padx=10, pady=(8,2))
            cb = ttk.Checkbutton(hdr, variable=self.app.step_vars[sid])
            cb.pack(side="left")
            tk.Label(hdr, text=label, bg=CARD, fg=INK, font=("Helvetica",12,"bold")
                     ).pack(side="left", padx=4)
            bcolor = BADGE_COLOR.get(badge, "#888")
            tk.Label(card, text=badge, bg=blend(bcolor,0.15,CARD), fg=bcolor,
                     font=("Helvetica",9), padx=6, pady=1).place(relx=1.0, y=8, anchor="ne", x=-10)
            tk.Label(card, text=desc, bg=CARD, fg=MUTED, font=("Helvetica",10),
                     wraplength=195, justify="left").pack(anchor="w", padx=10, pady=(0,8))
            for w in (card, hdr):
                w.bind("<Button-1>", lambda e, s=sid: self._select_step(s))
            self._step_cards[sid] = card

        tk.Frame(body, bg=BORDER, width=1).pack(side="left", fill="y")

        # 右侧内容
        right = tk.Frame(body, bg=CARD)
        right.pack(side="left", fill="both", expand=True)

        rhead = tk.Frame(right, bg=CARD, height=44)
        rhead.pack(fill="x"); rhead.pack_propagate(False)
        self._step_title = tk.Label(rhead, text="", bg=CARD, fg=INK,
                                    font=("Helvetica",12,"bold"))
        self._step_title.pack(side="left", padx=14)
        ttk.Button(rhead, text="✓ 应用此步", command=self._apply_step).pack(side="right", padx=10, pady=6)
        self._compare_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(rhead, text="前后对比", variable=self._compare_var,
                        command=self._refresh).pack(side="right", padx=6)

        tk.Frame(right, bg=BORDER, height=1).pack(fill="x")

        panes = tk.Frame(right, bg=CARD)
        panes.pack(fill="both", expand=True, padx=8, pady=8)

        lf = tk.Frame(panes, bg=CARD)
        lf.pack(side="left", fill="both", expand=True)
        tk.Label(lf, text="处理前", bg=CARD, fg=MUTED, font=("Helvetica",10)).pack(anchor="w")
        self._before = scrolledtext.ScrolledText(lf, font=("Helvetica",12), bg="#FAFAF8",
            fg=INK, relief="flat", bd=1, padx=10, pady=10, state="disabled", wrap="word")
        self._before.pack(fill="both", expand=True, pady=(4,0))

        self._divider = tk.Frame(panes, bg=BORDER, width=1)
        self._divider.pack(side="left", fill="y", padx=6)

        rf = tk.Frame(panes, bg=CARD)
        rf.pack(side="right", fill="both", expand=True)
        tk.Label(rf, text="处理后", bg=CARD, fg=MUTED, font=("Helvetica",10)).pack(anchor="w")
        self._after = scrolledtext.ScrolledText(rf, font=("Helvetica",12), bg="#FAFAF8",
            fg=INK, relief="flat", bd=1, padx=10, pady=10, state="disabled", wrap="word")
        self._after.pack(fill="both", expand=True, pady=(4,0))

        for w in (self._before, self._after):
            w.tag_configure("chapter", foreground="#4A3FA3", font=("Helvetica",12,"bold"))
            w.tag_configure("dialogue", foreground="#C0542F")
        self._after.tag_configure("modified", background="#FDF0C8")

        # 底部规则提示条
        tk.Frame(right, bg=BORDER, height=1).pack(fill="x")
        self._rules_bar = tk.Frame(right, bg="#FAFAF7", height=34)
        self._rules_bar.pack(fill="x"); self._rules_bar.pack_propagate(False)

        self._select_step(FORMATTER_STEPS[0][0])

    def _select_step(self, sid):
        self._active_step = sid
        for s, card in self._step_cards.items():
            card.config(highlightbackground=ACC if s==sid else BORDER,
                       highlightthickness=2 if s==sid else 1,
                       bg=ACC_BG if s==sid else CARD)
            for w in card.winfo_children():
                try: w.config(bg=ACC_BG if s==sid else CARD)
                except Exception: pass
        label = dict((s,l) for s,l,_,_ in FORMATTER_STEPS)[sid]
        desc  = dict((s,d) for s,_,_,d in FORMATTER_STEPS)[sid]
        self._step_title.config(text=f"{label} — {desc}")
        self._refresh_rules(sid)

    def _refresh_rules(self, sid):
        for w in self._rules_bar.winfo_children():
            w.destroy()
        RULES = {
            "clean_metadata": ["页码正则 /^\\d{1,6}$/", "跨页重复行检测"],
            "merge_sentences": ["末尾无句末符→合并", "章节标题不参与合并"],
            "remove_duplicates": ["相邻文本精确匹配", "对白全文去重"],
            "dialogue_restore": ["「…」『…』括号识别", "对白独立成行"],
            "detect_chapters": ["序章/第X章/幕間/後記 关键词"],
            "normalize_punct": ["...→……", "--→——", "半角→全角括号"],
        }
        tk.Label(self._rules_bar, text="规则:", bg="#FAFAF7", fg=MUTED,
                 font=("Helvetica",10)).pack(side="left", padx=(10,4))
        for r in RULES.get(sid, []):
            tk.Label(self._rules_bar, text=r, bg="#EFEEE9", fg="#555",
                     font=("Helvetica",10), padx=6, pady=1).pack(side="left", padx=2, pady=6)

    def _load_json(self):
        path = filedialog.askopenfilename(title="载入 JSON", filetypes=[("JSON","*.json")])
        if not path: return
        try:
            from models.document import UnifiedDocument
            with open(path, encoding="utf-8") as f:
                self.app.ocr_result_doc = UnifiedDocument.from_json(f.read())
            self._show_doc(self.app.ocr_result_doc, self._before)
            messagebox.showinfo("完成", f"已载入 {len(self.app.ocr_result_doc.blocks)} 个块")
        except Exception as e:
            messagebox.showerror("载入失败", str(e))

    def _undo(self):
        doc = self.app.fmt_result_doc
        if not doc or not doc.history:
            messagebox.showinfo("提示", "没有可撤销的步骤（Book 版本历史为空）")
            return
        target_version = doc.version - 1
        prev = doc.rollback_to(target_version)
        if prev is None:
            messagebox.showinfo("提示", "已经是最初版本，无法再撤销")
            return
        self.app.fmt_result_doc = prev
        self._show_doc(prev, self._after)
        self._update_version_label(prev)
        messagebox.showinfo("已撤销", f"已回退到 Book V{prev.version}")

    def _update_version_label(self, doc):
        if doc is None:
            self._version_lbl.config(text="")
        else:
            self._version_lbl.config(text=f"Book V{doc.version} · {len(doc.history)} 步历史")

    def _run_all(self):
        steps = [sid for sid,_,_,_ in FORMATTER_STEPS if self.app.step_vars[sid].get()]
        self._run_steps(steps, base_on_current=False)

    def _apply_step(self):
        self._run_steps([self._active_step], base_on_current=True)

    def _run_steps(self, steps, base_on_current=True):
        """
        base_on_current=True  : 在当前 fmt_result_doc（若存在）基础上叠加运行——
                                 用于"应用此步"这种逐步累积的场景，Undo 才有意义。
        base_on_current=False : 总是从 ocr_result_doc（Book V0）重新开始——
                                 用于"全部运行"，避免同一步骤被重复叠加。
        """
        base = None
        if base_on_current and self.app.fmt_result_doc is not None:
            base = self.app.fmt_result_doc
        else:
            base = self.app.ocr_result_doc

        if base is None:
            messagebox.showerror("错误", "请先完成 OCR，或载入 JSON 文件")
            return
        self._show_doc(self.app.ocr_result_doc or base, self._before)

        def _bg():
            try:
                from engine.formatter import run_pipeline
                result = run_pipeline(base, steps=steps, verbose=False)
                self.app.fmt_result_doc = result
                self.after(0, lambda: self._show_doc(result, self._after))
                self.after(0, lambda: self._update_version_label(result))
            except Exception as e:
                import traceback
                tb = traceback.format_exc()
                self.after(0, lambda: messagebox.showerror("处理失败", f"{e}\n\n{tb}"))
        threading.Thread(target=_bg, daemon=True).start()

    def _show_doc(self, doc, widget):
        widget.config(state="normal")
        widget.delete("1.0","end")
        for b in doc.blocks:
            if b.type.value == "image_ref":
                widget.insert("end", f"[图片: {Path(b.image_path).name}]\n", "dialogue")
            elif b.type.value == "chapter":
                widget.insert("end", f"\n▌ {b.text}\n\n", "chapter")
            elif b.type.value == "dialogue":
                widget.insert("end", f"  {b.text}\n", "dialogue")
            else:
                tag = "modified" if b.modified_by else ""
                widget.insert("end", f"  {b.text}\n", tag)
        widget.config(state="disabled")

    def _refresh(self):
        if self._compare_var.get():
            self._before.master.pack(side="left", fill="both", expand=True)
            self._divider.pack(side="left", fill="y", padx=6)
            self._after.master.pack(side="right", fill="both", expand=True)
        else:
            self._before.master.pack_forget()
            self._divider.pack_forget()
            self._after.master.pack(fill="both", expand=True)

    def _save_json(self):
        doc = self.app.fmt_result_doc or self.app.ocr_result_doc
        if not doc:
            messagebox.showinfo("提示","还没有处理结果"); return
        path = filedialog.asksaveasfilename(title="保存 JSON", defaultextension=".json",
                                            filetypes=[("JSON","*.json")])
        if path:
            with open(path,"w",encoding="utf-8") as f:
                f.write(doc.to_json())
            messagebox.showinfo("保存完成", f"已保存:\n{path}")


# ══════════════════════════════════════════════════════════════════════════════
#  Tab 4 — EPUB Builder  (参照图1: 顶部元数据条 + 左侧书籍结构树 + 右侧源码/预览)
# ══════════════════════════════════════════════════════════════════════════════

class EPUBTab(tk.Frame):
    def __init__(self, parent, app: App):
        super().__init__(parent, bg=BG)
        self.app = app
        self._build()

    def _build(self):
        root = tk.Frame(self, bg=CARD, highlightbackground=BORDER, highlightthickness=1)
        root.pack(fill="both", expand=True, padx=10, pady=10)

        # ── 顶部元数据条 ──────────────────────────────────────────────────────
        top = tk.Frame(root, bg=CARD, height=52)
        top.pack(fill="x"); top.pack_propagate(False)
        tk.Label(top, text="EPUB Builder", bg=CARD, fg=INK,
                 font=("Helvetica",13,"bold")).pack(side="left", padx=14)

        pills = tk.Frame(top, bg=CARD)
        pills.pack(side="left", padx=8)
        self._pill_title = self._make_pill(pills, "📖", self.app.book_title.get() or "未命名")
        self._pill_author = self._make_pill(pills, "👤", self.app.book_author.get() or "作者未填")
        self._pill_lang = self._make_pill(pills, "🌐", "ja · EPUB3")
        self._pill_mode = self._make_pill(pills, "✎", "vertical-rl")
        self._pill_pages = self._make_pill(pills, "🖼", "0 页 · 0 图")
        for p in (self._pill_title,self._pill_author,self._pill_lang,self._pill_mode,self._pill_pages):
            p.pack(side="left", padx=3)

        ttk.Button(top, text="⚙  Build EPUB", style="Accent.TButton",
                   command=self._build_epub).pack(side="right", padx=14, pady=10)

        tk.Frame(root, bg=BORDER, height=1).pack(fill="x")
        self._prog = ttk.Progressbar(root, mode="determinate", style="Horizontal.TProgressbar")
        self._prog.pack(fill="x")

        body = tk.Frame(root, bg=CARD)
        body.pack(fill="both", expand=True)

        # ── 左侧：书籍结构 / 设置 ─────────────────────────────────────────────
        left = tk.Frame(body, bg=CARD, width=230)
        left.pack(side="left", fill="y"); left.pack_propagate(False)

        tk.Label(left, text="Book Structure", bg=CARD, fg=MUTED,
                 font=("Helvetica",10)).pack(anchor="w", padx=14, pady=(12,4))
        self._tree = ttk.Treeview(left, show="tree", selectmode="browse", height=14)
        self._tree.pack(fill="x", padx=8)
        self._tree.insert("","end", text="点击 Build 生成", tags=("placeholder",))
        self._tree.bind("<<TreeviewSelect>>", self._on_tree_select)

        tk.Frame(left, bg=BORDER, height=1).pack(fill="x", pady=8)

        # 元数据编辑（折叠式简易表单）
        tk.Label(left, text="元数据", bg=CARD, fg=MUTED,
                 font=("Helvetica",10)).pack(anchor="w", padx=14, pady=(4,2))
        for lbl, var in [("书名", self.app.book_title), ("作者", self.app.book_author),
                        ("出版社", self.app.book_publisher), ("卷号", self.app.book_volume)]:
            tk.Label(left, text=lbl, bg=CARD, fg="#555", font=("Helvetica",10)
                     ).pack(anchor="w", padx=14, pady=(4,0))
            e = ttk.Entry(left, textvariable=var)
            e.pack(fill="x", padx=14, pady=(0,2))
            e.bind("<FocusOut>", lambda ev: self._update_pills())

        tk.Label(left, text="CSS 模板", bg=CARD, fg="#555", font=("Helvetica",10)
                 ).pack(anchor="w", padx=14, pady=(8,0))
        ttk.Combobox(left, textvariable=self.app.css_template, values=["denki","mf","web"],
                    state="readonly").pack(fill="x", padx=14, pady=(0,4))

        mf = tk.Frame(left, bg=CARD); mf.pack(fill="x", padx=14, pady=(2,10))
        ttk.Radiobutton(mf, text="竖排", variable=self.app.writing_mode, value="vertical",
                        command=self._update_pills).pack(side="left")
        ttk.Radiobutton(mf, text="横排", variable=self.app.writing_mode, value="horizontal",
                        command=self._update_pills).pack(side="left", padx=8)

        tk.Frame(body, bg=BORDER, width=1).pack(side="left", fill="y")

        # ── 右侧：源码 / 预览 ─────────────────────────────────────────────────
        right = tk.Frame(body, bg=CARD)
        right.pack(side="left", fill="both", expand=True)

        rhead = tk.Frame(right, bg=CARD, height=40)
        rhead.pack(fill="x"); rhead.pack_propagate(False)
        self._preview_title = tk.Label(rhead, text="选择左侧文件预览内容", bg=CARD, fg=MUTED,
                                       font=("Helvetica",11))
        self._preview_title.pack(side="left", padx=14)
        self._view = tk.StringVar(value="code")
        for val, label in [("code","源码"),("preview","预览")]:
            tk.Radiobutton(rhead, text=label, value=val, variable=self._view,
                          command=self._refresh_preview, bg=CARD, indicatoron=False,
                          relief="flat", padx=12, pady=5, font=("Helvetica",10),
                          selectcolor=ACC_BG).pack(side="right", padx=(0,4), pady=6)

        tk.Frame(right, bg=BORDER, height=1).pack(fill="x")

        self._code_view = scrolledtext.ScrolledText(right, font=("Menlo",11), bg="#FAFAF8",
            fg="#333", relief="flat", bd=0, padx=14, pady=12, state="disabled", wrap="none")
        self._code_view.pack(fill="both", expand=True)

        self._preview_frame = tk.Frame(right, bg="#F0EFEA")
        self._empty_preview = tk.Label(self._preview_frame,
            text="📦\n\n准备就绪\n点击「Build EPUB」生成文件结构",
            bg="#F0EFEA", fg=MUTED, font=("Helvetica",13), justify="center")
        self._empty_preview.pack(expand=True)

        # ── 底部统计 ──────────────────────────────────────────────────────────
        tk.Frame(root, bg=BORDER, height=1).pack(fill="x")
        stats = tk.Frame(root, bg=CARD, height=54)
        stats.pack(fill="x"); stats.pack_propagate(False)
        self._stat_labels = {}
        for key, label in [("files","文件数"),("chapters","章节"),("images","图片"),
                           ("size","估算大小"),("status","状态")]:
            f = tk.Frame(stats, bg=CARD, highlightbackground=BORDER, highlightthickness=1)
            f.pack(side="left", fill="both", expand=True)
            v = tk.Label(f, text="0" if key!="status" else "待构建", bg=CARD, fg=INK,
                        font=("Helvetica",15,"bold"))
            v.pack(pady=(8,0))
            tk.Label(f, text=label, bg=CARD, fg=MUTED, font=("Helvetica",10)).pack(pady=(0,8))
            self._stat_labels[key] = v

        self._log_lines = []

    def _make_pill(self, parent, icon, text):
        f = tk.Frame(parent, bg="#F5F4F0", highlightbackground=BORDER, highlightthickness=1)
        tk.Label(f, text=f"{icon} {text}", bg="#F5F4F0", fg="#444",
                 font=("Helvetica",10), padx=8, pady=3).pack()
        return f

    def _update_pills(self):
        def relabel(pill, icon, text):
            for w in pill.winfo_children():
                w.config(text=f"{icon} {text}")
        relabel(self._pill_title, "📖", self.app.book_title.get() or "未命名")
        relabel(self._pill_author, "👤", self.app.book_author.get() or "作者未填")
        relabel(self._pill_mode, "✎", "vertical-rl" if self.app.writing_mode.get()=="vertical" else "horizontal")
        doc = self.app.fmt_result_doc or self.app.ocr_result_doc
        n_pages = len(doc.pages) if doc else 0
        n_imgs  = len(doc.image_blocks()) if doc else 0
        relabel(self._pill_pages, "🖼", f"{n_pages} 页 · {n_imgs} 图")

    def _on_tree_select(self, event):
        sel = self._tree.selection()
        if not sel: return
        item = sel[0]
        text = self._tree.item(item, "text")
        data = self._tree_data.get(item) if hasattr(self, "_tree_data") else None
        if data:
            self._preview_title.config(text=data["name"])
            self._show_file(data)

    def _show_file(self, data):
        self._code_view.config(state="normal")
        self._code_view.delete("1.0","end")
        self._code_view.insert("1.0", data.get("content","(二进制文件，无法预览)"))
        self._code_view.config(state="disabled")
        self._current_file = data
        self._refresh_preview()

    def _refresh_preview(self):
        if self._view.get() == "code":
            self._preview_frame.pack_forget()
            self._code_view.pack(fill="both", expand=True)
        else:
            self._code_view.pack_forget()
            self._preview_frame.pack(fill="both", expand=True)

    def _build_epub(self):
        doc = self.app.fmt_result_doc or self.app.ocr_result_doc
        if not doc:
            messagebox.showerror("错误", "请先完成 OCR 和格式处理")
            return
        if not self.app.output_epub.get():
            path = filedialog.asksaveasfilename(title="保存 EPUB", defaultextension=".epub",
                                                filetypes=[("EPUB","*.epub")])
            if not path: return
            self.app.output_epub.set(path)

        if self.app.book_title.get():     doc.metadata.title     = self.app.book_title.get()
        if self.app.book_author.get():    doc.metadata.author    = self.app.book_author.get()
        if self.app.book_publisher.get(): doc.metadata.publisher = self.app.book_publisher.get()
        if self.app.book_volume.get():    doc.metadata.volume    = self.app.book_volume.get()

        self._update_pills()
        self._stat_labels["status"].config(text="构建中…")
        self._prog["value"] = 30
        threading.Thread(target=self._build_bg, args=(doc,), daemon=True).start()

    def _build_bg(self, doc):
        try:
            from builder.epub_builder import build_epub
            import io, contextlib, zipfile
            output = self.app.output_epub.get()
            tmpl = self.app.css_template.get()
            vert = self.app.writing_mode.get() == "vertical"
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                build_epub(doc, output_path=output, css_template=tmpl, vertical=vert, verbose=True)

            self.after(0, lambda: self._prog.configure(value=90))
            self.after(0, lambda: self._show_tree(output))
            size_kb = Path(output).stat().st_size // 1024
            n_ch = len(doc.toc)
            n_img = len(doc.image_blocks())
            self.after(0, lambda: self._stat_labels["files"].config(text="—"))
            self.after(0, lambda: self._stat_labels["chapters"].config(text=str(n_ch)))
            self.after(0, lambda: self._stat_labels["images"].config(text=str(n_img)))
            self.after(0, lambda: self._stat_labels["size"].config(text=f"{size_kb} KB"))
            self.after(0, lambda: self._stat_labels["status"].config(text="✓ 完成"))
            self.after(0, lambda: self._prog.configure(value=100))
            self.after(0, lambda: messagebox.showinfo("完成", f"EPUB 已生成:\n{output}"))
        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            self.after(0, lambda: self._stat_labels["status"].config(text="✗ 失败"))
            self.after(0, lambda: messagebox.showerror("生成失败", f"{e}\n\n{tb}"))

    def _show_tree(self, epub_path):
        import zipfile
        self._tree.delete(*self._tree.get_children())
        self._tree_data = {}
        try:
            with zipfile.ZipFile(epub_path) as zf:
                names = sorted(zf.namelist())
                contents = {}
                for n in names:
                    if n.endswith(('.xhtml','.opf','.css','.xml')):
                        try:
                            contents[n] = zf.read(n).decode('utf-8', errors='replace')
                        except Exception:
                            contents[n] = ""

            n_files = len(names)
            self._stat_labels["files"].config(text=str(n_files))

            root = self._tree.insert("","end", text=f"📦 {Path(epub_path).name}", open=True)
            dirs = {}
            for name in names:
                parts = name.split("/")
                if len(parts) == 1:
                    item = self._tree.insert(root,"end", text=f"📄 {name}")
                    self._tree_data[item] = {"name": name, "content": contents.get(name, "(二进制)")}
                else:
                    dname = parts[0]
                    if dname not in dirs:
                        dirs[dname] = self._tree.insert(root,"end", text=f"📁 {dname}/", open=True)
                    fname = "/".join(parts[1:])
                    if fname:
                        ext = Path(fname).suffix.lower().lstrip(".")
                        icon = {"xhtml":"📝","css":"🎨","jpg":"🖼","jpeg":"🖼",
                               "png":"🖼","opf":"⚙️","xml":"⚙️"}.get(ext,"📄")
                        item = self._tree.insert(dirs[dname],"end", text=f"{icon} {fname}")
                        self._tree_data[item] = {"name": name, "content": contents.get(name, "(二进制文件，无法预览)")}
        except Exception as e:
            self._tree.insert("","end", text=f"⚠️ 无法读取: {e}")


# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    app = App()
    app.mainloop()
