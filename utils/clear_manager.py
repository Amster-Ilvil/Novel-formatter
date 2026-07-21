from pathlib import Path
import shutil

class ClearManager:
    @staticmethod
    def clear_dir(path):
        if not path:
            return
        p=Path(path)
        if p.exists() and p.is_dir():
            for x in p.iterdir():
                try:
                    shutil.rmtree(x) if x.is_dir() else x.unlink()
                except Exception:
                    pass

    @classmethod
    def clear_pages(cls, tab):
        for n in ("page_images","page_overrides","selected_pages","thumb_cache"):
            if hasattr(tab,n):
                try: getattr(tab,n).clear()
                except Exception: pass
        for n,val in (("_last_loaded_raw_inputs",None),("_current_filter","all")):
            if hasattr(tab,n): setattr(tab,n,val)
        for n,text in (("_file_lbl","未打开文件"),("_count_lbl",""),("_sel_lbl","选中 0 页 · 标记为："),("_stat_label","")):
            if hasattr(tab,n):
                getattr(tab,n).setText(text)
        if hasattr(tab,"_prog"): tab._prog.setVisible(False)
        if hasattr(tab,"_render"): tab._render()

    @classmethod
    def clear_ocr(cls, tab):
        for n in ("_pending_inputs","_selected_inputs"):
            if hasattr(tab,n):
                try: getattr(tab,n).clear()
                except Exception: pass
        for n in ("_input_lbl","_log_view","_result_view"):
            if hasattr(tab,n):
                w=getattr(tab,n)
                if hasattr(w,"clear"): w.clear()
        if hasattr(tab,"_input_lbl"): tab._input_lbl.setText("（尚未选择输入）")
        if hasattr(tab,"_prog"): tab._prog.setVisible(False)
        if hasattr(tab,"_run_btn"): tab._run_btn.setEnabled(True)

    @classmethod
    def clear_formatter(cls, tab):
        for n in ("_ocr_doc","_fmt_doc","_paddle_doc","ocr_result_doc","formatted_result","current_document"):
            if hasattr(tab,n): setattr(tab,n,None)
        for n in ("_before","_after","_diff_view"):
            if hasattr(tab,n):
                try: getattr(tab,n).clear()
                except Exception: pass
        if hasattr(tab,"_update_version"):
            try: tab._update_version(None)
            except Exception: pass

    @classmethod
    def clear_epub(cls, tab):
        for n in ("_doc","_tree_data"):
            if hasattr(tab,n): setattr(tab,n,None if n=="_doc" else [])
        for n in ("_tree","_code_view"):
            if hasattr(tab,n):
                try: getattr(tab,n).clear()
                except Exception: pass
        if hasattr(tab,"_update_pills"):
            tab._update_pills()
