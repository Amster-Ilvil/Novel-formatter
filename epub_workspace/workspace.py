from PySide6.QtWidgets import QWidget, QHBoxLayout, QListWidget, QTextBrowser, QPushButton, QVBoxLayout

class EPUBWorkspace(QWidget):
    """v2.4.5 EPUB 工作区：页面树 + 实时HTML预览基础框架。"""
    def __init__(self, parent=None):
        super().__init__(parent)
        layout=QHBoxLayout(self)
        left=QVBoxLayout()
        self.pages=QListWidget()
        self.pages.addItems(["封面","扉页","目录","第一章","插图"])
        for t in ["+ 新建页面","🖼 添加图片","🗑 删除页面","⬆ 上移","⬇ 下移"]:
            left.addWidget(QPushButton(t))
        left.addWidget(self.pages)
        self.preview=QTextBrowser()
        self.preview.setHtml("<h2>EPUB Preview</h2><p>实时预览区域</p>")
        layout.addLayout(left,1)
        layout.addWidget(self.preview,3)
