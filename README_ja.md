<div align="center">

[简体中文](README.md) · [English](README_en.md) · **日本語**

</div>

<p align="center">
  <img src="assets/novel_formatter_banner.webp" alt="Novel Formatter" width="100%">
</p>

<h1 align="center">Novel Formatter</h1>

<p align="center"></p>

<p align="center">
  <img alt="macOS" src="https://img.shields.io/badge/macOS-Apple%20Silicon%20%2F%20Intel-black?logo=apple">
  <img alt="Windows" src="https://img.shields.io/badge/Windows-x64-0078D4?logo=windows11&logoColor=white">
  <img alt="PySide6" src="https://img.shields.io/badge/UI-PySide6-41CD52?logo=qt&logoColor=white">
  <img alt="Japanese OCR" src="https://img.shields.io/badge/OCR-Japanese-2f6feb">
  <img alt="EPUB" src="https://img.shields.io/badge/Output-EPUB-8A2BE2">
</p>

> 日本語縦書き OCR · 複数 OCR エンジンの比較・統合 · 画像とテキストの照合校正 · Formatter · EPUB · AI 校正パッケージ · Apple Vision

## 主な機能

- **画像・PDF の取り込み**：画像、画像フォルダー、PDF ページを処理します。
- **日本語縦書き OCR**：書籍の縦書き、列ごとの認識、複数列にまたがる文章に対応します。
- **複数 OCR エンジンの比較・統合**：Apple Vision、Windows Snipping OCR、NDLOCR-Lite、Hayai OCR、48px OCR、PaddleOCR などの結果を比較・確認できます。
- **画像とテキストの照合校正**：元画像と認識結果を照らし合わせ、誤字、脱字、信頼度の低い箇所、レイアウト上の問題を検出・確認します。
- **ルビ・柱・ノンブルの処理**：FindText CenterNet を利用してルビ領域を補助的に検出し、柱、ページ番号、列やページをまたぐ不要な断片を整理します。
- **Formatter（テキスト整形）**：段落、句読点、見出し、ページ間の文章の接続などを OCR 後に調整します。
- **EPUB の作成**：整形した本文、文書構造、リソースから EPUB を出力します。
- **AI 校正パッケージ**：OCR の根拠、本文、構造、リソースを出力し、GPT、Gemini、GLM、DeepSeek などの互換サービスで追加校正して出版向け EPUB の作成を支援します。
- **ローカルデータを優先**：ユーザーの原稿、モデルの重み、キャッシュ、ログ、データベースなどをソースリポジトリにコミットしません。

## v2.0 の主な更新内容

v1.3 と比べて、v2.0 では OCR、校正、PDF、EPUB のワークフローを大幅に拡充しました。

- **新しいプロジェクトワークスペース**：ページ、OCR 結果、校正判断、テキスト整形、EPUB 制作の状態を一元管理し、プロジェクトの復元や最近のプロジェクトの確認に対応しました。
- **複数 OCR と校正判断の改善**：列ごと・ページ全体の認識、結果の不一致の確認、画像との照合、手動校正を改善し、Hayai OCR、NDLOCR-Lite、48px OCR の連携を最適化しました。
- **OS 標準 OCR の拡充**：Windows 11 Snipping OCR に対応し、Intel Mac と Apple Silicon Mac で Apple Vision OCR の処理を改善しました。
- **PDF テキストレイヤー処理の刷新**：「ページ管理」に取り込んだ元 PDF と物理ページの対応を活用し、縦書きの読み順、ページ間の文章接続、段落構造、出版向けの文章整理を改善しました。
- **AI 校正判断の強化**：GPT、Gemini、GLM、DeepSeek などの互換サービスに対応し、校正パッケージ、OCR の根拠、手動修正の安全な往復処理を強化しました。
- **EPUB / TEI と UI の改善**：表紙、挿絵、改ページ、メタデータ、EPUB 構造、プレビュー、進捗表示、長編文書の処理安定性を向上させました。

## OCR エンジン

対応またはオプションで利用可能：

- Apple Vision（macOS）
- Windows Snipping OCR（Windows 11）
- NDLOCR-Lite
- Hayai OCR
- 48px OCR
- PaddleOCR
- Paddle VL MLX（macOS Apple Silicon）
- FindText CenterNet（ルビ領域の検出補助）

Apple Vision、Swift OCR Helper、Apple Pencil の手書き文字認識は macOS 専用です。Windows Snipping OCR は対応するシステム OCR コンポーネントがある Windows 11 環境で利用できます。

## 推奨ワークフロー

1. PDF、画像、または画像フォルダーを取り込みます。
2. 表紙、扉、目次、挿絵、本文、あとがきなどのページ種別を設定します。
3. OCR エンジンを選び、必要に応じて列ごとの認識を有効にします。
4. OCR を実行します。
5. 「OCR 比較」と「画像・テキスト照合」で、認識結果の不一致、脱字、ルビ、ページ番号、ページ間の文章接続を確認します。
6. 統合結果を適用し、Formatter で本文を整形します。
7. EPUB を出力するか、AI 校正パッケージを出力して追加校正します。

## AI 校正パッケージ

OCR の根拠資料、テキスト、文書構造、関連リソースをまとめることで、単なるテキストよりも元のページの文脈を参照した AI 校正ができます。

主な用途：

- OCR の誤字、脱字、句読点の修正
- 人名、地名、用語の表記統一
- ページや章をまたぐ文脈の確認
- EPUB の構造と組版の確認
- 出版品質に近い EPUB の仕上げ

## インストールと起動

Python 3.10 以降を推奨します。

### macOS

```bash
git clone https://github.com/Amster-Ilvil/Novel-formatter.git
cd Novel-formatter
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -r requirements.txt
python gui_pyside6.py
```

または、次のスクリプトを実行します。

```bash
./run_novel_formatter.command
```

### Windows

```powershell
git clone https://github.com/Amster-Ilvil/Novel-formatter.git
cd Novel-formatter
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
python -m pip install -r requirements.txt
python gui_pyside6.py
```

ビルド済みインストーラーは [Releases](https://github.com/Amster-Ilvil/Novel-formatter/releases) からダウンロードできます。

## プラットフォーム対応状況

| 機能 | macOS | Windows |
|---|---:|---:|
| メイン画面 | ✓ | ✓ |
| PDF・画像 OCR | ✓ | ✓ |
| 複数 OCR エンジン | ✓ | ✓ |
| Formatter / EPUB | ✓ | ✓ |
| AI 校正パッケージ | ✓ | ✓ |
| Apple Vision | ✓ | — |
| Windows Snipping OCR | — | Windows 11 |
| Swift OCR Helper | ✓ | — |
| Apple Pencil の手書き文字認識 | ✓ | — |

## モデルとローカルデータ

OCR モデルの重み、キャッシュ、仮想環境、ログ、データベース、ユーザーの生成物はリポジトリにコミットしません。一部の OCR エンジンは初回使用時に必要な依存関係やモデルをローカルに準備します。

## 参考プロジェクト・謝辞

本プロジェクトの UI、OCR 連携、文書処理、手書き文字認識では、次のプロジェクトや公式資料を利用または参考にしています。

- [Qt for Python / PySide6](https://doc.qt.io/qtforpython/)：デスクトップ GUI フレームワーク。
- [Apple Vision](https://developer.apple.com/documentation/vision)：macOS 標準の OCR と画像認識。
- [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)：汎用 OCR とクロスプラットフォーム認識。
- [NDLOCR-Lite](https://github.com/ndl-lab/ndlocr-lite)：日本語書籍向け OCR とレイアウト解析。
- [Open Model Zoo](https://github.com/openvinotoolkit/open_model_zoo)：オプションの手書き文字認識モデルと推論の参考資料。
- [PyMuPDF](https://pymupdf.readthedocs.io/)：PDF ページ、画像、テキストレイヤーの処理。
- [python-docx](https://python-docx.readthedocs.io/)：DOCX の読み込みと生成。
- [Pillow](https://python-pillow.org/)：画像の読み込み、切り出し、前処理。
- [jlect-jhr](https://github.com/ZacharyRead/jlect-jhr)：日本語手書き文字認識の補助リソース。ライセンスは `third_party/jlect_jhr/LICENSE.txt` を参照してください。

第三者のプロジェクト、モデル、リソースにはそれぞれのライセンスが適用されます。本リポジトリは OCR モデルの重みを再配布しません。

## Star History

<p align="center">
  <a href="https://www.star-history.com/?type=date&repos=Amster-Ilvil%2FNovel-formatter">
    <picture>
      <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=Amster-Ilvil/Novel-formatter&type=Date&theme=dark">
      <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=Amster-Ilvil/Novel-formatter&type=Date">
      <img alt="Novel Formatter Star History Chart" src="https://api.star-history.com/svg?repos=Amster-Ilvil/Novel-formatter&type=Date" width="800">
    </picture>
  </a>
</p>


## 利用目的とライセンス

独自のコードとドキュメントは [MIT License](LICENSE) で公開しています。条件を遵守する限り、個人利用、学習、研究、商用利用が可能です。

第三者の OCR エンジン、モデル、依存ライブラリ、プラットフォーム機能にはそれぞれのライセンスと利用規約が適用されます。詳しくは [ACKNOWLEDGEMENTS.md](ACKNOWLEDGEMENTS.md) をご覧ください。
