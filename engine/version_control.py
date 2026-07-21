#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
版本控制子命令（风格类似 Git），操作对象是 UnifiedDocument 的内容寻址仓库。

用法：
    # 查看某本书的处理历史
    python -m engine.version_control log --repo /tmp/novel_repo_xxxx

    # 把某个 JSON 文档的当前状态提交一次（一般不需要手动做，
    # run_pipeline / GUI 每步都会自动 commit；这个命令主要用于手动挂载
    # 一份外部编辑过的 JSON 到仓库里）
    python -m engine.version_control commit --repo .novel --doc book.json -m "手动校对"

    # 回退到某个 commit，把内容写回一个 JSON 文件
    python -m engine.version_control checkout --repo .novel <commit_id> --out restored.json

    # 比较两个 commit 之间 blocks 的差异（按文本内容做增删改判定）
    python -m engine.version_control diff --repo .novel <old_commit> <new_commit>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from models.document import UnifiedDocument, Repository


def cmd_init(args):
    """初始化一个空仓库（创建目录结构，并提交一个空文档作为根提交）"""
    doc = UnifiedDocument()
    commit_id = doc.commit(args.repo, "init", "Initial empty document")
    print(f"已初始化空仓库: {args.repo}")
    print(f"根提交: {commit_id}")


def cmd_commit(args):
    """把一份 JSON 文档的当前内容挂载/提交进仓库"""
    if not Path(args.doc).exists():
        print(f"❌ 文档不存在: {args.doc}")
        sys.exit(1)

    with open(args.doc, encoding="utf-8") as f:
        doc = UnifiedDocument.from_json(f.read())

    # 若这份 JSON 本身没有关联仓库，就挂到 --repo 指定的仓库上
    if doc.repo is None:
        doc.repo = Repository(args.repo)
    elif str(doc.repo.path) != str(Path(args.repo)):
        print(f"⚠️  该文档已关联仓库 {doc.repo.path}，将改用 --repo 指定的 {args.repo}")
        doc.repo = Repository(args.repo)

    commit_id = doc.commit(args.repo, args.step or "manual", args.message or "")
    print(f"已提交: {commit_id}")

    # 把带有新 commit 指针的 JSON 写回原文件，方便下次继续从这里提交
    with open(args.doc, "w", encoding="utf-8") as f:
        f.write(doc.to_json())


def cmd_log(args):
    """列出仓库里某个分支的提交历史"""
    repo = Repository(args.repo)
    head = repo.read_ref(args.branch)
    if not head:
        print(f"分支 {args.branch} 还没有任何提交")
        return

    commits = repo.log(head, max_count=args.max)
    for c in commits:
        import datetime
        ts = datetime.datetime.fromtimestamp(c["timestamp"]).strftime("%Y-%m-%d %H:%M:%S")
        print(f"commit {c['id']}")
        print(f"  步骤: {c['step']}")
        if c.get("message"):
            print(f"  说明: {c['message']}")
        print(f"  时间: {ts}")
        print()


def cmd_checkout(args):
    """把某个 commit 的内容还原出来，写入一个 JSON 文件"""
    repo = Repository(args.repo)
    try:
        commit_data = repo.read_commit(args.commit_id)
    except FileNotFoundError:
        print(f"❌ 提交不存在: {args.commit_id}")
        sys.exit(1)

    root = repo.read_blob(commit_data["root"])
    doc = UnifiedDocument.from_dict(root)
    doc.repo = repo
    doc.commit_id = args.commit_id

    out_path = args.out or f"checkout_{args.commit_id[:8]}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(doc.to_json())
    print(f"已还原提交 {args.commit_id[:8]} → {out_path}")
    print(f"  {len(doc.blocks)} 个块，{len(doc.toc)} 个章节")


def cmd_diff(args):
    """比较两个 commit 之间的 blocks 差异（按 (page, text) 做简单增删判定）"""
    repo = Repository(args.repo)

    def load(commit_id):
        commit_data = repo.read_commit(commit_id)
        root = repo.read_blob(commit_data["root"])
        return UnifiedDocument.from_dict(root)

    old_doc = load(args.old_commit)
    new_doc = load(args.new_commit)

    old_texts = [(b.page, b.type.value, b.text) for b in old_doc.blocks]
    new_texts = [(b.page, b.type.value, b.text) for b in new_doc.blocks]

    old_set = set(old_texts)
    new_set = set(new_texts)

    removed = [t for t in old_texts if t not in new_set]
    added = [t for t in new_texts if t not in old_set]

    print(f"{args.old_commit[:8]} → {args.new_commit[:8]}")
    print(f"  块数: {len(old_texts)} → {len(new_texts)}")
    print()
    if removed:
        print(f"删除 {len(removed)} 块:")
        for page, btype, text in removed[:30]:
            print(f"  - [p{page} {btype}] {text[:40]}")
        if len(removed) > 30:
            print(f"  ... 还有 {len(removed) - 30} 条")
        print()
    if added:
        print(f"新增 {len(added)} 块:")
        for page, btype, text in added[:30]:
            print(f"  + [p{page} {btype}] {text[:40]}")
        if len(added) > 30:
            print(f"  ... 还有 {len(added) - 30} 条")
    if not removed and not added:
        print("（内容完全一致）")


def main():
    parser = argparse.ArgumentParser(
        description="Novel Formatter 版本控制（Git 风格子命令）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    subparsers = parser.add_subparsers(dest="command")

    p_init = subparsers.add_parser("init", help="初始化一个新仓库")
    p_init.add_argument("--repo", required=True, help="仓库目录")

    p_commit = subparsers.add_parser("commit", help="提交一份 JSON 文档的当前状态")
    p_commit.add_argument("--repo", required=True, help="仓库目录")
    p_commit.add_argument("--doc", required=True, help="UnifiedDocument JSON 文件路径")
    p_commit.add_argument("-m", "--message", help="提交说明")
    p_commit.add_argument("--step", help="步骤标签（默认 manual）")

    p_log = subparsers.add_parser("log", help="查看提交历史")
    p_log.add_argument("--repo", required=True, help="仓库目录")
    p_log.add_argument("--branch", default="main", help="分支名（默认 main）")
    p_log.add_argument("--max", type=int, default=100, help="最多显示多少条")

    p_checkout = subparsers.add_parser("checkout", help="还原某个提交到 JSON 文件")
    p_checkout.add_argument("--repo", required=True, help="仓库目录")
    p_checkout.add_argument("commit_id", help="要还原的提交 id")
    p_checkout.add_argument("--out", help="输出 JSON 路径（默认 checkout_<id前8位>.json）")

    p_diff = subparsers.add_parser("diff", help="比较两个提交之间 blocks 的差异")
    p_diff.add_argument("--repo", required=True, help="仓库目录")
    p_diff.add_argument("old_commit", help="旧提交 id")
    p_diff.add_argument("new_commit", help="新提交 id")

    args = parser.parse_args()

    if args.command == "init":
        cmd_init(args)
    elif args.command == "commit":
        cmd_commit(args)
    elif args.command == "log":
        cmd_log(args)
    elif args.command == "checkout":
        cmd_checkout(args)
    elif args.command == "diff":
        cmd_diff(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
