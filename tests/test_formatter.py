from engine.formatter import (
    clean_metadata_blocks,
    merge_broken_sentences,
    remove_duplicates,
    restore_dialogue_breaks,
)
from models.document import Block, BlockType, Metadata, UnifiedDocument


def _document(*blocks: Block, source_engine: str = "apple_vision") -> UnifiedDocument:
    return UnifiedDocument(blocks=list(blocks), metadata=Metadata(source_engine=source_engine))


def test_preserves_repeated_burologue_title():
    title = "ブロローグ魔王敗れる。そしてー"
    doc = _document(
        Block(BlockType.PARAGRAPH, title, page=1),
        Block(BlockType.PARAGRAPH, title, page=1),
    )

    result = remove_duplicates(doc)

    assert [block.text for block in result.blocks] == [title, title]


def test_metadata_cleanup_preserves_repeated_burologue_title_across_pages():
    title = "ブロローグ魔王敗れる。そしてー"
    doc = _document(
        Block(BlockType.PARAGRAPH, title, page=1),
        Block(BlockType.PARAGRAPH, title, page=2),
    )

    result = clean_metadata_blocks(doc)

    assert [block.text for block in result.blocks] == [title, title]


def test_inline_ocr_quote_does_not_merge_past_sentence_end():
    doc = _document(
        Block(BlockType.PARAGRAPH, "漆黒の波動ーそれは人間の有する『怒り』「憎しみ』などのあらゆる負の感情。", page=1),
        Block(BlockType.PARAGRAPH, "もちろん次の文である。", page=1),
    )

    result = merge_broken_sentences(doc)

    assert [block.text for block in result.blocks] == [
        "漆黒の波動ーそれは人間の有する『怒り』「憎しみ』などのあらゆる負の感情。",
        "もちろん次の文である。",
    ]


def test_does_not_split_an_inline_quoted_term_as_dialogue():
    text = "漆黒の「何か」が絶えずあふれ出ている。"
    result = restore_dialogue_breaks(_document(Block(BlockType.PARAGRAPH, text)))

    assert [(block.type, block.text) for block in result.blocks] == [
        (BlockType.PARAGRAPH, text)
    ]


def test_splits_dialogue_at_a_sentence_boundary():
    result = restore_dialogue_breaks(_document(
        Block(BlockType.PARAGRAPH, "彼は立ち止まった。「何者だ？」彼は尋ねた。")
    ))

    assert [(block.type, block.text) for block in result.blocks] == [
        (BlockType.PARAGRAPH, "彼は立ち止まった。"),
        (BlockType.DIALOGUE, "「何者だ？」"),
        (BlockType.PARAGRAPH, "彼は尋ねた。"),
    ]


def test_merges_an_unfinished_ocr_sentence_across_pages():
    doc = _document(
        Block(BlockType.PARAGRAPH, "時間と共に次第に大きく、そしてその源", page=6),
        Block(BlockType.PARAGRAPH, "がドンドン近くなっていくことが感じられる。", page=7),
        source_engine="apple_vision",
    )

    result = merge_broken_sentences(doc)

    assert [block.text for block in result.blocks] == [
        "時間と共に次第に大きく、そしてその源がドンドン近くなっていくことが感じられる。"
    ]


def test_does_not_merge_a_completed_ocr_sentence_across_pages():
    doc = _document(
        Block(BlockType.PARAGRAPH, "第六页的句子结束。", page=6),
        Block(BlockType.PARAGRAPH, "第七页的新段落。", page=7),
        source_engine="apple_vision",
    )

    result = merge_broken_sentences(doc)

    assert [block.text for block in result.blocks] == ["第六页的句子结束。", "第七页的新段落。"]


def test_preserve_ocr_layout_skips_structure_changing_steps(tmp_path):
    from engine.formatter import run_pipeline

    doc = _document(
        Block(BlockType.PARAGRAPH, "途中で", page=1),
        Block(BlockType.PARAGRAPH, "続く。", page=1),
        Block(BlockType.PARAGRAPH, "彼は言った。「行こう」そして歩いた。", page=1),
    )
    doc.metadata.preserve_ocr_layout = True

    result = run_pipeline(
        doc,
        steps=["merge_sentences", "dialogue_restore", "restore_indents"],
        verbose=False,
        repo_path=str(tmp_path / "repo"),
    )

    assert [(block.type, block.text) for block in result.blocks] == [
        (BlockType.PARAGRAPH, "途中で"),
        (BlockType.PARAGRAPH, "続く。"),
        (BlockType.PARAGRAPH, "彼は言った。「行こう」そして歩いた。"),
    ]
    assert [log["step"] for log in result.processing_log[-3:]] == [
        "merge_sentences",
        "dialogue_restore",
        "restore_indents",
    ]


def test_preserve_ocr_layout_flag_round_trips_json():
    doc = _document(Block(BlockType.PARAGRAPH, "本文"))
    doc.metadata.preserve_ocr_layout = True

    loaded = UnifiedDocument.from_json(doc.to_json())

    assert loaded.metadata.preserve_ocr_layout is True


def test_semantic_duplicate_resolver_keeps_corrected_replacement():
    doc = _document(
        Block(BlockType.PARAGRAPH, "◼ とっとと終わらせるべく、俺はキノープと共にフエール家の邸宅に向かった", page=1),
        Block(BlockType.PARAGRAPH, "◼ とっとと終わらせるべく、俺はキノープと共にフェール家の邸宅に向かった。", page=1, ocr_raw="フエール家"),
    )

    result = remove_duplicates(doc)

    assert [block.text for block in result.blocks] == [
        "◼ とっとと終わらせるべく、俺はキノープと共にフェール家の邸宅に向かった。"
    ]
    assert result.processing_log[-1]["count"] == 1


def test_semantic_duplicate_resolver_keeps_longer_containing_sentence():
    doc = _document(
        Block(BlockType.PARAGRAPH, "キノープはどこか驚いた様子だ。", page=1),
        Block(BlockType.PARAGRAPH, "キノープはどこか驚いた様子だ。何だ？俺が無能な父親を消す心配しているのか？", page=1),
    )

    result = remove_duplicates(doc)

    assert [block.text for block in result.blocks] == [
        "キノープはどこか驚いた様子だ。何だ？俺が無能な父親を消す心配しているのか？"
    ]


def test_semantic_duplicate_resolver_only_checks_nearby_blocks():
    repeated = "「待って。」"
    doc = _document(
        Block(BlockType.DIALOGUE, repeated, page=1),
        Block(BlockType.PARAGRAPH, "彼は走った。", page=1),
        Block(BlockType.PARAGRAPH, "角を曲がった。", page=1),
        Block(BlockType.PARAGRAPH, "雨が降り出した。", page=2),
        Block(BlockType.PARAGRAPH, "息を整えた。", page=2),
        Block(BlockType.DIALOGUE, repeated, page=3),
    )

    result = remove_duplicates(doc)

    assert [block.text for block in result.blocks].count(repeated) == 2
