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
