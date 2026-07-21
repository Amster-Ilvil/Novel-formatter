from engine.formatter import merge_cross_page_sentences
from models.document import Block, BlockType, UnifiedDocument


def make_block(text, page_index=None, order_in_page=1, block_type=BlockType.PARAGRAPH):
    return Block(type=block_type, text=text, page_index=page_index, order_in_page=order_in_page)


def test_merge_word_split_across_pages():
    left = make_block("そのために玉座の主は、一見するとぼんやりと人の形をしているというだけで、その輪", 5, 8)
    right = make_block("郭すらハツキリとはせず、顔はもちろん年齢や性別すら判然としない", 6, 1)
    result = merge_cross_page_sentences(UnifiedDocument(blocks=[left, right]))
    assert len(result.blocks) == 1
    assert result.blocks[0].text == "そのために玉座の主は、一見するとぼんやりと人の形をしているというだけで、その輪郭すらハツキリとはせず、顔はもちろん年齢や性別すら判然としない"
    assert result.blocks[0].metadata["source_pages"] == [5, 6]


def test_merge_sentence_across_pages():
    left = make_block("最初はごくごく小さな地響きであったのが、時間と共に次第に大きく、そしてその源", 6, 8)
    right = make_block("がドンドン近くなっていくことが感じられる。", 7, 1)
    result = merge_cross_page_sentences(UnifiedDocument(blocks=[left, right]))
    assert len(result.blocks) == 1
    assert result.blocks[0].text.endswith("そしてその源がドンドン近くなっていくことが感じられる。")


def test_do_not_merge_completed_sentence():
    result = merge_cross_page_sentences(UnifiedDocument(blocks=[make_block("彼は静かに立ち上がった。", 10, 5), make_block("翌朝、彼らは街を出発した。", 11, 1)]))
    assert len(result.blocks) == 2


def test_do_not_merge_into_chapter_title():
    result = merge_cross_page_sentences(UnifiedDocument(blocks=[make_block("そして俺は新たな生を受けた", 20, 7), make_block("第一章 魔王、幼年期とその終わり", 21, 1, BlockType.CHAPTER)]))
    assert len(result.blocks) == 2


def test_do_not_cross_page_merge_same_page():
    result = merge_cross_page_sentences(UnifiedDocument(blocks=[make_block("これはまだ終わっていない", 8, 2), make_block("ように見える文章だ。", 8, 3)]))
    assert len(result.blocks) == 2


def test_only_merge_last_and_first_blocks():
    result = merge_cross_page_sentences(UnifiedDocument(blocks=[make_block("ページ途中の文章", 3, 1), make_block("本当に続いている文章の前半", 3, 5), make_block("であり、これは後半である。", 4, 1)]))
    assert result.blocks[0].text == "ページ途中の文章"
    assert result.blocks[1].text == "本当に続いている文章の前半であり、これは後半である。"


def test_skip_when_page_metadata_missing():
    result = merge_cross_page_sentences(UnifiedDocument(blocks=[make_block("途中で終わっている文章", None), make_block("の続きに見える。", None)]))
    assert len(result.blocks) == 2
