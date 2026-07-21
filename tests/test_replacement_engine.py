from engine.replacement_engine import replace_text
from models.document import Block, BlockType, Metadata, UnifiedDocument
from models.paragraph import Paragraph


def _document(*blocks: Block) -> UnifiedDocument:
    return UnifiedDocument(blocks=list(blocks), metadata=Metadata(source_engine="apple_vision"))


def test_preserve_ocr_layout_flattens_source_without_changing_block_structure():
    doc = _document(
        Block(BlockType.PARAGRAPH, "魔王 が 人 を 殺す", page=1),
        Block(BlockType.IMAGE_REF, image_path="page.png", page=1),
        Block(BlockType.PARAGRAPH, "当然 だ", page=2),
    )
    source = [
        Paragraph("魔王が人を\n　殺す", index=0, source="source"),
        Paragraph("当然だ", index=1, source="source"),
    ]

    result, report = replace_text(
        doc,
        source,
        match_threshold=0.0,
        preserve_ocr_layout=True,
    )

    assert report.replaced == 2
    assert [block.type for block in result.blocks] == [
        BlockType.PARAGRAPH,
        BlockType.IMAGE_REF,
        BlockType.PARAGRAPH,
    ]
    assert result.blocks[0].text == "魔王が人を 殺す"
    assert result.blocks[0].ocr_raw == "魔王 が 人 を 殺す"
    assert result.blocks[0].modified_by == "text_replacement_preserve_layout"
