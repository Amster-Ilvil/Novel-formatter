"""Persist GUI adjudication authority in a sealed multi-OCR text roundtrip.

The external text contract still allows changing only ``edited_text``.
Unedited ambiguous sentences must not silently turn into confirmed sentences
just because a 2,000-row JSON was exported and imported again.

Legacy packages without this extension retain the previous import behaviour.
"""
from __future__ import annotations

import hashlib
import re

_SCHEMA = 'novel_formatter.gui_roundtrip_authority.v1'


def _text_digest(value: str) -> str:
    return hashlib.sha256(str(value or '').encode('utf-8')).hexdigest()


def attach_gui_review_authority(package: dict, states) -> dict:
    """Add immutable row-wise source statuses and reseal the package."""
    rows = list(package.get('editable_items') or ())
    states = list(states or ())
    if len(rows) != len(states):
        raise ValueError('GUI 裁决状态与导出对齐行数不一致，拒绝导出。')
    manifest = []
    for index, (item, state) in enumerate(zip(rows, states)):
        if int(getattr(state, 'row_index', -1)) != index:
            raise ValueError(f'GUI 裁决状态第 {index + 1} 行身份错位，拒绝导出。')
        manifest.append({
            'row_id': str(item['row_id']),
            'exported_text_sha256': _text_digest(item.get('edited_text', '')),
            'was_unresolved': bool(getattr(state, 'unresolved', True)),
            'selection_origin': str(getattr(state, 'selection_origin', '') or '')[:96],
        })
    package['gui_review_authority'] = {'schema': _SCHEMA, 'rows': manifest}
    package['instructions']['gui_review_authority'] = (
        '原有待裁决行如果 edited_text 未变化，导回后仍为待裁决；'
        '只修改 editable_items[].edited_text，不要更改 gui_review_authority。'
        '保持原字但需要人工确认的行，仍须回 GUI 点击确认裁决。'
    )
    from engine.ocr_roundtrip_package import _seal_package
    return _seal_package(package)


def reapply_gui_review_authority(package: dict, states, lines) -> bool:
    """Restore confirmed/unresolved statuses; False means legacy package.

    Caller must first run the engine's complete sealed package import, which
    validates the immutable manifest and stable row identities before this step.
    """
    metadata = package.get('gui_review_authority')
    if metadata is None:
        return False
    if not isinstance(metadata, dict) or metadata.get('schema') != _SCHEMA:
        raise ValueError('GUI 裁决状态清单版本不受支持。')
    manifest = metadata.get('rows')
    items = package.get('editable_items') or ()
    if not isinstance(manifest, list) or len(manifest) != len(items) or len(states) != len(items) or len(lines) != len(items):
        raise ValueError('GUI 裁决清单与实际对齐行数不一致。')
    # Validate complete identity, source hashes and exact boolean types BEFORE
    # mutating a single fusion state: partial restores are forbidden.
    for index, (raw, item) in enumerate(zip(manifest, items)):
        if not isinstance(raw, dict) or str(raw.get('row_id')) != str(item.get('row_id')):
            raise ValueError(f'GUI 裁决清单第 {index + 1} 行 stable row ID 不匹配。')
        digest = raw.get('exported_text_sha256')
        if not isinstance(digest, str) or re.fullmatch(r'[0-9a-f]{64}', digest) is None:
            raise ValueError(f'GUI 裁决清单第 {index + 1} 行文本指纹不合法。')
        if type(raw.get('was_unresolved')) is not bool:
            raise ValueError(f'GUI 裁决清单第 {index + 1} 行缺少可靠的待裁决状态。')
        if not isinstance(raw.get('selection_origin'), str):
            raise ValueError(f'GUI 裁决清单第 {index + 1} 行裁决来源不合法。')
    from engine.ocr_compare_view_model import upsert_external_candidate
    for raw, item, state, text in zip(manifest, items, states, lines):
        value = str(text or '')
        unchanged = _text_digest(value) == raw['exported_text_sha256']
        if raw['was_unresolved'] and unchanged:
            state.selected_index = None
            state.selection_origin = ''
            state.requires_confirmation = True
            continue
        origin = (raw['selection_origin'] if unchanged and not raw['was_unresolved']
                  else 'external_ai_package')
        upsert_external_candidate(
            state, value,
            display_label='外部AI融合结果', select=True,
            allow_empty=item.get('delete_intentionally') is True,
            selection_origin=origin,
        )
    return True
