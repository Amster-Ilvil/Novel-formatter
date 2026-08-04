from __future__ import annotations
import hashlib, json, re, shutil
from pathlib import Path

from utils.atomic_io import atomic_write_json


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def checkpoint_book_identity(source_path: str, doc=None) -> tuple[str, str]:
    p = Path(source_path).expanduser()
    if source_path and p.exists() and p.is_file():
        content_hash = _file_sha256(p)
        payload = f"v2\0{p.resolve()}\0{p.stat().st_size}\0{content_hash}"
        return hashlib.sha256(payload.encode('utf-8')).hexdigest(), content_hash
    text = '\n'.join(str(getattr(b, 'text', '') or '') for b in getattr(doc, 'blocks', []) or [])
    content_hash = hashlib.sha256(text.encode('utf-8')).hexdigest()
    return hashlib.sha256((f"v2-doc\0{content_hash}").encode()).hexdigest(), content_hash


def safe_stem(value: str) -> str:
    return re.sub(r'[^\w.-]+', '_', value, flags=re.UNICODE).strip('._')[:64] or 'untitled'


def migrate_legacy_roots(base: Path, source_stem: str, target: Path) -> list[str]:
    migrated=[]
    if not base.exists(): return migrated
    prefixes=(source_stem+'_', safe_stem(source_stem)+'_')
    for old in sorted(base.iterdir()):
        if not old.is_dir() or old == target or not old.name.startswith(prefixes):
            continue
        for phase in ('correction','layout','correction_layout'):
            src=old/phase
            if src.exists():
                shutil.copytree(src, target/phase, dirs_exist_ok=True)
                migrated.append(str(old))
    return sorted(set(migrated))


def prepare_checkpoint_root(source_path: str, doc=None, override: str='') -> Path:
    identity, content_hash = checkpoint_book_identity(source_path, doc)
    source = Path(source_path).expanduser() if source_path else None
    if override:
        root=Path(override).expanduser()
    elif source and source.exists():
        root=source.parent/'.novel_formatter_ai'/f"{safe_stem(source.stem)}_{identity[:24]}"
    else:
        root=Path.home()/'.novel_formatter'/'ai_checkpoints'/identity[:24]
    root.mkdir(parents=True, exist_ok=True)
    migrated=[]
    if source and source.exists() and not override:
        migrated=migrate_legacy_roots(source.parent/'.novel_formatter_ai', source.stem, root)
    meta={'version':2,'source_path':str(source.resolve()) if source and source.exists() else source_path,
          'source_sha256':content_hash,'stable_identity':identity,'migrated_from':migrated}
    atomic_write_json(root/'checkpoint_book.json', meta, ensure_ascii=False, indent=2)
    return root
