# -*- coding: utf-8 -*-
"""Offline, read-only OCR model inventory for PR #12 adapted local builds.

Existing model installers (48px, NDL, Hayai, Paddle, OpenVINO) remain the
owners of installation and updates. This inspector centralizes *checks* for
pinned downloadable artifacts without causing network traffic or replacing any
runtime. Never infer integrity of an unpinned model from size alone.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path

ENV_MODELS_DIR = 'NOVEL_FORMATTER_MODELS_DIR'


class ModelState(str, Enum):
    READY = 'ready'
    MISSING = 'missing'
    CORRUPT = 'corrupt'


@dataclass(frozen=True)
class ModelArtifact:
    model_id: str
    relative_path: str
    sha256: str
    size: int = 0


MODEL_ARTIFACTS = {
    spec.model_id: spec for spec in (
        ModelArtifact('manga_48px.checkpoint', 'manga-48px-ar/ocr_ar_48px.ckpt',
                      '29daa46d080818bb4ab239a518a88338cbccff8f901bef8c9db191a7cb97671d', 204290192),
        ModelArtifact('manga_48px.dictionary', 'manga-48px-ar/alphabet-all-v7.txt',
                      'f5722368146aa0fbcc9f4726866e4efc3203318ebb66c811d8cbbe915576538a', 186651),
        ModelArtifact('openvino_handwriting.xml', 'handwriting_openvino/handwritten-japanese-recognition-0001.xml',
                      '12922031ae75163bc19c82e3ddc9dd8a5567ec50b2b62b72244dfc129ab820e4', 97226),
        ModelArtifact('openvino_handwriting.bin', 'handwriting_openvino/handwritten-japanese-recognition-0001.bin',
                      'afba3ba73b2d8e5a7a8f13107c77800fd3a96634cf37e84f1e0256e1b4b443a8', 17035726),
        ModelArtifact('openvino_handwriting.charlist', 'handwriting_openvino/kondate_nakayosi.txt',
                      '9851b52c412e2d8ae8a6ab3f9eaa69a53c7b3c9b87c2b5dd75f2ed8d7312ddf0', 17711),
    )
}


def default_models_dir() -> Path:
    configured = os.environ.get(ENV_MODELS_DIR, '').strip()
    if configured:
        return Path(configured).expanduser()
    # Do not migrate existing installations or silently create new paths.
    root = Path(__file__).resolve().parents[1]
    return root / '.model-cache'


def manga_48px_cache_dir() -> Path:
    legacy_override = os.environ.get('NOVEL_FORMATTER_CLOUD_48PX_CACHE', '').strip()
    if legacy_override:
        return Path(legacy_override).expanduser()
    return default_models_dir() / 'manga-48px-ar'


def _local_path(spec: ModelArtifact, models_dir: str | Path | None = None) -> Path:
    if models_dir is not None:
        return Path(models_dir).expanduser() / spec.relative_path
    if spec.model_id.startswith('manga_48px.'):
        return manga_48px_cache_dir() / Path(spec.relative_path).name
    configured = os.environ.get(ENV_MODELS_DIR, '').strip()
    if configured:
        return Path(configured).expanduser() / spec.relative_path
    return Path(__file__).resolve().parents[1] / 'models' / 'handwriting_openvino' / Path(spec.relative_path).name


def inspect_model(model_id: str, *, models_dir: str | Path | None = None, full_hash: bool = True) -> dict:
    spec = MODEL_ARTIFACTS[model_id]
    target = _local_path(spec, models_dir)
    result = {'model_id': model_id, 'path': str(target), 'expected_sha256': spec.sha256,
              'expected_size': spec.size}
    try:
        info = target.stat()
    except FileNotFoundError:
        return {**result, 'state': ModelState.MISSING.value}
    if not target.is_file() or spec.size and info.st_size != spec.size:
        return {**result, 'state': ModelState.CORRUPT.value, 'actual_size': info.st_size}
    if full_hash:
        sha = hashlib.sha256()
        with target.open('rb') as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                sha.update(chunk)
        actual_hash = sha.hexdigest()
        if actual_hash.lower() != spec.sha256.lower():
            return {**result, 'state': ModelState.CORRUPT.value, 'actual_sha256': actual_hash}
    return {**result, 'state': ModelState.READY.value, 'actual_size': info.st_size}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description='离线检查已下载 OCR 模型文件，不联网、不安装')
    parser.add_argument('--models-dir', help='可选的统一模型根目录；默认探测现有缓存')
    parser.add_argument('--quick', action='store_true', help='只检查文件大小，不计算完整 SHA-256（不能作为完整性验收）')
    args = parser.parse_args(argv)
    statuses = [inspect_model(key, models_dir=args.models_dir, full_hash=not args.quick) for key in MODEL_ARTIFACTS]
    print(json.dumps(statuses, ensure_ascii=False, indent=2))
    return 0 if all(item['state'] == ModelState.READY.value for item in statuses) else 2


if __name__ == '__main__':
    raise SystemExit(main())
