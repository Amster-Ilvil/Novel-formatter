#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PaddleOCR 适配器（macOS，跑在独立 venv 里）

背景：
    PaddlePaddle 目前没有 Python 3.14 的预编译包，而本项目其余部分跑在系统
    默认的 3.14 上。所以这里不直接 import paddleocr，而是通过 subprocess
    调用 .venv-paddle（Python 3.13，见 adapters/paddle_ocr_worker.py）里的
    解释器，跨进程拿到逐行识别结果（含像素坐标），再在这一侧（有 PIL）
    转换成归一化 bbox，组装成 UnifiedDocument。

    页眉检测 / 页面自动分类 / 章节正则复用 apple_vision_adapter 里已有的实现，
    避免重复一份几乎一样的规则。PaddleOCR 的检测顺序不是可靠的阅读顺序
    （尤其竖排日文），所以这里给每个 Block 都写入 bbox——真正的阅读顺序
    由 Formatter 里的 reading_order 步骤（engine/reading_order.py）用坐标
    重新算，这里不需要、也不应该自己排。

依赖：
    .venv-paddle/（用 Python 3.13 创建，装了 paddlepaddle + paddleocr）
    首次运行会创建：
        /Library/Frameworks/Python.framework/Versions/3.13/bin/python3 -m venv .venv-paddle
        .venv-paddle/bin/pip install paddlepaddle "paddleocr>=3.7,<4"

用法（命令行，供单独测试）：
    python adapters/paddle_ocr_adapter.py /图片文件夹 output.json
"""

from __future__ import annotations

import sys
import os
import json
import subprocess
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from models.document import (
    UnifiedDocument, Block, BlockType, PageInfo, BoundingBox, Metadata, TocEntry
)
from adapters.apple_vision_adapter import (
    detect_running_headers, auto_classify_pages, CHAPTER_RE,
)
from adapters.runtime_env import ensure_venv
from adapters.paddle_ocr_models import (
    PADDLE_DETECTION_MODEL,
    PADDLE_RECOGNITION_MODEL,
    PADDLEOCR_PACKAGE_SPEC,
    PADDLE_RUNTIME_SIGNATURE,
    PADDLE_MODEL_SOURCE_LABELS,
    normalize_paddle_model_source,
    paddle_model_source_attempts,
    paddle_source_environment,
    paddleocr_version_marker,
)

VENV_DIR = Path(__file__).parent.parent / ".venv-paddle"
VENV_PYTHON = VENV_DIR / "bin" / "python"
WORKER_SCRIPT = Path(__file__).parent / "paddle_ocr_worker.py"


def _venv_ready() -> bool:
    return VENV_PYTHON.exists()


def _has_structure_deps() -> bool:
    """paddlex[ocr] 额外依赖（PP-StructureV3 / PaddleOCRVL 都需要）是否已装"""
    result = subprocess.run(
        [str(VENV_PYTHON), "-c", "import openpyxl"],
        capture_output=True,
        timeout=30,
    )
    return result.returncode == 0


def setup_venv(verbose: bool = True, pipeline: str = "ocr") -> None:
    """创建/修复独立 PaddleOCR 环境，并自动跳过损坏的 Python 路径。"""
    ensure_venv(
        VENV_DIR,
        label="PaddleOCR",
        marker_code=paddleocr_version_marker(),
        packages=["paddlepaddle", PADDLEOCR_PACKAGE_SPEC],
        verbose=verbose,
    )

    # PP-StructureV3 / PaddleOCRVL need paddlex[ocr] extras. Install lazily.
    if pipeline in ("structure", "vl") and not _has_structure_deps():
        if verbose:
            print(f"📦  {pipeline} 模型需要额外依赖，首次使用自动安装 ...")
        subprocess.run(
            [str(VENV_PYTHON), "-m", "pip", "install", "paddlex[ocr]"],
            check=True,
            timeout=3600,
        )


def _page_size(image_path: str) -> tuple[int, int]:
    from PIL import Image
    with Image.open(image_path) as img:
        return img.size  # (w, h)


def _worker_command(
    image_paths: list[str],
    *,
    lang: str,
    pipeline: str,
    probe: bool = False,
    vl_runtime: dict | None = None,
) -> list[str]:
    cmd = [str(VENV_PYTHON), str(WORKER_SCRIPT), "--lang", lang, "--pipeline", pipeline]
    if pipeline == "vl":
        runtime = dict(vl_runtime or {})
        cmd.extend([
            "--vl-backend", "mlx" if runtime.get("backend") == "mlx" else "paddle",
            "--vl-server-url", str(runtime.get("server_url") or ""),
            "--vl-api-model-name", str(
                runtime.get("model") or "PaddlePaddle/PaddleOCR-VL-1.6"
            ),
        ])
    if probe:
        cmd.append("--probe")
    cmd.extend(image_paths)
    return cmd


def _prepare_vl_runtime(
    pipeline: str,
    vl_backend: str,
    *,
    verbose: bool = False,
    progress_callback=None,
) -> dict:
    if pipeline != "vl":
        return {}
    from adapters.paddle_vl_mlx import prepare_vl_backend
    return prepare_vl_backend(
        vl_backend,
        verbose=verbose,
        progress_callback=progress_callback,
    )


def _run_worker(
    image_paths: list[str],
    lang: str,
    pipeline: str = "ocr",
    cancel_check=None,
    model_source: str = "auto",
    status_callback=None,
    vl_backend: str = "auto",
):
    """Run PaddleOCR in a child process without pipe deadlocks or infinite waits."""
    from adapters.subprocess_watchdog import (
        LinePump, ProcessCancelled, env_seconds, isolated_process_kwargs,
        terminate_process,
    )

    failures: list[str] = []
    startup_timeout = env_seconds("NOVEL_FORMATTER_OCR_STARTUP_TIMEOUT", 900.0, minimum=60.0)
    request_timeout = env_seconds("NOVEL_FORMATTER_OCR_REQUEST_TIMEOUT", 300.0, minimum=30.0)
    vl_runtime = _prepare_vl_runtime(
        pipeline,
        vl_backend,
        verbose=False,
        progress_callback=status_callback,
    )
    for attempt_index, source in enumerate(paddle_model_source_attempts(model_source), start=1):
        label = PADDLE_MODEL_SOURCE_LABELS.get(source, source)
        if status_callback is not None:
            status_callback(f"PaddleOCR 模型源：{label}（第 {attempt_index} 次尝试）")
        proc = subprocess.Popen(
            _worker_command(
                image_paths, lang=lang, pipeline=pipeline, vl_runtime=vl_runtime
            ),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            env=paddle_source_environment(source),
            **isolated_process_kwargs(),
        )
        stdout_pump = LinePump(proc.stdout, name="paddle-ocr-stdout")
        stderr_pump = LinePump(proc.stderr, name="paddle-ocr-stderr")
        emitted_payload = False
        diagnostics: list[str] = []
        cancelled = False

        def add_diagnostic(value: str, *, report: bool = True) -> None:
            text = str(value or "").strip()
            if not text:
                return
            diagnostics.append(text)
            if len(diagnostics) > 400:
                del diagnostics[:200]
            if report and status_callback is not None:
                status_callback(text[-500:])

        def drain_stderr() -> None:
            for item in stderr_pump.get_nowait_lines():
                add_diagnostic(item)

        try:
            while True:
                timeout = request_timeout if emitted_payload else startup_timeout
                try:
                    line = stdout_pump.readline(
                        proc=proc,
                        timeout=timeout,
                        cancel_check=cancel_check,
                        label="PaddleOCR",
                        on_wait=drain_stderr,
                    )
                except ProcessCancelled:
                    cancelled = True
                    break
                if line is None:
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    add_diagnostic(line)
                    continue
                emitted_payload = True
                if data.get("ok"):
                    yield data.get("path", ""), data.get("blocks") or [], None
                else:
                    yield data.get("path", ""), None, data.get("error", "未知错误")
        finally:
            drain_stderr()
            if cancelled:
                terminate_process(proc)
            try:
                ret = proc.wait(timeout=20)
            except TypeError:
                ret = proc.wait()
            except Exception:
                ret = terminate_process(proc)
            stdout_pump.close()
            stderr_pump.close()

        if cancelled or ret == -15:
            return
        if ret == 0:
            return

        combined = "\n".join(diagnostics[-40:]).strip()
        failures.append(f"{label}: {combined or f'worker code={ret}'}")
        # Once page-level output has started, retrying would duplicate earlier
        # results. Source fallback is therefore limited to model startup.
        if emitted_payload:
            raise RuntimeError(
                f"PaddleOCR worker 在输出部分页面后异常退出 (code={ret})：\n{combined[-4000:]}"
            )

    detail = "\n\n".join(failures[-4:])
    raise RuntimeError(
        "PaddleOCR 模型下载/初始化失败。程序已经按设置尝试可用模型源。\n"
        "可在 OCR → 引擎 → Paddle 模型源中手动选择 ModelScope 或百度 BOS 后重试。\n\n"
        + detail[-8000:]
    )

def prepare_runtime(
    *,
    pipeline: str = "ocr",
    lang: str = "japan",
    model_source: str = "auto",
    vl_backend: str = "auto",
    verbose: bool = True,
    progress_callback=None,
) -> dict:
    """Install dependencies and initialise/download the selected Paddle model."""
    from adapters.subprocess_watchdog import (
        LinePump, env_seconds, isolated_process_kwargs, terminate_process,
    )

    if pipeline not in {"ocr", "structure", "vl"}:
        pipeline = "ocr"
    setup_venv(verbose=verbose, pipeline=pipeline)
    vl_runtime = _prepare_vl_runtime(
        pipeline,
        vl_backend,
        verbose=verbose,
        progress_callback=progress_callback,
    )
    failures: list[str] = []
    startup_timeout = env_seconds("NOVEL_FORMATTER_OCR_STARTUP_TIMEOUT", 900.0, minimum=60.0)
    for attempt_index, source in enumerate(paddle_model_source_attempts(model_source), start=1):
        label = PADDLE_MODEL_SOURCE_LABELS.get(source, source)
        if progress_callback is not None:
            progress_callback(f"正在通过 {label} 准备 PaddleOCR（尝试 {attempt_index}）…")
        proc = subprocess.Popen(
            _worker_command(
                [], lang=lang, pipeline=pipeline, probe=True, vl_runtime=vl_runtime
            ),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=paddle_source_environment(source),
            **isolated_process_kwargs(),
        )
        output_pump = LinePump(proc.stdout, name="paddle-prepare-output")
        payload = None
        output_lines: list[str] = []
        timed_out = None
        try:
            while True:
                try:
                    raw = output_pump.readline(
                        proc=proc,
                        timeout=startup_timeout,
                        label="PaddleOCR 模型初始化",
                    )
                except Exception as exc:
                    timed_out = exc
                    break
                if raw is None:
                    break
                line = raw.strip()
                if not line:
                    continue
                output_lines.append(line)
                if len(output_lines) > 500:
                    del output_lines[:250]
                try:
                    candidate = json.loads(line)
                except json.JSONDecodeError:
                    candidate = None
                if isinstance(candidate, dict) and candidate.get("probe"):
                    payload = candidate
                elif progress_callback is not None:
                    progress_callback(line[-500:])
        finally:
            try:
                ret = proc.wait(timeout=20)
            except TypeError:
                ret = proc.wait()
            except Exception:
                ret = terminate_process(proc)
            output_pump.close()

        if timed_out is not None:
            output_lines.append(str(timed_out))
        if ret == 0 and isinstance(payload, dict) and payload.get("ok"):
            component_id = {
                "ocr": "paddle_ocr",
                "structure": "paddle_structure",
                "vl": "paddle_vl",
            }.get(pipeline, "paddle_ocr")
            from adapters.ocr_runtime_catalog import mark_runtime_ready
            mark_runtime_ready(
                component_id,
                runtime_signature=PADDLE_RUNTIME_SIGNATURE,
                detection_model=PADDLE_DETECTION_MODEL,
                recognition_model=PADDLE_RECOGNITION_MODEL,
                model_source=source,
                model_profile=str(payload.get("model_profile") or ""),
                vl_backend=str(payload.get("vl_backend") or vl_runtime.get("backend") or ""),
                vl_requested_backend=str(vl_runtime.get("requested") or ""),
            )
            if progress_callback is not None:
                progress_callback(f"PaddleOCR 已就绪：{label}")
            return {
                **payload,
                "model_source": source,
                "vl_backend": str(payload.get("vl_backend") or vl_runtime.get("backend") or ""),
                "vl_backend_detail": str(vl_runtime.get("detail") or ""),
            }
        failures.append(f"{label}: " + "\n".join(output_lines[-40:])[-6000:])

    raise RuntimeError(
        "PaddleOCR 环境已安装，但模型仍无法下载或初始化。\n\n"
        + "\n\n".join(failures[-4:])[-10000:]
    )

def run(
    image_folder: str | None = None,
    page_overrides: dict[int, str] | None = None,
    lang: str = "japan",
    pipeline: str = "ocr",
    verbose: bool = True,
    input_paths: list[str] | None = None,
    progress_callback=None,
    cancel_check=None,
    crop_top: float = 0.0,
    crop_bottom: float = 0.0,
    crop_rect: tuple[float, float, float, float] | None = None,
    temp_crop_dir: str | None = None,
    reuse_existing_crops: bool = False,
    filter_running_headers: bool = True,
    model_source: str = "auto",
    vl_backend: str = "auto",
    ocr_mode: str = "ja_vertical",
    merge_horizontal_fragments: bool = True,
) -> UnifiedDocument:
    """
    核心函数：对输入执行 PaddleOCR，返回 UnifiedDocument。
    参数与 apple_vision_adapter.run() 保持一致（含 crop_rect 手动框选区域），
    方便 GUI 按适配器名切换调用。

    pipeline: "ocr"（默认，PP-OCR 纯文字识别，最快）/
              "structure"（PP-StructureV3，多一步文档方向矫正/版面分析，
              需要 pip install "paddlex[ocr]"）/
              "vl"（PaddleOCR-VL，视觉语言模型，识别质量通常更好但模型体积
              是前两者的好几倍，首次使用会下载数 GB 模型文件）。
    """
    from adapters.ocr_profiles import get_ocr_profile, normalize_ocr_mode
    ocr_mode = normalize_ocr_mode(ocr_mode)
    profile = get_ocr_profile(ocr_mode)
    # The profile is authoritative. A stale Japanese ``lang`` value from an
    # older project/session must not leak into Simplified-Chinese recognition.
    if not profile.vertical:
        lang = profile.paddle_lang
    elif not str(lang or "").strip():
        lang = profile.paddle_lang

    setup_venv(verbose=verbose, pipeline=pipeline)

    from adapters.ocr_engine_common import run_ocr_engine

    def worker_fn(ocr_paths, cancel_check):
        return _run_worker(
            ocr_paths,
            lang=lang,
            pipeline=pipeline,
            cancel_check=cancel_check,
            model_source=model_source,
            status_callback=progress_callback,
            vl_backend=vl_backend,
        )

    doc = run_ocr_engine(
        worker_fn,
        source_engine="paddle_ocr",
        image_folder=image_folder,
        page_overrides=page_overrides,
        verbose=verbose,
        input_paths=input_paths,
        progress_callback=progress_callback,
        cancel_check=cancel_check,
        crop_top=crop_top,
        crop_bottom=crop_bottom,
        crop_rect=crop_rect,
        temp_crop_dir=temp_crop_dir,
        reuse_existing_crops=reuse_existing_crops,
        filter_running_headers=filter_running_headers,
        ocr_mode=ocr_mode,
        merge_horizontal_fragments=merge_horizontal_fragments,
    )
    from adapters.ocr_runtime_catalog import mark_runtime_ready
    component_id = {"ocr": "paddle_ocr", "structure": "paddle_structure", "vl": "paddle_vl"}.get(
        pipeline, "paddle_ocr"
    )
    details = {}
    if component_id in {"paddle_ocr", "paddle_structure"}:
        details = {
            "runtime_signature": PADDLE_RUNTIME_SIGNATURE,
            "detection_model": PADDLE_DETECTION_MODEL,
            "recognition_model": PADDLE_RECOGNITION_MODEL,
            "model_source": normalize_paddle_model_source(model_source),
        }
    elif component_id == "paddle_vl":
        from adapters.paddle_vl_mlx import normalize_vl_backend
        details = {
            "pipeline_version": "v1.6",
            "vl_requested_backend": normalize_vl_backend(vl_backend),
        }
    mark_runtime_ready(component_id, **details)
    return doc


def main():
    parser = argparse.ArgumentParser(description="PaddleOCR 适配器 CLI")
    parser.add_argument("input", help="图片文件夹或文件路径")
    parser.add_argument("output", help="输出 JSON 路径")
    parser.add_argument("--lang", default="japan")
    args = parser.parse_args()

    doc = run(image_folder=args.input, lang=args.lang)
    Path(args.output).write_text(doc.to_json(), encoding="utf-8")
    print(f"已写入 {args.output}")


if __name__ == "__main__":
    main()
