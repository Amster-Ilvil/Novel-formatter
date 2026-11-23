"""Optional Gemini Batch API for already locally segmented Japanese OCR crops.

Explicit submit/check/collect commands only; no autonomous background polling.
These results are remote *candidates*, never direct authoritative EPUB writes.
"""
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from urllib.request import Request
from urllib.error import HTTPError, URLError

from .http_transport import open_direct

from ai.config import validate_api_key
from ai.gemini_ocr_transport import DEFAULT_MODEL, OCR_SCHEMA, GeminiJsonTransport, response_text, safe_model, validate_items
from adapters.gemini_ocr_adapter import PROMPT, _contact_sheet, plan_column_sheets
from adapters.vertical_crop_segments import looks_like_full_page
from core.ocr_segment_cache import file_sha256

API = "https://generativelanguage.googleapis.com/v1beta/"


def prepare(paths: list[str], output: Path, *, model: str = DEFAULT_MODEL,
            batch_items: int = 20) -> dict:
    """Create local evidence sheets with hash-pinned crop->model ID mappings."""
    model = safe_model(model)
    batch_items = max(1, min(int(batch_items), 30))
    output.mkdir(parents=True, exist_ok=True)
    items: list[dict] = []
    if not all(Path(path).is_file() for path in paths):
        raise ValueError("Batch 输入必须是已存在的本地切列图片")
    if any(looks_like_full_page(path) for path in paths):
        raise ValueError("Batch 禁止裸扫整页：必须先通过本地布局检测与物理切列")
    for sheet in plan_column_sheets(paths, batch_items):
        image, ids = _contact_sheet(sheet, output / f"sheet-{len(items) + 1:04d}.webp")
        items.append({"key": f"group-{len(items) + 1:04d}", "sheet": image.name,
                      "sheet_sha256": file_sha256(image),
                      "crops": [{"id": row_id, "path": str(Path(p).resolve()),
                                 "sha256": file_sha256(p)} for row_id, p in zip(ids, sheet.paths, strict=True)]})
    manifest = {"schema": "novel-formatter-gemini-batch-v1", "model": model,
                "prompt": PROMPT, "items": items}
    (output / "batch_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf8")
    return manifest


def load_manifest(folder: Path) -> dict:
    value = json.loads((folder / "batch_manifest.json").read_text(encoding="utf8"))
    if value.get("schema") != "novel-formatter-gemini-batch-v1":
        raise ValueError("不受支持的 Gemini OCR Batch 清单")
    if not isinstance(value.get("items"), list):
        raise ValueError("Batch 项目清单无效")
    return value


def make_request_body(folder: Path, manifest: dict) -> dict:
    requests = []
    for item in manifest["items"]:
        image = folder / item["sheet"]
        if image.name != item["sheet"] or file_sha256(image) != item["sheet_sha256"]:
            raise ValueError("证据拼图丢失或 SHA256 已发生变化")
        for crop in item["crops"]:
            if file_sha256(crop["path"]) != crop["sha256"]:
                raise ValueError("原始切列图已经变化；禁止沿用旧 Batch 锚点")
        body = GeminiJsonTransport.make_body(manifest["prompt"], [image], schema=OCR_SCHEMA,
                                             max_output_tokens=max(1500, 1000 * len(item["crops"])))
        requests.append({"request": body, "metadata": {"key": item["key"]}})
    if not requests:
        raise ValueError("没有任何待识别切列")
    result = {"batch": {"display_name": "novel-formatter-anchored-ocr",
                        "input_config": {"requests": {"requests": requests}}}}
    payload_size = len(json.dumps(result).encode("utf8"))
    if payload_size >= 18 * 1024 * 1024:
        raise ValueError("单个 Inline Batch 超过 18 MiB；请使用 make_request_bodies 自动拆任务")
    return result


def make_request_bodies(folder: Path, manifest: dict) -> list[dict]:
    """Shard oversized books into independent inline Batch jobs, no API calls."""
    outputs: list[dict] = []
    group: list[dict] = []
    for item in manifest["items"]:
        partial = {**manifest, "items": [item]}
        single = make_request_body(folder, partial)["batch"]["input_config"]["requests"]["requests"][0]
        attempt = group + [single]
        payload = {"batch": {"display_name": "novel-formatter-anchored-ocr",
                             "input_config": {"requests": {"requests": attempt}}}}
        if len(json.dumps(payload).encode("utf8")) >= 18 * 1024 * 1024:
            if not group:
                raise ValueError("单张视觉拼图过大，请重新切成更小拼图")
            outputs.append({"batch": {"display_name": "novel-formatter-anchored-ocr",
                                      "input_config": {"requests": {"requests": group}}}})
            group = [single]
        else:
            group = attempt
    if group:
        outputs.append({"batch": {"display_name": "novel-formatter-anchored-ocr",
                                  "input_config": {"requests": {"requests": group}}}})
    if not outputs:
        raise ValueError("没有任何待识别切列")
    return outputs


def _api_call(path: str, *, api_key: str, payload: dict | None = None) -> dict:
    key = validate_api_key(api_key)
    headers = {"Content-Type": "application/json"}
    req = Request(API + path, headers=headers, method="POST" if payload is not None else "GET",
                  data=json.dumps(payload).encode("utf8") if payload is not None else None)
    req.add_unredirected_header("x-goog-api-key", key)
    try:
        with open_direct(req, timeout=120) as stream:
            return json.loads(stream.read().decode("utf8"))
    except HTTPError as exc:
        exc.close()
        raise RuntimeError(f"Gemini Batch API HTTP {exc.code}") from None
    except (URLError, TimeoutError):
        raise RuntimeError("Gemini Batch API 网络请求失败") from None


def submit(folder: Path, *, api_key: str, allow_paid_batch: bool = False) -> dict:
    if not allow_paid_batch:
        raise RuntimeError("Batch 提交须显式传入 --allow-paid-batch，以确认接受可能产生的费用。")
    manifest = load_manifest(folder)
    batches = make_request_bodies(folder, manifest)
    record = {"job_names": [], "model": manifest["model"],
              "sheet_count": len(manifest["items"]), "state": "submitting"}
    for body in batches:
        try:
            result = _api_call(f"models/{safe_model(manifest['model'])}:batchGenerateContent",
                               api_key=api_key, payload=body)
            name = str(result.get("name") or "")
            if not re.fullmatch(r"batches/[A-Za-z0-9_.-]+", name):
                raise RuntimeError("Batch API 没有返回有效任务 ID")
            record["job_names"].append(name)
            # Write after each submitted task to preserve IDs on partial failure.
            (folder / "batch_job.json").write_text(
                json.dumps(record, ensure_ascii=False, indent=2), encoding="utf8")
        except Exception:
            record["state"] = "submission_incomplete"
            (folder / "batch_job.json").write_text(
                json.dumps(record, ensure_ascii=False, indent=2), encoding="utf8")
            raise
    record["state"] = "submitted"
    (folder / "batch_job.json").write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf8")
    return record


def check(folder: Path, *, api_key: str) -> dict:
    record = json.loads((folder / "batch_job.json").read_text(encoding="utf8"))
    names = record.get("job_names") or [record.get("job_name")]
    if not names or any(not re.fullmatch(r"batches/[A-Za-z0-9_.-]+", str(name or "")) for name in names):
        raise ValueError("无效 Batch 任务 ID")
    responses = [_api_call(name, api_key=api_key) for name in names]
    if len(responses) == 1:
        return responses[0]
    return {"jobs": responses, "name": "multiple-batches", "job_count": len(responses)}


def collect(folder: Path, response: dict) -> dict:
    manifest = load_manifest(folder)
    responses = response.get("jobs") if isinstance(response.get("jobs"), list) else [response]
    outputs = []
    for job in responses:
        dest = job.get("dest") or job.get("destination") or {}
        outputs.extend(dest.get("inlinedResponses") or dest.get("inlined_responses") or [])
    index = {i["key"]: i for i in manifest["items"]}
    results = {}
    for response_item in outputs:
        key = str((response_item.get("metadata") or {}).get("key") or "")
        if key not in index or key in results:
            raise ValueError("Batch 返回缺少或重复任务 key")
        raw = response_text(response_item.get("response") or {})
        values = validate_items(raw, [row["id"] for row in index[key]["crops"]])
        results[key] = [{"crop_sha256": r["sha256"], "crop_path": r["path"],
                         "text": values[r["id"]], "source": "gemini_batch_unverified"}
                        for r in index[key]["crops"]]
    if set(results) != set(index):
        raise ValueError("Batch 部分任务未返回；拒绝整体导入")
    value = {"schema": "novel-formatter-gemini-batch-candidates-v1",
             "model": manifest["model"], "candidates": results}
    (folder / "unverified_candidates.json").write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf8")
    return value


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="mode", required=True)
    a = sub.add_parser("prepare", help="从本地切列图片建立批量证据，零网络调用")
    a.add_argument("--output", type=Path, required=True)
    a.add_argument("--model", default=DEFAULT_MODEL)
    a.add_argument("images", nargs="+")
    for name in ("submit", "check", "collect"):
        q = sub.add_parser(name)
        q.add_argument("--folder", type=Path, required=True)
    submit_parser = sub.choices["submit"]
    submit_parser.add_argument("--allow-paid-batch", action="store_true",
                               help="确认接受可能产生的 Batch API 费用")
    args = p.parse_args()
    if args.mode == "prepare":
        result = prepare(args.images, args.output, model=args.model)
        print("PREPARED_GROUPS", len(result["items"]))
        return
    key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or ""
    if args.mode == "submit":
        print(json.dumps(submit(args.folder, api_key=key, allow_paid_batch=args.allow_paid_batch), ensure_ascii=False))
        return
    result = check(args.folder, api_key=key)
    if args.mode == "check":
        jobs = result.get("jobs") if isinstance(result.get("jobs"), list) else [result]
        print(json.dumps({"jobs": [{"name": job.get("name"),
                                    "state": job.get("state") or (job.get("metadata") or {}).get("state")}
                                   for job in jobs]}, ensure_ascii=False))
    else:
        print("CANDIDATES", sum(len(x) for x in collect(args.folder, result)["candidates"].values()))


if __name__ == "__main__":
    main()
