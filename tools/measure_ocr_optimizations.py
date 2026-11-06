#!/usr/bin/env python3
"""Scoped, offline before/after measurements. Never removes an output directory."""
from __future__ import annotations

import argparse
import cProfile
import csv
import dataclasses
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import platform
import pstats
import shlex
import statistics
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).absolute().parent.parent
SCENARIOS = ("components", "columns", "fake_column_pipeline", "real_session")
OFFLINE = {
    "NOVEL_FORMATTER_MODEL_AUTO_DOWNLOAD": "0",
    "NOVEL_FORMATTER_OCR_OFFLINE": "1",
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "HF_HUB_DISABLE_TELEMETRY": "1",
    "NOVEL_FORMATTER_MODEL_CACHE_SIZE": "2",
    "NOVEL_FORMATTER_NDLOCR_PROVIDER": "cpu",
    "NOVEL_FORMATTER_OCR_STARTUP_TIMEOUT": "120",
    "PYTHONHASHSEED": "0",
}


def digest(value):
    data = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def create_output(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=False)
    return path


def distribution(values):
    if not values:
        raise ValueError("No timing samples")
    median = statistics.median(values)
    return {
        "n": len(values), "median_s": median, "min_s": min(values), "max_s": max(values),
        "mad_s": statistics.median(abs(value - median) for value in values),
    }


def real_signature(rows):
    if not rows:
        raise ValueError("Real OCR returned no image results")
    result = []
    for path, (blocks, error) in sorted(rows.items()):
        if error or blocks is None:
            raise ValueError(f"Real OCR failed for {Path(path).name}: {error}")
        result.append({"file": Path(path).name, "blocks": blocks})
    if not any(item["blocks"] for item in result):
        raise ValueError("Real OCR returned no blocks on the text fixture")
    return result


def redact(text, replacements):
    def original_length(pair):
        original, _replacement = pair
        return len(str(original))
    for original, replacement in sorted(replacements, key=original_length, reverse=True):
        text = text.replace(str(original), replacement)
    return text


def child(args):
    repo = Path(args.repo).resolve()
    os.chdir(repo)
    sys.path.insert(0, str(repo))
    started = time.perf_counter()
    from PIL import Image, __version__ as pillow_version
    import adapters.column_ocr_adapter as column
    bench = load_file("measurement_bench_helpers", ROOT / "tools/bench_pipeline.py")
    imports_s = time.perf_counter() - started
    inputs = Path(args.inputs).resolve()
    fixture = load_file("measurement_equivalence", ROOT / "tests/test_performance_equivalence.py")

    if args.scenario == "equivalence":
        segmentation = fixture.segmentation_signature()
        images = fixture.column_image_signature()
        expected_seg = json.loads(fixture.REFERENCE.read_text())
        expected_images = json.loads(fixture.COLUMN_REFERENCE.read_text())
        return {
            "components_equal_reference": segmentation["components"] == expected_seg["components"],
            "columns_equal_reference": segmentation["columns"] == expected_seg["columns"],
            "column_images_equal_reference": images == expected_images,
            "components_sha256": digest(segmentation["components"]),
            "columns_sha256": digest(segmentation["columns"]),
            "column_images_sha256": digest(images),
            "components": len(segmentation["components"]),
            "columns": len(segmentation["columns"]),
            "column_images": len(images["images"]),
        }

    with Image.open(inputs / "segmentation.png") as opened:
        page = opened.convert("RGB")
    mask = fixture.ink_mask(page.convert("L"))
    restore = None
    if args.scenario == "components":
        def operation():
            return column._scanline_connected_ink_components(mask), {}
        def signature(value):
            return [dataclasses.asdict(item) for item in value]
    elif args.scenario == "columns":
        def operation():
            return column.detect_vertical_columns(page), {}
        def signature(value):
            return [dataclasses.asdict(item) for item in value]
    elif args.scenario == "fake_column_pipeline":
        restore = bench.install_stub_recognizer()
        def operation():
            return column.run(
                recognition_engine="apple_vision",
                input_paths=[str(inputs / "segmentation.png")], verbose=False,
            ), {}
        signature = bench.document_signature
    elif args.scenario == "real_session":
        def operation():
            session = column._RecognizerSession(
                "ndlocr_lite", shortcut_name="ExtractText", cancel_check=None,
                verbose=False, temp=Path(os.environ["TMPDIR"]),
                engine_options={"provider": "cpu"},
            )
            t0 = time.perf_counter()
            session.__enter__()
            t1 = time.perf_counter()
            try:
                worker = session._persistent
                rows = session.recognize_blocks([str(inputs / "ocr-text.png")])
                t2 = time.perf_counter()
                details = {
                    "setup_s": t1 - t0, "recognition_s": t2 - t1,
                    "worker_pid": worker.process.pid, "provider": worker.provider_info,
                }
            finally:
                session.__exit__(*sys.exc_info())
            details["teardown_s"] = time.perf_counter() - t2
            return rows, details
        signature = real_signature
    else:
        raise ValueError(args.scenario)

    def run_once():
        start = time.perf_counter()
        value, details = operation()
        elapsed = time.perf_counter() - start
        normalized = signature(value)
        return {
            "total_s": elapsed, "signature_sha256": digest(normalized),
            "details": details,
        }, normalized

    try:
        if args.profile:
            profiler = cProfile.Profile()
            profiler.enable()
            value, details = operation()
            profiler.disable()
            normalized = signature(value)
            stream = io.StringIO()
            pstats.Stats(profiler, stream=stream).strip_dirs().sort_stats("cumulative").print_stats(30)
            return {
                "profile_top30": stream.getvalue(), "signature_sha256": digest(normalized),
                "details": details,
                "scope": "Parent main thread only; worker inference and preparation threads are not profiled.",
            }
        cold, _ = run_once()
        warmup, _ = run_once()
        warm, _ = run_once()
        verification_value, _ = operation()
        verification = signature(verification_value)
        hashes = {
            cold["signature_sha256"], warmup["signature_sha256"],
            warm["signature_sha256"], digest(verification),
        }
        if len(hashes) != 1:
            raise ValueError("Output changed across cold/warmup/warm/untimed verification")
        sources = {}
        for module in list(sys.modules.values()):
            filename = getattr(module, "__file__", None)
            if filename:
                path = Path(filename).resolve()
                if path.suffix == ".py" and path.is_relative_to(repo):
                    sources[path.relative_to(repo).as_posix()] = file_sha(path)
        result = {
            "cold": cold, "warmup": warmup, "warm": warm,
            "verification_sha256": digest(verification),
            "verification_output": verification,
            "imports_s": imports_s, "python": platform.python_version(),
            "pillow": pillow_version, "sources": sources,
            "prepare_workers": bench._prepare_workers(repo),
        }
        if args.scenario == "real_session":
            pids = [item["details"]["worker_pid"] for item in (cold, warmup, warm)]
            result["worker_pids"] = pids
            result["session_reused"] = len(set(pids)) == 1
        return result
    finally:
        if restore:
            restore()
        page.close()
        mask.close()
        cache = sys.modules.get("adapters.recognizer_session_cache")
        if cache:
            cache.invalidate_recognizer_sessions()
        from utils.session_temp import cleanup_session_temp
        cleanup_session_temp()


def summarize(samples):
    summary = []
    scenarios = sorted({sample["scenario"] for sample in samples})
    for scenario in scenarios:
        relevant = [sample for sample in samples if sample["scenario"] == scenario]
        hashes = {sample["result"]["verification_sha256"] for sample in relevant}
        equivalent = len(hashes) == 1
        for phase in ("cold", "warm"):
            groups = {}
            for variant in ("baseline", "current"):
                values = [
                    sample["result"][phase]["total_s"] for sample in relevant
                    if sample["variant"] == variant
                ]
                groups[variant] = distribution(values)
            baseline, current = groups["baseline"], groups["current"]
            summary.append({
                "scenario": scenario, "phase": phase, **groups,
                "equivalent": equivalent,
                "baseline_over_current": (
                    baseline["median_s"] / current["median_s"]
                    if equivalent and current["median_s"] > 0 else None
                ),
                "ranges_overlap": max(baseline["min_s"], current["min_s"])
                                  <= min(baseline["max_s"], current["max_s"]),
            })
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline")
    parser.add_argument("--inputs", required=True)
    parser.add_argument("--out")
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--runtime-python")
    parser.add_argument("--runtime-source")
    parser.add_argument("--scenarios", nargs="+", choices=SCENARIOS, default=list(SCENARIOS))
    parser.add_argument("--child", action="store_true")
    parser.add_argument("--repo")
    parser.add_argument("--scenario", choices=(*SCENARIOS, "equivalence"))
    parser.add_argument("--child-output")
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()
    if args.child:
        result = child(args)
        Path(args.child_output).write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
        return 0
    if args.runs < 5:
        parser.error("At least five measured repetitions are required")
    if not args.baseline or not args.out:
        parser.error("--baseline and --out are required")
    if "real_session" in args.scenarios and not (args.runtime_python and args.runtime_source):
        parser.error("Real OCR requires explicit local --runtime-python and --runtime-source")
    inputs = Path(args.inputs).resolve()
    for name in ("segmentation.png", "ocr-text.png"):
        if not (inputs / name).is_file():
            parser.error(f"Missing input: {name}")
    out = create_output(args.out).resolve()
    baseline = Path(args.baseline).resolve()
    samples, profiles, reference_checks, sources = [], {}, {}, {}
    report = {
        "environment": {
            "python": platform.python_version(), "platform": platform.platform(),
            "machine": platform.machine(), "cpu_count": os.cpu_count(), "controls": OFFLINE,
        },
        "inputs": {name: file_sha(inputs / name) for name in ("segmentation.png", "ocr-text.png")},
        "runs": args.runs, "samples": samples, "profiles": profiles,
        "reference_checks": reference_checks, "sources": sources,
        "method": {
            "cold": "First operation in a fresh parent process; imports measured separately; OS file cache not flushed.",
            "warm": "Third operation in the same process after one recorded, excluded warmup.",
            "order": "Paired baseline/current, reversed on odd repetition indices; sequential, never concurrent.",
            "real_scope": "Recognizer-session acquire/start, full-page NDLOCR recognition, session release; no PDF, column cropping, formatter or export.",
            "fake_scope": "Single PNG through column isolation, fake recognizer and document assembly; no real inference, PDF, formatter or export.",
            "equivalence": "Signatures computed outside timers; additional untimed operation; historical reference checks in separate processes.",
        },
    }
    for variant, repo in (("baseline", baseline), ("current", ROOT)):
        report[variant + "_head"] = subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    with tempfile.TemporaryDirectory(prefix="nf-measure-") as temp:
        scratch = Path(temp)
        replacements = [
            (str(ROOT), "<workspace>"), (str(inputs), "<inputs>"),
            (str(baseline), "<baseline>"), (str(scratch), "<scratch>"),
            (str(Path.home()), "<user-home>"),
        ]
        env = dict(os.environ)
        env.pop("PYTHONPATH", None)
        env.update(OFFLINE)
        if args.runtime_python:
            launcher = scratch / "runtime-python"
            launcher.write_text("#!/bin/sh\nexec " + shlex.quote(str(Path(args.runtime_python).absolute())) + ' "$@"\n')
            launcher.chmod(0o700)
            env["NOVEL_FORMATTER_NDLOCR_PYTHON"] = str(launcher)
            env["NOVEL_FORMATTER_NDLOCR_SOURCE_DIR"] = str(Path(args.runtime_source).resolve())
        def persist():
            text = json.dumps(report, ensure_ascii=False, indent=2)
            (out / "measurements.json").write_text(redact(text, replacements) + "\n")
        def launch(variant, scenario, tag, profile=False):
            work = scratch / tag
            work.mkdir()
            home, tmp = work / "home", work / "tmp"
            home.mkdir()
            tmp.mkdir()
            child_env = dict(env, HOME=str(home), TMPDIR=str(tmp),
                             NOVEL_FORMATTER_OCR_RUNTIME_HOME=str(work / "runtime"))
            repo = baseline if variant == "baseline" else ROOT
            target = work / "result.json"
            command = [
                sys.executable, str(Path(__file__).absolute()), "--child",
                "--repo", str(repo), "--inputs", str(inputs), "--scenario", scenario,
                "--child-output", str(target),
            ]
            if profile:
                command.append("--profile")
            proc = subprocess.run(command, cwd=repo, env=child_env,
                                  capture_output=True, text=True, timeout=300)
            if proc.returncode:
                report["failure"] = {
                    "tag": tag, "returncode": proc.returncode,
                    "stdout": proc.stdout, "stderr": proc.stderr,
                }
                persist()
                raise RuntimeError(f"{tag} failed; see measurements.json")
            return json.loads(target.read_text())
        try:
            for variant in ("baseline", "current"):
                reference_checks[variant] = launch(variant, "equivalence", variant + "-equivalence")
                persist()
            for scenario in args.scenarios:
                for repeat in range(args.runs):
                    order = ("baseline", "current") if repeat % 2 == 0 else ("current", "baseline")
                    for variant in order:
                        tag = f"{scenario}-{repeat}-{variant}"
                        result = launch(variant, scenario, tag)
                        key = variant + ":" + scenario
                        current_sources = result.pop("sources")
                        if key in sources and sources[key] != current_sources:
                            raise RuntimeError("Measured source files changed between repetitions")
                        sources[key] = current_sources
                        output = result.pop("verification_output")
                        artifact = out / f"{scenario}-{variant}-output.json"
                        if not artifact.exists():
                            artifact.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n")
                        samples.append({"scenario": scenario, "variant": variant, "repeat": repeat, "result": result})
                        persist()
                        print(tag, "cold", result["cold"]["total_s"], "warm", result["warm"]["total_s"], flush=True)
                for variant in ("baseline", "current"):
                    profiles[variant + ":" + scenario] = launch(variant, scenario, f"profile-{scenario}-{variant}", True)
                    persist()
            report["summary"] = summarize(samples)
            report["harness_sha256"] = file_sha(Path(__file__))
            persist()
        except BaseException as exc:
            report.setdefault("failure", {"type": type(exc).__name__, "message": str(exc)})
            persist()
            raise
    with (out / "timings.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "scenario", "variant", "repeat", "phase", "total_s", "setup_s", "recognition_s", "teardown_s",
        ])
        writer.writeheader()
        for sample in samples:
            for phase in ("cold", "warmup", "warm"):
                value = sample["result"][phase]
                row = {k: sample[k] for k in ("scenario", "variant", "repeat")}
                row.update(phase=phase, total_s=value["total_s"])
                row.update({k: v for k, v in value["details"].items() if k.endswith("_s")})
                writer.writerow(row)
    text = "\n\n".join(f"## {key}\n\n{value['scope']}\n\n```text\n{value['profile_top30']}\n```" for key, value in profiles.items())
    (out / "profiles.md").write_text(text + "\n")
    print(json.dumps(report["summary"], indent=2))
    if not all(row["equivalent"] for row in report["summary"]):
        return 1
    if not all(value for checks in reference_checks.values() for key, value in checks.items() if key.endswith("_reference")):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
