"""Safe, low-coupling GitHub source updater for Novel Formatter.

Design goals:
- never check or update automatically;
- update only from the locked official repository/main branch;
- real Git checkouts are clean-worktree + fast-forward only;
- portable ZIP builds use a staged, validated, transactional source swap;
- virtualenvs, model caches, OCR runtimes, user books, outputs and logs are outside
  the managed source surface and are never deleted by this module.

The module intentionally has no Qt/OCR/AI imports so an updater failure cannot
change any business workflow.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import ast
import base64
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
from typing import Callable
from urllib.parse import quote, urlencode, urlparse
import urllib.request
import zipfile

DEFAULT_REPOSITORY = "Amster-Ilvil/Novel-formatter"
DEFAULT_BRANCH = "main"
API_BASE = "https://api.github.com/repos"
STATE_RELATIVE = Path(".runtime") / "source_update_state.json"
MAX_ARCHIVE_FILES = 120_000
MAX_ARCHIVE_BYTES = 2 * 1024**3
MAX_DOWNLOAD_BYTES = 768 * 1024**2

# Only source/release-controlled paths. Runtime/user data deliberately does not
# appear here. Directories are replaced as units so removed upstream modules do
# not linger and create mixed-version imports.
MANAGED_PATHS = (
    "adapters", "ai", "assets", "builder", "core", "docs", "engine", "models",
    "native", "packaging", "tests", "third_party", "tools", "ui", "utils",
    ".github",
    ".gitignore", "ACKNOWLEDGEMENTS.md", "LICENSE", "README.md", "VERSION",
    "bootstrap.py", "gui_pyside6.py", "requirements.txt",
    "requirements-handwriting-openvino.txt", "run.py", "run_novel_formatter.command",
    "update_novel_formatter.command", "启动Mac.command", "启动Windows.bat",
    "build_apple_vision_helper.command", "prepare_48px_ar.command", "pytest.ini",
)
REQUIRED_PATHS = (
    "gui_pyside6.py", "run.py", "bootstrap.py", "requirements.txt",
    "adapters", "core", "engine", "models", "ui", "utils",
)
ProgressCallback = Callable[[str], None]


def _emit(progress: ProgressCallback | None, message: str) -> None:
    if progress is not None:
        progress(str(message).strip())


def normalize_repository(value: str | None = None) -> str:
    raw = str(value or DEFAULT_REPOSITORY).strip().rstrip("/")
    if "://" in raw:
        parsed = urlparse(raw)
        if parsed.hostname not in {"github.com", "www.github.com"}:
            raise ValueError("Git 更新仓库已锁定为 Novel Formatter 官方 GitHub 仓库")
        raw = parsed.path.strip("/")
    raw = raw.removesuffix(".git").strip("/")
    if raw != DEFAULT_REPOSITORY:
        raise ValueError(f"Git 更新仓库已锁定：{DEFAULT_REPOSITORY}")
    return DEFAULT_REPOSITORY


def normalize_branch(value: str | None = None) -> str:
    branch = str(value or DEFAULT_BRANCH).strip()
    if branch != DEFAULT_BRANCH:
        raise ValueError(f"Git 更新分支已锁定：{DEFAULT_BRANCH}")
    return DEFAULT_BRANCH


def github_repository_url(repo: str | None = None) -> str:
    return f"https://github.com/{normalize_repository(repo)}.git"


def discover_project_root(start: str | Path | None = None) -> Path:
    candidates: list[Path] = []
    if start is not None:
        candidates.append(Path(start).expanduser().resolve())
    candidates.extend((Path(__file__).resolve(), Path.cwd().resolve()))
    seen: set[Path] = set()
    for candidate in candidates:
        probe = candidate if candidate.is_dir() else candidate.parent
        for root in (probe, *probe.parents):
            if root in seen:
                continue
            seen.add(root)
            if (root / "gui_pyside6.py").is_file() and (root / "run.py").is_file() and (root / "adapters").is_dir():
                return root
    raise RuntimeError("无法定位 Novel Formatter 程序目录")


def _version_tuple(value: str) -> tuple[int, int, int, int]:
    raw = str(value or "0").strip().lstrip("vV").split("-", 1)[0]
    parts: list[int] = []
    for token in raw.split("."):
        match = re.search(r"\d+", token)
        parts.append(int(match.group(0)) if match else 0)
    return tuple((parts + [0, 0, 0, 0])[:4])  # type: ignore[return-value]


def read_project_version(root: str | Path) -> str:
    base = Path(root)
    version_file = base / "VERSION"
    if version_file.is_file():
        value = version_file.read_text(encoding="utf-8", errors="replace").strip()
        if value:
            return value
    gui = base / "gui_pyside6.py"
    if gui.is_file():
        match = re.search(r'^VERSION\s*=\s*["\']([^"\']+)["\']', gui.read_text(encoding="utf-8", errors="replace"), re.M)
        if match:
            return match.group(1).strip()
    raise RuntimeError("无法读取本地 VERSION")


def _headers() -> dict[str, str]:
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "Novel-Formatter-Source-Updater",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    token = str(os.environ.get("NOVEL_FORMATTER_GITHUB_TOKEN", "")).strip()
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _request_json(url: str, *, timeout: float = 12.0) -> object:
    request = urllib.request.Request(url, headers=_headers())
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def _github_file(repo: str, path: str, branch: str, *, timeout: float) -> bytes:
    query = urlencode({"ref": branch})
    data = _request_json(f"{API_BASE}/{repo}/contents/{quote(path)}?{query}", timeout=timeout)
    if not isinstance(data, dict) or str(data.get("encoding") or "") != "base64":
        raise RuntimeError(f"无法读取远端文件：{path}")
    content = str(data.get("content") or "")
    if not content:
        raise RuntimeError(f"远端文件为空：{path}")
    return base64.b64decode(content, validate=False)


def _remote_version(repo: str, branch: str, *, timeout: float) -> str:
    value = _github_file(repo, "VERSION", branch, timeout=timeout).decode("utf-8", errors="replace").strip()
    if not value:
        raise RuntimeError("远端 VERSION 为空")
    return value


def _run_git(args: list[str], *, cwd: str | Path | None = None, timeout: float = 90.0) -> str:
    git = shutil.which("git")
    if not git:
        raise RuntimeError("系统未安装 Git")
    proc = subprocess.run([git, *args], cwd=str(cwd) if cwd else None, text=True,
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          timeout=timeout, check=False)
    output = str(proc.stdout or "")
    if proc.returncode != 0:
        raise RuntimeError(f"Git 命令失败：git {' '.join(args)}\n{output[-6000:]}")
    return output.strip()


def _git_head(root: Path) -> str:
    if not (root / ".git").exists() or not shutil.which("git"):
        return ""
    try:
        return _run_git(["rev-parse", "HEAD"], cwd=root, timeout=12.0)
    except Exception:
        return ""


def _state_path(root: Path) -> Path:
    return root / STATE_RELATIVE


def read_source_state(root: str | Path) -> dict[str, object]:
    try:
        data = json.loads(_state_path(Path(root)).read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError, TypeError):
        return {}


def _write_source_state(root: Path, *, commit: str, version: str, method: str) -> None:
    path = _state_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "novel_formatter.source_update_state.v1",
        "repository": DEFAULT_REPOSITORY,
        "branch": DEFAULT_BRANCH,
        "commit": str(commit),
        "version": str(version),
        "method": str(method),
        "updated_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
    }
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temp, path)


def _requirements_snapshot(root: Path) -> dict[str, bytes]:
    result: dict[str, bytes] = {}
    for path in sorted(root.glob("requirements*.txt")):
        if path.is_file():
            try:
                result[path.name] = path.read_bytes()
            except OSError:
                pass
    return result


@dataclass(frozen=True, slots=True)
class SourceUpdateInfo:
    repository: str
    branch: str
    local_version: str
    remote_version: str
    local_commit: str
    remote_commit: str
    remote_date: str
    remote_message: str
    available: bool
    reason: str
    project_root: Path
    install_layout: str

    @property
    def local_short(self) -> str:
        return self.local_commit[:10] if self.local_commit else "未记录"

    @property
    def remote_short(self) -> str:
        return self.remote_commit[:10]


@dataclass(frozen=True, slots=True)
class SourceUpdateResult:
    project_root: Path
    old_version: str
    new_version: str
    commit: str
    method: str
    restart_required: bool = True
    requirements_changed: bool = False


def check_source_update(repo: str | None = None, branch: str | None = None, *,
                        project_root: str | Path | None = None,
                        timeout: float = 12.0) -> SourceUpdateInfo:
    repository = normalize_repository(repo)
    branch_name = normalize_branch(branch)
    root = discover_project_root(project_root)
    local_version = read_project_version(root)
    commit_data = _request_json(f"{API_BASE}/{repository}/commits/{quote(branch_name, safe='')}", timeout=timeout)
    if not isinstance(commit_data, dict):
        raise RuntimeError("GitHub commit 响应异常")
    remote_commit = str(commit_data.get("sha") or "").strip()
    meta = commit_data.get("commit") if isinstance(commit_data.get("commit"), dict) else {}
    remote_message = str((meta or {}).get("message") or "").splitlines()[0].strip()
    author = (meta or {}).get("author") if isinstance((meta or {}).get("author"), dict) else {}
    remote_date = str((author or {}).get("date") or "").strip()
    if not remote_commit:
        raise RuntimeError("GitHub 没有返回远端 commit")
    remote_version = _remote_version(repository, branch_name, timeout=timeout)

    git_head = _git_head(root)
    layout = "git" if git_head else "portable"
    local_commit = git_head
    if not local_commit:
        state = read_source_state(root)
        if str(state.get("repository") or "") == repository and str(state.get("branch") or "") == branch_name:
            local_commit = str(state.get("commit") or "").strip()

    local_v = _version_tuple(local_version)
    remote_v = _version_tuple(remote_version)
    if remote_v > local_v:
        available, reason = True, f"发现新版本 v{remote_version}"
    elif remote_v < local_v:
        available, reason = False, f"本地 v{local_version} 比仓库 v{remote_version} 更新；禁止自动降级"
    elif local_commit and local_commit != remote_commit:
        available, reason = True, "版本号相同，但 main 分支已有新的代码提交"
    elif local_commit and local_commit == remote_commit:
        available, reason = False, "当前代码已经与 GitHub main 分支同步"
    else:
        # Stricter than Folirina's portable first-sync policy: a manually patched
        # ZIP can differ from Git while sharing the same version. Without a known
        # commit baseline, never overwrite it merely because the commit is unknown.
        available = False
        reason = "当前为无 Git 基线的便携源码包；同版本不会自动覆盖。仓库版本号提升后可安全更新"

    return SourceUpdateInfo(repository, branch_name, local_version, remote_version,
                            local_commit, remote_commit, remote_date, remote_message,
                            available, reason, root, layout)


def _safe_extract_zip(archive: zipfile.ZipFile, destination: Path) -> None:
    destination = destination.resolve()
    infos = archive.infolist()
    if len(infos) > MAX_ARCHIVE_FILES:
        raise RuntimeError("仓库归档文件数量异常，已停止更新")
    if sum(max(0, int(x.file_size)) for x in infos) > MAX_ARCHIVE_BYTES:
        raise RuntimeError("仓库归档解压体积异常，已停止更新")
    for info in infos:
        raw = str(info.filename).replace("\\", "/")
        candidate = Path(raw)
        if candidate.is_absolute() or any(part == ".." for part in candidate.parts):
            raise RuntimeError(f"仓库归档包含不安全路径：{info.filename}")
        output = (destination / candidate).resolve()
        try:
            output.relative_to(destination)
        except ValueError as exc:
            raise RuntimeError(f"仓库归档路径越界：{info.filename}") from exc
    archive.extractall(destination)


def _download_archive(repo: str, ref: str, destination: Path, *, timeout: float = 120.0) -> None:
    # Use the immutable commit returned by the preceding check, not a moving
    # branch name. This closes the check/download race for systems without Git.
    url = f"{API_BASE}/{repo}/zipball/{quote(ref, safe='')}"
    request = urllib.request.Request(url, headers=_headers())
    with urllib.request.urlopen(request, timeout=timeout) as response, destination.open("wb") as fh:
        total = 0
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            total += len(chunk)
            if total > MAX_DOWNLOAD_BYTES:
                raise RuntimeError("Git 仓库下载体积异常，已停止更新")
            fh.write(chunk)


def _candidate_root(extract_root: Path) -> Path:
    roots = [p for p in extract_root.iterdir() if p.is_dir()]
    for candidate in roots:
        if (candidate / "gui_pyside6.py").is_file() and (candidate / "adapters").is_dir():
            return candidate
    if (extract_root / "gui_pyside6.py").is_file():
        return extract_root
    raise RuntimeError("下载的 Git 仓库中没有找到 Novel Formatter 项目根目录")


def _validate_python_tree(candidate: Path) -> int:
    count = 0
    for path in candidate.rglob("*.py"):
        # Ignore virtualenv/build outputs if a contributor accidentally has them
        # in an archive; managed copy would not include them anyway.
        rel = path.relative_to(candidate).as_posix()
        if any(part.startswith(".venv") for part in Path(rel).parts) or "/build/" in f"/{rel}/":
            continue
        ast.parse(path.read_text(encoding="utf-8", errors="strict"), filename=str(path))
        count += 1
    if count < 20:
        raise RuntimeError("远端 Python 源码数量异常，已停止更新")
    return count


def _validate_candidate(candidate: Path, *, current_version: str) -> str:
    missing = [rel for rel in REQUIRED_PATHS if not (candidate / rel).exists()]
    if missing:
        raise RuntimeError("远端源码结构不完整：" + "、".join(missing))
    for rel in MANAGED_PATHS:
        path = candidate / rel
        if not path.exists() and not path.is_symlink():
            continue
        if path.is_symlink():
            raise RuntimeError(f"远端源码包含不允许的符号链接：{rel}")
        if path.is_dir():
            for child in path.rglob("*"):
                if child.is_symlink():
                    raise RuntimeError(f"远端源码包含不允许的符号链接：{child.relative_to(candidate)}")
    version = read_project_version(candidate)
    if _version_tuple(version) < _version_tuple(current_version):
        raise RuntimeError(f"远端 v{version} 低于本地 v{current_version}，已阻止降级")
    _validate_python_tree(candidate)
    return version


def _portable_candidate(info: SourceUpdateInfo, temp_root: Path, progress: ProgressCallback | None) -> tuple[Path, str]:
    checkout = temp_root / "checkout"
    if shutil.which("git"):
        _emit(progress, "正在从官方 Git 仓库克隆 main 分支到临时校验区…")
        _run_git(["clone", "--depth", "1", "--single-branch", "--branch", info.branch,
                  github_repository_url(info.repository), str(checkout)], timeout=240.0)
        commit = _run_git(["rev-parse", "HEAD"], cwd=checkout, timeout=12.0)
        return checkout, commit
    _emit(progress, "系统未安装 Git；改用 GitHub main 分支 ZIP 归档…")
    archive_path = temp_root / "repository.zip"
    _download_archive(info.repository, info.remote_commit, archive_path)
    extract = temp_root / "archive"
    extract.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path) as archive:
        _safe_extract_zip(archive, extract)
    return _candidate_root(extract), info.remote_commit


def _copy_managed(candidate: Path, stage: Path) -> list[str]:
    copied: list[str] = []
    stage.mkdir(parents=True, exist_ok=True)
    for rel in MANAGED_PATHS:
        source = candidate / rel
        if not source.exists():
            continue
        target = stage / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        if source.is_dir() and not source.is_symlink():
            shutil.copytree(source, target, symlinks=True)
        else:
            shutil.copy2(source, target, follow_symlinks=False)
        copied.append(rel)
    if not all(required in copied or any(required.startswith(x + "/") for x in copied) for required in ("gui_pyside6.py", "adapters", "core", "engine", "utils")):
        raise RuntimeError("远端源码缺少核心程序路径，已停止更新")
    return copied


def _remove_path(path: Path) -> None:
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path, ignore_errors=True)
    else:
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass


def _ensure_launcher_permissions(root: Path) -> None:
    if os.name == "nt":
        return
    for rel in (
        "run_novel_formatter.command", "update_novel_formatter.command",
        "启动Mac.command", "build_apple_vision_helper.command",
        "prepare_48px_ar.command",
        # Python zipfile extraction does not reliably restore POSIX mode bits.
        # Keep the bundled Swift OCR helper executable after portable updates.
        "tools/apple_vision_helper/bin/apple_vision_helper",
    ):
        path = root / rel
        if path.is_file():
            try:
                path.chmod(path.stat().st_mode | 0o111)
            except OSError:
                pass


def _install_portable(info: SourceUpdateInfo, *, progress: ProgressCallback | None) -> SourceUpdateResult:
    root = info.project_root.resolve()
    old_version = read_project_version(root)
    old_requirements = _requirements_snapshot(root)
    with tempfile.TemporaryDirectory(prefix="novel-formatter-source-update-") as temp_name:
        temp_root = Path(temp_name)
        candidate, candidate_commit = _portable_candidate(info, temp_root, progress)
        new_version = _validate_candidate(candidate, current_version=old_version)
        commit = candidate_commit or info.remote_commit
        if commit != info.remote_commit:
            raise RuntimeError("下载到的 Git commit 与检查结果不一致；请重新检查更新")
        _emit(progress, f"远端源码校验通过：v{new_version} · {commit[:10]}")

        # Same-filesystem stage/backup makes each top-level move atomic. The
        # backup is retained only during the transaction and is restored on any
        # exception before being deleted.
        stage = Path(tempfile.mkdtemp(prefix=f".{root.name}.nf-stage-", dir=str(root.parent)))
        backup = Path(tempfile.mkdtemp(prefix=f".{root.name}.nf-backup-", dir=str(root.parent)))
        installed: list[str] = []
        moved_old: list[str] = []
        try:
            copied = _copy_managed(candidate, stage)
            _validate_candidate(stage, current_version=old_version)
            copied_set = set(copied)
            _emit(progress, "事务更新区校验通过；正在替换程序文件（用户数据不会进入替换列表）…")
            # Mirror the locked program-managed surface exactly. If upstream
            # intentionally removed an old program file/directory, leaving that
            # stale path behind could mix two code versions. Every old managed
            # path is therefore backed up first; only paths present in the
            # verified candidate are installed.
            for rel in MANAGED_PATHS:
                current = root / rel
                incoming = stage / rel
                saved = backup / rel
                if current.exists() or current.is_symlink():
                    saved.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(current), str(saved))
                    moved_old.append(rel)
                if rel in copied_set:
                    current.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(incoming), str(current))
                    installed.append(rel)
            _validate_candidate(root, current_version=old_version)
            _ensure_launcher_permissions(root)
            _write_source_state(root, commit=commit, version=new_version, method="portable-transaction")
        except Exception:
            _emit(progress, "更新失败；正在自动回滚到更新前程序代码…")
            for rel in reversed(installed):
                _remove_path(root / rel)
            for rel in reversed(moved_old):
                saved, current = backup / rel, root / rel
                if saved.exists() or saved.is_symlink():
                    current.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(saved), str(current))
            raise
        finally:
            shutil.rmtree(stage, ignore_errors=True)
            shutil.rmtree(backup, ignore_errors=True)
    requirements_changed = old_requirements != _requirements_snapshot(root)
    _emit(progress, "程序源码更新完成；请重新启动 Novel Formatter。")
    if requirements_changed:
        _emit(progress, "依赖清单已变化；为保持现有安全合同，本次没有自动运行 pip。重启前请按项目安装说明更新主依赖。")
    return SourceUpdateResult(root, old_version, new_version, commit, "portable-transaction", True, requirements_changed)


def _git_clean(root: Path) -> bool:
    # Ignored runtime/model/output paths stay invisible; every other tracked or
    # untracked source change blocks self-update to prevent accidental overwrite.
    return not bool(_run_git(["status", "--porcelain"], cwd=root, timeout=15.0).strip())


def _install_git(info: SourceUpdateInfo, *, progress: ProgressCallback | None) -> SourceUpdateResult:
    root = info.project_root.resolve()
    if not _git_clean(root):
        raise RuntimeError("当前 Git 工作区有未提交修改；为避免覆盖本地代码，已停止更新")
    old_commit = _run_git(["rev-parse", "HEAD"], cwd=root, timeout=12.0)
    old_version = read_project_version(root)
    old_requirements = _requirements_snapshot(root)
    _emit(progress, "正在 fetch 官方 main 分支…")
    _run_git(["fetch", "--no-tags", github_repository_url(info.repository), info.branch], cwd=root, timeout=240.0)
    target = _run_git(["rev-parse", "FETCH_HEAD"], cwd=root, timeout=12.0)
    if target != info.remote_commit:
        raise RuntimeError("远端 commit 已变化；请重新检查更新后再安装")
    git = shutil.which("git") or "git"
    ancestor = subprocess.run([git, "merge-base", "--is-ancestor", old_commit, target], cwd=str(root), check=False)
    if ancestor.returncode != 0:
        raise RuntimeError("本地 Git 历史与官方 main 已分叉；自动更新只允许 fast-forward，未修改任何文件")
    _emit(progress, "历史校验通过；执行 Git fast-forward…")
    _run_git(["merge", "--ff-only", target], cwd=root, timeout=180.0)
    try:
        new_version = _validate_candidate(root, current_version=old_version)
    except Exception:
        # A fast-forward can be safely moved back because the worktree was clean
        # and no user commit was created by the updater.
        _run_git(["reset", "--hard", old_commit], cwd=root, timeout=60.0)
        raise
    _write_source_state(root, commit=target, version=new_version, method="git-fast-forward")
    requirements_changed = old_requirements != _requirements_snapshot(root)
    _emit(progress, "Git fast-forward 更新完成；请重新启动 Novel Formatter。")
    if requirements_changed:
        _emit(progress, "依赖清单已变化；本次没有自动运行 pip，请按项目安装说明更新主依赖。")
    return SourceUpdateResult(root, old_version, new_version, target, "git-fast-forward", True, requirements_changed)


def install_source_update(info: SourceUpdateInfo, *, progress: ProgressCallback | None = None) -> SourceUpdateResult:
    if normalize_repository(info.repository) != info.repository or normalize_branch(info.branch) != info.branch:
        raise RuntimeError("更新目标与锁定仓库不一致")
    root = discover_project_root(info.project_root)
    if root.resolve() != info.project_root.resolve():
        raise RuntimeError("程序目录发生变化；请重新检查更新")
    current_version = read_project_version(root)
    if _version_tuple(info.remote_version) < _version_tuple(current_version):
        raise RuntimeError("远端版本低于当前版本，禁止自动降级")
    if not info.available:
        raise RuntimeError("当前检查结果没有可安装更新")
    if (root / ".git").exists() and shutil.which("git"):
        return _install_git(info, progress=progress)
    return _install_portable(info, progress=progress)


def diagnostic_summary(root: str | Path | None = None) -> str:
    base = discover_project_root(root)
    state = read_source_state(base)
    head = _git_head(base)
    lines = [
        "Novel Formatter 诊断信息",
        f"版本: {read_project_version(base)}",
        f"程序目录: {base}",
        f"安装形态: {'Git checkout' if head else 'portable/source ZIP'}",
        f"Git HEAD: {head[:12] if head else '未记录'}",
        f"更新基线: {str(state.get('commit') or '')[:12] or '未记录'}",
        f"更新仓库: {DEFAULT_REPOSITORY}",
        f"更新分支: {DEFAULT_BRANCH}",
    ]
    return "\n".join(lines)


__all__ = [
    "DEFAULT_REPOSITORY", "DEFAULT_BRANCH", "MANAGED_PATHS", "STATE_RELATIVE",
    "SourceUpdateInfo", "SourceUpdateResult", "check_source_update",
    "diagnostic_summary", "discover_project_root", "github_repository_url",
    "install_source_update", "normalize_branch", "normalize_repository",
    "read_project_version", "read_source_state", "_safe_extract_zip",
]
