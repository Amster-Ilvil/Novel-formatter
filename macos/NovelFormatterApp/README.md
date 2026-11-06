# NovelFormatterApp — native SwiftUI shell

This is the macOS shell for Novel Formatter Studio. It is a thin SwiftUI
wrapper: all OCR, formatting, EPUB export and AI logic stays in the Python
codebase (`gui_pyside6.py`, `bootstrap.py`), launched through the same
`bootstrap.py --install-main-deps --launch` entry point as the classic
`packaging/launcher.zsh`.

## What the shell does

1. On launch, `PythonRuntime` prepares the standalone CPython 3.12.12
   (python-build-standalone, same mirrors and SHA-256 verification as
   `launcher.zsh`) into
   `~/Library/Application Support/Novel Formatter Studio/native-runtime-v1`.
2. It copies the bundled Python source from
   `Contents/Resources/app` into `~/Library/Application Support/Novel Formatter Studio/app-source`
   (same location as the zsh launcher, so both shells share the runtime and
   the writable source tree).
3. The window shows the preparation log. "Open workspace" starts the Python
   GUI as a child process with a cleaned environment (mirrors
   `clean_injected_environment()` from `launcher.zsh`).

## What it deliberately does not do

- No OCR models are bundled or downloaded here. The policy is unchanged:
  OCR runtimes/models install only after the user starts an OCR and confirms
  the prompt inside the Python GUI.
- No Python logic is reimplemented in Swift; the shell owns process
  lifecycle, logging and Finder integration only.

## Build

`packaging/build_macos_swift_app.sh` compiles this with a single `swiftc`
invocation (no Xcode project required), embeds `Resources/app`, ad-hoc
signs the bundle and packages DMG + app.zip under `package/`.
Requires macOS 14+ at runtime (SwiftUI `.onChange` without the `initial:`
parameter). The GitHub Actions job `macos-swift` in
`.github/workflows/release.yml` runs it on every release, and
`.github/workflows/build-swift-app.yml` builds it on demand.
