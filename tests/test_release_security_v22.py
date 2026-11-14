"""Offline v2.2 release-security regression tests (stdlib only)."""
from __future__ import annotations

import ctypes
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ai import secure_store


class FakeSecurity:
    def __init__(self):
        self.data, self.items, self.buffers = {}, {}, {}
        self.seq = 100

    def SecKeychainAddGenericPassword(self, _, nsvc, svc, nacct, acct, n, data, _item):
        key = (bytes(svc)[:nsvc], bytes(acct)[:nacct])
        if key in self.data:
            return -25299
        self.seq += 1
        self.data[key] = ctypes.string_at(data, n)
        self.items[self.seq] = key
        return 0

    def SecKeychainFindGenericPassword(self, _, nsvc, svc, nacct, acct, length, data, item):
        key = (bytes(svc)[:nsvc], bytes(acct)[:nacct])
        if key not in self.data:
            return -25300
        ident = next(i for i, k in self.items.items() if k == key)
        ctypes.cast(item, ctypes.POINTER(ctypes.c_void_p)).contents.value = ident
        if length is not None:
            value = self.data[key]
            self.buffers[ident] = ctypes.create_string_buffer(value)
            ctypes.cast(length, ctypes.POINTER(ctypes.c_uint32)).contents.value = len(value)
            ctypes.cast(data, ctypes.POINTER(ctypes.c_void_p)).contents.value = ctypes.addressof(self.buffers[ident])
        return 0

    def SecKeychainItemModifyAttributesAndData(self, item, _attr, n, data):
        self.data[self.items[item.value]] = ctypes.string_at(data, n)
        return 0

    def SecKeychainItemDelete(self, item):
        self.data.pop(self.items.pop(item.value))
        return 0

    def SecKeychainItemFreeContent(self, _attr, _data):
        return 0


class FakeCF:
    def CFRelease(self, _item):
        pass


class SecurityTests(unittest.TestCase):
    def test_keychain_create_update_read_delete_without_cli(self):
        native = FakeSecurity()
        with mock.patch.object(secure_store, "_mac_security_framework", return_value=(native, FakeCF())), \
             mock.patch.object(secure_store.subprocess, "run", side_effect=AssertionError("CLI called")):
            self.assertTrue(secure_store._mac_save("机密🔒", "NovelFormatter.TEST", "token"))
            self.assertEqual(secure_store._mac_load("NovelFormatter.TEST", "token"), "机密🔒")
            self.assertTrue(secure_store._mac_save("second", "NovelFormatter.TEST", "token"))
            self.assertEqual(secure_store._mac_load("NovelFormatter.TEST", "token"), "second")
            secure_store._mac_delete("NovelFormatter.TEST", "token")
            self.assertEqual(secure_store._mac_load("NovelFormatter.TEST", "token"), "")
            secure_store._mac_delete("NovelFormatter.TEST", "token")

    def test_keychain_failure_never_falls_back_to_argv(self):
        with mock.patch.object(secure_store, "_mac_security_framework", side_effect=OSError("missing")), \
             mock.patch.object(secure_store.subprocess, "run", side_effect=AssertionError("CLI called")):
            with self.assertRaises(OSError):
                secure_store._mac_save("secret")

    def test_python_checksums_cannot_come_from_mirror(self):
        mac = (ROOT / "packaging/launcher.zsh").read_text()
        win = (ROOT / "packaging/launcher_windows.ps1").read_text()
        self.assertIn('fetch_file "$SUMS_PATH" "$OFFICIAL_ROOT/SHA256SUMS"', mac)
        self.assertNotIn('"$MIRROR_ROOT/SHA256SUMS"', mac)
        self.assertIn('Invoke-Download -Urls @("$officialRoot/SHA256SUMS")', win)
        self.assertNotIn('"$mirrorRoot/SHA256SUMS"', win)

    def test_appimagetool_pinned_asset_ids(self):
        shell = (ROOT / "packaging/build_linux_release.sh").read_text()
        self.assertIn("APPIMAGETOOL_ASSET_ID=98605504", shell)
        self.assertIn("APPIMAGETOOL_ASSET_ID=98605483", shell)
        self.assertIn('validate_appimagetool_metadata "$APPIMAGETOOL_META"', shell)
        self.assertIn("sha256sum --check --status", shell)

    def test_appimagetool_changed_asset_metadata_fails(self):
        shell = (ROOT / "packaging/build_linux_release.sh").read_text()
        func = "validate_appimagetool_metadata() {\n" + shell.split("validate_appimagetool_metadata() {\n", 1)[1].split("\n\ncurl --proto", 1)[0]
        common = {
            "id": 98605504,
            "name": "appimagetool-x86_64.AppImage",
            "size": 8811712,
            "updated_at": "2025-07-26T07:31:24Z",
            "state": "uploaded",
            "browser_download_url": "https://github.com/AppImage/AppImageKit/releases/download/continuous/appimagetool-x86_64.AppImage",
        }
        prefix = "APPIMAGETOOL_ASSET_ID=98605504; APPIMAGETOOL_NAME=appimagetool-x86_64.AppImage; APPIMAGETOOL_EXPECTED_SIZE=8811712; APPIMAGETOOL_UPDATED_AT=2025-07-26T07:31:24Z;\n"
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "metadata.json"
            command = prefix + func + "\nvalidate_appimagetool_metadata " + str(path)
            for payload, should_pass in [(common, True), ({**common, "id": 98605505}, False), ({**common, "size": 1}, False)]:
                path.write_text(json.dumps(payload))
                proc = subprocess.run(["bash", "-c", command], capture_output=True, text=True)
                self.assertEqual(proc.returncode == 0, should_pass, proc.stderr)

    def test_version_is_22(self):
        self.assertEqual((ROOT / "VERSION").read_text().strip(), "2.2")
        self.assertIn('VERSION = "2.2"', (ROOT / "core/app_meta.py").read_text())
        self.assertIn('self._make_card(f"Novel Formatter {VERSION}")', (ROOT / "ui/settings/tab.py").read_text())


if __name__ == "__main__":
    unittest.main()
