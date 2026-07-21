# -*- coding: utf-8 -*-
import json
from pathlib import Path

def save_replacement_report(report, path="replacement_report.json"):
    data = report.to_dict() if hasattr(report,"to_dict") else report
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return path
