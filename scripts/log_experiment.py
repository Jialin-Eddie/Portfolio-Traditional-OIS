"""PostToolUse hook: log experiment metrics after main.py runs.

Trigger condition:
  - CLAUDE_TOOL_NAME == "Bash"
  - CLAUDE_TOOL_INPUT contains "main.py"
  - outputs/backtest_summary.csv modified within last 120 seconds

Appends one JSON line to outputs/experiment_log.jsonl.
"""
import json, os, sys, time, re, subprocess
from pathlib import Path
from datetime import datetime, timezone

PROJECT = Path(__file__).resolve().parent.parent
OUTPUTS = PROJECT / "outputs"
CONFIG_PY = PROJECT / "src" / "config.py"
BACKTEST_CSV = OUTPUTS / "backtest_summary.csv"
OOS_CSV = OUTPUTS / "metrics_summary_OOS_2021_2024.csv"
FULL_CSV = OUTPUTS / "metrics_summary_Full.csv"
LOG_FILE = OUTPUTS / "experiment_log.jsonl"
STALENESS_SEC = 120


def should_run():
    tool_name = os.environ.get("CLAUDE_TOOL_NAME", "")
    tool_input = os.environ.get("CLAUDE_TOOL_INPUT", "")
    if tool_name != "Bash":
        return False
    if "main.py" not in tool_input:
        return False
    if not BACKTEST_CSV.exists():
        return False
    age = time.time() - BACKTEST_CSV.stat().st_mtime
    return age < STALENESS_SEC


def parse_csv_metrics(path):
    if not path.exists():
        return {}
    metrics = {}
    with open(path, "r") as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) >= 2 and parts[0] and parts[0] != "":
                key = parts[0].strip()
                val = parts[1].strip()
                try:
                    metrics[key] = round(float(val), 6)
                except ValueError:
                    pass
    return metrics


def parse_hyperparams():
    if not CONFIG_PY.exists():
        return {}
    text = CONFIG_PY.read_text(encoding="utf-8")
    params = {}
    xgb_block = re.search(r"XGB_PARAMS\s*=\s*dict\((.*?)\)", text, re.DOTALL)
    if xgb_block:
        for m in re.finditer(r"(\w+)\s*=\s*([^,\)]+)", xgb_block.group(1)):
            k, v = m.group(1).strip(), m.group(2).strip()
            try:
                params[k] = int(v) if "." not in v else float(v)
            except ValueError:
                params[k] = v
    for pat in [r"XGB_EARLY_STOPPING_ROUNDS\s*=\s*(\d+)", r"XGB_VAL_FRACTION\s*=\s*([\d.]+)"]:
        m = re.search(pat, text)
        if m:
            name = pat.split(r"\s")[0]
            val_str = m.group(1)
            try:
                params[name] = int(val_str) if "." not in val_str else float(val_str)
            except ValueError:
                params[name] = val_str
    return params


def get_git_hash():
    try:
        result = subprocess.run(
            ["git", "-C", str(PROJECT), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5
        )
        return result.stdout.strip() if result.returncode == 0 else "no-git"
    except Exception:
        return "no-git"


def main():
    if not should_run():
        return
    record = {
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "git_hash": get_git_hash(),
        "hyperparams": parse_hyperparams(),
        "oos_metrics": parse_csv_metrics(OOS_CSV),
        "full_metrics": parse_csv_metrics(FULL_CSV),
        "backtest_summary": parse_csv_metrics(BACKTEST_CSV),
        "notes": "",
    }
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
