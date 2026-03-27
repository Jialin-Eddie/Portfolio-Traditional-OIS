"""Stop hook: log key src file changes as decisions at session end.

Checks git diff (or file mtime fallback) for changes to key source files.
If any changed, appends a decision record to outputs/decision_log.jsonl.
"""
import json, os, subprocess, sys
from pathlib import Path
from datetime import datetime, timezone

PROJECT = Path(__file__).resolve().parent.parent
OUTPUTS = PROJECT / "outputs"
LOG_FILE = OUTPUTS / "decision_log.jsonl"

KEY_FILES = [
    "src/config.py",
    "src/signals.py",
    "src/optimizer.py",
    "src/preprocessing.py",
    "src/data_loader.py",
    "src/backtest.py",
    "src/constraints.py",
]


def git_available():
    try:
        r = subprocess.run(
            ["git", "-C", str(PROJECT), "status", "--porcelain"],
            capture_output=True, text=True, timeout=5
        )
        return r.returncode == 0
    except Exception:
        return False


def get_changed_files():
    try:
        r = subprocess.run(
            ["git", "-C", str(PROJECT), "diff", "--name-only"],
            capture_output=True, text=True, timeout=5
        )
        staged = subprocess.run(
            ["git", "-C", str(PROJECT), "diff", "--name-only", "--cached"],
            capture_output=True, text=True, timeout=5
        )
        all_changed = set(r.stdout.strip().split("\n") + staged.stdout.strip().split("\n"))
        all_changed.discard("")
        return [f for f in all_changed if f in KEY_FILES]
    except Exception:
        return []


def get_diff_summary(files):
    summaries = []
    for f in files:
        try:
            r = subprocess.run(
                ["git", "-C", str(PROJECT), "diff", "--stat", f],
                capture_output=True, text=True, timeout=5
            )
            stat = r.stdout.strip()
            d = subprocess.run(
                ["git", "-C", str(PROJECT), "diff", "-U0", f],
                capture_output=True, text=True, timeout=5
            )
            changes = []
            for line in d.stdout.split("\n"):
                if line.startswith("+") and not line.startswith("+++"):
                    changes.append(line[1:].strip())
                elif line.startswith("-") and not line.startswith("---"):
                    changes.append(f"(removed) {line[1:].strip()}")
            short = "; ".join(changes[:5])
            if len(changes) > 5:
                short += f" ... (+{len(changes)-5} more)"
            summaries.append(f"{f}: {short}" if short else f"{f}: {stat}")
        except Exception:
            summaries.append(f"{f}: (diff unavailable)")
    return " | ".join(summaries)


def get_git_hash():
    try:
        r = subprocess.run(
            ["git", "-C", str(PROJECT), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5
        )
        return r.stdout.strip() if r.returncode == 0 else "uncommitted"
    except Exception:
        return "uncommitted"


def main():
    if not git_available():
        return
    changed = get_changed_files()
    if not changed:
        return
    record = {
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "files_changed": changed,
        "diff_summary": get_diff_summary(changed),
        "git_hash": get_git_hash(),
    }
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
