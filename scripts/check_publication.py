"""Audit Git's index and reachable history without printing matched secrets."""

import argparse
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ROOT_FILES = {
    "README.md",
    "README.en.md",
    "LICENSE",
    "SECURITY.md",
    ".gitignore",
    ".gitattributes",
    ".env.example",
    "pyproject.toml",
    "requirements.txt",
    "requirements-demo.txt",
    "requirements-dev.txt",
    "requirements-lock-demo.txt",
    "apply_pipeline.py",
    "experiment.py",
    "preprocessing.py",
    "io_utils.py",
}
SPECIAL_FILES = {
    "datasets/demo_tickets.csv",
    "datasets/README.md",
    "docs/benchmark_results.csv",
    "docs/experiment_summary.json",
    "docs/benchmark_nmi_chart.png",
    "docs/RESULTS.md",
    "scripts/check_publication.py",
    "scripts/plot_benchmark.py",
    ".github/workflows/ci.yml",
}
SECRET_PATTERNS = {
    "API credential": rb"\b(?:sk-(?:or-v1-|proj-)?|hf_|gh[pousr]_)[A-Za-z0-9_-]{20,}",
    "GitHub fine-grained credential": rb"\bgithub_pat_[A-Za-z0-9_]{20,}",
    "private key": rb"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----",
    "local user path": rb"(?i)[a-z]:[\\/]Users[\\/][^\r\n]+",
    "email address": rb"[A-Za-z0-9_.+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}",
    "IP address": rb"\b(?:\d{1,3}\.){3}\d{1,3}\b",
}


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT)


def allowed_path(name):
    path = Path(name)
    return name in ROOT_FILES | SPECIAL_FILES or (
        path.parts[0] == "tests" and path.suffix == ".py" and "__pycache__" not in path.parts
    )


def inspect_blob(data):
    # Generated chart pixels do not contain source text; source files are all UTF-8.
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return []
    return [kind for kind, pattern in SECRET_PATTERNS.items() if re.search(pattern, data)]


def audit(history=False):
    problems = []
    names = git("ls-files", "-z").decode("utf-8").split("\0")
    for name in filter(None, names):
        if not allowed_path(name):
            problems.append((name, "file is outside the public allowlist"))
        for kind in inspect_blob(git("show", f":{name}")):
            problems.append((name, kind))
    if history:
        for commit in git("rev-list", "--all").decode().splitlines():
            for name in git("ls-tree", "-r", "--name-only", commit).decode("utf-8").splitlines():
                if not allowed_path(name):
                    problems.append((name, "unexpected file in Git history"))
        object_ids = {
            line.split()[0] for line in git("rev-list", "--objects", "--all").decode().splitlines()
        }
        for object_id in object_ids:
            if git("cat-file", "-t", object_id).strip() == b"blob":
                for kind in inspect_blob(git("cat-file", "blob", object_id)):
                    problems.append((object_id[:12], f"{kind} in Git history"))
    return sorted(set(problems))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history", action="store_true")
    args = parser.parse_args()
    findings = audit(args.history)
    for location, reason in findings:
        print(f"FAIL: {location}: {reason}; matched value withheld")
    if findings:
        raise SystemExit(1)
    print("Publication audit passed: allowlisted files; no credential/contact/path patterns found.")
