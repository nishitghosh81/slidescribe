"""Repairs the SlideScribe repo layout.

The Python modules were committed to the repo root instead of into a
``slidescribe/`` package directory, and ``__init__.py`` was never committed at
all. Setuptools therefore finds no package, builds an empty wheel, and pip
reports a successful install of nothing -- which surfaces later as
``ModuleNotFoundError: No module named 'slidescribe'``.

This script moves the files into the correct layout using ``git mv`` so history
is preserved, writes the missing ``__init__.py``, and untracks files that should
never have been committed.

Run once from the repo root:

    python fix_layout.py

It prints what it will do and asks before touching anything. Nothing is deleted
from disk -- ``sample_output.pdf`` is only untracked from git.
"""

import pathlib
import subprocess
import sys

PACKAGE_MODULES = [
    "cli.py",
    "config.py",
    "llm.py",
    "pdf.py",
    "pipeline.py",
    "slides.py",
    "transcribe.py",
]

TEST_MODULES = ["test_slidescribe.py"]

# Committed by accident: build output and a stray duplicate of the notebook.
UNTRACK = ["sample_output.pdf", "SlideScribe_Colab.ipynb"]

INIT_PY = '''"""SlideScribe — turn conference recordings into illustrated PDF transcripts.

Public API:
    from slidescribe import process_video, Config
    process_video("meeting.mp4", "meeting.pdf")
"""

from slidescribe.config import Config, LLMConfig
from slidescribe.pipeline import process_video, ProcessResult

__version__ = "0.1.0"
__all__ = ["process_video", "Config", "LLMConfig", "ProcessResult"]
'''


def git(*args, check=True):
    """Run a git command, returning stdout."""
    result = subprocess.run(
        ["git", *args], capture_output=True, text=True, check=False
    )
    if check and result.returncode != 0:
        print(f"  git {' '.join(args)}\n    {result.stderr.strip()}")
        return None
    return result.stdout.strip()


def tracked() -> set:
    listing = git("ls-files")
    if listing is None:
        sys.exit("Not a git repository, or git is not on PATH.")
    return set(listing.splitlines())


def main() -> int:
    if not pathlib.Path(".git").is_dir():
        sys.exit("Run this from the repo root (the folder containing .git).")

    files = tracked()
    plan = []

    for name in PACKAGE_MODULES:
        if name in files:
            plan.append(("move", name, f"slidescribe/{name}"))
        elif f"slidescribe/{name}" not in files:
            plan.append(("missing", name, ""))

    for name in TEST_MODULES:
        if name in files:
            plan.append(("move", name, f"tests/{name}"))

    if "slidescribe/__init__.py" not in files:
        plan.append(("create", "slidescribe/__init__.py", ""))

    for name in UNTRACK:
        if name in files:
            plan.append(("untrack", name, ""))

    if not plan:
        print("Layout already correct — nothing to do.")
        return 0

    print("Planned changes:\n")
    labels = {
        "move": "move    ",
        "create": "create  ",
        "untrack": "untrack ",
        "missing": "MISSING ",
    }
    for action, src, dst in plan:
        arrow = f" -> {dst}" if dst else ""
        note = "  (not tracked anywhere — check your working copy)" if action == "missing" else ""
        print(f"  {labels[action]}{src}{arrow}{note}")

    if any(a == "missing" for a, _, _ in plan):
        print("\nSome modules are not in the repo at all. Add them before continuing.")

    print()
    if input("Proceed? [y/N] ").strip().lower() not in ("y", "yes"):
        print("Aborted. Nothing changed.")
        return 1

    print()
    pathlib.Path("slidescribe").mkdir(exist_ok=True)
    pathlib.Path("tests").mkdir(exist_ok=True)

    for action, src, dst in plan:
        if action == "move":
            if git("mv", src, dst) is not None:
                print(f"  moved   {src} -> {dst}")
        elif action == "create":
            pathlib.Path(src).write_text(INIT_PY, encoding="utf-8")
            git("add", src)
            print(f"  created {src}")
        elif action == "untrack":
            git("rm", "--cached", "-q", src)
            print(f"  untracked {src} (still on disk)")

    print("\nDone. Verify, then commit:\n")
    print("  git status")
    print("  pip install -e .")
    print('  python -c "import slidescribe; print(slidescribe.__version__)"')
    print("  pytest -q")
    print('  git commit -m "Fix package layout: move modules into slidescribe/"')
    print("  git push origin main")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
