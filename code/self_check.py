"""
self_check.py
==============
Verifies the repo's docs and notebook stay consistent with each other and
with the original working notebook. Run with:

    python code/self_check.py

Prints PASS or FAIL for each check, with specifics (file, line, what's
wrong) on any failure. Exits with status 1 if anything failed.
"""

import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
LESSONS_DIR = REPO_ROOT / "lessons"
NOTEBOOK_PATH = REPO_ROOT / "mini_model_full_notebook.ipynb"
LESSON07_PATH = LESSONS_DIR / "07_complete_code.md"
README_PATH = REPO_ROOT / "README.md"
DATASETS_README_PATH = REPO_ROOT / "datasets" / "README.md"

# The original, hand-written source notebook this whole repo was built from.
# It lives outside the repo (a sibling of the repo directory) since it's the
# author's personal working copy, not a project deliverable.
ORIGINAL_NOTEBOOK_PATH = REPO_ROOT.parent / "mini_model.ipynb"

BANNED_PHRASES = [
    "cylindrical onion",
    "firework exploding",
    "wet sand",
    "whispering",
    "the coach",
]

# Lessons that are expected to carry a "Full code for this lesson" section.
# Lesson 0 is pure concept (no code), and lesson 7 is itself a pure-code
# reference page with its own Step headers, so neither one is in scope here.
CODE_BEARING_LESSONS = [
    "01_what_is_a_jet.md",
    "02_finding_the_truth_labels.md",
    "03_preparing_the_data.md",
    "04_building_mini_part.md",
    "05_training_the_model.md",
    "06_evaluating_the_model.md",
]

failures_total = 0


def report(name, ok, details=None):
    global failures_total
    status = "PASS" if ok else "FAIL"
    print(f"[{status}] {name}")
    if not ok:
        failures_total += 1
        for d in details or []:
            print(f"       {d}")


def all_md_files():
    return sorted(REPO_ROOT.rglob("*.md"), key=lambda p: str(p))


def is_excluded(path):
    # Check components *relative to the repo root* only, so the repo
    # directory itself (also named "mini_model_parT") doesn't match.
    rel_parts = set(path.relative_to(REPO_ROOT).parts)
    return bool(rel_parts & {".venv", ".git", "mini_model_parT", ".claude"})


def load_notebook_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def cell_source_text(cell):
    src = cell.get("source", "")
    if isinstance(src, list):
        return "".join(src)
    return src


def code_cells(nb_json):
    return [
        cell_source_text(c)
        for c in nb_json.get("cells", [])
        if c.get("cell_type") == "code"
    ]


def comment_lines(source):
    return [line.strip() for line in source.splitlines() if line.strip().startswith("#")]


def extract_python_blocks(text, after_heading=None):
    if after_heading is not None:
        idx = text.find(after_heading)
        if idx == -1:
            return []
        text = text[idx:]
    return re.findall(r"```python\n(.*?)```", text, re.DOTALL)


# ---------------------------------------------------------------------------
# Check 1: zero em dash characters in any .md file
# ---------------------------------------------------------------------------
def check_no_em_dash():
    violations = []
    for path in all_md_files():
        if is_excluded(path):
            continue
        rel = path.relative_to(REPO_ROOT)
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if "—" in line:
                violations.append(f"{rel}:{lineno}: {line.strip()}")
    report("No em dash characters in any .md file", not violations, violations)


# ---------------------------------------------------------------------------
# Check 2: banned phrases not present in lessons/*.md (case-insensitive)
# ---------------------------------------------------------------------------
def check_no_banned_phrases():
    violations = []
    for path in sorted(LESSONS_DIR.glob("*.md")):
        rel = path.relative_to(REPO_ROOT)
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            low = line.lower()
            for phrase in BANNED_PHRASES:
                if phrase in low:
                    violations.append(f"{rel}:{lineno}: contains \"{phrase}\" -> {line.strip()}")
    report("No banned phrases in lessons/*.md", not violations, violations)


# ---------------------------------------------------------------------------
# Check 3: every code-bearing lesson file has the "Full code" heading
# ---------------------------------------------------------------------------
def check_full_code_heading():
    heading = "## Full code for this lesson"
    violations = []
    for name in CODE_BEARING_LESSONS:
        path = LESSONS_DIR / name
        if not path.exists():
            violations.append(f"lessons/{name}: file does not exist")
            continue
        if heading not in path.read_text(encoding="utf-8"):
            violations.append(f"lessons/{name}: missing heading '{heading}'")
    report(
        "lessons/01-06 each contain '## Full code for this lesson'",
        not violations,
        violations,
    )


# ---------------------------------------------------------------------------
# Check 4: notebook is valid JSON, nbformat 4, every code cell compiles
# ---------------------------------------------------------------------------
def check_notebook_valid():
    violations = []
    try:
        nb = load_notebook_json(NOTEBOOK_PATH)
    except (OSError, json.JSONDecodeError) as e:
        report("mini_model_full_notebook.ipynb is valid JSON / nbformat 4 / compiles", False,
               [f"failed to load/parse {NOTEBOOK_PATH.name}: {e}"])
        return None

    if nb.get("nbformat") != 4:
        violations.append(f"nbformat is {nb.get('nbformat')!r}, expected 4")

    for i, cell in enumerate(nb.get("cells", [])):
        if cell.get("cell_type") != "code":
            continue
        src = cell_source_text(cell)
        try:
            compile(src, f"<notebook cell {i}>", "exec")
        except SyntaxError as e:
            violations.append(f"cell {i}: SyntaxError: {e}")

    report(
        "mini_model_full_notebook.ipynb is valid JSON, nbformat 4, all code cells compile",
        not violations,
        violations,
    )
    return nb


# ---------------------------------------------------------------------------
# Check 5: every comment line in each original cell is present in the
# matching notebook cell / lesson "Full code" block
# ---------------------------------------------------------------------------
def check_comments_preserved(nb):
    if not ORIGINAL_NOTEBOOK_PATH.exists():
        report(
            "All comment lines from mini_model.ipynb preserved in notebook + lessons",
            False,
            [f"original notebook not found at {ORIGINAL_NOTEBOOK_PATH}"],
        )
        return

    orig = load_notebook_json(ORIGINAL_NOTEBOOK_PATH)
    orig_cells = [c for c in code_cells(orig) if c.strip()]

    violations = []

    if nb is not None:
        nb_code_cells = code_cells(nb)
        if len(nb_code_cells) != len(orig_cells):
            violations.append(
                f"notebook has {len(nb_code_cells)} code cells, original has {len(orig_cells)}"
            )
        for i, orig_src in enumerate(orig_cells):
            if i >= len(nb_code_cells):
                break
            nb_lines = set(l.strip() for l in nb_code_cells[i].splitlines())
            for c in comment_lines(orig_src):
                if c not in nb_lines:
                    violations.append(f"notebook cell {i}: missing comment line: {c!r}")

    lesson_map = [
        ("01_what_is_a_jet.md", [0]),
        ("02_finding_the_truth_labels.md", [1]),
        ("03_preparing_the_data.md", [2]),
        ("04_building_mini_part.md", [3]),
        ("05_training_the_model.md", [4]),
        ("06_evaluating_the_model.md", [5, 6, 7, 8]),
    ]
    for fname, cell_idxs in lesson_map:
        path = LESSONS_DIR / fname
        if not path.exists():
            violations.append(f"lessons/{fname}: file does not exist")
            continue
        blocks = extract_python_blocks(path.read_text(encoding="utf-8"), "## Full code for this lesson")
        for bi, ci in enumerate(cell_idxs):
            if ci >= len(orig_cells):
                continue
            if bi >= len(blocks):
                violations.append(f"lessons/{fname}: missing code block for original cell {ci}")
                continue
            block_lines = set(l.strip() for l in blocks[bi].splitlines())
            for c in comment_lines(orig_cells[ci]):
                if c not in block_lines:
                    violations.append(f"lessons/{fname} block {bi} (cell {ci}): missing comment line: {c!r}")

    report(
        "All comment lines from mini_model.ipynb preserved in notebook + lessons",
        not violations,
        violations,
    )


# ---------------------------------------------------------------------------
# Check 6: README.md repo structure tree lists the notebook + check_entries.py
# ---------------------------------------------------------------------------
def check_readme_tree():
    text = README_PATH.read_text(encoding="utf-8")
    m = re.search(r"## Repo structure\s*\n```(.*?)```", text, re.DOTALL)
    tree = m.group(1) if m else ""
    violations = []
    if "mini_model_full_notebook.ipynb" not in tree:
        violations.append("repo structure tree does not list mini_model_full_notebook.ipynb")
    if "check_entries.py" not in tree:
        violations.append("repo structure tree does not list code/check_entries.py")
    report(
        "README.md repo structure tree lists the notebook and code/check_entries.py",
        not violations,
        violations,
    )


# ---------------------------------------------------------------------------
# Check 7: datasets/README.md mentions check_entries.py
# ---------------------------------------------------------------------------
def check_datasets_readme_mentions_check_entries():
    text = DATASETS_README_PATH.read_text(encoding="utf-8")
    ok = "check_entries.py" in text
    report(
        "datasets/README.md mentions check_entries.py",
        ok,
        [] if ok else ["'check_entries.py' not found in datasets/README.md"],
    )


# ---------------------------------------------------------------------------
# Check 8: README.md and datasets/README.md both mention Jupyter and Colab
# ---------------------------------------------------------------------------
def check_jupyter_colab_mentions():
    violations = []
    for path in (README_PATH, DATASETS_README_PATH):
        text = path.read_text(encoding="utf-8")
        rel = path.relative_to(REPO_ROOT)
        if "Jupyter" not in text:
            violations.append(f"{rel}: does not mention 'Jupyter'")
        if "Colab" not in text:
            violations.append(f"{rel}: does not mention 'Colab'")
    report(
        "README.md and datasets/README.md both mention Jupyter and Colab",
        not violations,
        violations,
    )


# ---------------------------------------------------------------------------
# Check 9: lessons/07_complete_code.md exists and matches the notebook's
# 9 code cells exactly, in order, including comments
# ---------------------------------------------------------------------------
def check_lesson07(nb):
    if not LESSON07_PATH.exists():
        report("lessons/07_complete_code.md exists and matches the notebook's 9 code cells", False,
               ["lessons/07_complete_code.md does not exist"])
        return

    violations = []
    lesson_blocks = extract_python_blocks(LESSON07_PATH.read_text(encoding="utf-8"))

    if nb is None:
        violations.append("notebook could not be loaded, cannot compare")
        report("lessons/07_complete_code.md exists and matches the notebook's 9 code cells", False, violations)
        return

    nb_cells = code_cells(nb)

    if len(lesson_blocks) != 9:
        violations.append(f"expected 9 code blocks in lesson 07, found {len(lesson_blocks)}")
    if len(nb_cells) != 9:
        violations.append(f"expected 9 code cells in the notebook, found {len(nb_cells)}")

    for i in range(min(len(lesson_blocks), len(nb_cells))):
        a = lesson_blocks[i].rstrip("\n")
        b = nb_cells[i].rstrip("\n")
        if a != b:
            violations.append(f"block {i}: lesson 07 code does not exactly match notebook cell {i}")

    report(
        "lessons/07_complete_code.md exists and matches the notebook's 9 code cells exactly, in order",
        not violations,
        violations,
    )


def main():
    check_no_em_dash()
    check_no_banned_phrases()
    check_full_code_heading()
    nb = check_notebook_valid()
    check_comments_preserved(nb)
    check_readme_tree()
    check_datasets_readme_mentions_check_entries()
    check_jupyter_colab_mentions()
    check_lesson07(nb)

    print()
    if failures_total:
        print(f"{failures_total} check(s) FAILED")
        sys.exit(1)
    else:
        print("All checks PASSED")
        sys.exit(0)


if __name__ == "__main__":
    main()
