---
name: lint-and-validate
description: Ensure code quality before commits by running linters and checking file sizes.
---

# Lint and Validate Skill

## Value Proposition
Prevents "broken builds" and "garbage code" from entering the repo. Ensures that large files (common in ML) don't accidentally get committed to git without LFS.

## When to Use
-   **Pre-Commit**: Before running `git commit`.
-   **Code Review**: When the user asks to "check my code".
-   **New Files**: When creating new Python scripts.

## Instructions
1.  **Linting**:
    -   Check for basic syntax errors `python -m py_compile [file]`.
    -   If `flake8` or `pylint` is available, run it on modified files.
2.  **File Size Check**:
    -   **CRITICAL**: Check if any new file is >50MB.
    -   If >50MB, check if it's tracked by Git LFS (`git lfs track "[pattern]"`).
    -   If not tracked, warn the user and suggest adding to `.gitignore` or LFS.
3.  **Auto-Fix**: If the error is simple (e.g., trailing whitespace, missing newline), fix it automatically.

## Best Practices
-   **Passive Safety**: Don't block the user if they explicitly override, but always warn.
-   **Scope**: Only lint files that have been modified or created in the current task.
