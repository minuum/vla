---
name: git-collaboration
description: Enforce structured Git collaboration practices for shared repositories, including knowledge sharing, environment synchronization, and experiment history tracking.
---

# Git Collaboration and Knowledge Sharing Skill

## Value Proposition
When multiple researchers or servers share the same Git repository, maintaining a consistent state of knowledge, environment, and experimental history is critical. This skill ensures that essential context is committed to the repository in a structured format, allowing seamless handover and synchronization between different environments (e.g., Training Server <-> Inference Jetson).

## When to Use
- **Before Pushing**: When preparing to push changes that include new experimental results, configuration changes, or architectural decisions.
- **Environment Setup**: When setting up a new environment (e.g., pulling the repo on a new Jetson).
- **Handover**: When finishing a session and wanting to leave a clear state for the next researcher (or yourself on another machine).

## Instructions

### 1. Mandatory Documentation Check (The "Knowledge Sync")
Before any significant push, ensure the following Markdown files are up-to-date in the `docs/` directory. If they don't exist, create them.

#### A. Experiment Master List (`docs/ALL_EXPERIMENTS_MASTER_LIST.md`)
- **Purpose**: Single source of truth for all experiments.
- **Content**: Table with columns `[Exp ID, Description, Config Summary, Status, Result Metrics, Key Insight]`.

#### B. Deployment Guide (`docs/JETSON_DEPLOYMENT_GUIDE.md` or similar)
- **Purpose**: Instructions for deploying specific models to edge devices.
- **Content**: Exact paths to weights, config files, required dependencies, and run commands.

#### C. Current State Snapshot (`docs/CURRENT_STATE_SNAPSHOT.md`)
- **Purpose**: A "sticky note" for the next person.
- **Content**:
    - **Latest Commit**: What was just changed?
    - **Active Model**: Which model is currently the "Champion"?
    - **Next Steps**: What should be done next? (e.g., "Run EXP-12 inference test")
    - **Known Issues**: Any bugs or caveats?

### 2. Environment Synchronization (`docs/ENV_SYNC.md`)
- **Purpose**: Ensure `requirements.txt` or environment variables are consistent.
- **Content**:
    - List of key environment variables (e.g., `VLA_MODEL_NAME`, `VLA_API_KEY`).
    - Specific library versions if critical (e.g., `torch==2.1.0`).
    - **Secrets Handling**: Explicitly mention *where* to find secrets (e.g., "See `secrets.sh.template`", never commit real secrets).

### 3. Git Operations
- **Ignore Large Files**: Always verify `.gitignore` covers new large artifacts (`*.h5`, `*.ckpt`, `*.pth`, `logs/*.log`). Use `git check-ignore -v [file]` to test.
- **Commit Messages**: Use structured commit messages:
    - `feat:` for new features/models.
    - `docs:` for documentation updates.
    - `fix:` for bug fixes.
    - **Body**: Include the `Exp ID` and key metrics if relevant (e.g., "EXP-17 reached 94.72% accuracy").

## Best Practices
- **Relative Paths**: Always use relative paths in documentation and scripts (e.g., `./scripts/train.sh` instead of `/home/user/project/...`) to ensure portability.
- **No Hardcoded Secrets**: double-check for API keys or passwords before adding files.
- **Self-Contained Scripts**: Scripts should check for their dependencies or environment variables and fail gracefully with helpful messages.
