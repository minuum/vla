---
name: experiment-tracking
description: Standardize the logging of VLA experiments, ensuring data-driven insights and reproducibility.
---

# Experiment Tracking Skill

## Value Proposition
Systematic tracking of VLA (Vision-Language-Action) model experiments is crucial for research. This skill ensures that every experiment is logged with its configuration, results, and insights, making it easier to write papers later.

## When to Use
-   **After Training**: When a training run completes.
-   **After Evaluation**: When evaluation metrics (success rate, accuracy) are available.
-   **Insights**: When the user discusses findings from a specific experiment ID (e.g., "EXP-17").

## Instructions
1.  **Target File**: The primary log file is `docs/EXPERIMENT_HISTORY_AND_INSIGHTS.md`.
2.  **Format**: Always use a consistent table format for results.
    ```markdown
    | Exp ID | Model | Epochs | Success Rate | Failure Modes | Commit Hash |
    | :--- | :--- | :--- | :--- | :--- | :--- |
    | EXP-XX | [Model Name] | [N] | [XX.X]% | [Brief Description] | [Short Hash] |
    ```
3.  **Argumentation**: When adding insights, follow the "Claim -> Evidence" structure.
    -   **Claim**: "Model X performs better on initial frames."
    -   **Evidence**: "Analysis of `initial_frame_accuracy.json` shows 94% accuracy vs 82% for baseline."
4.  **Formatting**: Ensure all numbers are formatted consistently (e.g., 2 decimal places for percentages).

## Best Practices
-   **Link to Artifacts**: Always link to the raw log files or JSON results.
-   **Auto-Update**: If the user provides a new log file, parse it and update the history table automatically.
-   **Consistency**: Ensure `Exp ID` is unique and sequential or descriptive.
