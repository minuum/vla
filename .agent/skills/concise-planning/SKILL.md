---
name: concise-planning
description: Enforce a structured planning phase before execution, generating checklists and risk assessments.
---

# Concise Planning Skill

## Value Proposition
This skill ensures that every complex task starts with a clear, step-by-step plan. It prevents "jumping into code" without understanding the implications, especially for new features or major refactors.

## When to Use
- **Start of a new feature**: When the user asks for a new capability (e.g., "Add a retry mechanism").
- **Complex Refactor**: When modifying core logic that affects multiple files.
- **Ambiguous Requests**: When the user's request is high-level (e.g., "Improve performance").

## Instructions
1.  **Analyze**: Before writing any code, analyze the user's request and the current codebase state.
2.  **Plan**: create or update `implementation_plan.md` using the standard template.
    -   **Goal**: What are we trying to achieve?
    -   **Proposed Changes**: List specific files and functions to modify.
    -   **Verification**: How will we know it worked?
3.  **Review**: If the task is complex, explicitly ask the user to review the plan via `notify_user`.

## Best Practices
-   Keep the plan **concise**. Bullet points are better than paragraphs.
-   Identify **risks** early (e.g., "This change might break the API for v1 clients").
-   Define a **rollback strategy** if things go wrong.
