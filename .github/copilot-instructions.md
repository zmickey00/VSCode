<!--
Project appears to have no existing AI-agent guidance files in the workspace.
This file is a compact, actionable guide to help AI coding agents be immediately productive
once the codebase is present. It describes discovery steps, places to look, and the
project-specific behaviors agents should follow. Update with concrete examples once
source files (package manifests, source roots, CI, etc.) exist.
-->

# Copilot / AI agent instructions (starter)

Purpose
- Provide concise, project-specific guidance for an AI coding agent to explore,
  reason about, and implement changes in this repository.

Quick facts
- Repository currently contains no source files (no package manifests detected).
- Primary action for the agent: discover project type, locate entrypoints, then
  infer build/test/debug commands and coding conventions.

Discovery checklist (run in this order)
- Look for language manifests (examples): `package.json`, `pyproject.toml`, `requirements.txt`, `go.mod`, `Cargo.toml`.
- Identify source roots: `src/`, `app/`, `lib/`, `pkg/`, or language-specific layouts.
- Look for Dockerfile, `.github/workflows/`, `Makefile`, or `Procfile` for CI/build commands.
- Find tests in `tests/`, `spec/`, `__tests__/`, or language-specific test patterns.
- If present, open top-level `README.md` and `CHANGELOG.md` for architectural context.

How to infer the "big picture"
- If a `package.json` exists: prefer `scripts` to learn build/test commands.
- If `pyproject.toml` or `setup.cfg` exists: locate `tool.poetry`, `tool.flit`, or `setup.cfg` sections.
- If `Dockerfile` or `Procfile` present: inspect to understand run-time services and ports.
- Map components by following imports/exports across files to identify service boundaries.

Project-specific conventions (starter rules)
- Favor small, focused pull requests: change one feature or bug at a time.
- Preserve existing formatting and linter rules; look for `.editorconfig`, `.prettierrc`, `.clang-format`.
- Tests: run local unit tests first; if none found, ask the user where tests live.

Developer workflows (what to try first)
- Discover build/test commands from manifests or CI workflow; if none, ask the user.
- Run unit tests before changing behavior; report failures and root-cause hypotheses.
- When adding code, include minimal tests demonstrating the behavior.

Integration points to inspect
- External services: check for `*.env`, `.env.example`, `config/`, or `settings` files indicating external APIs or DBs.
- Database migrations: look for `migrations/`, `alembic/`, `schema.sql`.
- Message queues or background workers: search for `rabbitmq`, `kafka`, `celery`, `sidekiq` in code or CI.

What the agent should ask the human
- "Point me to the files that implement the core service or the top-level `README.md` to infer architecture." 
- "Which commands do you use locally to build, test, and run the app?"
- "Are there private registry credentials, or secrets I should avoid touching?"

When to modify this file
- Replace this starter content with concrete examples once source files are present.
- Add snippets showing the exact `npm`, `pip`, `cargo`, `go`, or other commands discovered in the repo.

Next steps for the user
- If you want an AI to proceed now, point me to the repository root files (package manifest, README, or src directory).
