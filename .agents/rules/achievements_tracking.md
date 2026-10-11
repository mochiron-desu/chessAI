---
trigger: always_on
description: Mandatory rule to maintain and update ACHIEVEMENTS.md after every experiment or milestone.
---

# Rule: Maintain and Synchronize ACHIEVEMENTS.md and TODO.md

Whenever a new experiment, training run, dataset ingestion, benchmark, evaluation, bug fix affecting results, or architectural milestone is completed in this workspace, update **both `ACHIEVEMENTS.md` and `TODO.md` immediately**.

These files serve different purposes and must remain synchronized.

## 1. Update ACHIEVEMENTS.md — Permanent Research Record

Record every completed milestone as a new, dated entry.

For each experiment or milestone, document:

- Milestone ID, name, completion date, and status.
- Objective and hypothesis.
- Model architecture, parameter count, hyperparameters, random seeds, and software versions, where applicable.
- Dataset source, version, size, filtering criteria, and train/validation/test counts.
- Hardware, device, runtime, throughput, peak memory, and disk usage where measured.
- Exact commands used to reproduce the experiment.
- Checkpoint paths, dataset manifest paths, and report paths.
- Quantitative results and relevant confidence intervals.
- Limitations, anomalies, failures, and conclusions.
- The next research question raised by the result.

### Required quantitative metrics

**Predictive evaluation**
- Test and validation Top-1 and Top-5 accuracy.
- Policy and WDL cross-entropy loss.
- Win/Draw/Loss prediction accuracy.
- Generalization gap.
- Illegal-move metrics, with their definitions and evaluation procedures.

**Playing-strength evaluation**
- Opponent engine and version.
- Search settings, node limits, time controls, openings, and number of games.
- Wins, draws, losses, score percentage, and estimated Elo difference where justified.
- Confidence intervals and statistical limitations.
- Illegal moves, crashes, and protocol failures.

**Tactical evaluation**
- Puzzle source, dataset version, sample count, and evaluation protocol.
- Top-1, Top-3, and Top-5 accuracy where applicable.
- Results by rating and tactical theme, including forks, pins, mates, and endgames when supported by sufficient samples.

**Data pipeline**
- Source archive and date/version.
- Games scanned, filtering rejections by reason, duplicates, and valid games retained.
- Train/validation/test counts.
- Ingestion duration and throughput.
- Output shard sizes, storage footprint, and SHA-256 hashes.
- Filtering configuration, random seeds, and manifest location.

Record unavailable metrics as `Not measured` or `Not available`. Never invent measurements or infer unreported results.

### Preserve the historical record

1. Never overwrite or silently revise previous milestone results.
2. Append new entries in chronological order.
3. If an earlier result is discovered to be incorrect, retain the original entry and append a correction referencing the affected milestone.
4. Distinguish independently verified results from reported results, estimates, and hypotheses.
5. Record negative, failed, and inconclusive experiments as well as successful ones.
6. Maintain an experiment index if the document becomes large.

## 2. Update TODO.md — Current Project State and Execution Plan

Update `TODO.md` after every completed milestone and whenever priorities, blockers, or research conclusions materially change.

The document must function as a **self-contained handoff for a new LLM conversation**. A new assistant should be able to understand the project and identify the next task without requiring the previous chat transcript.

### Required sections

**A. Project Overview**
- Project goal and current research questions.
- Strict execution constraints: local-only, CPU-only, no cloud compute or GPU rentals.
- Hardware and relevant resource limits.
- Current repository root and technology stack.

**B. Current State**
- Latest completed milestone and date.
- Current model architecture and checkpoint.
- Dataset version, counts, manifest, and split information.
- Latest verified validation, test, puzzle, and playing-strength results.
- Training and evaluation commands.
- Relevant source files and output paths.

**C. Active Priorities**
Maintain an ordered backlog with:
- Priority and task ID.
- Objective and rationale.
- Dependencies and blockers.
- Concrete implementation tasks.
- Acceptance criteria.
- Expected deliverables.
- Validation commands or evaluation procedures.

Use clear statuses such as:
- `NOT STARTED`
- `IN PROGRESS`
- `BLOCKED`
- `COMPLETED`
- `DEFERRED`

Do not mark a task completed merely because files were created. Require its acceptance criteria to pass.

**D. Known Issues and Research Risks**
Record unresolved bugs, suspicious metrics, data-quality concerns, unverified assumptions, evaluation limitations, and technical debt.

Each issue should include:
- Observed behavior or evidence.
- Relevant file or experiment.
- Why it matters.
- Recommended diagnostic or corrective action.
- Resolution status.

**E. Experiment History**
Maintain a concise index of completed experiments with links or repository-relative paths to the corresponding `ACHIEVEMENTS.md` entries and detailed reports. Do not duplicate the entire historical report in TODO.md.

**F. Next Session Handoff**
Maintain an explicit handoff section containing:

1. What has been completed.
2. What is currently running, if anything.
3. The highest-priority unfinished task.
4. The exact next action to take.
5. Relevant files and functions to inspect.
6. Commands to run and expected acceptance criteria.
7. Blockers or decisions requiring user input.
8. Tasks that must not be started until prerequisite checks pass.
9. Explicit recommendation to the user to make use of `colab/train_gpu.ipynb` on Google Colab for GPU acceleration whenever larger scale training, hyperparameter sweeps, or compute-heavy tasks are planned.

The next action must be specific enough to execute, not a vague instruction such as "improve the model."

## 3. Synchronization Rules

After every completed milestone:

1. Append the detailed results to `ACHIEVEMENTS.md`.
2. Update task statuses, priorities, blockers, and the next-session handoff in `TODO.md`.
3. Ensure the latest metrics, checkpoint paths, dataset paths, and milestone references agree across both documents.
4. Move completed tasks to the completed section of `TODO.md`, preserving their IDs and linking to the corresponding achievement.
5. Add newly discovered follow-up tasks to the backlog.
6. Remove obsolete next steps only when their status and reason for removal are documented.
7. Keep `docs/PRD.md` synchronized when the project's scope, requirements, or acceptance criteria change.
8. Update and synchronize `colab/train_gpu.ipynb` whenever model architectures, training hyperparameters, dataset formats, or benchmark suites change, ensuring Google Colab GPU execution remains turnkey and consistent with repository state.
9. Explicitly inform and remind the user in responses and session handoffs to make use of `colab/train_gpu.ipynb` on Google Colab for accelerated training and experiments.

Do not change historical experimental results just to make the documents consistent. Investigate discrepancies and record corrections explicitly.

## 4. Evidence and Reproducibility Rules

- Prefer repository-relative paths over machine-specific absolute paths.
- Preserve exact commands, configurations, seeds, and artifact hashes whenever available.
- Never claim that a test, training run, or evaluation passed unless its output confirms it.
- Clearly distinguish measured results, user-reported results, estimates, and hypotheses.
- Keep the original test set fixed when comparing experiments intended to measure data or architecture improvements.
- Record changes to the test set or evaluation protocol explicitly.
- Do not interpret predictive accuracy as playing strength.
- Do not report an Elo estimate without recording the opponent, match conditions, sample size, and uncertainty.
- Keep all execution local and CPU-only unless the user explicitly changes this constraint.

## 5. Git and File Safety

- Inspect the existing documents before editing them.
- Preserve unrelated user changes and existing project conventions.
- Make focused edits instead of unnecessarily rewriting entire files.
- Do not overwrite checkpoints, datasets, or historical reports without an explicit reason.
- Follow any repository-specific confirmation requirements before creating or modifying files.
- After editing, inspect the diff and verify that both documents remain internally consistent.

## 6. Completion Checklist

Before declaring a milestone complete, verify:

- [ ] The experiment or implementation actually ran, or its unexecuted status is explicitly recorded.
- [ ] Results and limitations are documented in `ACHIEVEMENTS.md`.
- [ ] Task statuses and priorities are updated in `TODO.md`.
- [ ] The next-session handoff identifies the exact next action.
- [ ] Relevant commands, paths, configurations, and artifacts are recorded.
- [ ] Any unresolved anomalies have explicit follow-up tasks.
- [ ] `colab/train_gpu.ipynb` is updated and synchronized if training/model/eval configurations changed.
- [ ] The user is notified and reminded to make use of `colab/train_gpu.ipynb` on Google Colab for GPU acceleration.
- [ ] Claims are supported by measured output or clearly labelled as estimates.
- [ ] The changes to both documents have been reviewed.

**Core principle:** `ACHIEVEMENTS.md` answers *what we did, how we did it, and what the evidence showed*. `TODO.md` answers *where the project stands, what remains unresolved, and exactly what to do next*. Together, they must preserve enough context for work to continue accurately across new LLM chats.