# Workspace Rules: ChessLab Research Framework

## Rule 1: Mandatory Achievements Tracking (`ACHIEVEMENTS.md`)

The file `ACHIEVEMENTS.md` at the repository root is the project's single source of truth for all completed engineering milestones, training experiments, benchmarks, and data pipelines.

Whenever you or the user complete a new engineering task, model training run, benchmark evaluation, or data pipeline execution:

1. **Immediate Update to `ACHIEVEMENTS.md`**:
   - Log the newly completed milestone before finishing the response or concluding the turn.
   - Record the timestamp/date, configuration parameters, dataset size, compute runtime, and empirical metrics.
2. **Quantitative Precision**:
   - Always record exact numeric measurements:
     - Predictive: Top-1 accuracy, Top-5 accuracy, Cross-entropy loss, WDL accuracy, Generalization gap.
     - Strength: Head-to-head score, win/draw/loss counts, estimated Delta Elo with 95% confidence intervals, illegal move rate.
     - Benchmarks: Tactical puzzle accuracy overall and by theme (fork, pin, mate).
     - Pipeline: Games scanned, filter rejection rate, duplicates, valid kept, ingestion throughput (games/sec), storage bytes, and SHA-256 shard hashes.
3. **Preserve Historical Continuity**:
   - Never silently delete or overwrite past milestone entries.
   - Append new milestones chronologically and update the active status table.

---

## Rule 2: Mandatory Task & Backlog Tracking (`TODO.md`)

The file `TODO.md` at the repository root is the project's live task list and research backlog.

Whenever new work is planned, started, or completed:

1. **Keep `TODO.md` Synchronized**:
   - When a task is completed, mark it as `[x]` and archive it with a reference to `ACHIEVEMENTS.md`.
   - When new research questions, user requests, or engineering tasks are identified, add them to the appropriate section ("Immediate Active Focus", "Upcoming Research Milestones", or "Engineering Backlog") with a concrete goal and deliverable.
   - Maintain clear task IDs (`TODO-XX`, `DONE-XX`).
2. **Focus Priority**:
   - Always keep the top 1–2 items in Section 1 ("Immediate Active Focus") aligned with the active conversation goals.

---

## Rule 3: Git Safety Compliance

In accordance with global git safety rules, do not run `git commit`, `git push`, or alter git state without explicit user confirmation.
