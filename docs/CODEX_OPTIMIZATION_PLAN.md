# DeepRevision Codex Optimization Plan

> Branch target: `dev`
>
> This document is written for Codex to execute repository optimizations incrementally.
> The goal is **not** to rewrite the whole project, but to extend the current architecture with the highest-value learning-system features while keeping the existing chat / RAG / quiz / exam / planner flows stable.

---

## 0. Current Baseline

The current repository already has these capabilities:

- FastAPI chat entry with SSE / `stream_v2`
- Supervisor-based multi-route architecture: `rag / quiz / exam / ops / planner / history / chitchat`
- Hybrid RAG: BM25 + vector + RRF + rewrite + rewrite_guard + fast/full + HyDE + context cache + semantic cache
- Quiz / Exam generation with critique / revise workflow and quality guards
- Practice record submission, history, similar-question retrieval, weak-point stats
- Dual-track memory: recent / trash / graph nodes
- User profile / persona injection

The next step is to evolve DeepRevision from a "reactive AI assistant" into a **data-driven personalized revision system**.

---

## 1. Execution Principles

Codex must follow these principles while implementing changes:

1. **Do not break existing routes**. Existing endpoints and payload shapes should remain backward compatible unless explicitly stated.
2. **Prefer additive changes** over destructive refactors.
3. **Keep session isolation intact**. All new learning data must remain scoped by `session_id` unless explicitly designed as cross-session user data.
4. **Do not rewrite the whole memory system**. Extend the existing SQLite-backed memory and stats pipeline.
5. **Add instrumentation** for every major new feature.
6. **Implement in phases**. Each phase should be shippable independently.

---

## 2. Optimization Roadmap

Execution order:

1. Mastery score model
2. Spaced repetition scheduler
3. Review endpoints + planner integration
4. Coverage dashboard
5. Section-aware chunk metadata
6. Clickable citations
7. Sprint pack generator

The first 4 are the highest priority.

---

## Phase 1 — Mastery Score Model

### Goal
Replace the current weak-point-only logic with a richer **knowledge point mastery model**.

### Why
The current system mainly relies on accuracy / weak_points. That is useful but coarse. We need a score that reflects:

- historical correctness
- recent wrong streaks
- recency / forgetting
- review urgency

### Files to inspect / modify

Primary:

- `utils/memory_service.py`
- `api/routers/chat.py`
- `agent/multi_agent/supervisor.py`

Possible new file:

- `utils/mastery_service.py`

### Proposed data model
Add a new per-session table, for example:

```sql
CREATE TABLE IF NOT EXISTS knowledge_mastery (
  session_id TEXT NOT NULL,
  knowledge_point TEXT NOT NULL,
  attempt_count INTEGER NOT NULL DEFAULT 0,
  correct_count INTEGER NOT NULL DEFAULT 0,
  recent_wrong_streak INTEGER NOT NULL DEFAULT 0,
  recent_correct_streak INTEGER NOT NULL DEFAULT 0,
  last_seen_at INTEGER NOT NULL DEFAULT 0,
  last_correct_at INTEGER,
  last_wrong_at INTEGER,
  mastery_score REAL NOT NULL DEFAULT 0,
  next_review_at INTEGER,
  PRIMARY KEY (session_id, knowledge_point)
);
```

### Mastery scoring rule (initial version)
Codex should implement a simple, explainable formula first:

- `accuracy_component = correct_count / max(attempt_count, 1)`
- `streak_penalty = min(recent_wrong_streak * 0.12, 0.36)`
- `recency_penalty = min(days_since_last_seen * 0.015, 0.20)`
- `mastery_score = clamp(accuracy_component - streak_penalty - recency_penalty, 0, 1)`

This does not need to be perfect yet. It just needs to be stable and inspectable.

### Required code changes

1. When practice records are submitted, update mastery stats for each normalized knowledge point.
2. Provide a helper like:

- `get_mastery_snapshot(session_id)`
- `get_priority_review_points(session_id, limit=10)`
- `upsert_mastery_from_practice_record(...)`

3. Update planner and history logic so they can consume mastery-based ordering.

### Acceptance criteria

- After submitting practice records, mastery rows are created / updated.
- A knowledge point that is repeatedly answered wrong gets a lower mastery score.
- A knowledge point with old stale history becomes more review-urgent than a recently reviewed one at the same accuracy.
- Existing `knowledge_point_stats` API behavior remains available.

### Non-goals

- No machine learning model
- No heavy Bayesian estimation yet

---

## Phase 2 — Spaced Repetition Scheduler

### Goal
Introduce a **review schedule** per knowledge point so the system can answer: "what should the student review today?"

### Files to inspect / modify

Primary:

- `utils/memory_service.py`
- `agent/multi_agent/supervisor.py`
- `api/routers/chat.py`

Possible new file:

- `utils/review_scheduler.py`

### Proposed data model
Add a table:

```sql
CREATE TABLE IF NOT EXISTS review_schedule (
  session_id TEXT NOT NULL,
  knowledge_point TEXT NOT NULL,
  interval_days INTEGER NOT NULL DEFAULT 1,
  ease_factor REAL NOT NULL DEFAULT 2.5,
  next_review_at INTEGER NOT NULL,
  last_review_at INTEGER,
  due_state TEXT NOT NULL DEFAULT 'scheduled',
  PRIMARY KEY (session_id, knowledge_point)
);
```

### Scheduling policy (simple initial version)

- Wrong answer:
  - interval resets to 1 day
  - next_review_at = now + 1 day
- Correct answer on low mastery point:
  - interval = min(previous_interval * 2, 14)
- Correct answer on stable point:
  - interval = min(previous_interval * ease_factor, 21)

### Required code changes

1. On practice submit, update review schedule for each affected knowledge point.
2. Add helper methods:

- `get_due_review_points(session_id, now_ts=None, limit=20)`
- `get_upcoming_review_points(session_id, within_days=7)`

3. Planner should prefer due review points before general weak points.
4. History / analysis report should surface due points.

### Acceptance criteria

- Repeatedly wrong knowledge points get short review intervals.
- Well-mastered points get longer intervals.
- There is a stable API helper that returns "due today" points.

---

## Phase 3 — Review Endpoints and Planner Integration

### Goal
Expose spaced repetition and mastery to the application layer.

### Files to inspect / modify

- `api/routers/chat.py`
- `agent/multi_agent/supervisor.py`

Optional new router:

- `api/routers/review.py`

### New endpoints to add

#### 1. GET `/review/today`
Return:

- due knowledge points
- mastery score
- suggested action

#### 2. GET `/review/upcoming`
Return next 7 days review schedule.

#### 3. GET `/mastery`
Return sorted mastery table for a session.

### Planner integration requirements

In `planner_subagent_node`:

1. Merge current practice snapshot with due-review snapshot.
2. Planner should prioritize:
   - due today points
   - low mastery points
   - then uncovered points if available
3. The markdown plan should visibly distinguish:
   - review tasks
   - practice tasks
   - recap tasks

### Acceptance criteria

- `/review/today` returns actionable data.
- Planner output changes when due reviews exist.
- Existing planner payload remains backward compatible.

---

## Phase 4 — Knowledge Coverage Dashboard

### Goal
Add a course/session-level **coverage view**.

### Why
The project already knows what the student practiced. It does not yet clearly show what was never covered.

### Files to inspect / modify

- `utils/memory_service.py`
- `agent/multi_agent/quiz_agent.py`
- `api/routers/chat.py`

Potential new file:

- `utils/coverage_service.py`

### Required data sources
Use existing and new sources:

- extracted topic pool / dynamic topics from quiz/exam generation
- question bank knowledge points
- mastery table
- practice history

### New output shape
A coverage snapshot should contain:

- `total_known_points`
- `practiced_points`
- `mastered_points`
- `weak_points`
- `uncovered_points`
- `coverage_ratio`

### API suggestion
Add:

- GET `/coverage`

### Acceptance criteria

- Coverage can be computed per session.
- The system can list uncovered points.
- This endpoint does not break existing APIs.

---

## Phase 5 — Section-Aware Chunk Metadata

### Goal
Improve retrieval quality by making chunks more structure-aware.

### Current limitation
Chunking is already decent, but still mostly based on recursive character splitting after file-to-text extraction. We need better metadata for downstream retrieval and citation.

### Files to inspect / modify

- `utils/file_handler.py`
- `rag/vector_store.py`
- `rag/rag_service.py`
- `config/chroma.yml`

### Required improvement
Do **not** rewrite the whole parser. Instead:

1. Preserve / infer a `section_title` where possible.
2. Add `chunk_type` when obvious:
   - `definition`
   - `example`
   - `table`
   - `formula`
   - `procedure`
   - `general`
3. Persist `chunk_index` and `section_title` in metadata.

### Implementation hint
For PPT/PDF chunks:

- use page title or nearest heading-like line as `section_title`
- if content contains `【表格】`, mark as `table`
- if content contains formula markers, mark as `formula`
- if content contains steps / process language, mark as `procedure`

### Acceptance criteria

- New chunks include richer metadata.
- RAG context formatting can show section title when present.
- Existing retrieval pipeline still works.

---

## Phase 6 — Clickable Citation Support

### Goal
Upgrade citations from "reference blocks" to precise, clickable source anchors.

### Files to inspect / modify

- `rag/rag_service.py`
- `api/routers/chat.py`
- `api/message_protocol.py`
- possibly front-end payload contracts if already mirrored here

### Required metadata
Ensure retrieval results carry:

- `source_filename`
- `page` or `slide`
- `chunk_index`
- `section_title`

### Required response upgrades
For citation payloads, return:

```json
{
  "source": "os-final-review.pdf",
  "page": 12,
  "chunk_index": 5,
  "section_title": "进程调度",
  "quote": "..."
}
```

### Acceptance criteria

- Citation payload contains enough information for a front-end jump.
- Grounded evidence logic remains intact.
- Old clients can still read `source` + `quote`.

---

## Phase 7 — One-Click Sprint Pack

### Goal
Create a high-visibility product feature: **generate a final revision sprint pack**.

### Why
This turns current underlying capabilities into a single compelling outcome.

### Suggested route
Either:

- add a new supervisor route `sprint_pack`
- or implement as an `ops` / `planner` style callable action first

Recommended first step: implement as an `ops`-style action or planner extension to reduce routing risk.

### Contents of sprint pack
Generate a structured artifact containing:

1. top weak knowledge points
2. must-review concepts
3. a 7-day sprint plan
4. a mini quiz set
5. common mistakes summary
6. optionally a markdown export payload

### Files to inspect / modify

- `agent/multi_agent/supervisor.py`
- `agent/multi_agent/quiz_agent.py`
- `api/routers/chat.py`
- `utils/memory_service.py`

Possible new file:

- `agent/multi_agent/sprint_pack.py`

### Acceptance criteria

- Can generate a markdown sprint pack from current session data.
- Works even when practice history is sparse (with graceful fallback).
- Output is structured enough to export later.

---

## 3. Metrics to Add

Codex should extend observability for new features.

### Add counters / samples for

- `mastery_rows_total`
- `review_due_count`
- `review_completed_count`
- `coverage_ratio`
- `sprint_pack_generated_total`
- `citation_with_page_count`

### Where
- extend runtime metrics in `api/routers/chat.py`
- keep helper functions small and composable

---

## 4. Suggested Commit Order

Codex should not submit all changes in one giant patch. Use this order:

1. schema migrations + helpers in `memory_service`
2. mastery update path wired into practice submit
3. review scheduler helpers
4. planner integration
5. coverage endpoint
6. chunk metadata enrichment
7. citation payload upgrade
8. sprint pack generation

---

## 5. Definition of Done

This optimization plan is considered complete when:

1. Existing chat / RAG / quiz / exam flows still work.
2. Practice submission updates mastery + review schedule.
3. There are stable APIs for mastery / today-review / coverage.
4. Planner consumes mastery + due-review data.
5. Citations expose page/slide-level anchors when available.
6. A sprint pack can be generated from a session.

---

## 6. Explicit Non-Goals

Codex should **not** do these in this optimization cycle:

- replace SQLite with another database
- rewrite all existing router contracts
- replace LangGraph workflow design
- train a recommendation model
- introduce external services unless absolutely necessary

---

## 7. First Task for Codex

Start with **Phase 1 only**.

Concrete first PR scope:

1. add `knowledge_mastery` table support in `utils/memory_service.py`
2. update practice submit flow to maintain mastery rows
3. expose a helper method to fetch mastery snapshot
4. add a minimal GET endpoint for mastery inspection
5. add tests or at least deterministic validation paths for score updates

Do not start spaced repetition or sprint pack in the first PR.
