# SIM & CRSI Architecture

This guide documents the Self-Improvement Memory (SIM) ledger and the Continual Recursive Self-Improvement (CRSI) outer loop shipped with Nexus. It is an engineering reference for the schema, optimizer contract, transfer rules, and operational lifecycle.

## 1. Why SIM Exists

A chat transcript is ephemeral: it records what a model said, not why a harness change was made, which failures it predicted, or whether it actually helped. When the policy model, benchmark, or evaluator changes, a memoryless optimizer inherits evolved code with no provenance. It then re-tightens old guards, repeats disproven patches, and invents a false history for code it did not write. This is **amnesiac transfer**.

SIM is a structured laboratory ledger for the outer-loop optimizer, not episodic memory for the task-solving agent. It preserves:

- the root-cause hypothesis behind each harness problem;
- every attempt, including rejected proposals, misses, regressions, and accepted edits that fixed nothing;
- the falsifiable prediction made *before* scoring;
- the measured task-level outcomes and aggregate score;
- the policy model, optimizer model, benchmark, round, and diff reference that produced the evidence;
- the provenance chain when a ledger is carried into a new run.

That evidence makes optimization falsifiable. A proposal can be judged against what it claimed it would fix, not against a plausible narrative. It also prevents a new model or a new dataset from silently discarding hard-won knowledge, while still allowing obsolete, model-specific patches to be pruned deliberately.

SIM assumes a shared, compatible executable harness interface. It does not transfer experience across incompatible tools or verifiers. When the interface itself changes, record an explicit schema break instead of pretending the old evidence still applies.

## 2. Canonical SIM JSON Schema

The ledger is indexed by recurring harness problems (`P1`, `P2`, ...), not by flat chronological commits. Nexus stores it at `~/.local/state/nexus/sim.json` (`NEXUS_SIM_PATH` overrides the path) and writes it atomically with `tempfile` + `os.replace`.

```json
{
  "schema_version": "1.0",
  "interface": {
    "name": "nexus-harness",
    "version": "1.0"
  },
  "carried_from": {
    "from": "run-a/benchmark-v1/model-a",
    "into": "run-b/benchmark-v2/model-b",
    "parent": null,
    "carried_at": 1790000000.0
  },
  "schema_breaks": [
    {
      "ts": 1790000000.0,
      "from_version": "1.0",
      "to_version": "2.0",
      "reason": "routing verifier now returns per-task traces"
    }
  ],
  "context": {
    "domain": "routing",
    "policy_model": "opencode/deepseek-v4.1-flash-free",
    "benchmark": "nexus-routing-v1",
    "created_at": 1790000000.0
  },
  "aliases": {
    "P6": "P5"
  },
  "problems": [
    {
      "id": "P5",
      "name": "explicit-only routes leak into implicit keyword routing",
      "opened_round": 3,
      "tier": "adaptable",
      "invariant_summary": "explicit-only routes leak into implicit keyword routing: gate explicit-only matches before keyword scoring",
      "model_notes": {
        "opencode/deepseek-v4.1-flash-free": [
          "r3 revise: gate explicit-only matches before keyword scoring"
        ]
      },
      "dead_ends": [
        "[zero-fix] r4 add a broader premium keyword list"
      ],
      "archived_attempts_count": 2,
      "attempts": [
        {
          "round": 3,
          "sample": "1",
          "action": "revise",
          "accepted": true,
          "backfilled": false,
          "domain": "routing",
          "what": "gate explicit-only matches before keyword scoring",
          "files": ["config/routes.json"],
          "predicted": ["e-noimplicit-local", "e-noimplicit-premium"],
          "fixed": ["e-noimplicit-local", "e-noimplicit-premium"],
          "missed": [],
          "regressed": [],
          "passed": 20,
          "total": 22,
          "score": 0.9091,
          "policy_model": "opencode/deepseek-v4.1-flash-free",
          "optimizer_model": "opencode-go/glm-5.3",
          "benchmark": "nexus-routing-v1",
          "diff_ref": "sandbox:config/routes.json",
          "ts": 1790000000.0
        }
      ]
    }
  ]
}
```

### Field semantics

| Field | Meaning |
| --- | --- |
| `schema_version` | Ledger schema generation. Bump only through an explicit schema break. |
| `interface` | Shared harness contract (`name`, `version`). Transfer is valid only inside one compatible interface. |
| `carried_from` | Source and target run, plus a nested `parent` chain, so later readers can detect domain or model drift. |
| `schema_breaks[]` | Human-auditable record of interface changes that invalidate naive continuation. |
| `context` | Domain, active policy model, benchmark, and creation time for this run. |
| `aliases` | Canonical merges (`"P6": "P5"`). Alias resolution is transitive and rejects cycles. |
| `problems[]` | Durable root-cause containers. IDs are never reused, including after merges. |
| `problem.name` | Falsifiable root-cause hypothesis, not a vague topic label. |
| `problem.opened_round` | First diagnosis round within the originating run. |
| `problem.tier` | `invariant`, `adaptable`, or `model_specific` (see Section 4). |
| `problem.attempts[]` | Append-only evidence. Earlier attempts are never overwritten or removed. |
| `problem.invariant_summary` | Compaction output: the durable lesson distilled from accepted, fixing attempts. |
| `problem.model_notes` | Compaction output: per-policy-model implementation notes. |
| `problem.dead_ends` | Compaction output: rejected, regressed, and zero-fix attempts retained as negative evidence. |
| `problem.archived_attempts_count` | Number of raw attempts trimmed after distillation. The raw ledger is archived before trimming. |

Every attempt is tagged with four-tuple provenance: `policy_model`, `optimizer_model`, `benchmark`, and `round`. `sample` distinguishes candidates proposed in the same round.

Attempt fields:

- `action`: `new`, `retain`, `revise`, `replace`, or `remove`.
- `accepted`: whether the attempt won selection. Rejected attempts remain in the ledger.
- `backfilled`: whether the attempt was recorded after the fact from an older run.
- `predicted`: exact evolve task IDs the edit was expected to fix, logged before scoring.
- `fixed`: predicted tasks that flipped to pass, followed by any other newly passing tasks.
- `missed`: predicted tasks that still fail.
- `regressed`: tasks that passed before the edit and fail after it.
- `passed` / `total` / `score`: aggregate evolve-split outcome.
- `diff_ref`: sandbox or patch reference for the edit.

`compute_task_outcomes()` credits unpredicted fixes too, so an edit is not penalized for solving more than it claimed. It still records every miss and regression separately.

## 3. The Ten Canonical Proposer/Selector Directives

Every CRSI prompt injects all ten directives. They are not optional style guidance; they define the evidence contract.

1. **Inject the complete ledger into both optimizer roles.** The full SIM JSON goes into every Proposer and Selector prompt on every round. Do not silently truncate problems, attempts, aliases, or outcomes.
2. **Enforce strict Evolve/Held-Out isolation.** Only evolve-split scores, traces, and errors reach the Proposer, the Selector, or SIM updates. Held-out outcomes stay completely outside the optimization loop.
3. **Require explicit root-cause attribution before editing.** Every candidate must cite an existing `problem_id` or open a new problem carrying a testable root-cause hypothesis.
4. **Log pre-evaluation predictions before scoring.** The Proposer must name the exact evolve task IDs it expects the edits to fix before any evaluation runs.
5. **Read provenance before judging relevance.** Interpret `id`, `opened_round`, `domain`, `policy_model`, and `carried_from` in their recorded run context. Never assume an old finding transfers unchanged.
6. **Choose explicitly among inheritance actions.** Decide whether inherited code should be `retain`ed, `revise`d, `replace`d, or `remove`d (or open a new problem). Never inherit silently.
7. **Preserve history when code is replaced or removed.** Rewriting a guard must not erase its SIM history. Append the new attempt and outcome under the existing problem so later rounds know why the old guard existed.
8. **Merge duplicate diagnoses through aliases.** Merge matching hypotheses and record the mapping in `aliases` while preserving every historical attempt.
9. **Retain failures, misses, and regressions explicitly.** Keep zero-fix accepted edits (`fixed == []`), rejected proposals, and regressions so the optimizer can separate causal repairs from score noise.
10. **Lock security and release gates outside editable scope.** Authorization, sandboxing, verifiers, and approval gates must remain permanently outside the optimizer's editable files.

### Immutable `ScopePolicy`

`config/sim_policy.json` (plus `config/policies/self_rewrite.json` and `config/policies/polyglot.json`) supplies `editable_files`, optional `editable_dirs` (for `config/profiles` or compiled `native/` hot paths), optional trusted `build_cmd` / `build_timeout_s`, `locked_files`, and `max_file_bytes`. `ScopePolicy.for_scope()` selects named presets (`routes`, `prompts`, `full`, `polyglot`), always merging immutable defaults subject to one hard rule: **locks always win**. Built-in immutable defaults cover `nexus/sim.py`, `nexus/evolve.py`, `nexus/drops.py`, `config/sim_policy.json`, `config/policies`, `config/evals`, `tests`, `.git`, and `.env`.

Before any write, `ScopePolicy.validate_paths()` rejects:

- empty or non-string paths;
- absolute paths and `..` traversal;
- symlinks at the destination, or parent symlinks that escape the workspace;
- anything at, inside, or shadowing a locked path (so editing `config` would shadow `config/evals` or `config/policies`);
- anything outside `editable_files` and `editable_dirs`.

Edits are applied inside a throwaway sandbox, never directly to the live workspace. When `build_cmd` is configured on the policy or benchmark, it runs inside the sandbox before verification; failing builds are rejected and logged as negative evidence.

### Multi-Axis Improvement Ladder & `beats_baseline()`

Proposer prompts expose a four-step ladder (`prompt_tune` -> `profile_tune` -> `code_refactor` -> `polyglot_rewrite`) tuned toward squeezing open/free/local models and harness speed under `--objective {accuracy,speed,efficiency,open-model,balanced}`. With zero regressions (`regressed == []`) and a verified build, `beats_baseline()` promotes strictly better evolve scores or equal-accuracy measurable wins (`>= 5%` median `latency_ms` speedup, `>= 5%` `tokens_total` drop, or `>= +0.01` `open_model_rate`). `detect_improvement_opportunities()` passively flags failing open routes, premium escalations, latency hotspots, and recurring SIM dead ends (`nexus evolve suggest`).

## 4. Three-Tier Behavioral Transfer Canon

Inherited material is classified by how well it survives a model or dataset change.

| Tier | Name | What belongs here | Action on transfer |
| --- | --- | --- | --- |
| `invariant` | Tier 1: Durable Invariants | Core execution loops, indexing scaffolds, prompt infrastructure, interface contracts, structural checks. | Retain and verify the invariant still holds. Do not strip it during a broad refactor. |
| `adaptable` | Tier 2: Portable Rationales Requiring New Code | Completion gates, transport guards, retrieval thresholds, context/step budgets. The rationale transfers; the mechanism may not. | Revise or replace the implementation while preserving the root-cause rationale in SIM. Re-verify evaluator and interface contracts. |
| `model_specific` | Tier 3: Policy-Locked & Domain-Locked Traps | Tool-call parser repairs, model-quirk patches, inactive domain normalizers, false-positive wins. | Remove or prune dead predicates and foreign-model patches. Inspect `fixed == []` records before reuse. |

### Zero-fix pruning rule

A record with `fixed == []` is not evidence that an edit worked. Such attempts are tagged `[zero-fix]` during compaction and moved into `dead_ends`. They stay visible so a later model does not mistake an accepted-but-ineffective patch for a validated fix. The same applies to rejected proposals (`[rejected]`) and edits that caused regressions (`[regressed]`).

Compaction is lossless in the operational sense: raw attempts are archived to `sim_archive_<UTC timestamp>.json` before the working ledger is reduced. Distillation keeps the invariant, the per-model note, and the dead end, so pruning removes noise without removing the lesson.

## 5. Multi-Model Consolidation & Lifecycle

### Unified Synthesis, not sequential drift

When several policy models share one harness, pool their evolve-split trajectories inside each round. Nexus implements this by scoring every task once per active policy route and tagging results as `"{route}::{task_id}"`. The Selector then sees the pooled evidence and can steer toward Tier 1 invariants and adaptable Tier 2 checks.

Do not chain adaptations sequentially (`model A → model B → model C`). Sequential chaining induces recency drift: later models over-weight the most recent edits and can lose durable invariants established earlier. Pooled synthesis keeps every active model's evidence in the same decision context.

### Carry-forward

`carry_forward(source_ledger, from_run=..., into_run=...)` deep-copies a ledger and prepends a `carried_from` node. The nested `parent` chain preserves the full lineage across repeated migrations. Carry forward when the interface is compatible; record a schema break when it is not.

### Compaction

`compact_sim()` is the maturity path for a large ledger:

1. Archive the uncompacted ledger to the archive directory (when one is supplied).
2. Distill each problem into `invariant_summary` (durable root cause and accepted fixes), `model_notes` (per-policy-model implementation guidance), and `dead_ends` (rejected, regressed, zero-fix attempts).
3. Trim raw `attempts[]` to the newest `keep_recent_attempts` entries and increment `archived_attempts_count`.

Compaction is idempotent for distilled fields: running it twice does not duplicate invariant segments, model notes, or dead ends. This keeps optimizer prompts inside context limits without discarding the evidence that matters.

### Cold-start challenger

Historical memory can help an incremental upgrade and hurt a frontier leap. Before onboarding a major next-generation model, run both lineages:

1. Clone the evolved workspace and continue it with `carry_forward` ("continued").
2. Clone a pristine seed workspace and start it with a fresh ledger ("fresh").
3. Run the same evolve loop on both, scoring only the evolve split.
4. Evaluate each final workspace exactly once on the held-out split.
5. Promote strictly on held-out score: retain the inherited SIM if it wins, prefer a cold start if the fresh lineage wins, and keep the inherited SIM on a tie to preserve provenance at no measurable cost.

`run_cold_start_challenger()` returns `winner`, `continued_held_out`, `fresh_held_out`, `delta`, both ledgers, and a recommendation. It never copies `.git` into the comparison trees.

## 6. CLI Reference

```bash
# Create, inspect, and validate the ledger
nexus sim init
nexus sim show
nexus sim validate

# Archive and distill a mature ledger
nexus sim compact --archive-dir ~/.local/state/nexus/archive

# Validate the full CRSI loop without calling any backend
nexus evolve --dry-run --rounds 1
nexus evolve --scope polyglot --objective speed --dry-run --rounds 1

# Scan telemetry & SIM dead ends for concrete optimization suggestions
nexus evolve suggest

# Score the held-out split exactly once
nexus evolve eval --split held-out

# Compare a continued SIM lineage against a fresh cold start on held-out tasks
nexus evolve challenge --seed-dir /path/to/pristine-checkout
```

Operational notes:

- `nexus sim show` renders a width-safe summary: schema, interface, domain, benchmark, policy, carry chain, aliases, and per-problem accepted/rejected/zero-fix counts.
- `nexus sim validate` checks problem IDs, names, tiers, attempt actions, list fields, score bounds, alias targets, and cycles. It reports every error rather than stopping at the first.
- `nexus evolve --dry-run --rounds 1` exercises the Proposer, sandbox, verifier, Selector, and SIM write path with a deterministic no-op proposal.
- `nexus evolve eval --split held-out` is a read-only measurement. Do not feed its output back into a prompt.
- `nexus evolve challenge --seed-dir ...` runs both lineages and reports held-out deltas. Use the recommendation as evidence, not as an automatic deployment gate; deterministic verifiers and human approval still decide promotion.

For the benchmark and policy inputs, see `config/evals/routing_benchmark.json` and `config/sim_policy.json`. For the ledger and loop implementations, see `nexus/sim.py` and `nexus/evolve.py`.

## 7. Proactive Research Directives & LifeOS-Inspired Evolution

Alongside the ten canonical SIM directives (`D1`–`D10`), the Proposer injects four **Proactive Research Directives (`R1`–`R4`)** to eliminate passive stagnation even when baseline accuracy reaches `100%`:

- **`R1` Unanswered Questions:** Treat every failure, roadblock, or dead end as an unanswered question with an explicit falsifiable hypothesis before editing.
- **`R2` Research & OSS Grounding:** Check recent AI/CS literature and proven open-source harnesses before inventing custom mechanisms.
- **`R3` Explicit `open_question`s:** Every suggestion emitted by `detect_improvement_opportunities()` / `nexus evolve suggest` carries an `open_question` (rendered as `ask: ...`) surfacing unresolved bottlenecks or assumptions.
- **`R4` Proactive Squeeze:** Hunt higher-leverage gains in prompt framing, latency/compression, and compiled `native/` speedups even when functional accuracy is green.

Queued in `tasks/todo.md` under LifeOS's **ISA (Ideal State Articulation) + Errata** architecture:

- **Verbatim Errata & False-Pass Capture (`nexus/errata.py`):** Records user complaints and post-`DONE` false passes verbatim into `~/.local/state/nexus/errata.jsonl` joined directly to `trace_id`, surfaced in `nexus evolve suggest` without placing LLMs on the interactive hot path.
- **Assay Core Quality & Shadow Calibration (`nexus/assay.py`, `config/evals/core_quality_benchmark.json`):** Binds deterministic multi-domain checks (routing, >240-char tool-output compression, 3-scope health classification, latency budget) with SHA-256 receipts, staged gating (`UNMEASURED` on empty stages), and shadow router agreement tracking.
- **Multi-Objective & GEPA Evolution:** Expand `nexus/evolve.py` with MAP-Elites/Pareto archives, trajectory-feedback prompt co-evolution, and sandbox network-egress / anti-reward-hacking guards.
