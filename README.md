# Nexus (`nexus`)

Multi-model CLI harness entrypoint, specialist model router, and real-time delegation tree visualizer for OpenCode, local OpenAI-compatible HTTP servers, local Ollama, and an optional external CLI backend.

## Features
- **Unified Entity Mode (`nexus/entity.py`, default in TUI & REPL):** Present one continuous assistant identity to the user while internally routing each turn to the best specialist model, injecting `system_persona`, reusing per-route `backend_session_id`s, and handing off structured conversation summaries (`build_handoff_context` for CLI backends, bounded `history_messages` for HTTP backends) across route switches. Pass `--sticky`, `/mode sticky`, or `/pin <route>` (`/switch <route>`) to lock a single model.
- **Pluggable Configuration Profiles (`config/profiles/` & `~/.config/nexus/profiles/`):** Hot-swap behavior and routing presets without losing conversation memory (`default`, `free-only`, `local-only`, `ensemble`, plus custom profiles via `nexus profile init`). In `ensemble` mode (or `nexus council --unified`), internal specialist seats run in parallel and synthesize into one cohesive response.
- **Performance Telemetry & Multi-Config Benchmark (`nexus stats`, `nexus bench`):** Aggregate latency (`p50`/`p95`), success rate, token throughput, and zero-cost ratio from `events.jsonl` by `profile`, `route`, or `model`, and benchmark multiple profiles head-to-head against the disjoint `evolve`/`held-out` routing splits.
- **Budget-Aware Specialist Routing (`config/routes.json`):** Routes coding, refactoring, deep reasoning/security audits, long-context reading, and vision tasks to free (`opencode/*-free`), subscription (`opencode-go/*`, including `kimi-k3.1-pro` on `@refactor` and `glm-5.3` on `@reason` with `qwen4-max` fallbacks), local OpenAI-compatible HTTP (`@local`, `127.0.0.1:8000`), local Ollama (`@ollama`, `127.0.0.1:11434`), or an optional external CLI on the `PREMIUM` tier (`@premium`). Heavy routes carry 300s timeouts.
- **1-Hop Fallback & Safe Unattended Runs:** Failed non-cancelled/non-timed-out OpenCode calls automatically retry once against the route's `fallback` pin (`--no-fallback` disables it) and pass `--auto` unless `auto_approve: false`.
- **Self-Improvement Memory & CRSI (`nexus sim` / `nexus evolve`):** An append-only diagnostic ledger records every accepted and rejected optimization attempt, its falsifiable prediction, measured fixes/misses/regressions, and its model/benchmark provenance. The outer loop runs a Proposer/Selector pair with a strict evolve/held-out firewall, sandboxed edits, scope presets (`--scope routes|prompts|full|polyglot`), multi-axis promotion (`--objective accuracy|speed|efficiency|open-model|balanced` via `beats_baseline`), sandbox build verification for `native/` Rust/C/Zig/Go rewrites (`config/policies/*.json`), passive improvement scanning (`nexus evolve suggest`), and immutable security boundaries. See [docs/sim-architecture.md](docs/sim-architecture.md).
- **Loud Validation & Environment Audit (`nexus routes --validate [--local]`, `nexus doctor`):** Verifies specialist, `fallback`, `decision_model`, and `council_roles` pins against live `opencode models`, optionally probes local Ollama and OpenAI-compatible HTTP endpoints (`--local`), and audits binaries, local endpoint reachability/models, and state files (`nexus doctor [--prune-events] [--keep-lines N]`).
- **First-Class Streaming Local HTTP (`ollama` & `openai-compat`) + Sandboxed Drops:** Streams NDJSON/SSE reasoning and tokens incrementally with `--think`/`/think`, bounded 12-turn/16k-char conversation history in REPL/TUI, text (`<=64 KiB`) and base64 image (`-f`/`/attach`) attachments, native sandboxed `send_file` tool-calling into the drop folder (`~/.local/share/nexus/drops/` by default, `NEXUS_DROP_DIR` override), `-o`/`/save [file]` code extraction from any route, fast `socket.shutdown()` cancellation, and partial output/token preservation on timeout.
- **Harness-Agnostic JSONL Telemetry Bus (`~/.local/state/nexus/events.jsonl`):** Every parent or child span emits atomic `O_APPEND` JSONL events (`SPAWN`, `RUNNING`, `DONE`, `FAIL`, `TIMEOUT`). `read_events()` tail-seeks both `limit` and `trace_id` queries, and `prune_events()` compacts logs atomically. Token accounting sums per-step usage and folds OpenCode's cache reads into the prompt size.
- **Bounded Execution That Actually Bounds:** Both child pipes are drained completely before stopping readers, wait-polling enforces deadlines across process groups, and multi-line JSON errors are scanned instead of leaking raw `step_start` JSON.
- **Live Delegation Tree Visualizer (`nexus` & `nexus watch`):** Renders parent → router → child harness/model nodes in real time with token counts, elapsed latency, and quota badges (`[FREE]`, `[GO]`, `[LOCAL]`, and red `[PREMIUM]`). Both single turns and councils emit proper parent/child span trees across CLI and TUI.
- **Live Stderr Thinking & Status Lines:** Front-desk decision (`◆`), spawn target (`▸`), numbered per-tool steps, live TTY spinner char-count/preview, live wrapped `┆` thinking lines on `stderr` at default verbosity (`>= 1`), and a one-line outcome receipt (`thought Nc` included) — while `stdout` stays pure answer text.
- **Three Verbosity Levels (`-q` / `-v`, `/quiet` + `/verbose` in REPL/TUI):** `0` (`quiet`) = answer and errors only (`stderr` silent, TUI thinking collapsed), `1` (`default`) = status lines, delegation tree, and visible thinking traces, `2` (`verbose`) = adds decision rationale, intent, prompt preview, trace id, and cached token breakdown. Subparser `-q`/`-v`/`--no-color` work cleanly after subcommands, and `--no-color` strips every ANSI code.
- **Full-Screen Textual TUI (`nexus` on a TTY, or `nexus tui`):** A 0-gap charcoal console with multi-tab route-isolated sessions, `/think` and `/attach` meta strip indicators, multiline composer with `/` and `@` popups, default-visible thinking/tool groups (`ToolActivityGroup` retains up to 16k chars with O(1) spinner ticks), streaming Markdown answers with live `thinking · Nc` counters and active-tab `/copy`, 3-seat budget-checked parallel council grid, lazy inspector sidebar, and five dark themes.

## Quick Start
```bash
# Audit environment binaries, route pins, local endpoints, and state files
nexus doctor
nexus routes --validate
nexus routes --validate --local

# One-shot specialist execution (auto-routed, @route prefixed, thinking level, attachments)
nexus "@code Write a POSIX-safe atomic file writer in Python"
nexus run -r ollama --think low "Explain tail-seek indexing concisely"
nexus run -r code -o out.py "Write a tiny echo script in Python"
nexus run -r reason -f diagram.png "Audit this architecture"
nexus run -r code --no-fallback "Single attempt only"

# Verbosity: quiet (no status/thinking) / default (live thinking + tree) / verbose (rationale + trace id)
nexus run -q -r code "..."
nexus -v run -r code "..."

# Self-Improvement Memory and the CRSI outer loop
nexus sim init
nexus sim show
nexus sim validate
nexus sim compact
nexus evolve --dry-run --rounds 1
nexus evolve suggest
nexus evolve --scope polyglot --objective speed --dry-run --rounds 1
nexus evolve eval --split held-out
nexus evolve challenge --seed-dir /path/to/pristine-checkout

# Parallel 3-model Tripartite Council (Pioneer / Red-Teamer / Synthesizer) or unified synthesis
nexus council "Compare append-only JSONL vs SQLite WAL for multi-agent telemetry"
nexus council --unified "Compare append-only JSONL vs SQLite WAL for multi-agent telemetry"

# Unified entity profiles, telemetry stats & side-by-side profile benchmarking
nexus profile list
nexus profile show free-only
nexus profile init my-profile --from free-only --mode entity
nexus --profile free-only "Write a POSIX-safe helper in Python"
nexus --profile ensemble "Design an idempotent retry queue"
nexus stats --by route
nexus bench --profiles default,free-only,local-only --split held-out

# Watch live delegation trees across any terminal session
nexus watch                       # live, Ctrl-C to exit
nexus watch --once --trace tr_xx  # single snapshot of one trace
nexus watch --interval 1.0        # slower poll

# Interactive UI
nexus                             # full-screen TUI (Unified Entity Mode by default)
nexus --sticky --repl             # REPL with classic sticky route takeover
nexus --repl --profile free-only  # line-oriented REPL under the free-only profile
nexus tui --theme oled-black      # start the TUI on a named theme
nexus tui --no-sidebar --dry-run  # TUI without the inspector, no backend calls
```

## Self-Improvement Memory (SIM) & CRSI
`nexus sim` owns the durable diagnostic ledger at `~/.local/state/nexus/sim.json` (`NEXUS_SIM_PATH` overrides it). It is an evidence ledger for the outer-loop optimizer, not chat history: each problem records its root-cause hypothesis, every attempt, the tasks it predicted it would fix, what it actually fixed, missed, or regressed, and the policy/optimizer models and benchmark involved.

`nexus evolve` runs Continual Recursive Self-Improvement against `config/evals/routing_benchmark.json`. Evolve-split outcomes drive proposals; held-out tasks are scored exactly once per challenger run and never reach the Proposer, the Selector, or the SIM. Edits are applied inside a throwaway sandbox and only promoted when they beat the baseline without regressions.

- `nexus sim init` creates an empty canonical ledger; `nexus sim show` renders a width-safe summary; `nexus sim validate` checks schema and alias integrity; `nexus sim compact` distills invariants, per-model notes, and dead ends, and `--archive-dir <path>` saves the uncompacted ledger first.
- `nexus evolve --dry-run --rounds 1` validates the full loop without any backend call.
- `nexus evolve eval --split held-out` scores the held-out split once, and `nexus evolve challenge --seed-dir ...` compares a continued SIM lineage against a fresh cold start head-to-head on held-out tasks.
- Security boundaries (`nexus/sim.py`, `nexus/evolve.py`, `nexus/drops.py`, `config/sim_policy.json`, `config/evals`, `tests`, `.git`, `.env`) are permanently locked outside editable scope.

## Customization & Sharing
Nexus ships generic defaults so a fresh clone runs without private infrastructure. Personal choices stay local:

- **Route overrides:** copy `config/routes.json` to `~/.config/nexus/routes.json` and change models, endpoints, hosts, timeouts, or explicit-only flags there. The local file wins at load time; commit only your generic defaults.
- **Drop folder:** `NEXUS_DROP_DIR` overrides the default `~/.local/share/nexus/drops/`. Generated files from `send_file` and `/save` land there.
- **Protected paths:** `NEXUS_PROTECTED_PATHS` is an `os.pathsep`-delimited list of directories where artifact writes are refused (for example, read-only vaults or source trees).
- **SIM ledger:** `NEXUS_SIM_PATH` overrides `~/.local/state/nexus/sim.json`, useful for per-benchmark experiments.
- **Benchmarks and policy:** `config/evals/routing_benchmark.json` and `config/sim_policy.json` are plain JSON; fork them for your own domain, but keep the evolve and held-out task IDs disjoint.

## Interactive TUI (`nexus tui`)
Bare `nexus` on an interactive TTY opens the full-screen Textual TUI. `nexus --repl`, and every non-TTY invocation (piped stdin, captured output, `cron`), falls back to the line-oriented REPL. `nexus tui` starts the TUI explicitly and accepts `--theme <name>`, `--no-sidebar` (inspector starts collapsed), and `--dry-run` (routing decisions and TUI rendering only, no backend calls).

- **Multi-tab session bar (`SessionTabBar`):** One pill per session with a status dot — `⠋` running, `●` active, `○` idle — plus a `+`/`×` control pair. The right-hand status strip carries the active route and model, a running `in→out` token counter, the sidebar state, and a premium-quota guard, which flips to a red `PREMIUM! <n>` badge as soon as the session accumulates a span on the `PREMIUM` tier.
- **Multiline composer (`ComposerTextArea`):** Auto-grows from 1 to 6 lines, `Enter` sends, `Shift+Enter` or `Ctrl+J` inserts a newline, and `Up`/`Down` at the document edge walk prompt history from `~/.local/state/nexus/history` (shared with the REPL, last 500 entries). Typing `/` opens the slash-command popup; typing `@` opens the route popup showing backend, model pin, and quota badge. `Tab`/`Enter` accepts a suggestion, arrows navigate, `Esc` dismisses.
- **Thinking and tool groups (`ToolActivityGroup`):** Each assistant turn folds its reasoning trace (retaining up to 16k chars once done, and live-streaming with O(1) title updates) and numbered tool-call steps into one `Collapsible`. Turns with thinking or tools stay open by default (`verbosity >= 1`), quiet mode (`-q` / `/quiet`) collapses them, and `Ctrl+O` expands or collapses every group at once.
- **Streaming answers (`AssistantMessageCard`):** Live `thinking · Nc` counter while waiting, followed by tokens streaming cleanly into a `Markdown` widget. Each card closes with a one-line receipt and a `copy` button (matching `/copy`, which reads the active tab directly and survives session switches and restarts).
- **Composer controls (`/think`, `/attach` & `/save`):** `/think [off|low|medium|high|default]` overrides reasoning effort and shows `· think:<lvl>` in the composer strip; `/attach <path|clear>` attaches one file/image (`· +<name>`) cleanly for the next prompt; `/save [filename]` extracts the largest fenced code block from the last response into the drop folder.
- **Council grid (`CouncilGridCard`):** `/council` fans out to three seats — `PIONEER`, `RED-TEAMER`, `SYNTHESIZER` — each streaming into its own bordered seat with its own receipt. The grid is 3 columns wide and stacks to a single column below 110 terminal columns.
- **Inspector sidebar (`InspectorSidebar`):** Toggled with `Ctrl+B`, and auto-hidden below 100 columns unless you toggle it manually at that width. Three tabs: `Tree` (live delegation tree, or a border-free compact tree in a narrow pane), `Routes` (the specialist table with `@route`, backend, model, quota badge, and missing-pin markers) and `Square` (pinned `BOARD.md` plus recent town-square posts and a quick-post field).
- **Five dark themes (`Ctrl+T`):** `nexus-charcoal` (default), `oled-black`, `opencode-dark`, `github-slate`, and `catppuccin-mocha`. The picker live-previews as you move and restores the previous theme on `Esc`. Themes are registered from `nexus/tui/themes.py`, and `nexus/tui/styles.tcss` is written against theme variables only, so it stays valid under all five.
- **Modals and command palette:** `Ctrl+P` command palette, `Ctrl+K` route picker (with `r` for front-desk and `v` to re-validate pins), `Ctrl+L` session picker (`n` new, `d` delete), `Ctrl+N` new tab, `Ctrl+T` theme picker, `Ctrl+Tab`/`Alt+Right` next tab, `Ctrl+Shift+Tab`/`Alt+Left` previous tab, `Esc` cancels the active run or dismisses a popup, `F1`/`?`/`/help` opens the two-column keymap, `Ctrl+Q` quits, and a double `Ctrl+C` quits from anywhere.
- **Persistent state (`nexus/sessions.py`):** Transcripts and per-session token totals survive restarts in `~/.local/state/nexus/tui_sessions.json` (most recent 30 sessions), while theme, sidebar state, routing mode, and verbosity persist in `~/.config/nexus/tui.json`. Both files are written atomically (`tempfile` + `os.replace`) and tolerate missing or corrupt content.

### Keymap
| Keys | Action |
| --- | --- |
| `Enter` | Send prompt |
| `Shift+Enter`, `Ctrl+J` | Newline in the composer |
| `Up` / `Down` | Prompt history at the document edge; popup navigation while a popup is open |
| `Tab` | Accept the highlighted `/` or `@` suggestion |
| `Ctrl+P` | Command palette |
| `Ctrl+K` | Route picker (`r` front-desk, `v` validate pins) |
| `Ctrl+L` | Session picker (`n` new, `d` delete) |
| `Ctrl+N` | New session tab |
| `Ctrl+Tab`, `Alt+Right` | Next session tab |
| `Ctrl+Shift+Tab`, `Alt+Left` | Previous session tab |
| `Ctrl+B` | Toggle inspector sidebar |
| `Ctrl+O` | Expand / collapse all thinking and tool groups |
| `Ctrl+T` | Theme picker |
| `Esc` | Cancel the active run, or dismiss a popup |
| `F1`, `?`, `/help` | Keymap and slash-command reference |
| `Ctrl+Q` | Quit |
| `Ctrl+C` twice | Quit from anywhere |

## Delegation Budgets & Route Precedence (`config/routes.json`)
- **Budgets** (`max_spans_per_trace: 32`, `max_children: 8`, `max_depth: 2`) are enforced before dispatch: a council that would exceed `max_children` is capped, and a trace that would exceed its span budget is refused rather than truncated silently.
- **Precedence** is an ordered list used only to break keyword ties, so a prompt matching both `vision` and `code` resolves to the more specific route instead of relying on JSON key order.
- **Route principles** for the three overlapping coding routes are stated in the table (`code` = existing codebase, `fast-code` = single snippet, `refactor` = multi-file with call-site moves).
- **`local`, `ollama`, and `premium` are explicit-only** (`@local`, `@ollama`, `@premium`); free cloud models win by default, per the free-first model policy.

## Evaluations
`tests/test_events_and_router.py` carries a table-driven route-accuracy eval (intent text → expected route), explicit-only assertions for local and premium backends, budget-overrun tests, and a precedence tie-break test. `tests/test_sim.py` covers the SIM schema, aliases, compaction, validation, the evolve/held-out firewall, scope policy, and the cold-start challenger — the cheapest form of the "assert from the transcript" pattern Anthropic's agent guidance recommends.
