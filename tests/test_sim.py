"""Tests for the Self-Improvement Memory (SIM) ledger."""

from __future__ import annotations

import json
from pathlib import Path
import stat
import sys

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from nexus.sim import (
    ACTION_VALUES,
    DEFAULT_SIM_PATH,
    SCHEMA_VERSION,
    TIER_VALUES,
    carry_forward,
    compact_sim,
    compute_task_outcomes,
    find_problem,
    format_sim_for_prompt,
    load_sim,
    merge_problems,
    new_ledger,
    next_problem_id,
    open_problem,
    record_attempt,
    record_schema_break,
    render_sim_summary,
    resolve_problem_id,
    resolve_sim_path,
    save_sim,
    validate_sim,
)
from nexus.visualizer import strip_ansi, visible_len


# --------------------------------------------------------------------------- helpers


def _attempt(
    ledger: dict,
    problem_id: str,
    *,
    round_num: int,
    what: str,
    accepted: bool = True,
    fixed: list[str] | None = None,
    predicted: list[str] | None = None,
    regressed: list[str] | None = None,
    action: str = "new",
    sample: str = "s0",
    policy_model: str | list[str] = "policy-a",
    domain: str = "code",
    passed: int = 1,
    total: int = 2,
    **kwargs: object,
) -> dict:
    return record_attempt(
        ledger,
        problem_id=problem_id,
        round_num=round_num,
        sample=sample,
        action=action,
        accepted=accepted,
        domain=domain,
        what=what,
        files=["nexus/mod.py"],
        predicted=predicted if predicted is not None else ["t1"],
        fixed=fixed if fixed is not None else [],
        missed=[],
        passed=passed,
        total=total,
        regressed=regressed,
        policy_model=policy_model,
        **kwargs,
    )


# --------------------------------------------------------------------------- paths


def test_resolve_sim_path_precedence(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    explicit = tmp_path / "explicit.json"
    assert resolve_sim_path(explicit) == explicit

    monkeypatch.setenv("NEXUS_SIM_PATH", str(tmp_path / "env.json"))
    assert resolve_sim_path() == tmp_path / "env.json"
    assert resolve_sim_path(explicit) == explicit

    monkeypatch.setenv("NEXUS_SIM_PATH", "")
    assert resolve_sim_path() == DEFAULT_SIM_PATH

    monkeypatch.delenv("NEXUS_SIM_PATH")
    assert resolve_sim_path() == DEFAULT_SIM_PATH


def test_constants_are_canonical() -> None:
    assert SCHEMA_VERSION == "1.0"
    assert ACTION_VALUES == ("new", "retain", "revise", "replace", "remove")
    assert TIER_VALUES == ("invariant", "adaptable", "model_specific")


# --------------------------------------------------------------------------- construction


def test_new_ledger_shape() -> None:
    ledger = new_ledger(
        domain="math",
        policy_model="p/x",
        benchmark="bench",
        interface_name="iface",
        interface_version="9.9",
    )
    assert ledger["schema_version"] == SCHEMA_VERSION
    assert ledger["interface"] == {"name": "iface", "version": "9.9"}
    assert ledger["carried_from"] is None
    assert ledger["schema_breaks"] == []
    assert ledger["context"]["domain"] == "math"
    assert ledger["context"]["policy_model"] == "p/x"
    assert ledger["context"]["benchmark"] == "bench"
    assert isinstance(ledger["context"]["created_at"], float)
    assert ledger["aliases"] == {}
    assert ledger["problems"] == []


# --------------------------------------------------------------------------- persistence


def test_save_load_atomic_roundtrip(tmp_path: Path) -> None:
    target = tmp_path / "nested" / "sim.json"
    ledger = new_ledger()
    open_problem(ledger, "Parser drift")
    _attempt(ledger, "P1", round_num=1, what="tighten lexer", fixed=["t1"])

    returned = save_sim(ledger, target)
    assert returned == target
    assert target.exists()
    assert target.read_text(encoding="utf-8").endswith("\n")
    assert stat.S_IMODE(target.stat().st_mode) == 0o644
    assert not list(target.parent.glob("*.tmp"))

    loaded = load_sim(target)
    assert loaded == ledger
    assert load_sim(target, strict=True) == ledger


def test_load_missing_and_corrupt(tmp_path: Path) -> None:
    missing = tmp_path / "absent.json"
    fresh = load_sim(missing)
    assert fresh["schema_version"] == SCHEMA_VERSION
    assert fresh["problems"] == []
    assert fresh["context"]["domain"] == "nexus"
    with pytest.raises(FileNotFoundError):
        load_sim(missing, strict=True)

    corrupt = tmp_path / "corrupt.json"
    corrupt.write_text("{not json", encoding="utf-8")
    assert load_sim(corrupt)["problems"] == []
    with pytest.raises(ValueError):
        load_sim(corrupt, strict=True)

    not_dict = tmp_path / "list.json"
    not_dict.write_text("[1, 2, 3]", encoding="utf-8")
    assert load_sim(not_dict)["schema_version"] == SCHEMA_VERSION
    with pytest.raises(ValueError):
        load_sim(not_dict, strict=True)


def test_load_defaults_missing_optional_keys(tmp_path: Path) -> None:
    target = tmp_path / "partial.json"
    target.write_text(json.dumps({"problems": []}), encoding="utf-8")
    loaded = load_sim(target)
    for key in (
        "schema_version",
        "interface",
        "carried_from",
        "schema_breaks",
        "context",
        "aliases",
        "problems",
    ):
        assert key in loaded
    assert loaded["aliases"] == {}
    assert loaded["schema_breaks"] == []


def test_load_repairs_wrong_types(tmp_path: Path) -> None:
    target = tmp_path / "bad_types.json"
    target.write_text(
        json.dumps(
            {
                "interface": "nope",
                "context": [],
                "aliases": [],
                "problems": "nope",
                "schema_breaks": {},
                "schema_version": 7,
            }
        ),
        encoding="utf-8",
    )
    loaded = load_sim(target)
    assert isinstance(loaded["interface"], dict)
    assert isinstance(loaded["context"], dict)
    assert loaded["aliases"] == {}
    assert loaded["problems"] == []
    assert loaded["schema_breaks"] == []
    assert loaded["schema_version"] == SCHEMA_VERSION


# --------------------------------------------------------------------------- ids and lookup


def test_resolve_problem_id_transitive_and_cycle() -> None:
    ledger = new_ledger()
    ledger["aliases"] = {"P6": "P5", "P5": "P2"}
    assert resolve_problem_id(ledger, "P6") == "P2"
    assert resolve_problem_id(ledger, "P2") == "P2"
    assert resolve_problem_id(ledger, "P9") == "P9"

    ledger["aliases"] = {"P1": "P2", "P2": "P3", "P3": "P1"}
    with pytest.raises(ValueError, match="alias cycle detected"):
        resolve_problem_id(ledger, "P1")


def test_find_problem_follows_aliases() -> None:
    ledger = new_ledger()
    problem = open_problem(ledger, "target", problem_id="P2")
    ledger["aliases"] = {"P6": "P5", "P5": "P2"}
    assert find_problem(ledger, "P6") is problem
    assert find_problem(ledger, "P1") is None


def test_next_problem_id_never_reuses_aliases() -> None:
    ledger = new_ledger()
    assert next_problem_id(ledger) == "P1"

    open_problem(ledger, "one", problem_id="P3")
    ledger["aliases"] = {"P1": "P2", "P2": "P3"}
    assert next_problem_id(ledger) == "P4"

    ledger["aliases"] = {"P9": "P3"}
    assert next_problem_id(ledger) == "P10"


# --------------------------------------------------------------------------- problems


def test_open_problem_allocates_and_validates() -> None:
    ledger = new_ledger()
    first = open_problem(ledger, "  Parser drift  ", opened_round=4, tier="invariant")
    assert first["id"] == "P1"
    assert first["name"] == "Parser drift"
    assert first["opened_round"] == 4
    assert first["tier"] == "invariant"
    assert first["invariant_summary"] == ""
    assert first["model_notes"] == {}
    assert first["dead_ends"] == []
    assert first["attempts"] == []
    assert ledger["problems"] == [first]

    second = open_problem(ledger, "Second", problem_id="P7")
    assert second["id"] == "P7"

    auto = open_problem(ledger, "Auto", problem_id="")
    assert auto["id"] == "P8"

    with pytest.raises(ValueError):
        open_problem(ledger, "   ")
    with pytest.raises(ValueError):
        open_problem(ledger, "bad tier", tier="nope")
    with pytest.raises(ValueError):
        open_problem(ledger, "duplicate", problem_id="P1")

    ledger["aliases"]["P4"] = "P1"
    with pytest.raises(ValueError):
        open_problem(ledger, "aliased id", problem_id="P4")


def test_merge_problems_moves_attempts_and_aliases() -> None:
    ledger = new_ledger()
    target = open_problem(ledger, "target", problem_id="P2")
    source = open_problem(ledger, "source", problem_id="P1")
    target["model_notes"] = {"policy-a": ["kept"]}
    source["model_notes"] = {"policy-a": ["new note"], "policy-b": ["other"]}
    target["dead_ends"] = ["[rejected] old"]
    source["dead_ends"] = ["[rejected] old", "[zero-fix] fresh"]

    _attempt(ledger, "P2", round_num=2, what="target edit", sample="s2")
    _attempt(ledger, "P1", round_num=3, what="source later", sample="s3")
    _attempt(ledger, "P1", round_num=1, what="source earlier", sample="s1")

    merged = merge_problems(ledger, "P1", "P2")

    assert merged is target
    assert [a["round"] for a in merged["attempts"]] == [1, 2, 3]
    assert ledger["aliases"] == {"P1": "P2"}
    assert [p["id"] for p in ledger["problems"]] == ["P2"]
    assert merged["model_notes"]["policy-a"] == ["kept", "new note"]
    assert merged["model_notes"]["policy-b"] == ["other"]
    assert merged["dead_ends"] == ["[rejected] old", "[zero-fix] fresh"]
    assert find_problem(ledger, "P1") is merged

    with pytest.raises(ValueError):
        merge_problems(ledger, "P2", "P2")
    with pytest.raises(ValueError):
        merge_problems(ledger, "P9", "P2")
    with pytest.raises(ValueError):
        merge_problems(ledger, "P2", "P9")


# --------------------------------------------------------------------------- outcomes


def test_compute_task_outcomes_full() -> None:
    before = {"a": False, "b": True, "c": False, "d": False}
    after = {"a": True, "b": False, "c": False, "d": True, "e": True}
    outcome = compute_task_outcomes(before, after, ["a", "c"])
    assert outcome["fixed"] == ["a", "d", "e"]
    assert outcome["missed"] == ["c"]
    assert outcome["regressed"] == ["b"]
    assert outcome["passed"] == 3
    assert outcome["total"] == 5
    assert outcome["score"] == pytest.approx(0.6)


def test_compute_task_outcomes_empty_and_no_change() -> None:
    assert compute_task_outcomes({}, {}, []) == {
        "fixed": [],
        "missed": [],
        "regressed": [],
        "passed": 0,
        "total": 0,
        "score": 0.0,
    }

    same = {"a": True, "b": False}
    outcome = compute_task_outcomes(same, dict(same), ["a", "b"])
    assert outcome["fixed"] == []
    assert outcome["missed"] == ["b"]
    assert outcome["regressed"] == []
    assert outcome["score"] == pytest.approx(0.5)


# --------------------------------------------------------------------------- attempts


def test_record_attempt_fields_and_append_only() -> None:
    ledger = new_ledger()
    open_problem(ledger, "parser")

    first = _attempt(
        ledger,
        "P1",
        round_num=1,
        what="  tighten lexer  ",
        predicted=["t1", "t2"],
        fixed=["t1"],
        passed=3,
        total=4,
        policy_model=["a", "b"],
        optimizer_model="opt",
        benchmark="bench",
        diff_ref="sha:1",
    )
    assert first["round"] == 1
    assert first["sample"] == "s0"
    assert first["action"] == "new"
    assert first["accepted"] is True
    assert first["backfilled"] is False
    assert first["what"] == "tighten lexer"
    assert first["predicted"] == ["t1", "t2"]
    assert first["fixed"] == ["t1"]
    assert first["missed"] == []
    assert first["regressed"] == []
    assert first["passed"] == 3
    assert first["total"] == 4
    assert first["score"] == pytest.approx(0.75)
    assert first["policy_model"] == ["a", "b"]
    assert first["optimizer_model"] == "opt"
    assert first["benchmark"] == "bench"
    assert first["diff_ref"] == "sha:1"
    assert isinstance(first["ts"], float)

    second = _attempt(
        ledger,
        "P1",
        round_num=2,
        what="revert",
        accepted=False,
        fixed=["t2"],
        total=0,
        passed=0,
    )
    assert second["score"] == 0.0
    problem = find_problem(ledger, "P1")
    assert problem is not None
    assert problem["attempts"] == [first, second]

    explicit = _attempt(
        ledger,
        "P1",
        round_num=3,
        what="score override",
        score=0.123,
    )
    assert explicit["score"] == pytest.approx(0.123)


def test_record_attempt_validations() -> None:
    ledger = new_ledger()
    open_problem(ledger, "parser")

    with pytest.raises(ValueError):
        _attempt(ledger, "P1", round_num=1, what="x", action="bogus")
    with pytest.raises(ValueError):
        _attempt(ledger, "P1", round_num=1, what="   ")
    with pytest.raises(ValueError):
        record_attempt(
            ledger,
            problem_id="P1",
            round_num=1,
            sample="s",
            action="new",
            accepted=True,
            domain="code",
            what="x",
            files=[],
            predicted=None,  # type: ignore[arg-type]
            fixed=[],
            missed=[],
            passed=0,
            total=0,
        )
    with pytest.raises(KeyError):
        _attempt(ledger, "P9", round_num=1, what="x")


def test_record_attempt_auto_opens_problem() -> None:
    ledger = new_ledger()
    attempt = _attempt(
        ledger,
        "P5",
        round_num=4,
        what="auto open",
        problem_name="Discovered issue",
        tier="model_specific",
    )
    problem = find_problem(ledger, "P5")
    assert problem is not None
    assert problem["name"] == "Discovered issue"
    assert problem["opened_round"] == 4
    assert problem["tier"] == "model_specific"
    assert problem["attempts"] == [attempt]


# --------------------------------------------------------------------------- carry forward


def test_carry_forward_builds_parent_chain_without_mutation() -> None:
    source = new_ledger(domain="code")
    open_problem(source, "parser", problem_id="P1")
    _attempt(source, "P1", round_num=1, what="first fix", fixed=["t1"])
    source["carried_from"] = {"from": "run0", "into": "run1", "parent": None, "carried_at": 1.0}
    source["aliases"] = {"P2": "P1"}

    carried = carry_forward(source, from_run="run1", into_run="run2")

    assert carried is not source
    assert carried["carried_from"]["from"] == "run1"
    assert carried["carried_from"]["into"] == "run2"
    assert carried["carried_from"]["parent"] == source["carried_from"]
    assert isinstance(carried["carried_from"]["carried_at"], float)
    assert carried["aliases"] == {"P2": "P1"}
    assert carried["problems"][0]["opened_round"] == 1
    assert carried["problems"][0]["attempts"][0]["what"] == "first fix"

    carried["problems"][0]["attempts"][0]["what"] = "mutated"
    assert source["problems"][0]["attempts"][0]["what"] == "first fix"

    again = carry_forward(carried, from_run="run2", into_run="run3")
    assert again["carried_from"]["parent"]["from"] == "run1"
    assert again["carried_from"]["parent"]["parent"]["from"] == "run0"


# --------------------------------------------------------------------------- schema breaks


def test_record_schema_break() -> None:
    ledger = new_ledger(interface_version="1.0")
    record = record_schema_break(ledger, new_version="2.0", reason="field rename")
    assert record["from_version"] == "1.0"
    assert record["to_version"] == "2.0"
    assert record["reason"] == "field rename"
    assert isinstance(record["ts"], float)
    assert ledger["schema_breaks"] == [record]
    assert ledger["interface"]["version"] == "2.0"


# --------------------------------------------------------------------------- compaction


def test_compact_sim_archives_and_distills(tmp_path: Path) -> None:
    ledger = new_ledger()
    problem = open_problem(ledger, "parser drift", problem_id="P1")
    _attempt(
        ledger,
        "P1",
        round_num=1,
        what="bad patch",
        accepted=False,
        fixed=[],
        policy_model="policy-a",
    )
    _attempt(
        ledger,
        "P1",
        round_num=2,
        what="noop patch",
        accepted=True,
        fixed=[],
        policy_model="policy-a",
    )
    _attempt(
        ledger,
        "P1",
        round_num=3,
        what="good patch",
        accepted=True,
        fixed=["t1"],
        policy_model=["policy-a", "policy-b"],
    )
    _attempt(
        ledger,
        "P1",
        round_num=4,
        what="broke patch",
        accepted=True,
        fixed=["t2"],
        regressed=["t1"],
        policy_model="policy-b",
    )
    ledger["aliases"] = {"P2": "P1"}

    compacted, archive_path = compact_sim(
        ledger, archive_dir=tmp_path, keep_recent_attempts=1
    )

    assert archive_path is not None
    assert archive_path.exists()
    assert archive_path.name.startswith("sim_archive_")
    assert archive_path.name.endswith(".json")
    archived = json.loads(archive_path.read_text(encoding="utf-8"))
    assert len(archived["problems"][0]["attempts"]) == 4

    compacted_problem = compacted["problems"][0]
    assert len(compacted_problem["attempts"]) == 1
    assert compacted_problem["attempts"][0]["what"] == "broke patch"
    assert compacted_problem["archived_attempts_count"] == 3
    assert "compacted_at" in compacted

    summary = compacted_problem["invariant_summary"]
    assert summary.startswith("parser drift: ")
    assert "good patch" in summary

    dead_ends = compacted_problem["dead_ends"]
    assert any(entry.startswith("[rejected]") and "bad patch" in entry for entry in dead_ends)
    assert any(entry.startswith("[zero-fix]") and "noop patch" in entry for entry in dead_ends)
    assert any(entry.startswith("[regressed]") and "broke patch" in entry for entry in dead_ends)

    notes = compacted_problem["model_notes"]
    assert "policy-a" in notes
    assert "policy-a, policy-b" in notes
    assert any("noop patch" in note for note in notes["policy-a"])
    assert any("good patch" in note for note in notes["policy-a, policy-b"])
    assert compacted["aliases"] == {"P2": "P1"}

    # The source ledger is never mutated.
    assert len(problem["attempts"]) == 4


def test_compact_sim_without_archive_and_zero_keep() -> None:
    ledger = new_ledger()
    open_problem(ledger, "parser", problem_id="P1")
    _attempt(ledger, "P1", round_num=1, what="a", fixed=["t1"])
    _attempt(ledger, "P1", round_num=2, what="b", fixed=["t2"])

    compacted, archive_path = compact_sim(ledger, keep_recent_attempts=0)
    assert archive_path is None
    assert compacted["problems"][0]["attempts"] == []
    assert compacted["problems"][0]["archived_attempts_count"] == 2
    assert len(ledger["problems"][0]["attempts"]) == 2


def test_compact_sim_is_idempotent_for_distilled_fields(tmp_path: Path) -> None:
    ledger = new_ledger()
    open_problem(ledger, "parser", problem_id="P1")
    _attempt(ledger, "P1", round_num=1, what="bad", accepted=False)
    _attempt(ledger, "P1", round_num=2, what="good", fixed=["t1"])

    once, _ = compact_sim(ledger, archive_dir=tmp_path, keep_recent_attempts=1)
    twice, _ = compact_sim(once, archive_dir=tmp_path, keep_recent_attempts=1)

    assert twice["problems"][0]["dead_ends"] == once["problems"][0]["dead_ends"]
    assert twice["problems"][0]["model_notes"] == once["problems"][0]["model_notes"]
    assert twice["problems"][0]["invariant_summary"] == once["problems"][0]["invariant_summary"]


# --------------------------------------------------------------------------- validation


def test_validate_sim_accepts_good_ledger() -> None:
    ledger = new_ledger()
    open_problem(ledger, "parser", problem_id="P1")
    _attempt(ledger, "P1", round_num=1, what="fix", fixed=["t1"])
    ledger["aliases"] = {"P2": "P1"}
    ok, errors = validate_sim(ledger)
    assert ok is True
    assert errors == []


def test_validate_sim_detects_structural_errors() -> None:
    ok, errors = validate_sim([])  # type: ignore[arg-type]
    assert ok is False
    assert errors

    ledger = new_ledger()
    open_problem(ledger, "one", problem_id="P1")
    ledger["problems"].append(
        {
            "id": "P1",
            "name": "",
            "opened_round": "first",
            "tier": "nope",
            "attempts": [
                {
                    "action": "bogus",
                    "what": "",
                    "predicted": "t1",
                    "fixed": None,
                    "missed": [],
                    "passed": 5,
                    "total": 2,
                    "policy_model": "",
                    "domain": "",
                }
            ],
        }
    )
    ok, errors = validate_sim(ledger)
    assert ok is False
    joined = "\n".join(errors)
    assert "duplicate problem id 'P1'" in joined
    assert "invalid 'opened_round'" in joined
    assert "invalid tier" in joined
    assert "invalid action" in joined
    assert "missing 'what'" in joined
    assert "must be a list" in joined
    assert "exceeds total" in joined
    assert "missing 'policy_model' and 'domain'" in joined


def test_validate_sim_detects_alias_problems() -> None:
    ledger = new_ledger()
    open_problem(ledger, "one", problem_id="P1")
    ledger["aliases"] = {"P1": "P2", "P2": "P1"}
    ok, errors = validate_sim(ledger)
    assert ok is False
    assert any("alias cycle detected" in error for error in errors)

    ledger = new_ledger()
    open_problem(ledger, "one", problem_id="P1")
    ledger["aliases"] = {"P1": "P9"}
    ok, errors = validate_sim(ledger)
    assert ok is False
    assert any("missing problem 'P9'" in error for error in errors)


# --------------------------------------------------------------------------- prompt and summary


def test_format_sim_for_prompt_is_complete_json() -> None:
    ledger = new_ledger()
    open_problem(ledger, "parser", problem_id="P1")
    _attempt(ledger, "P1", round_num=1, what="x" * 500, fixed=["t1"])

    text = format_sim_for_prompt(ledger)
    assert json.loads(text) == ledger
    assert "\n  " in text
    assert not text.endswith("\n")
    assert "x" * 500 in text


def test_render_sim_summary_width_and_content() -> None:
    ledger = new_ledger(domain="code", policy_model="policy-a", benchmark="bench")
    open_problem(ledger, "a very long parser drift problem name", problem_id="P1")
    _attempt(ledger, "P1", round_num=1, what="bad", accepted=False)
    _attempt(ledger, "P1", round_num=2, what="good", fixed=["t1"])
    ledger["aliases"] = {"P2": "P1"}
    ledger["carried_from"] = {
        "from": "run1",
        "into": "run2",
        "parent": {"from": "run0", "into": "run1", "parent": None},
        "carried_at": 1.0,
    }

    for width in (60, 72, 96, 120):
        rendered = render_sim_summary(ledger, color=True, width=width)
        lines = rendered.splitlines()
        assert all(visible_len(line) <= width for line in lines)
        assert "SIM LEDGER" in rendered
        assert "P1" in rendered
        assert "carried" in rendered
        assert "run0→run1" in rendered
        assert "aliases" in rendered
        assert "P2→P1" in rendered

    plain = render_sim_summary(ledger, color=False, width=60)
    assert plain == strip_ansi(plain)
    assert "P1" in plain


def test_render_sim_summary_empty_ledger() -> None:
    rendered = render_sim_summary(new_ledger(), color=False, width=60)
    lines = rendered.splitlines()
    assert all(visible_len(line) <= 60 for line in lines)
    assert "no problems recorded" in rendered
    assert "aliases none" in rendered
