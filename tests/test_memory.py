"""Unit tests for the agent's two-layer memory."""
from __future__ import annotations

from src.agent.memory import Lesson, LongTermMemory, ShortTermMemory, TrialRecord


def _rec(i, ret, **kw):
    return TrialRecord(trial_id=i, config={"x": i},
                       summary={"mean_return": ret, **kw})


def test_shortterm_best_picks_max_return():
    m = ShortTermMemory()
    m.add(_rec(1, 10.0)); m.add(_rec(2, 42.0)); m.add(_rec(3, 7.0))
    assert m.best().trial_id == 2


def test_shortterm_empty_context():
    assert ShortTermMemory().to_context() == "No trials yet."


def test_shortterm_context_mentions_trials():
    m = ShortTermMemory(); m.add(_rec(1, 5.0))
    ctx = m.to_context()
    assert "Trial 1" in ctx and "return" in ctx


def test_longterm_persists_across_instances(tmp_path):
    p = str(tmp_path / "lt.json")
    lt = LongTermMemory(p)
    lt.add(Lesson(task="t", diagnosis="lr too high", recommendation="lower lr"))
    lt2 = LongTermMemory(p)
    assert len(lt2.lessons) == 1
    assert lt2.lessons[0].diagnosis == "lr too high"


def test_longterm_dedupes_same_diagnosis(tmp_path):
    p = str(tmp_path / "lt.json")
    lt = LongTermMemory(p)
    lt.add(Lesson(task="t", diagnosis="d", recommendation="r1"))
    lt.add(Lesson(task="t", diagnosis="d", recommendation="r2"))
    assert len(lt.lessons) == 1
    assert lt.lessons[0].recommendation == "r2"


def test_longterm_filters_by_task(tmp_path):
    lt = LongTermMemory(str(tmp_path / "lt.json"))
    lt.add(Lesson(task="a", diagnosis="da", recommendation="ra"))
    lt.add(Lesson(task="b", diagnosis="db", recommendation="rb"))
    assert [x.diagnosis for x in lt.for_task("a")] == ["da"]


def test_longterm_survives_corrupt_file(tmp_path):
    p = str(tmp_path / "lt.json")
    with open(p, "w") as fh:
        fh.write("{ not valid json")
    lt = LongTermMemory(p)          # must not raise
    assert lt.lessons == []
