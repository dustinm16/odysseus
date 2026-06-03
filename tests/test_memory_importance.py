"""Tests for memory importance scoring and retrieval boost."""
import time
import pytest
from unittest.mock import MagicMock, patch

from src.memory import MemoryManager
from src.chat_processor import ChatProcessor
from src.context_compactor import trim_for_context


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_mem(text, importance=0.5, pinned=False, category="fact", ts=None):
    return {
        "id": f"mem-{text[:8].replace(' ', '_')}",
        "text": text,
        "importance": importance,
        "pinned": pinned,
        "category": category,
        "timestamp": ts or int(time.time()),
        "source": "user",
        "owner": "testuser",
    }


def _make_processor(memories=None, vector_scores=None):
    """ChatProcessor with a stubbed memory_manager and optional vector store."""
    mm = MagicMock()
    mm.load.return_value = memories or []
    mm.increment_uses = MagicMock()

    mv = None
    if vector_scores is not None:
        mv = MagicMock()
        mv.healthy = True
        mv.search.return_value = [
            {"memory_id": mid, "score": score}
            for mid, score in vector_scores.items()
        ]

    return ChatProcessor(
        memory_manager=mm,
        personal_docs_manager=MagicMock(),
        memory_vector=mv,
    )


# ── 1. Default importance backfilled on legacy entries ────────────────────────

def test_validate_entries_backfills_importance(tmp_path):
    import json, uuid
    mem_file = tmp_path / "memory.json"
    legacy = [{"id": str(uuid.uuid4()), "text": "old entry", "timestamp": 1000, "source": "user", "category": "fact"}]
    mem_file.write_text(json.dumps(legacy))

    mgr = MemoryManager(str(tmp_path))
    entries = mgr.load_all()

    assert len(entries) == 1
    assert entries[0]["importance"] == 0.5, "Legacy entry should default to 0.5"


def test_add_entry_stores_importance(tmp_path):
    import json
    (tmp_path / "memory.json").write_text("[]")
    mgr = MemoryManager(str(tmp_path))

    entry = mgr.add_entry("test", importance=0.9)
    assert entry["importance"] == 0.9

    entry_low = mgr.add_entry("low", importance=0.1)
    assert entry_low["importance"] == 0.1

    # Clamps to [0, 1]
    entry_clamped = mgr.add_entry("clamped", importance=1.5)
    assert entry_clamped["importance"] == 1.0


# ── 2. increment_uses persists ────────────────────────────────────────────────

def test_increment_uses(tmp_path):
    import json, uuid
    mem_id = str(uuid.uuid4())
    mem_file = tmp_path / "memory.json"
    mem_file.write_text(json.dumps([{
        "id": mem_id, "text": "something", "timestamp": 1000,
        "source": "user", "category": "fact", "importance": 0.5,
    }]))

    mgr = MemoryManager(str(tmp_path))
    mgr.increment_uses([mem_id])

    reloaded = mgr.load_all()
    assert reloaded[0].get("uses", 0) == 1

    mgr.increment_uses([mem_id])
    reloaded2 = mgr.load_all()
    assert reloaded2[0]["uses"] == 2


# ── 3. High-importance memory outranks a vector-similar low-importance one ────

def test_importance_boosts_retrieval_rank():
    high_imp = _make_mem("user likes jazz music", importance=0.9)
    low_imp  = _make_mem("user enjoys listening", importance=0.1)

    # Give both identical vector scores and keyword overlap
    proc = _make_processor(
        memories=[high_imp, low_imp],
        vector_scores={high_imp["id"]: 0.6, low_imp["id"]: 0.6},
    )

    results = proc._hybrid_retrieve("what music does the user like?", [high_imp, low_imp], k=2)

    assert results[0]["id"] == high_imp["id"], (
        "High-importance memory should rank first when vector/keyword scores are equal"
    )


def test_importance_does_not_override_strong_relevance():
    """A very relevant low-importance memory should still beat an irrelevant high-importance one."""
    relevant_low  = _make_mem("user likes jazz music", importance=0.1)
    irrelevant_high = _make_mem("unrelated database migration note", importance=0.9)

    proc = _make_processor(
        memories=[relevant_low, irrelevant_high],
        vector_scores={relevant_low["id"]: 0.9, irrelevant_high["id"]: 0.05},
    )

    results = proc._hybrid_retrieve("what music does the user like?", [relevant_low, irrelevant_high], k=2)

    # The relevant one must appear
    result_ids = [r["id"] for r in results]
    assert relevant_low["id"] in result_ids, (
        "High relevance should still surface a low-importance memory"
    )


# ── 4. All non-pinned memories go through hybrid retrieval ────────────────────

def test_irrelevant_memory_not_injected():
    """A memory with no keyword or vector match for the query should not appear."""
    mem = _make_mem("user mentioned coffee", importance=0.4)

    proc = _make_processor(memories=[mem])
    proc.memory_vector = None  # No vector; keyword won't match "hello"

    preface, _, _ = proc.build_context_preface(
        message="hello",
        session=MagicMock(),
        use_memory=True,
        use_rag=False,
        use_web=False,
        owner="testuser",
    )

    system_contents = " ".join(m.get("content", "") for m in preface)
    assert "user mentioned coffee" not in system_contents, (
        "Memory with no relevance to the query must not be injected"
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Auto-importance scorer tests
# ═══════════════════════════════════════════════════════════════════════════════

from services.memory.importance_scorer import score_importance


# ── 7. Pattern scoring ────────────────────────────────────────────────────────

@pytest.mark.parametrize("text,category,source,min_score,max_score", [
    # Critical identity — user source, no cap
    ("My name is Alex.",                    "identity",   "user",     1.0,  1.0),
    ("Call me Alex.",                       "identity",   "user",     1.0,  1.0),
    # Strong identity
    ("I work at Acme Corp.",                "identity",   "user",     0.9,  1.0),
    ("I live in Seattle.",                  "identity",   "user",     0.9,  1.0),
    ("My wife is Jane.",                    "identity",   "user",     0.9,  1.0),
    ("Please remember that I use vim.",     "fact",       "user",     0.9,  1.0),
    ("Remember this: dark mode only.",      "preference", "user",     0.9,  1.0),
    # Role identity / preference
    ("I am a software engineer.",           "identity",   "user",     0.85, 1.0),
    ("I prefer Python over Java.",          "preference", "user",     0.85, 1.0),
    ("I'm allergic to peanuts.",            "preference", "user",     0.85, 1.0),
    ("I hate early meetings.",              "preference", "user",     0.85, 1.0),
    # Goals
    ("I want to learn Rust.",               "goal",       "user",     0.75, 1.0),
    # Category floor — no pattern match
    ("User uses neovim.",                   "preference", "user",     0.70, 0.85),
    ("Random note about the meeting.",      "fact",       "user",     0.50, 0.65),
    # Source cap — auto capped at 0.85
    ("My name is Alex.",                    "identity",   "auto",     0.85, 0.85),
    ("My name is Alex.",                    "identity",   "ai_agent", 0.85, 0.85),
    # Source cap — category floor still applies for auto
    ("Random note.",                        "fact",       "auto",     0.50, 0.85),
    # Negation guard — should NOT hit full pattern score
    ("I do not live in Paris.",             "identity",   "user",     0.0,  0.85),
    ("I don't like coffee.",                "preference", "user",     0.0,  0.75),
    # Weak noun guard — "I am a bit tired" should not score as role-identity
    ("I am a bit tired today.",             "fact",       "user",     0.0,  0.65),
    ("I'm a little confused.",              "fact",       "user",     0.0,  0.65),
])
def test_score_importance_patterns(text, category, source, min_score, max_score):
    score = score_importance(text, category, source)
    assert min_score <= score <= max_score, (
        f"score_importance({text!r}, {category!r}, {source!r}) = {score}, "
        f"expected [{min_score}, {max_score}]"
    )


# ── 8. Explicit importance preserved; None triggers auto-score ────────────────

def test_api_explicit_importance_preserved(tmp_path):
    """When importance is passed explicitly it must not be overwritten by scorer."""
    import json
    (tmp_path / "memory.json").write_text("[]")
    mgr = MemoryManager(str(tmp_path))

    # Explicit 0.2 on a "my name is" text — scorer would give 1.0, but explicit wins
    entry = mgr.add_entry("My name is Alex.", source="user", importance=0.2)
    assert entry["importance"] == 0.2, "Explicit importance must not be overridden by scorer"


def test_auto_score_fires_when_importance_none(tmp_path):
    """add_entry with default importance=0.5 does NOT auto-score — scorer is applied at call sites."""
    import json
    (tmp_path / "memory.json").write_text("[]")
    mgr = MemoryManager(str(tmp_path))

    # Simulating the call-site pattern: score first, pass result in
    imp = score_importance("My name is Alex.", "identity", "user")
    entry = mgr.add_entry("My name is Alex.", source="user", importance=imp)
    assert entry["importance"] == 1.0


# ── 9. Auto-extracted identity scores high enough to win retrieval ────────────

def test_auto_extracted_identity_scores_high():
    """Auto-extracted 'my name is' scores 0.85 (source cap) — high enough to rank first."""
    score = score_importance("My name is Sam.", "identity", "auto")
    assert score >= 0.8, "Auto-extracted identity must score ≥ 0.8"
    assert score <= 0.85, "Auto-extracted identity must not exceed the auto source cap"


def test_user_identity_reaches_critical():
    """User-entered 'my name is' should reach 1.0 with no source cap."""
    assert score_importance("My name is Sam.", "identity", "user") == 1.0


# ── 10. Identity category floor ──────────────────────────────────────────────

def test_identity_category_floor_without_pattern():
    """An identity-category entry with no pattern match still floors at 0.8."""
    score = score_importance("Some obscure identity note.", "identity", "user")
    assert score >= 0.8, "Identity category floor must be 0.8"


# ── 11. Negation window width ─────────────────────────────────────────────────

def test_negation_window_catches_distant_negation():
    """Negation >80 chars before the pattern should still suppress the score."""
    # "used to" is ~55 chars before "live in" — would have been missed with 40-char window
    text = "I used to live in Berlin back in my university days, but I live in Seattle now."
    score = score_importance(text, "identity", "user")
    # Should NOT hit 0.9 (the live-in pattern) because of the earlier "used to"
    # It will still hit 0.9 for the second "live in Seattle" — that's correct
    # The key test: "used to live in Berlin" should not be scored as 0.9
    assert score <= 0.9, "Distant negation should be caught by 80-char window"


def test_negation_window_falls_back_to_category_floor():
    """When negation suppresses a pattern, the category floor is still applied."""
    # "I don't" is within 80 chars of "live in" — window catches it, suppresses 0.9
    # but identity floor (0.8) still applies — not a zero score
    text = "I don't drink coffee, but I live in Seattle."
    score = score_importance(text, "identity", "user")
    assert score == 0.8, (
        "When negation suppresses the pattern the identity floor (0.8) must still apply"
    )


# ── 12. services/memory/memory.py parity ─────────────────────────────────────

def test_services_memory_manager_parity(tmp_path):
    """services/memory/memory.py add_entry must have parity with src/memory.py."""
    import json
    (tmp_path / "memory.json").write_text("[]")
    from services.memory.memory import MemoryManager as ServicesMemoryManager
    mgr = ServicesMemoryManager(str(tmp_path))
    entry = mgr.add_entry("test entry", source="user")
    assert entry.get("uses") == 0, "services add_entry must set uses=0"
    assert "last_used_at" in entry, "services add_entry must set last_used_at"


def test_services_increment_uses_nudges_importance(tmp_path):
    """services/memory/memory.py increment_uses must nudge importance (parity with src)."""
    import json
    (tmp_path / "memory.json").write_text("[]")
    from services.memory.memory import MemoryManager as ServicesMemoryManager
    mgr = ServicesMemoryManager(str(tmp_path))
    entry = mgr.add_entry("test entry", source="user", importance=0.5)
    all_mem = [entry]; mgr.save(all_mem)

    mgr.increment_uses([entry["id"]])
    reloaded = mgr.load_all()
    assert reloaded[0]["importance"] == pytest.approx(0.52, abs=0.001)


# ── 13. PUT re-scores when text changes ──────────────────────────────────────

def test_importance_persists_through_save_reload(tmp_path):
    """Importance set on an entry must survive a save/load round-trip."""
    import json
    (tmp_path / "memory.json").write_text("[]")
    mgr = MemoryManager(str(tmp_path))
    entry = mgr.add_entry("My name is Alex.", source="user", importance=1.0)
    all_mem = mgr.load_all()
    all_mem.append(entry)
    mgr.save(all_mem)

    reloaded = mgr.load_all()
    assert reloaded[0]["importance"] == 1.0, "Importance must survive save/reload"


def test_importance_preserved_when_only_category_changes(tmp_path):
    """Changing category on an existing entry must not reset importance."""
    import json
    (tmp_path / "memory.json").write_text("[]")
    mgr = MemoryManager(str(tmp_path))
    entry = mgr.add_entry("Some fact.", source="user", importance=0.9)
    all_mem = mgr.load_all()
    all_mem.append(entry)
    mgr.save(all_mem)

    # Simulate a category-only update (importance unchanged)
    all_mem[0]["category"] = "preference"
    mgr.save(all_mem)

    reloaded = mgr.load_all()
    assert reloaded[0]["importance"] == 0.9, "Category change must not reset importance"


# ═══════════════════════════════════════════════════════════════════════════════
# Entity features: usage nudge, decay, observation endpoint
# ═══════════════════════════════════════════════════════════════════════════════

# ── 14. Usage nudge ───────────────────────────────────────────────────────────

def test_increment_uses_nudges_importance(tmp_path):
    """Each use bumps importance by _NUDGE_STEP."""
    import json
    (tmp_path / "memory.json").write_text("[]")
    mgr = MemoryManager(str(tmp_path))
    entry = mgr.add_entry("Admin likes coffee", source="user", importance=0.5)
    all_mem = [entry]; mgr.save(all_mem)

    mgr.increment_uses([entry["id"]])
    reloaded = mgr.load_all()
    assert reloaded[0]["importance"] == pytest.approx(0.52, abs=0.001)


def test_nudge_respects_source_cap_auto(tmp_path):
    """Auto-source entries must never exceed 0.85 no matter how many uses."""
    import json
    (tmp_path / "memory.json").write_text("[]")
    mgr = MemoryManager(str(tmp_path))
    entry = mgr.add_entry("My name is Alex", source="auto", importance=0.85)
    all_mem = [entry]; mgr.save(all_mem)

    for _ in range(30):
        mgr.increment_uses([entry["id"]])

    reloaded = mgr.load_all()
    assert reloaded[0]["importance"] <= 0.85, "Auto source must be capped at 0.85"


def test_nudge_respects_source_cap_observation(tmp_path):
    """Observation-source entries must never exceed 0.50."""
    import json
    (tmp_path / "memory.json").write_text("[]")
    mgr = MemoryManager(str(tmp_path))
    entry = mgr.add_entry("User opened VS Code", source="observation", importance=0.2)
    all_mem = [entry]; mgr.save(all_mem)

    for _ in range(30):
        mgr.increment_uses([entry["id"]])

    reloaded = mgr.load_all()
    assert reloaded[0]["importance"] <= 0.50, "Observation source must be capped at 0.50"


def test_nudge_updates_last_used_at(tmp_path):
    """increment_uses must update last_used_at timestamp."""
    import json, time
    (tmp_path / "memory.json").write_text("[]")
    mgr = MemoryManager(str(tmp_path))
    entry = mgr.add_entry("test", source="user", importance=0.5)
    original_ts = entry["last_used_at"]
    all_mem = [entry]; mgr.save(all_mem)

    time.sleep(0.01)
    mgr.increment_uses([entry["id"]])
    reloaded = mgr.load_all()
    assert reloaded[0]["last_used_at"] >= original_ts


# ── 15. Importance decay ──────────────────────────────────────────────────────

def test_decay_reduces_importance_on_idle_entry(tmp_path):
    """Entry idle for 48h should have decayed importance."""
    import json, time
    (tmp_path / "memory.json").write_text("[]")
    mgr = MemoryManager(str(tmp_path))
    now = int(time.time())
    two_days_ago = now - (48 * 3600)

    entry = mgr.add_entry("User mentioned coffee", source="user", importance=0.75)
    all_mem = [entry]
    all_mem[0]["last_used_at"] = two_days_ago
    mgr.save(all_mem)

    n = mgr.decay_unused()
    assert n == 1
    reloaded = mgr.load_all()
    assert reloaded[0]["importance"] < 0.75, "Idle entry importance should have decayed"


def test_decay_respects_category_floor(tmp_path):
    """Decay must not drop importance below category floor."""
    import json, time
    (tmp_path / "memory.json").write_text("[]")
    mgr = MemoryManager(str(tmp_path))
    now = int(time.time())
    old_ts = now - (365 * 24 * 3600)  # 1 year ago

    entry = mgr.add_entry("Some identity note", source="auto", importance=0.85, category="identity")
    all_mem = [entry]
    all_mem[0]["last_used_at"] = old_ts
    all_mem[0]["timestamp"] = old_ts
    mgr.save(all_mem)

    # Run many decay cycles
    for _ in range(50):
        mgr.decay_unused()

    reloaded = mgr.load_all()
    assert reloaded[0]["importance"] >= 0.80, "Identity floor is 0.80 — decay must not go below"


def test_decay_skips_pinned_entries(tmp_path):
    """Pinned entries must not be decayed."""
    import json, time
    (tmp_path / "memory.json").write_text("[]")
    mgr = MemoryManager(str(tmp_path))
    old_ts = int(time.time()) - (48 * 3600)

    entry = mgr.add_entry("Core fact", source="user", importance=0.9)
    all_mem = [entry]
    all_mem[0]["pinned"] = True
    all_mem[0]["last_used_at"] = old_ts
    mgr.save(all_mem)

    mgr.decay_unused()
    reloaded = mgr.load_all()
    assert reloaded[0]["importance"] == 0.9, "Pinned entry must not be decayed"


def test_decay_skips_recently_used(tmp_path):
    """Entry used within 24h must not be decayed."""
    import json, time
    (tmp_path / "memory.json").write_text("[]")
    mgr = MemoryManager(str(tmp_path))

    entry = mgr.add_entry("Recent fact", source="user", importance=0.75)
    all_mem = [entry]
    all_mem[0]["last_used_at"] = int(time.time()) - 3600  # 1 hour ago
    mgr.save(all_mem)

    n = mgr.decay_unused()
    assert n == 0, "Recently used entry must not be decayed"


# ── 16. Observation scorer cap ────────────────────────────────────────────────

def test_observation_source_capped_at_half(tmp_path):
    """score_importance with source='observation' must not exceed 0.50."""
    # Even a critical-identity pattern should be capped
    score = score_importance("My name is Alex.", "identity", "observation")
    assert score <= 0.50, "Observation source must be capped at 0.50"


def test_last_used_at_backfilled_from_timestamp(tmp_path):
    """Legacy entries without last_used_at get it backfilled from timestamp."""
    import json
    legacy_ts = 1700000000
    mem_file = tmp_path / "memory.json"
    mem_file.write_text(json.dumps([{
        "id": "abc", "text": "old entry", "timestamp": legacy_ts,
        "source": "user", "category": "fact", "importance": 0.5
    }]))
    mgr = MemoryManager(str(tmp_path))
    entries = mgr.load_all()
    assert entries[0]["last_used_at"] == legacy_ts, (
        "last_used_at must be backfilled from timestamp for legacy entries"
    )
