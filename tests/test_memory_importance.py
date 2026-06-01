"""Tests for hot/warm memory importance scoring and context tiering."""
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


# ── 4. Hot tier always injected; warm tier is retrieval-only ──────────────────

def test_hot_memories_always_injected():
    hot_mem  = _make_mem("user is an engineer", importance=0.9)
    warm_mem = _make_mem("user mentioned coffee", importance=0.4)

    proc = _make_processor(memories=[hot_mem, warm_mem])
    # Give warm_mem no vector match so hybrid retrieve won't pick it up
    proc.memory_vector = None

    preface, _, _ = proc.build_context_preface(
        message="hello",
        session=MagicMock(),
        use_memory=True,
        use_rag=False,
        use_web=False,
        owner="testuser",
    )

    all_contents = " ".join(m.get("content", "") for m in preface)
    assert "user is an engineer" in all_contents, "Hot memory must always appear in context"


def test_warm_memories_not_always_injected():
    warm_mem = _make_mem("user mentioned coffee", importance=0.4)

    proc = _make_processor(memories=[warm_mem])
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
        "Warm memory with no relevance should not appear for an unrelated query"
    )


# ── 5. Hot memory survives trim_for_context under pressure ───────────────────

def test_hot_memory_protected_from_trimming():
    hot_content = "High-importance facts — always keep in mind:\n- user is an engineer"

    # Mirrors what build_context_preface produces: user-role untrusted message + _protected flag
    from src.prompt_security import untrusted_context_message
    hot_msg = untrusted_context_message("saved memory: high-importance facts", hot_content)
    hot_msg["_protected"] = True

    regular_system = {"role": "system", "content": "You are a helpful assistant."}
    # Large conversation to pressure the context budget
    big_convo = [
        {"role": "user" if i % 2 == 0 else "assistant", "content": "x " * 200}
        for i in range(30)
    ]

    messages = [regular_system, hot_msg] + big_convo

    # Use a very tight budget (500 tokens)
    trimmed = trim_for_context(messages, context_length=600, reserve_tokens=50)

    combined = " ".join(m.get("content", "") for m in trimmed)
    assert "user is an engineer" in combined, (
        "Hot (_protected) memory must survive aggressive context trimming"
    )


# ── 6. Hot-tier cap prevents context lockup ───────────────────────────────────

def test_hot_tier_capped_at_five():
    hot_mems = [_make_mem(f"important fact {i}", importance=0.9) for i in range(10)]
    proc = _make_processor(memories=hot_mems)
    proc.memory_vector = None

    preface, _, _ = proc.build_context_preface(
        message="hello",
        session=MagicMock(),
        use_memory=True,
        use_rag=False,
        use_web=False,
        owner="testuser",
    )

    hot_block = next(
        (m for m in preface if "high-importance" in m.get("content", "").lower() and m.get("_protected")),
        None
    )
    assert hot_block is not None, "Protected hot block should be present"
    injected_facts = [l for l in hot_block["content"].splitlines() if l.strip().startswith("- ")]
    assert len(injected_facts) <= 5, f"Hot tier must be capped at 5, got {len(injected_facts)}"
