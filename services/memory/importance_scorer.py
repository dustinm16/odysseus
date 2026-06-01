"""
Importance scorer for memory entries.

Pure function — no side effects, no imports from the memory pipeline.
Call score_importance() at every add_entry() call site; pass the result
as the importance= argument so it lands in the stored entry.

Scoring layers (applied in order, highest wins):
  1. Content patterns   — regex triggers on the memory text
  2. Category floor     — minimum score per category when no pattern fires
  3. Source cap         — auto/ai_agent sources are capped at 0.85 so the
                          agent cannot auto-promote entries to critical (1.0)
"""

import re
from typing import Optional

# ---------------------------------------------------------------------------
# Pattern table  (most-specific / highest-scoring first)
# Each entry: (compiled_regex, score, optional_category_hint)
# category_hint is informational only — scoring fires regardless of category.
# ---------------------------------------------------------------------------

_WEAK_NOUNS = frozenset(
    "bit little lot few good bad big very quite rather just really fairly "
    "pretty much more less sure right wrong okay fine great cool nice".split()
)

_PATTERNS: list[tuple[re.Pattern, float, Optional[str]]] = [
    # ── Critical identity (1.0) ──────────────────────────────────────────
    (re.compile(r"\bmy name is\s+\w", re.I),             1.0, "identity"),
    (re.compile(r"\bcall me\s+\w", re.I),                1.0, "identity"),

    # ── Strong identity / explicit remember request (0.9) ────────────────
    (re.compile(r"\bi (?:work|worked) (?:at|for|as)\b", re.I), 0.9, "identity"),
    (re.compile(r"\bi (?:live|am based|'?m based) in\b", re.I), 0.9, "identity"),
    (re.compile(r"\bbased in\b", re.I),                  0.9, "identity"),
    (re.compile(
        r"\bmy (?:wife|husband|partner|girlfriend|boyfriend|spouse|"
        r"son|daughter|mother|father|mom|dad|brother|sister|child|kids?)\b",
        re.I,
    ),                                                    0.9, "identity"),
    (re.compile(r"\b(?:please )?(?:remember|memorize|save|note|store) this\b", re.I), 0.9, None),
    (re.compile(r"\b(?:please )?(?:remember|memorize|save|note|store) that\b", re.I), 0.9, None),
    (re.compile(r"\bplease (?:remember|save|note)\b",    re.I), 0.9, None),

    # ── Role identity / preference (0.85) ────────────────────────────────
    (re.compile(r"\bi am (?:a|an) (\w+)", re.I),         0.85, "identity"),  # guarded below
    (re.compile(r"\bi'?m (?:a|an) (\w+)", re.I),         0.85, "identity"),  # guarded below
    (re.compile(r"\bi (?:prefer|like|love)\b", re.I),    0.85, "preference"),
    (re.compile(r"\bi'?m allergic to\b", re.I),          0.85, "preference"),
    (re.compile(r"\bi (?:hate|can'?t stand|cannot stand|dislike)\b", re.I), 0.85, "preference"),

    # ── Goals (0.75) ──────────────────────────────────────────────────────
    (re.compile(r"\bi (?:want|plan|hope|intend|need) to\b", re.I), 0.75, "goal"),
]

# Negation window: if one of these appears in the 40 chars before a pattern
# match, the pattern score is suppressed back to the category floor.
_NEGATION_RE = re.compile(
    r"\b(not|don'?t|doesn'?t|never|no longer|isn'?t|aren'?t|wasn'?t|weren'?t|used to)\b",
    re.I,
)

# ---------------------------------------------------------------------------
# Category base floors (used when no pattern fires, or after negation)
# ---------------------------------------------------------------------------

_CATEGORY_FLOOR: dict[str, float] = {
    "identity":   0.80,
    "preference": 0.70,
    "goal":       0.70,
    "project":    0.60,
    "contact":    0.60,
    "task":       0.55,
    "fact":       0.50,
}

# ---------------------------------------------------------------------------
# Source cap — auto-added memories can't reach 1.0
# ---------------------------------------------------------------------------

_SOURCE_CAP: dict[str, float] = {
    "auto":     0.85,
    "ai_agent": 0.85,
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def score_importance(
    text: str,
    category: str = "fact",
    source: str = "user",
) -> float:
    """Return an importance score in [0.0, 1.0] for a memory entry.

    Parameters
    ----------
    text     : the memory text to score
    category : memory category (identity/preference/goal/fact/etc.)
    source   : "user" | "auto" | "ai_agent"
               user    — manual input, no source cap
               auto    — LLM extractor, capped at 0.85
               ai_agent — MCP / agent path, capped at 0.85
    """
    text = (text or "").strip()
    floor = _CATEGORY_FLOOR.get(category, 0.5)
    cap = _SOURCE_CAP.get(source, 1.0)
    raw = floor

    for pattern, score, _ in _PATTERNS:
        m = pattern.search(text)
        if not m:
            continue

        # Guard: "I am a/an <noun>" — skip weak nouns like "bit", "little"
        if pattern.groups and m.lastindex:
            noun = (m.group(1) or "").lower().strip()
            if noun in _WEAK_NOUNS:
                continue

        # Negation window: look at the 40 chars before the match start
        window = text[max(0, m.start() - 40) : m.start()]
        if _NEGATION_RE.search(window):
            continue

        if score > raw:
            raw = score

    return round(min(raw, cap), 4)
