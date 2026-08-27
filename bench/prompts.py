"""The prompt set. Same prompts, same order, every system.

DIFFICULTY (dimension 1). Five tiers, five prompts each, assigned a priori --
before any system was run, and never revised after seeing a result. The tiers
are an ordinal design variable, not a measurement:

    1  trivial     answerable without thinking (2+2, days in a week)
    2  recall      one known fact, retrieved not derived
    3  explain     a constructed causal explanation
    4  reason      two or more steps of arithmetic or inference
    5  open        no correct answer exists

THE CONFOUND THIS SET EXISTS TO KILL. A gap that grows with difficulty is only
interesting if it is not just growing with prompt length. A longer prompt is
more audio to transcribe, and every cascade agent's ASR stage is roughly linear
in input duration -- so an unmatched prompt set manufactures a positive
correlation that has nothing to do with the agent modelling difficulty.

So every prompt here is 7-9 words. ``demo()`` asserts that, and the runner
additionally measures the *rendered audio duration* of each prompt and reports
its correlation with the gap alongside the difficulty correlation. If the two
correlations look alike, the difficulty result is confounded and the report
says so. Length is controlled by design and checked by measurement, not
assumed.

A second confound is left in on purpose and measured rather than removed: a
harder question tends to get a longer *reply*, and a cascade that synthesises
the whole utterance before speaking pays for that in the gap. That is a real
mechanism by which a gap can track difficulty without anything in the system
modelling difficulty, so the runner records reply token count per turn and the
gap scorer partials it out. See bench/gap.py.

AFFECT (dimension 4). A separate five-prompt set chosen to pull affectively
distinct *replies* out of a system. We cannot control what a system says; we
can only ask things that a person would answer in visibly different registers,
then measure whether the voice moves at all.
"""

from __future__ import annotations

TIERS = {
    1: "trivial",
    2: "recall",
    3: "explain",
    4: "reason",
    5: "open",
}

# (id, tier, text). Order here is the canonical order; the runner cycles it so
# every tier is equally represented at every point in a run, which matters
# because this machine's own speed drifts within a session.
DIFFICULTY_PROMPTS = [
    ("t1a", 1, "What is two plus two, in total?"),
    ("t1b", 1, "How many days are there in a week?"),
    ("t1c", 1, "What colour is fresh snow in winter?"),
    ("t1d", 1, "How many legs does a normal dog have?"),
    ("t1e", 1, "What day comes right after Monday each week?"),

    ("t2a", 2, "What is the capital city of France?"),
    ("t2b", 2, "Who wrote the play Romeo and Juliet?"),
    ("t2c", 2, "What is the largest ocean on Earth?"),
    ("t2d", 2, "In what year did the Titanic finally sink?"),
    ("t2e", 2, "Which planet in our system is the largest?"),

    ("t3a", 3, "Why does bread rise while it is baking?"),
    ("t3b", 3, "Why does the moon change shape each month?"),
    ("t3c", 3, "How does a refrigerator keep the food cold?"),
    ("t3d", 3, "Why does salt melt the ice on roads?"),
    ("t3e", 3, "Why do we feel tired after a meal?"),

    ("t4a", 4, "Three friends split twelve dollars, how much each?"),
    ("t4b", 4, "What day comes five days after a Wednesday?"),
    ("t4c", 4, "What is forty dollars with half taken off?"),
    ("t4d", 4, "I bought six eggs and broke two, how many?"),
    ("t4e", 4, "If March has thirty one days, when is Easter?"),

    ("t5a", 5, "What do you think the meaning of life is?"),
    ("t5b", 5, "Is it ever right to lie to someone?"),
    ("t5c", 5, "What makes a person genuinely happy in life?"),
    ("t5d", 5, "Should machines ever be allowed to make laws?"),
    ("t5e", 5, "What would you change about the world today?"),
]

# (id, intended affect of a natural reply, text)
AFFECT_PROMPTS = [
    ("a_joy", "joy", "I just got the job I wanted, isn't that wonderful?"),
    ("a_grief", "grief", "My dog died this morning and I feel awful."),
    ("a_alarm", "alarm", "The kitchen is on fire right now, what do I do?"),
    ("a_neutral", "neutral", "Please tell me the current local time in London."),
    ("a_flat", "flat", "Read me the numbers one through five, slowly please."),
]

# One text, repeated. Used as the measurement floor for dimension 4: the same
# system saying the same thing twice should produce near-identical prosody, and
# whatever spread survives is the noise floor of the extractor plus the stack.
FLOOR_TEXT = "The package arrived at the loading dock this morning."


def difficulty_cycle(n: int):
    """n prompts, cycling the set so tiers stay balanced at any prefix length.

    Interleaved by tier rather than blocked: turn 0 is tier 1, turn 1 tier 2,
    ... turn 5 back to tier 1. A blocked order would alias tier against
    wall-clock time, and on this machine wall-clock time moves the gap.
    """
    by_tier = {t: [p for p in DIFFICULTY_PROMPTS if p[1] == t] for t in TIERS}
    out, i = [], 0
    while len(out) < n:
        t = (i % len(TIERS)) + 1
        row = by_tier[t][(i // len(TIERS)) % len(by_tier[t])]
        out.append(row)
        i += 1
    return out


def demo():
    """Self-check: the length control actually holds, and the cycle balances."""
    lens = {}
    for pid, tier, text in DIFFICULTY_PROMPTS:
        w = len(text.split())
        assert 7 <= w <= 9, f"{pid}: {w} words, breaks the 7-9 length control"
        lens.setdefault(tier, []).append(w)
    assert len(DIFFICULTY_PROMPTS) == 25, len(DIFFICULTY_PROMPTS)
    assert len({p[0] for p in DIFFICULTY_PROMPTS}) == 25, "duplicate prompt ids"
    means = {t: sum(v) / len(v) for t, v in lens.items()}
    spread = max(means.values()) - min(means.values())
    assert spread <= 1.0, f"tier mean word counts spread {spread:.2f}, too far apart: {means}"

    c = difficulty_cycle(20)
    assert len(c) == 20
    counts = {t: sum(1 for p in c if p[1] == t) for t in TIERS}
    assert set(counts.values()) == {4}, f"unbalanced cycle at n=20: {counts}"
    # and balanced at every prefix that is a multiple of 5
    for k in (5, 10, 15):
        cc = {t: sum(1 for p in c[:k] for _ in [0] if p[1] == t) for t in TIERS}
        assert set(cc.values()) == {k // 5}, f"unbalanced at prefix {k}: {cc}"
    assert len({p[0] for p in c}) == 20, "cycle repeated a prompt inside 20 turns"

    assert len(AFFECT_PROMPTS) == 5
    print(f"prompts self-check OK  (25 difficulty prompts, tier mean word counts "
          f"{ {t: round(m, 1) for t, m in means.items()} }, spread {spread:.2f})")


if __name__ == "__main__":
    demo()
