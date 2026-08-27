"""Dimension 1 -- turn-taking timing, and whether the gap ADAPTS to difficulty.

WHAT IS COMPUTED

For each turn: ``gap_ms`` = offset of user speech -> onset of agent speech,
both silence-trimmed. Taken directly from ``live.loop.run_turn``, which
measures it with ``harness.audio.segments`` exactly as ``harness/exchange.py``
defines it. This scorer does not recompute the gap; if it did, the number would
no longer be comparable to the sibling repo's, which is the whole point of
having one definition.

Two things are then reported, and they are different questions:

1. SPEED -- median gap and IQR. Never a bare mean.

2. ADAPTATION -- Spearman rho between gap and the a-priori difficulty tier
   (1 trivial .. 5 open-ended). A person takes longer to answer "what is the
   meaning of life" than "what is two plus two". An agent that answers both in
   the same time is broken in a way listeners feel and cannot name. rho ~ 0 is
   a real and reportable finding, not a failed measurement.

   Reported alongside it, always, because otherwise the number is not
   interpretable:

   - ``tier5_minus_tier1_ms`` -- the effect in milliseconds, which is what a
     listener would actually experience.
   - ``rho_gap_vs_prompt_audio`` -- the length confound. All prompts are 7-9
     words by design (bench/prompts.py), but design is a claim and this is the
     measurement. If the gap tracks prompt *duration* as strongly as it tracks
     difficulty, the difficulty result is confounded and this says so.
   - ``rho_gap_vs_reply_tokens`` and ``rho_gap_vs_tier_partialling_reply`` --
     the mechanism check. A cascade that synthesises the whole reply before
     speaking pays for a longer reply inside the gap. So a gap CAN track
     difficulty with nothing in the system modelling difficulty: harder
     question -> wordier answer -> longer TTS -> longer gap. The partial
     correlation removes reply length from the tier-gap relationship. If rho
     collapses when reply length is partialled out, the adaptation was
     incidental, and the honest description is "the gap moves, but as a side
     effect of reply length".

NO COMPOSITE. Speed and adaptation are reported separately. A fast agent with
rho = 0 and a slow agent with rho = 0.6 are different animals and averaging
them into one "timing score" would hide the finding.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bench import prompts as P  # noqa: E402
from bench.common import audio_mod, spearman, stats, write_result  # noqa: E402
from bench.runner import (  # noqa: E402
    base_result, close_systems, interleave, open_systems, turn_stamp,
)

DIMENSION = "turn-taking timing (gap + adaptation to difficulty)"


def _partial_spearman(x, y, z) -> dict:
    """Spearman rho of x,y controlling for z, on ranks.

    rho_xy.z = (r_xy - r_xz r_yz) / sqrt((1 - r_xz^2)(1 - r_yz^2))

    First-order partial on Spearman ranks. It assumes the rank relationships
    are roughly linear, which is the standard assumption for a partial
    correlation and is stated here rather than buried.
    """
    rxy = spearman(x, y).get("rho")
    rxz = spearman(x, z).get("rho")
    ryz = spearman(y, z).get("rho")
    if None in (rxy, rxz, ryz):
        return {"rho": None, "note": "a component correlation was undefined"}
    den = ((1 - rxz**2) * (1 - ryz**2)) ** 0.5
    if den < 1e-9:
        return {"rho": None, "note": "control variable explains a component completely"}
    return {
        "rho": round((rxy - rxz * ryz) / den, 3),
        "components": {"rho_xy": rxy, "rho_xz": rxz, "rho_yz": ryz},
        "method": "first-order partial on Spearman ranks",
    }


def collect(systems, n_per_system: int, progress=True) -> list[dict]:
    """Run the difficulty cycle on every system, interleaved."""
    items = P.difficulty_cycle(n_per_system)
    a = audio_mod()
    # Prompt audio is rendered once and reused for every system, so no system
    # is answering a slightly different recording of the same sentence.
    rendered = {}
    for pid, tier, text in items:
        if pid not in rendered:
            x = systems[0].render_prompt(text)
            b = a.segments(x, merge_gap_ms=30.0, min_len_ms=20.0)
            rendered[pid] = {
                "audio": x,
                "speech_ms": round(b[-1][1] - b[0][0], 1) if b else None,
                "text": text,
                "tier": tier,
            }

    rows, i = [], 0
    for s, (pid, tier, text) in interleave(systems, items):
        r = rendered[pid]
        try:
            t = s.turn(r["audio"], label=text, record=False)
            rows.append({
                "system": s.name, "prompt_id": pid, "tier": tier,
                "tier_name": P.TIERS[tier], "prompt": text,
                "prompt_words": len(text.split()),
                "prompt_speech_ms": r["speech_ms"],
                "gap_ms": t["gap_ms"],
                "acoustic_gap_ms": t.get("acoustic_gap_ms"),
                "reply": t["reply"],
                "reply_tokens": t.get("lm_tokens"),
                "reply_words": len(str(t["reply"]).split()),
                "reply_audio_ms": t.get("reply_audio_ms"),
                "transcript": t.get("transcript"),
                "wer": t.get("wer"),
                "truncated": t.get("truncated"),
                "speculated": t.get("speculated"),
                "stage_ms": t.get("stage_ms"),
                "work_ms": t.get("work_ms"),
                **turn_stamp(),
            })
            if progress:
                print(f"  [{i:3d}] {s.name:<18} tier{tier} gap {t['gap_ms']:7.1f}ms  "
                      f"{str(t['reply'])[:40]!r}", flush=True)
        except Exception as e:  # noqa: BLE001 -- a failed turn is data
            rows.append({"system": s.name, "prompt_id": pid, "tier": tier,
                         "prompt": text, "gap_ms": None,
                         "error": f"{type(e).__name__}: {e}", **turn_stamp()})
            if progress:
                print(f"  [{i:3d}] {s.name:<18} tier{tier} FAILED {type(e).__name__}: {e}",
                      flush=True)
        i += 1
    return rows


def score(rows) -> dict:
    """Aggregate per system. Every number here traces to a turn in `rows`."""
    out = {}
    for name in sorted({r["system"] for r in rows}):
        rs = [r for r in rows if r["system"] == name]
        ok = [r for r in rs if r.get("gap_ms") is not None]
        if not ok:
            out[name] = {"n": 0, "status": "not-measured",
                         "reason": "every turn errored",
                         "errors": sorted({r.get("error", "?") for r in rs})}
            continue
        gaps = [r["gap_ms"] for r in ok]
        tiers = [r["tier"] for r in ok]
        plen = [r["prompt_speech_ms"] for r in ok]
        rtok = [r["reply_tokens"] for r in ok]

        per_tier = {}
        for t in sorted(set(tiers)):
            g = [r["gap_ms"] for r in ok if r["tier"] == t]
            per_tier[str(t)] = {"tier_name": P.TIERS[t], **stats(g)}

        t1 = [r["gap_ms"] for r in ok if r["tier"] == 1]
        t5 = [r["gap_ms"] for r in ok if r["tier"] == 5]
        eff = None
        if t1 and t5:
            import statistics as st
            eff = round(st.median(t5) - st.median(t1), 1)

        out[name] = {
            "n": len(ok),
            "n_failed": len(rs) - len(ok),
            "gap_ms": stats(gaps),
            "acoustic_gap_ms": stats([r["acoustic_gap_ms"] for r in ok
                                      if r.get("acoustic_gap_ms") is not None]),
            "per_tier_gap_ms": per_tier,
            "adaptation": {
                "rho_gap_vs_tier": spearman(tiers, gaps),
                "tier5_minus_tier1_ms": eff,
                "rho_gap_vs_prompt_audio": spearman(plen, gaps),
                "rho_gap_vs_reply_tokens": spearman(rtok, gaps),
                "rho_gap_vs_tier_partialling_reply_tokens":
                    _partial_spearman(tiers, gaps, rtok),
            },
            "reply_tokens": stats([r["reply_tokens"] for r in ok
                                   if r.get("reply_tokens") is not None]),
            "reply_audio_ms": stats([r["reply_audio_ms"] for r in ok
                                     if r.get("reply_audio_ms") is not None]),
            # a shorter gap bought by cutting the talker off is not a faster
            # agent, so these never travel apart from the gap
            "endpointing": {
                "false_endpoints": sum(1 for r in ok if r.get("truncated")),
                "n": len(ok),
                "mean_wer_vs_prompt": round(
                    sum(r["wer"] for r in ok if r.get("wer") is not None)
                    / max(1, sum(1 for r in ok if r.get("wer") is not None)), 4),
            },
            "turns_speculated": sum(1 for r in ok if r.get("speculated")),
        }
    return out


def run(system_names, n: int = 25, out=None, device=None) -> dict:
    systems, info, player = open_systems(system_names, device=device)
    if not systems:
        res = base_result(DIMENSION, info, {"rows": [], "score": {},
                                            "note": "no system opened"})
        return write_result(out or "results/gap.json", res) and res
    t0 = time.perf_counter()
    try:
        rows = collect(systems, n)
    finally:
        close_systems(systems, player)
    res = base_result(DIMENSION, info, {
        "n_per_system": n,
        "prompt_set": "bench/prompts.py DIFFICULTY_PROMPTS, 25 prompts, 5 tiers, "
                      "7-9 words each, tiers assigned before any system ran",
        "gap_definition": "user speech offset -> agent speech onset, both "
                          "silence-trimmed, via harness.audio.segments "
                          "(merge_gap_ms=30, min_len_ms=20)",
        "elapsed_s": round(time.perf_counter() - t0, 1),
        "score": score(rows),
        "rows": rows,
    })
    write_result(out or "results/gap.json", res)
    return res


def report(res) -> str:
    L = []
    L.append(f"{'system':<20} {'n':>3} {'gap median [IQR]':>24} {'rho(gap,tier)':>14} "
             f"{'p':>7} {'t5-t1 ms':>9} {'rho|reply':>10}")
    for name, s in res["score"].items():
        if not s.get("n"):
            L.append(f"{name:<20} {'-':>3}  not-measured: {s.get('reason', '?')}")
            continue
        g, a = s["gap_ms"], s["adaptation"]
        rho = a["rho_gap_vs_tier"]
        pr = a["rho_gap_vs_tier_partialling_reply_tokens"].get("rho")
        L.append(
            f"{name:<20} {g['n']:>3} {g['median']:>10.0f} [{g['p25']:.0f}-{g['p75']:.0f}]"
            f"{'':>4} {str(rho.get('rho')):>14} {str(rho.get('p')):>7} "
            f"{str(a['tier5_minus_tier1_ms']):>9} {str(pr):>10}")
    return "\n".join(L)


def demo():
    """Self-check on synthetic turns: the scorer must find adaptation when it is
    there and must NOT find it when it is not. A scorer that cannot fail to
    detect an effect is not measuring anything."""
    import random
    rng = random.Random(0)

    flat = [{"system": "flat", "tier": t, "prompt_id": f"p{i}", "prompt": "x",
             "prompt_speech_ms": 1000 + rng.gauss(0, 20), "gap_ms": 800 + rng.gauss(0, 60),
             "reply_tokens": 10 + rng.randint(0, 3), "reply": "hi", "wer": 0.0}
            for i, t in enumerate([1, 2, 3, 4, 5] * 8)]
    adapt = [{"system": "adaptive", "tier": t, "prompt_id": f"p{i}", "prompt": "x",
              "prompt_speech_ms": 1000 + rng.gauss(0, 20),
              "gap_ms": 400 + 200 * t + rng.gauss(0, 60),
              "reply_tokens": 10 + rng.randint(0, 3), "reply": "hi", "wer": 0.0}
             for i, t in enumerate([1, 2, 3, 4, 5] * 8)]
    s = score(flat + adapt)

    f, a = s["flat"], s["adaptive"]
    assert f["n"] == 40 and a["n"] == 40
    assert abs(f["adaptation"]["rho_gap_vs_tier"]["rho"]) < 0.35, f["adaptation"]
    assert f["adaptation"]["rho_gap_vs_tier"]["p"] > 0.05, "flat system produced a false positive"
    assert a["adaptation"]["rho_gap_vs_tier"]["rho"] > 0.8, a["adaptation"]
    assert a["adaptation"]["rho_gap_vs_tier"]["p"] < 0.01, a["adaptation"]
    assert a["adaptation"]["tier5_minus_tier1_ms"] > 600, a["adaptation"]
    # medians and IQR present, never a bare mean
    assert "median" in f["gap_ms"] and "iqr" in f["gap_ms"]
    print(f"gap scorer self-check OK  (flat rho={f['adaptation']['rho_gap_vs_tier']['rho']} "
          f"p={f['adaptation']['rho_gap_vs_tier']['p']}, "
          f"adaptive rho={a['adaptation']['rho_gap_vs_tier']['rho']} "
          f"p={a['adaptation']['rho_gap_vs_tier']['p']})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--systems", nargs="+", default=["cascade-serial"])
    ap.add_argument("--n", type=int, default=25, help="turns per system (>=20)")
    ap.add_argument("--out", default="results/gap.json")
    ap.add_argument("--device", default=None)
    ap.add_argument("--selfcheck", action="store_true")
    a = ap.parse_args()
    if a.selfcheck:
        demo()
    else:
        r = run(a.systems, n=a.n, out=a.out, device=a.device)
        print()
        print(report(r))
        print(f"\nwrote {a.out}")
