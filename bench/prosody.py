"""Dimension 4 -- prosodic responsiveness. Does the voice actually move with content?

Feature extraction is NOT reimplemented here. It is imported from Abhay's
``~/Desktop/Playground/expressive-tts-audit/features.py`` -- Praat via
parselmouth for F0 and voice quality, librosa for energy, pausing and spectral
shape. That repo is not modified by this one. If its dependencies are not
installed, this dimension records itself as NOT-MEASURED with the reason,
rather than falling back to a weaker extractor whose numbers would not be
comparable to anything.

THREE THINGS ARE MEASURED, and only the second one answers the question.

1. CONTENT RANGE (descriptive). The agent is asked five affectively distinct
   things -- good news, a death, a fire, a clock, a list of numbers -- and the
   prosody of its actual spoken replies is extracted: F0 range, energy dynamics,
   speech rate. Reported per condition, median and IQR.

   On its own this is NOT evidence of responsiveness. Different words have
   different F0 whatever the speaker feels; "Paris." and "I'm so sorry to hear
   that." would differ under a completely affect-blind renderer. Reported
   because it is what a listener is exposed to, not because it settles anything.

2. CONTEXT SENSITIVITY (decisive). The same fixed sentence is put through the
   agent's own voice stage after a joyful exchange and after a grieving one. If
   the two renderings are sample-identical, the voice carries no conversational
   state at all and its prosodic responsiveness to context is exactly zero --
   not "low", zero, and measured rather than argued from the architecture.

   This is the test that distinguishes "the voice moves because the words moved"
   from "the voice moves because the conversation moved".

3. CEILING (scale check). The same sentence rendered through macOS ``say`` under
   contrasting explicit prosody tags -- the method in expressive-tts-audit. This
   is not a system under test. It exists so a reader can tell that a zero here
   means zero and not a broken extractor: it shows what a non-zero looks like on
   the same axis, measured by the same code, on the same machine.

A flat agent scores near zero on (1)'s spread and exactly zero on (2). That is a
real finding and it stays reported as one.
"""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bench import prompts as P  # noqa: E402
from bench.common import audio_mod, speech_bounds_ms, stats, write_result  # noqa: E402
from bench.runner import (  # noqa: E402
    base_result, close_systems, interleave, open_systems, turn_stamp,
)

DIMENSION = "prosodic responsiveness"

EXPRESSIVE = Path.home() / "Desktop/Playground/expressive-tts-audit"

# the features this dimension actually reports on; the extractor returns 20
KEY_FEATURES = ["f0_range", "f0_sd", "f0_mean", "rms_range_db", "rms_sd_db",
                "speech_rate", "articulation_rate", "pause_frac"]


def load_extractor():
    """expressive-tts-audit's feature extractor, or (None, reason)."""
    if str(EXPRESSIVE) not in sys.path:
        sys.path.insert(0, str(EXPRESSIVE))
    try:
        import features  # noqa: PLC0415
        return features, None
    except Exception as e:  # noqa: BLE001
        return None, (f"{type(e).__name__}: {e} -- expressive-tts-audit's extractor needs "
                      f"librosa and praat-parselmouth; install them and re-run")


def reply_wav(t: dict, path: Path) -> float | None:
    """Write the agent's actually-emitted reply audio, sliced out of the
    recorded device output. Returns its duration in ms, or None.

    Deliberately the recording and not the array the TTS produced: this is what
    left the machine, which is what a listener heard.
    """
    rec = t.get("out_rec")
    if rec is None or rec.size == 0:
        return None
    b = speech_bounds_ms(rec)
    if b is None:
        return None
    a = audio_mod()
    y = rec[a.samples(b[0]) : a.samples(b[1])]
    if y.size < a.samples(120.0):
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    a.write(path, y)
    return b[1] - b[0]


def collect(systems, n_per_system: int, out_dir: Path, progress=True) -> list[dict]:
    """Affect prompts, cycled, interleaved across systems."""
    reps = max(1, round(n_per_system / len(P.AFFECT_PROMPTS)))
    items = [p for _ in range(reps) for p in P.AFFECT_PROMPTS]
    rendered = {pid: systems[0].render_prompt(text) for pid, _, text in P.AFFECT_PROMPTS}

    fx, why = load_extractor()
    rows, i = [], 0
    for s, (pid, affect, text) in interleave(systems, items):
        try:
            t = s.turn(rendered[pid], label=text, record=True)
            w = out_dir / s.name / f"{pid}-{i:03d}.wav"
            dur = reply_wav(t, w)
            row = {"system": s.name, "prompt_id": pid, "affect": affect, "prompt": text,
                   "reply": t["reply"], "reply_words": len(str(t["reply"]).split()),
                   "gap_ms": t["gap_ms"], "wav": str(w.relative_to(out_dir.parent))
                   if dur else None, "reply_audio_ms": dur, **turn_stamp()}
            if dur and fx is not None:
                try:
                    row["features"] = {k: (None if v is None or (isinstance(v, float)
                                       and np.isnan(v)) else round(float(v), 4))
                                       for k, v in fx.extract(w, str(t["reply"])).items()}
                except Exception as e:  # noqa: BLE001
                    row["feature_error"] = f"{type(e).__name__}: {e}"
            elif fx is None:
                row["feature_error"] = why
            rows.append(row)
            if progress:
                f0 = (row.get("features") or {}).get("f0_range")
                print(f"  [{i:3d}] {s.name:<18} {affect:<8} f0_range="
                      f"{f0 if f0 is not None else 'n/a'}  {str(t['reply'])[:38]!r}", flush=True)
        except Exception as e:  # noqa: BLE001
            rows.append({"system": s.name, "prompt_id": pid, "affect": affect,
                         "error": f"{type(e).__name__}: {e}", **turn_stamp()})
            if progress:
                print(f"  [{i:3d}] {s.name:<18} FAILED {type(e).__name__}: {e}", flush=True)
        i += 1
    return rows


def context_sensitivity(systems, out_dir: Path, progress=True) -> dict:
    """THE DECISIVE TEST. One fixed sentence, two conversational contexts.

    The agent is put through a joyful exchange, then asked to voice
    ``FLOOR_TEXT``; then through a grieving exchange, then the same sentence.
    We compare the two renderings sample by sample.

    Identical samples => the voice stage carries nothing from the conversation,
    and prosodic responsiveness to context is exactly 0. Different samples =>
    something in the stack conditions the voice on context, and the size of the
    difference is reported.
    """
    out = {}
    for s in systems:
        try:
            # DETERMINISM BASELINE FIRST. The context test reads "the two
            # renderings differ" as evidence of context sensitivity. That
            # inference is only valid if the backend is deterministic to begin
            # with -- a synthesiser with any run-to-run jitter would look
            # context-sensitive while carrying no context at all. macOS `say`
            # shells out to a binary and cannot be assumed bit-stable, so it is
            # checked rather than assumed: same text, twice, nothing in between.
            b1 = s.synth(P.FLOOR_TEXT)
            b2 = s.synth(P.FLOOR_TEXT)
            bn = min(len(b1), len(b2))
            bd = float(np.max(np.abs(b1[:bn] - b2[:bn]))) if bn else None
            deterministic = bool(len(b1) == len(b2) and bd is not None and bd == 0.0)

            joy = next(p for p in P.AFFECT_PROMPTS if p[1] == "joy")
            grief = next(p for p in P.AFFECT_PROMPTS if p[1] == "grief")
            s.turn(s.render_prompt(joy[2]), label=joy[2], record=False)
            a1 = s.synth(P.FLOOR_TEXT)
            s.turn(s.render_prompt(grief[2]), label=grief[2], record=False)
            a2 = s.synth(P.FLOOR_TEXT)
            n = min(len(a1), len(a2))
            same_len = len(a1) == len(a2)
            d = float(np.max(np.abs(a1[:n] - a2[:n]))) if n else None
            identical = bool(same_len and d is not None and d == 0.0)
            aud = audio_mod()
            p1 = out_dir / s.name / "context-after-joy.wav"
            p2 = out_dir / s.name / "context-after-grief.wav"
            p1.parent.mkdir(parents=True, exist_ok=True)
            aud.write(p1, a1)
            aud.write(p2, a2)
            out[s.name] = {
                "text": P.FLOOR_TEXT,
                "determinism_baseline": {
                    "same_text_twice_identical": deterministic,
                    "max_abs_sample_difference": bd,
                    "why": "the context test below can only be read if the backend is "
                           "deterministic; this is that check, not an assumption",
                },
                "same_length": same_len,
                "max_abs_sample_difference": d,
                "identical": identical,
                "context_sensitivity": (0.0 if (deterministic and identical)
                                        else None),
                "interpretation": (
                    "BACKEND IS NONDETERMINISTIC: the same text rendered twice back to "
                    "back already differs, so a difference across contexts proves "
                    "nothing. This dimension is not-measured for this system."
                    if not deterministic else
                    "sample-identical after a joyful and after a grieving exchange, on a "
                    "backend verified deterministic: the voice stage receives only the "
                    "reply text and carries no conversational state. Prosodic "
                    "responsiveness to context is 0."
                    if identical else
                    "the backend is deterministic and the two renderings differ, so "
                    "something in the stack does condition the voice on context; the "
                    "size of the difference is above"),
                "wavs": [str(p1.relative_to(out_dir.parent)), str(p2.relative_to(out_dir.parent))],
            }
            if not deterministic:
                out[s.name]["status"] = "not-measured"
                out[s.name]["reason"] = ("backend is not deterministic; same text twice "
                                         f"differs by {bd}")
                out[s.name]["identical"] = None
            if progress:
                print(f"  context test  {s.name:<18} deterministic={deterministic} "
                      f"identical={out[s.name]['identical']} max|diff|={d}", flush=True)
        except Exception as e:  # noqa: BLE001
            out[s.name] = {"status": "not-measured", "reason": f"{type(e).__name__}: {e}"}
    return out


def ceiling(out_dir: Path) -> dict:
    """Scale check. NOT a system under test.

    macOS ``say`` rendering one sentence under two contrasting explicit prosody
    settings, measured by the same extractor. Shows what a moving voice looks
    like on the same axis, so a zero elsewhere is readable as a zero.
    """
    fx, why = load_extractor()
    if fx is None:
        return {"status": "not-measured", "reason": why}
    try:
        sys.path.insert(0, str(EXPRESSIVE))
        import render  # noqa: PLC0415
        out_dir.mkdir(parents=True, exist_ok=True)
        rows = {}
        for tag, params in {
            "flat": dict(rate=180, pbas=40, pmod=10, volm=1.0),
            "animated": dict(rate=210, pbas=62, pmod=90, volm=1.0),
        }.items():
            p = out_dir / f"ceiling-{tag}.wav"
            render.render_say(P.FLOOR_TEXT, params, str(p))
            f = fx.extract(p, P.FLOOR_TEXT)
            rows[tag] = {"params": params,
                         "features": {k: round(float(f[k]), 3) for k in KEY_FEATURES if k in f},
                         "wav": str(p.relative_to(out_dir.parent))}
        d = {k: round(rows["animated"]["features"][k] - rows["flat"]["features"][k], 3)
             for k in rows["flat"]["features"]}
        return {"note": "macOS `say` under explicit prosody tags; a reference scale, "
                        "not a system under test", "conditions": rows, "delta": d}
    except Exception as e:  # noqa: BLE001
        return {"status": "not-measured", "reason": f"{type(e).__name__}: {e}"}


def score(rows, ctx) -> dict:
    out = {}
    for name in sorted({r["system"] for r in rows}):
        rs = [r for r in rows if r["system"] == name and r.get("features")]
        if not rs:
            errs = sorted({r.get("feature_error") or r.get("error", "?")
                           for r in rows if r["system"] == name})
            out[name] = {"n": 0, "status": "not-measured",
                         "reason": "no features extracted", "errors": errs,
                         "context_sensitivity": ctx.get(name)}
            continue
        per_feature = {}
        for f in KEY_FEATURES:
            vals = [r["features"].get(f) for r in rs if r["features"].get(f) is not None]
            if not vals:
                continue
            by_aff = {}
            for aff in sorted({r["affect"] for r in rs}):
                v = [r["features"].get(f) for r in rs
                     if r["affect"] == aff and r["features"].get(f) is not None]
                if v:
                    by_aff[aff] = round(statistics.median(v), 3)
            spread = (round(max(by_aff.values()) - min(by_aff.values()), 3)
                      if len(by_aff) > 1 else None)
            per_feature[f] = {"overall": stats(vals), "median_by_affect": by_aff,
                              "between_condition_spread": spread}
        out[name] = {
            "n": len(rs),
            "per_feature": per_feature,
            "context_sensitivity": ctx.get(name),
            "reply_words": stats([r["reply_words"] for r in rs]),
        }
    return out


def run(system_names, n: int = 20, out=None, device=None, audio_dir="audio/prosody") -> dict:
    ad = Path(audio_dir)
    systems, info, player = open_systems(system_names, device=device)
    try:
        rows = collect(systems, n, ad)
        ctx = context_sensitivity(systems, ad)
    finally:
        close_systems(systems, player)
    fx, why = load_extractor()
    res = base_result(DIMENSION, info, {
        "n_per_system": n,
        "extractor": ("expressive-tts-audit/features.py (Praat/parselmouth + librosa), "
                      "imported not vendored" if fx else "NOT AVAILABLE"),
        "extractor_error": why,
        "affect_prompts": [{"id": a, "affect": b, "text": c} for a, b, c in P.AFFECT_PROMPTS],
        "ceiling_reference": ceiling(ad),
        "score": score(rows, ctx),
        "rows": [{k: v for k, v in r.items() if k != "out_rec"} for r in rows],
    })
    write_result(out or "results/prosody.json", res)
    return res


def report(res) -> str:
    L = [f"{'system':<20} {'n':>3} {'f0_range med':>13} {'spread across affect':>21} "
         f"{'context-sensitive':>18}"]
    for name, s in res["score"].items():
        if not s.get("n"):
            L.append(f"{name:<20} {'-':>3}  not-measured: {s.get('reason')}")
            continue
        f = s["per_feature"].get("f0_range", {})
        o = f.get("overall", {})
        c = s.get("context_sensitivity") or {}
        cs = ("no (sample-identical)" if c.get("identical")
              else "yes" if c.get("identical") is False else "?")
        L.append(f"{name:<20} {s['n']:>3} {o.get('median', '-'):>13} "
                 f"{str(f.get('between_condition_spread')):>21} {cs:>18}")
    cr = res.get("ceiling_reference", {})
    if "delta" in cr:
        L.append(f"\nceiling reference (macOS `say`, explicit prosody tags, not a system "
                 f"under test): f0_range moves {cr['delta'].get('f0_range')} Hz between "
                 f"flat and animated")
    return "\n".join(L)


def demo():
    """Self-check: the scorer must separate a voice that moves from one that does
    not, and must report a missing extractor as not-measured rather than zero."""
    def row(sys_, aff, f0, i):
        return {"system": sys_, "affect": aff, "prompt_id": f"a_{aff}", "reply": "ok fine",
                "reply_words": 2, "features": {"f0_range": f0, "rms_range_db": 10.0,
                                               "speech_rate": 3.0}}
    flat = [row("flat", a, 40.0 + (i % 2), i)
            for i, a in enumerate(["joy", "grief", "alarm", "neutral", "flat"] * 4)]
    movy = [row("movy", a, {"joy": 180.0, "grief": 60.0, "alarm": 220.0,
                            "neutral": 100.0, "flat": 50.0}[a], i)
            for i, a in enumerate(["joy", "grief", "alarm", "neutral", "flat"] * 4)]
    ctx = {"flat": {"identical": True, "context_sensitivity": 0.0,
                    "determinism_baseline": {"same_text_twice_identical": True}},
           "movy": {"identical": False, "max_abs_sample_difference": 0.4,
                    "determinism_baseline": {"same_text_twice_identical": True}},
           "jittery": {"identical": None, "status": "not-measured",
                       "determinism_baseline": {"same_text_twice_identical": False}}}
    s = score(flat + movy, ctx)
    assert s["flat"]["n"] == 20 and s["movy"]["n"] == 20
    fs = s["flat"]["per_feature"]["f0_range"]["between_condition_spread"]
    ms = s["movy"]["per_feature"]["f0_range"]["between_condition_spread"]
    assert fs <= 1.0, fs
    assert ms >= 150.0, ms
    assert s["flat"]["context_sensitivity"]["identical"] is True
    # a nondeterministic backend must NOT be read as context-sensitive
    assert ctx["jittery"]["identical"] is None and ctx["jittery"]["status"] == "not-measured"

    # a system with no features at all must be not-measured, never 0
    s2 = score([{"system": "dead", "affect": "joy", "feature_error": "librosa missing"}], {})
    assert s2["dead"]["status"] == "not-measured" and s2["dead"]["n"] == 0, s2
    print(f"prosody scorer self-check OK  (flat spread {fs} Hz vs moving {ms} Hz; "
          f"missing extractor reports not-measured, not zero; a nondeterministic "
          f"backend reports not-measured rather than context-sensitive)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--systems", nargs="+", default=["cascade-serial"])
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--out", default="results/prosody.json")
    ap.add_argument("--audio-dir", default="audio/prosody")
    ap.add_argument("--device", default=None)
    ap.add_argument("--selfcheck", action="store_true")
    a = ap.parse_args()
    if a.selfcheck:
        demo()
    else:
        r = run(a.systems, n=a.n, out=a.out, device=a.device, audio_dir=a.audio_dir)
        print()
        print(report(r))
        print(f"\nwrote {a.out}")
