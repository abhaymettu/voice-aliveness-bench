"""Dimension 2 -- interruption handling. Barge in mid-reply and see what happens.

THE PROBE

One input stream carries the whole exchange, exactly as a real conversation
would: ``[prompt] [silence] [barge-in utterance] [silence]``. The barge-in is
placed so its speech lands ``into_reply_ms`` after the agent starts speaking,
using the system's own measured median gap to aim. Nothing about the system is
special-cased -- it is handed one audio stream and we watch its output.

WHAT IS COMPUTED, all of it from the RECORDED OUTPUT of the real device
callback (systems/cascade.py::RecordingPlayer), never from what the system said
it would do:

  ``stop_latency_ms``  barge-in speech onset -> the moment agent output speech
                       ceases. This is the headline number: how many ms of the
                       agent's voice does the person talk over before it yields.
  ``played_to_completion``  did the output run the full length of the reply the
                       agent had synthesised? If yes, it never yielded -- it
                       simply finished. ``stop_latency_ms`` is then a fact about
                       the reply's length, not about interruption handling, and
                       the report says so rather than quietly reporting a
                       responsive-looking number.
  ``heard_barge_in``   did the barge-in reach the system at all? Measured by
                       looking for the barge-in's distinctive content words in
                       the transcript the system produced. A system that neither
                       stops nor hears has lost the turn entirely.

CLOCK ALIGNMENT. Every landmark below is derived from measured quantities on
one ``perf_counter`` clock; nothing is assumed. The recorder gives agent speech
onset inside the recording; ``run_turn`` gives the gap and the user speech
offset relative to the input stream's start; those two pin the input stream's
start into the recording's frame, and the barge-in's position inside the input
wav (measured with the same VAD, not the position we intended) does the rest.

A turn whose barge-in lands outside the reply -- before the agent starts, or
after it has already finished -- is DISCARDED with its reason recorded, never
counted as a non-response. Aiming with a median gap does not always hit.

WHICH NUMBER IS A PROPERTY OF THE SYSTEM. ``stop_latency_ms`` is not. On a
system that never yields it is just whatever was left of the reply, so it is a
property of the PROBE (where the barge-in was aimed) and of the reply's length,
not of the agent. The system properties are the categorical ones --
``played_to_completion``, ``yielded``, ``heard_barge_in`` -- and those are what
the verdict is built from.

That distinction is not theoretical; it is what an independent measurement
showed. A separate effort (~/Desktop/Playground/fullduplex-voice,
``cascade_bargein.py``) measured the same agent on the same machine through its
own probe and reported a median stop latency of 1173 ms [1060-1264], n=20,
0/20 stopped early. This harness, n=73 landed across three configurations,
measures 1992 ms [511-2785], 0/73 stopped early, 0/73 heard.

The two agree exactly on every categorical result and differ 2x on the
milliseconds, and the difference is fully accounted for: they interrupt at a
fixed 900 ms and draw from a five-prompt set; this probe aims 350 ms into the
reply (measured median 361 ms) over 25 prompts whose replies run to a median of
2445 ms. ``stop_latency ~= reply_audio_ms - barge_offset`` gives 2084 ms here,
against a measured 1992 ms. Two probes, one behaviour, two different numbers --
which is exactly why the verdict does not rest on the number.

Their mechanism note was verified here in the source rather than taken on
trust: ``live/loop.py`` line 360 allocates ``q = queue.Queue()`` *inside*
``capture()`` (line 346), so the queue is rebuilt every turn and speech arriving
during playback is discarded rather than buffered. The agent talks over you and
then never hears what you said. This harness measures that consequence
independently, via the transcript: 0 of 73 landed barge-ins left any trace.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bench import prompts as P  # noqa: E402
from bench.common import audio_mod, speech_bounds_ms, stats, write_result  # noqa: E402
from bench.runner import (  # noqa: E402
    base_result, close_systems, interleave, open_systems, turn_stamp,
)

DIMENSION = "interruption handling (barge-in)"

# Lexically distinctive so that "did it hear this?" is unambiguous. None of
# these words appear in any difficulty prompt or any plausible reply to one.
BARGE_TEXT = "Actually wait, stop, tell me about penguins instead."
BARGE_KEYWORDS = ("penguin", "penguins", "stop", "instead", "actually", "wait")

# where in the reply the barge-in should land
INTO_REPLY_MS = 350.0
# output is "ceased" once it has been below the speech floor this long; shorter
# than any inter-word gap inside a piper utterance (measured: longest 65 ms)
CEASE_HOLD_MS = 120.0


def build_source(prompt_audio: np.ndarray, barge_audio: np.ndarray,
                 barge_at_ms: float) -> tuple[np.ndarray, float]:
    """Prompt, then the barge-in placed so its *speech* starts at `barge_at_ms`
    from the start of the stream. Returns (stream, measured_barge_speech_ms)."""
    a = audio_mod()
    b = speech_bounds_ms(barge_audio)
    lead_in_barge = b[0] if b else 0.0
    start = a.samples(barge_at_ms) - a.samples(lead_in_barge)
    if start < len(prompt_audio):
        start = len(prompt_audio)
    n = max(start + len(barge_audio), len(prompt_audio)) + a.samples(1500.0)
    x = np.zeros(n, np.float32)
    x[: len(prompt_audio)] = prompt_audio
    x[start : start + len(barge_audio)] += barge_audio
    # measure where the barge-in speech actually is, rather than trusting the
    # arithmetic above
    segs = a.segments(x, merge_gap_ms=30.0, min_len_ms=20.0)
    barge_speech_ms = None
    for s0, _ in segs:
        if s0 > a.millis(len(prompt_audio)) - 200.0:
            barge_speech_ms = float(s0)
            break
    return x, barge_speech_ms


def analyse(t: dict, barge_speech_ms: float) -> dict:
    """Everything measured, from the recorded output of the real device."""
    rec, rec_t0 = t.get("out_rec"), t.get("out_rec_t0")
    if rec is None or rec_t0 is None or rec.size == 0:
        return {"status": "discarded", "reason": "output tap recorded nothing"}
    b = speech_bounds_ms(rec)
    if b is None:
        return {"status": "discarded", "reason": "no agent speech in the recorded output"}
    on_rec, off_rec = b

    # pin the input stream's t0 into the recording's frame
    # agent_onset_perf = t_stream0 + (speech_offset_ms + gap_ms)/1000
    #                  = rec_t0 + on_rec/1000
    stream0_in_rec_ms = on_rec - (t["speech_offset_ms"] + t["gap_ms"])
    barge_in_rec_ms = stream0_in_rec_ms + barge_speech_ms

    played_ms = off_rec - on_rec
    reply_ms = t.get("reply_audio_ms") or played_ms
    # "played to completion" = the tap heard the whole reply the agent made.
    # 60 ms covers VAD frame quantisation plus one output block.
    complete = abs(played_ms - reply_ms) <= 60.0

    if barge_in_rec_ms < on_rec:
        return {"status": "discarded", "reason": "barge-in landed before the agent began speaking",
                "barge_offset_into_reply_ms": round(barge_in_rec_ms - on_rec, 1)}
    if barge_in_rec_ms > off_rec:
        return {"status": "discarded", "reason": "barge-in landed after the reply had finished",
                "barge_offset_into_reply_ms": round(barge_in_rec_ms - on_rec, 1),
                "reply_audio_ms": reply_ms}

    tr = (t.get("transcript") or "").lower()
    heard = [k for k in BARGE_KEYWORDS if k in tr]

    return {
        "status": "ok",
        "barge_offset_into_reply_ms": round(barge_in_rec_ms - on_rec, 1),
        "agent_speech_ms": round(played_ms, 1),
        "reply_audio_ms": reply_ms,
        "played_to_completion": bool(complete),
        # from barge-in speech onset to the agent's output falling silent
        "stop_latency_ms": round(off_rec - barge_in_rec_ms, 1),
        # what the agent still had left to say when it was interrupted
        "remaining_reply_ms": round(off_rec - barge_in_rec_ms, 1),
        "yielded": bool(not complete),
        "heard_barge_in": bool(heard),
        "barge_keywords_in_transcript": heard,
        "transcript": t.get("transcript"),
        "reply": t.get("reply"),
    }


def collect(systems, n_per_system: int, median_gap: dict, progress=True) -> list[dict]:
    items = P.difficulty_cycle(n_per_system)
    barge = {s.name: s.render_prompt(BARGE_TEXT, lead_ms=0.0, tail_ms=600.0) for s in systems}
    rendered = {}
    for pid, tier, text in items:
        if pid not in rendered:
            rendered[pid] = systems[0].render_prompt(text)

    rows, i = [], 0
    for s, (pid, tier, text) in interleave(systems, items):
        pa = rendered[pid]
        pb = speech_bounds_ms(pa)
        # aim: prompt speech offset + this system's median gap + into-reply
        aim = pb[1] + median_gap.get(s.name, 800.0) + INTO_REPLY_MS
        src, barge_ms = build_source(pa, barge[s.name], aim)
        try:
            t = s.turn(src, label=text, record=True)
            a = analyse(t, barge_ms)
            rows.append({"system": s.name, "prompt_id": pid, "tier": tier, "prompt": text,
                         "aimed_barge_at_ms": round(aim, 1),
                         "measured_barge_speech_ms": round(barge_ms, 1) if barge_ms else None,
                         "gap_ms": t["gap_ms"], **a, **turn_stamp()})
            if progress:
                if a["status"] == "ok":
                    print(f"  [{i:3d}] {s.name:<18} barge +{a['barge_offset_into_reply_ms']:6.0f}ms "
                          f"into reply, stop {a['stop_latency_ms']:6.0f}ms, "
                          f"completed={a['played_to_completion']}, heard={a['heard_barge_in']}",
                          flush=True)
                else:
                    print(f"  [{i:3d}] {s.name:<18} discarded: {a['reason']}", flush=True)
        except Exception as e:  # noqa: BLE001
            rows.append({"system": s.name, "prompt_id": pid, "status": "error",
                         "reason": f"{type(e).__name__}: {e}", **turn_stamp()})
            if progress:
                print(f"  [{i:3d}] {s.name:<18} FAILED {type(e).__name__}: {e}", flush=True)
        i += 1
    return rows


def score(rows) -> dict:
    out = {}
    for name in sorted({r["system"] for r in rows}):
        rs = [r for r in rows if r["system"] == name]
        ok = [r for r in rs if r.get("status") == "ok"]
        disc = [r for r in rs if r.get("status") != "ok"]
        if not ok:
            out[name] = {"n": 0, "status": "not-measured",
                         "reason": "no barge-in landed inside a reply",
                         "discarded": [r.get("reason") for r in disc]}
            continue
        yielded = [r for r in ok if r["yielded"]]
        out[name] = {
            "n": len(ok),
            "n_discarded": len(disc),
            "discard_reasons": sorted({r.get("reason", "?") for r in disc}),
            "barge_offset_into_reply_ms": stats([r["barge_offset_into_reply_ms"] for r in ok]),
            "n_yielded": len(yielded),
            "yield_rate": round(len(yielded) / len(ok), 3),
            "n_played_to_completion": sum(1 for r in ok if r["played_to_completion"]),
            "n_heard_barge_in": sum(1 for r in ok if r["heard_barge_in"]),
            # only meaningful for turns that actually stopped early; for the
            # rest this is the length of the reply's tail, not a response time
            "stop_latency_ms_when_yielded": stats([r["stop_latency_ms"] for r in yielded]),
            "talked_over_ms": stats([r["stop_latency_ms"] for r in ok]),
            "verdict": (
                "never yields: output ran to the end of the reply on every landed "
                "barge-in, and the barge-in never reached the transcript"
                if not yielded and not any(r["heard_barge_in"] for r in ok)
                else "never yields, but the barge-in did reach the transcript"
                if not yielded
                else f"yielded on {len(yielded)}/{len(ok)} turns"
            ),
        }
    return out


def run(system_names, n: int = 20, out=None, device=None, median_gap=None) -> dict:
    systems, info, player = open_systems(system_names, device=device)
    median_gap = median_gap or {}
    try:
        # aim with each system's own gap, measured now rather than assumed
        missing = [s for s in systems if s.name not in median_gap]
        if missing:
            print("pilot: measuring each system's median gap to aim the barge-in", flush=True)
            for s in missing:
                g = []
                for pid, tier, text in P.difficulty_cycle(4):
                    t = s.turn(s.render_prompt(text), label=text, record=False)
                    g.append(t["gap_ms"])
                median_gap[s.name] = statistics.median(g)
                print(f"  {s.name:<18} pilot median gap {median_gap[s.name]:.0f}ms (n={len(g)})",
                      flush=True)
        t0 = time.perf_counter()
        rows = collect(systems, n, median_gap)
    finally:
        close_systems(systems, player)
    res = base_result(DIMENSION, info, {
        "n_per_system": n,
        "barge_in_text": BARGE_TEXT,
        "aim": f"barge-in speech placed {INTO_REPLY_MS:.0f}ms after the agent's expected "
               f"speech onset, using each system's pilot median gap",
        "pilot_median_gap_ms": {k: round(v, 1) for k, v in median_gap.items()},
        "measurement": "all landmarks from the recorded output of the real device callback",
        "elapsed_s": round(time.perf_counter() - t0, 1),
        "score": score(rows),
        "rows": [{k: v for k, v in r.items() if k not in ("out_rec",)} for r in rows],
    })
    write_result(out or "results/interrupt.json", res)
    return res


def report(res) -> str:
    L = [f"{'system':<20} {'n':>3} {'yielded':>8} {'heard':>6} {'talked over ms':>18}  verdict"]
    for name, s in res["score"].items():
        if not s.get("n"):
            L.append(f"{name:<20} {'-':>3}  not-measured: {s.get('reason')}")
            continue
        t = s["talked_over_ms"]
        L.append(f"{name:<20} {s['n']:>3} {s['n_yielded']:>8} {s['n_heard_barge_in']:>6} "
                 f"{t['median']:>10.0f} [{t['p25']:.0f}-{t['p75']:.0f}]  {s['verdict']}")
    return "\n".join(L)


def demo():
    """Self-check: the analyser must call a real yield a yield, and must NOT
    call a reply that merely ended a yield."""
    a = audio_mod()

    def fake(rec_speech_ms, reply_audio_ms, gap_ms=800.0, speech_offset_ms=1200.0,
             transcript="what is the capital city of france"):
        # a recording: 100 ms silence, then rec_speech_ms of tone, then silence
        t = np.arange(a.samples(rec_speech_ms)) / a.SR
        rec = np.concatenate([
            np.zeros(a.samples(100.0), np.float32),
            (0.3 * np.sin(2 * np.pi * 200 * t)).astype(np.float32),
            np.zeros(a.samples(400.0), np.float32)])
        return {"out_rec": rec, "out_rec_t0": 0.0, "gap_ms": gap_ms,
                "speech_offset_ms": speech_offset_ms, "reply_audio_ms": reply_audio_ms,
                "transcript": transcript, "reply": "Paris."}

    # agent onset sits at 100 ms in the recording, so stream0 is at
    # 100 - (1200 + 800) = -1900 ms. A barge-in at 2250 ms in the stream lands
    # at 350 ms into the reply (2350 - 1900 = 450 in the recording, 350 past onset).
    # played 1000 ms, synthesised 1000 ms -> ran to completion, did not yield
    r = analyse(fake(1000.0, 1000.0), 2350.0)
    assert r["status"] == "ok", r
    assert abs(r["barge_offset_into_reply_ms"] - 350.0) < 15, r
    assert r["played_to_completion"] and not r["yielded"], r
    assert not r["heard_barge_in"], r

    # played 500 ms of a 1000 ms reply -> it stopped early, that is a yield
    r2 = analyse(fake(500.0, 1000.0), 2350.0)
    assert r2["status"] == "ok" and r2["yielded"] and not r2["played_to_completion"], r2
    assert abs(r2["stop_latency_ms"] - 150.0) < 15, r2

    # a barge-in after the reply finished must be discarded, not scored as a miss
    r3 = analyse(fake(1000.0, 1000.0), 4000.0)
    assert r3["status"] == "discarded", r3
    # and one before the agent starts
    r4 = analyse(fake(1000.0, 1000.0), 1000.0)
    assert r4["status"] == "discarded", r4

    # transcript detection
    r5 = analyse(fake(1000.0, 1000.0, transcript="actually wait stop tell me about penguins"),
                 2250.0)
    assert r5["heard_barge_in"] and "penguins" in r5["barge_keywords_in_transcript"], r5

    s = score([{"system": "x", **analyse(fake(1000.0, 1000.0), 2350.0)},
               {"system": "x", **analyse(fake(1000.0, 1000.0), 2350.0)}])
    assert s["x"]["n"] == 2 and s["x"]["n_yielded"] == 0, s
    assert "never yields" in s["x"]["verdict"], s
    print("interrupt scorer self-check OK  (yield detected, completion detected, "
          "out-of-window barge-ins discarded not counted)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--systems", nargs="+", default=["cascade-serial"])
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--out", default="results/interrupt.json")
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
