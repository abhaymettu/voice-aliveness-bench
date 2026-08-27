"""Dimension 3 -- silence behaviour. Say nothing, and see what it does.

A person who gets no reply does something: waits, then checks in ("still
there?"), then eventually gives up. The question is what an agent does with the
same silence, and when.

TWO PROBES, because "silence" means two different situations.

  A. COLD SILENCE. The system is handed a stream that is nothing but silence
     for ``window_s``. It was never spoken to.

  B. POST-REPLY SILENCE. The system completes a normal exchange, and then the
     next stream is silence for ``window_s``. This is the realistic one: you
     asked, it answered, and now you say nothing.

WHAT IS COMPUTED, for each probe:

  ``outcome``            one of: spoke | returned | error | hung
  ``time_to_outcome_ms`` from the start of the silence stream to whatever
                         happened -- speech, a clean return, an exception, or
                         the guard expiring
  ``agent_audio_ms``     any output the tap heard during the silence window.
                         This is what settles "did it re-prompt?" -- a
                         re-prompt is audible, so it is answered with audio.

THE HANG GUARD. "Does it hang?" is one of the questions, so it cannot be
answered by the probe itself hanging. Each probe runs on a daemon thread with a
join deadline of ``window_s + GUARD_S``. If the thread is still alive at the
deadline the outcome is ``hung``, recorded with the time waited, and the run
moves on. The thread is abandoned, not killed -- Python cannot kill a thread,
and pretending otherwise would be worse than saying so.

An exception is a RESULT, not a failure of the harness. A stack that raises on
an empty stream does not wait, does not re-prompt and does not time out
gracefully; it falls over. That is reported as what it is.
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bench import prompts as P  # noqa: E402
from bench.common import audio_mod, speech_bounds_ms, stats, write_result  # noqa: E402
from bench.runner import (  # noqa: E402
    base_result, close_systems, interleave, open_systems, turn_stamp,
)

DIMENSION = "silence behaviour"

WINDOW_S = 8.0    # how long we say nothing
GUARD_S = 12.0    # extra grace before we call it hung
ABS_FLOOR_DBFS = -55.0


def _silence(window_s: float) -> np.ndarray:
    a = audio_mod()
    return np.zeros(a.samples(window_s * 1000.0), np.float32)


def probe(system, window_s: float = WINDOW_S) -> dict:
    """Hand `system` a stream of pure silence and watch. Never blocks forever."""
    a = audio_mod()
    x = _silence(window_s)
    box: dict = {}
    system.player.arm()

    def work():
        t0 = time.perf_counter()
        try:
            t = system.turn(x, label="", record=False)
            box["outcome"] = "returned"
            box["turn"] = {k: v for k, v in t.items() if k not in ("out_rec",)}
        except Exception as e:  # noqa: BLE001 -- an exception here is the finding
            box["outcome"] = "error"
            box["error"] = f"{type(e).__name__}: {e}"
        box["ms"] = (time.perf_counter() - t0) * 1000.0

    t0 = time.perf_counter()
    th = threading.Thread(target=work, daemon=True)
    th.start()
    th.join(window_s + GUARD_S)
    hung = th.is_alive()
    waited_ms = (time.perf_counter() - t0) * 1000.0
    # give the output stream a beat to flush anything queued late
    time.sleep(0.2)
    _, rec = system.player.take()

    audio_ms, peak = 0.0, None
    r = a.frame_rms(rec, frame_ms=5.0) if rec.size else np.zeros(0, np.float32)
    if r.size:
        above = r >= 10 ** (ABS_FLOOR_DBFS / 20.0)
        audio_ms = float(above.sum()) * 5.0
        peak = round(20 * float(np.log10(max(float(r.max()), 1e-12))), 1)
        b = speech_bounds_ms(rec)
    else:
        b = None

    outcome = "hung" if hung else box.get("outcome", "returned")
    if not hung and audio_ms > 50.0:
        outcome = "spoke"

    return {
        "window_s": window_s,
        "outcome": outcome,
        "time_to_outcome_ms": round(waited_ms if hung else box.get("ms", waited_ms), 1),
        "hung": bool(hung),
        "guard_s": GUARD_S,
        "agent_audio_ms": round(audio_ms, 1),
        "agent_speech_onset_ms": round(b[0], 1) if b else None,
        "output_peak_dbfs": peak,
        "re_prompted": bool(audio_ms > 50.0),
        "error": box.get("error"),
        "reply": (box.get("turn") or {}).get("reply"),
        "transcript": (box.get("turn") or {}).get("transcript"),
    }


def collect(systems, n_per_system: int, window_s: float = WINDOW_S, progress=True) -> list[dict]:
    """Probe A (cold) and probe B (after a real reply), interleaved."""
    items = P.difficulty_cycle(n_per_system)
    rendered = {}
    for pid, _t, text in items:
        if pid not in rendered:
            rendered[pid] = systems[0].render_prompt(text)

    rows, i = [], 0
    # A: cold silence, once per system per round
    for s, (pid, tier, text) in interleave(systems, items):
        # B first needs a real exchange to follow
        pre = None
        try:
            t = s.turn(rendered[pid], label=text, record=False)
            pre = {"reply": t["reply"], "gap_ms": t["gap_ms"]}
        except Exception as e:  # noqa: BLE001
            pre = {"error": f"{type(e).__name__}: {e}"}
        r = probe(s, window_s)
        rows.append({"system": s.name, "probe": "post-reply", "prompt_id": pid,
                     "preceding_turn": pre, **r, **turn_stamp()})
        if progress:
            print(f"  [{i:3d}] {s.name:<18} post-reply  {r['outcome']:<9} "
                  f"{r['time_to_outcome_ms']:8.0f}ms  audio {r['agent_audio_ms']:.0f}ms"
                  f"{'  ' + str(r['error'])[:60] if r.get('error') else ''}", flush=True)
        i += 1
    # B: cold silence, no preceding exchange
    for s in systems:
        r = probe(s, window_s)
        rows.append({"system": s.name, "probe": "cold", "preceding_turn": None,
                     **r, **turn_stamp()})
        if progress:
            print(f"  [ - ] {s.name:<18} cold        {r['outcome']:<9} "
                  f"{r['time_to_outcome_ms']:8.0f}ms  audio {r['agent_audio_ms']:.0f}ms"
                  f"{'  ' + str(r['error'])[:60] if r.get('error') else ''}", flush=True)
    return rows


def score(rows) -> dict:
    out = {}
    for name in sorted({r["system"] for r in rows}):
        rs = [r for r in rows if r["system"] == name]
        per = {}
        for probe_name in sorted({r["probe"] for r in rs}):
            ps = [r for r in rs if r["probe"] == probe_name]
            outcomes = {}
            for r in ps:
                outcomes[r["outcome"]] = outcomes.get(r["outcome"], 0) + 1
            spoke = [r for r in ps if r["outcome"] == "spoke"]
            per[probe_name] = {
                "n": len(ps),
                "outcomes": outcomes,
                "time_to_outcome_ms": stats([r["time_to_outcome_ms"] for r in ps]),
                "n_re_prompted": len(spoke),
                "re_prompt_rate": round(len(spoke) / len(ps), 3) if ps else None,
                "agent_audio_ms": stats([r["agent_audio_ms"] for r in ps]),
                "errors": sorted({r["error"] for r in ps if r.get("error")}),
            }
        allr = rs
        spoke_any = any(r["outcome"] == "spoke" for r in allr)
        hung_any = any(r["hung"] for r in allr)
        errs = [r for r in allr if r["outcome"] == "error"]
        out[name] = {
            "n": len(allr),
            "by_probe": per,
            "verdict": (
                "hangs: the probe's guard expired" if hung_any else
                f"never speaks into silence; falls over instead -- "
                f"{len(errs)}/{len(allr)} probes raised"
                if errs and not spoke_any else
                "never speaks into silence; returns quietly and waits for the next turn"
                if not spoke_any else
                f"speaks into silence on {sum(1 for r in allr if r['outcome'] == 'spoke')}"
                f"/{len(allr)} probes"),
            "re_prompts": spoke_any,
            "hangs": hung_any,
        }
    return out


def run(system_names, n: int = 20, out=None, device=None, window_s: float = WINDOW_S) -> dict:
    systems, info, player = open_systems(system_names, device=device)
    t0 = time.perf_counter()
    try:
        rows = collect(systems, n, window_s)
    finally:
        close_systems(systems, player)
    res = base_result(DIMENSION, info, {
        "n_per_system": n,
        "window_s": window_s,
        "guard_s": GUARD_S,
        "probes": {
            "cold": "a stream of pure silence, the system having never been spoken to",
            "post-reply": "a normal exchange completes, then the next stream is silence",
        },
        "measurement": "output audio during the silence window comes from the recorded "
                       "device callback; a re-prompt is audible, so it is answered with audio",
        "elapsed_s": round(time.perf_counter() - t0, 1),
        "score": score(rows),
        "rows": rows,
    })
    write_result(out or "results/silence.json", res)
    return res


def report(res) -> str:
    L = [f"{'system':<20} {'probe':<12} {'n':>3} {'outcomes':<34} {'median ms':>10}  re-prompt"]
    for name, s in res["score"].items():
        for pn, p in s["by_probe"].items():
            oc = ", ".join(f"{k}x{v}" for k, v in sorted(p["outcomes"].items()))
            L.append(f"{name:<20} {pn:<12} {p['n']:>3} {oc:<34} "
                     f"{p['time_to_outcome_ms']['median']:>10}  {p['n_re_prompted']}")
        L.append(f"{'':<20} -> {s['verdict']}")
    return "\n".join(L)


def demo():
    """Self-check on a fake system: waiting, re-prompting, erroring and hanging
    must each be reported as themselves."""
    a = audio_mod()

    class FakePlayer:
        def __init__(self, out): self.out, self._armed = out, False
        def arm(self): self._armed = True
        def take(self): return 0.0, self.out

    class Fake:
        def __init__(self, name, behaviour, out=None):
            self.name, self.behaviour = name, behaviour
            self.player = FakePlayer(out if out is not None else np.zeros(100, np.float32))
        def turn(self, x, label="", record=False):
            if self.behaviour == "error":
                raise RuntimeError("no speech found in captured input")
            if self.behaviour == "hang":
                time.sleep(30.0)
            return {"reply": "hello?", "gap_ms": 500.0, "transcript": ""}

    tone = (0.3 * np.sin(2 * np.pi * 200 * np.arange(a.samples(400.0)) / a.SR)).astype(np.float32)
    # a recording too short to frame must read as "no audio", not crash
    assert probe(Fake("stub", "ok", out=np.zeros(4, np.float32)),
                 window_s=0.05)["agent_audio_ms"] == 0.0

    quiet = probe(Fake("quiet", "ok"), window_s=0.2)
    assert quiet["outcome"] == "returned" and not quiet["re_prompted"], quiet

    talker = probe(Fake("talker", "ok", out=tone), window_s=0.2)
    assert talker["outcome"] == "spoke" and talker["re_prompted"], talker
    assert talker["agent_audio_ms"] > 300, talker

    boom = probe(Fake("boom", "error"), window_s=0.2)
    assert boom["outcome"] == "error" and "no speech found" in boom["error"], boom

    # the guard must fire rather than the probe hanging with it
    global GUARD_S
    old, GUARD_S = GUARD_S, 0.3
    try:
        stuck = probe(Fake("stuck", "hang"), window_s=0.2)
    finally:
        GUARD_S = old
    assert stuck["outcome"] == "hung" and stuck["hung"], stuck
    assert stuck["time_to_outcome_ms"] < 3000, stuck

    s = score([{"system": "boom", "probe": "cold", **boom},
               {"system": "boom", "probe": "post-reply", **boom}])
    assert "falls over" in s["boom"]["verdict"], s
    assert not s["boom"]["re_prompts"] and not s["boom"]["hangs"], s
    print("silence scorer self-check OK  (waits, re-prompts, errors and hangs each "
          "reported as themselves; guard fires without the harness hanging)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--systems", nargs="+", default=["cascade-serial"])
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--window", type=float, default=WINDOW_S)
    ap.add_argument("--out", default="results/silence.json")
    ap.add_argument("--device", default=None)
    ap.add_argument("--selfcheck", action="store_true")
    a = ap.parse_args()
    if a.selfcheck:
        demo()
    else:
        r = run(a.systems, n=a.n, out=a.out, device=a.device, window_s=a.window)
        print()
        print(report(r))
        print(f"\nwrote {a.out}")
