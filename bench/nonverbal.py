"""Dimension 5 -- non-verbal presence. Does anything at all fill the gap?

Breath, a filled pause ("uh"), a backchannel ("mm-hm"), a verbal stall, or dead
air. A person who is thinking is audibly still there. An agent that goes
completely silent for 800 ms has, to a listener, left the room.

WHAT IS COMPUTED

The gap window is located inside the RECORDED OUTPUT of the real device
callback, using the one gap definition this repo has:

    agent speech onset (in the recording, by VAD)  ->  call that `onset`
    gap window = [onset - gap_ms, onset]

with ``gap_ms`` taken from the agent's own turn record. Then, inside that
window:

  ``filled_fraction``   fraction of 5 ms frames above the absolute floor
                        (-55 dBFS, the same floor harness/audio.py uses)
  ``peak_dbfs``         loudest frame in the gap
  ``n_events``          VAD segments inside the gap, i.e. distinct sounds
  ``verdict``           "dead air" if nothing clears the floor, else the
                        measured events

POSITIVE CONTROL, and this dimension is worthless without it. "We detected
nothing" is only a finding if the detector can detect something. So the same
detector is run over a synthetic gap containing each of the four cues from the
sibling repo's ``harness/cues.py`` -- real audio, real sample offsets. If the
control fires on all four and the systems fire on none, the zero is real.

The control is NOT a system under test and is never mixed into the leaderboard.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bench import prompts as P  # noqa: E402
from bench.common import audio_mod, reply_bounds_ms, speech_bounds_ms, stats, write_result  # noqa: E402
from bench.runner import (  # noqa: E402
    base_result, close_systems, interleave, open_systems, turn_stamp,
)

DIMENSION = "non-verbal presence (what fills the gap)"

ABS_FLOOR_DBFS = -55.0  # same absolute floor harness/audio.py::speech_mask uses
FRAME_MS = 5.0
# The VAD rounds a segment's end UP to the next frame boundary, so a reply's
# measured offset sits up to one frame late and the onset anchored from it
# inherits that. Without a guard the last frame of the gap window contains the
# reply's first sample, and every dead-air gap reads as one hot frame at speech
# level -- a false positive on exactly the measurement this dimension exists to
# make. 15 ms covers the quantisation (harness/STATUS.md measures the systematic
# error at +5 to +10 ms) and costs under 4% of a 400 ms window.
GUARD_MS = 15.0


def analyse_gap(rec: np.ndarray, gap_ms: float, reply_audio_ms=None) -> dict:
    """Measure the gap window inside a recorded output timeline.

    The reply's onset is anchored from the reply's END (see
    common.reply_bounds_ms). Anchoring it to the first sound in the recording
    would break on precisely the systems this dimension is looking for: a cue
    in the gap is speech to the VAD, and would be mistaken for the reply.
    """
    a = audio_mod()
    if rec is None or rec.size == 0:
        return {"status": "discarded", "reason": "output tap recorded nothing"}
    b = reply_bounds_ms(rec, reply_audio_ms)
    if b is None:
        return {"status": "discarded", "reason": "no agent speech in the recorded output"}
    onset, _off, anchor = b
    # keep the reply's own first frames out of the gap window; see GUARD_MS
    onset = onset - GUARD_MS
    start = onset - gap_ms + GUARD_MS
    clipped = start < 0.0
    start = max(0.0, start)
    if onset - start < 50.0:
        return {"status": "discarded", "reason": "gap window shorter than 50ms after clipping"}

    g = rec[a.samples(start) : a.samples(onset)]
    r = a.frame_rms(g, frame_ms=FRAME_MS)
    if r.size == 0:
        return {"status": "discarded", "reason": "gap window too short to frame"}
    floor = 10 ** (ABS_FLOOR_DBFS / 20.0)
    above = r >= floor
    peak = float(r.max())
    segs = a.segments(g, merge_gap_ms=30.0, min_len_ms=20.0)

    return {
        "status": "ok",
        "gap_ms": round(gap_ms, 1),
        "reply_onset_anchor": anchor,
        "guard_ms": GUARD_MS,
        "window_ms": round(onset - start, 1),
        "window_clipped_by_recording_start": bool(clipped),
        "filled_fraction": round(float(above.mean()), 4),
        "n_frames": int(r.size),
        "n_frames_above_floor": int(above.sum()),
        "peak_dbfs": round(20 * float(np.log10(max(peak, 1e-12))), 1),
        "floor_dbfs": ABS_FLOOR_DBFS,
        "n_events": len(segs),
        "events_ms": [(round(s, 1), round(e, 1)) for s, e in segs],
        "verdict": "dead air" if not above.any() else f"{len(segs)} audible event(s)",
    }


def positive_control(out_dir: Path) -> dict:
    """Run the same detector over gaps that DO contain something.

    The cues come from the sibling repo's ``harness/cues.py`` -- filled pause,
    breath, backchannel, verbal stall -- rendered into an 800 ms gap between two
    utterances by ``harness.synthesize_exchange``. If the detector misses these,
    every zero in this dimension is a measurement failure, not a finding.
    """
    try:
        from harness import cues, synthesize_exchange  # noqa: PLC0415
        a = audio_mod()
        out_dir.mkdir(parents=True, exist_ok=True)
        rows = {}
        for cue in cues.CUES:
            p = out_dir / f"control-{cue}.wav"
            info = synthesize_exchange(
                "What time do you close on Sunday?", "We close at six on Sundays.",
                800.0, cue, str(p))
            x = a.read(p)
            # locate the gap the same way the live path does: agent speech onset
            # is the start of the last segment, the gap is the 800 ms before it
            segs = a.segments(x, merge_gap_ms=30.0, min_len_ms=20.0)
            onset = segs[-1][0]
            g = x[a.samples(onset - 800.0) : a.samples(onset)]
            r = a.frame_rms(g, frame_ms=FRAME_MS)
            above = r >= 10 ** (ABS_FLOOR_DBFS / 20.0)
            gs = a.segments(g, merge_gap_ms=30.0, min_len_ms=20.0)
            rows[cue] = {
                "filled_fraction": round(float(above.mean()), 4),
                "peak_dbfs": round(20 * float(np.log10(max(float(r.max()), 1e-12))), 1),
                "n_events": len(gs),
                "detected": bool(above.any()),
                "cue_duration_ms": round(info["cue_duration_ms"], 1),
                "cue_source": info["cue_source"],
                "wav": str(p.relative_to(out_dir.parent)),
            }
        detectable = [c for c, v in rows.items() if c != "none" and v["detected"]]
        return {
            "note": "sibling repo's harness/cues.py rendered into an 800ms gap and run "
                    "through this dimension's detector. Not a system under test.",
            "cues": rows,
            "detector_fires_on": detectable,
            "detector_silent_on_none_cue": not rows.get("none", {}).get("detected", True),
            "valid": len(detectable) == len(cues.CUES) - 1
                     and not rows.get("none", {}).get("detected", True),
        }
    except Exception as e:  # noqa: BLE001
        return {"status": "not-run", "reason": f"{type(e).__name__}: {e}"}


def collect(systems, n_per_system: int, out_dir: Path, progress=True) -> list[dict]:
    items = P.difficulty_cycle(n_per_system)
    rendered = {}
    for pid, tier, text in items:
        if pid not in rendered:
            rendered[pid] = systems[0].render_prompt(text)
    a = audio_mod()
    rows, i, saved = [], 0, set()
    for s, (pid, tier, text) in interleave(systems, items):
        try:
            t = s.turn(rendered[pid], label=text, record=True)
            r = analyse_gap(t.get("out_rec"), t["gap_ms"], t.get("reply_audio_ms"))
            # keep one gap recording per system as a listenable demo
            if r["status"] == "ok" and s.name not in saved:
                rec = t["out_rec"]
                on, off, _ = reply_bounds_ms(rec, t.get("reply_audio_ms"))
                lo = max(0.0, on - t["gap_ms"] - 150.0)
                hi = min(a.millis(len(rec)), off + 150.0)
                p = out_dir / f"{s.name}-gap-demo.wav"
                p.parent.mkdir(parents=True, exist_ok=True)
                a.write(p, rec[a.samples(lo) : a.samples(hi)])
                r["demo_wav"] = str(p.relative_to(out_dir.parent))
                r["demo_note"] = ("the recorded output around one real gap: the agent's "
                                  "reply preceded by the gap exactly as it played")
                saved.add(s.name)
            rows.append({"system": s.name, "prompt_id": pid, "tier": tier,
                         "reply": t["reply"], **r, **turn_stamp()})
            if progress:
                print(f"  [{i:3d}] {s.name:<18} gap {t['gap_ms']:6.0f}ms  "
                      f"filled {r.get('filled_fraction', '-')}  peak "
                      f"{r.get('peak_dbfs', '-')} dBFS  {r.get('verdict', r.get('reason'))}",
                      flush=True)
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
        if not ok:
            out[name] = {"n": 0, "status": "not-measured",
                         "reason": "no gap window could be measured",
                         "errors": sorted({r.get("reason", "?") for r in rs})}
            continue
        filled = [r["filled_fraction"] for r in ok]
        dead = [r for r in ok if r["filled_fraction"] == 0.0]
        out[name] = {
            "n": len(ok),
            "n_failed": len(rs) - len(ok),
            "filled_fraction": stats(filled),
            "peak_dbfs": stats([r["peak_dbfs"] for r in ok]),
            "n_events": stats([r["n_events"] for r in ok]),
            "n_dead_air": len(dead),
            "dead_air_rate": round(len(dead) / len(ok), 3),
            "gap_ms": stats([r["gap_ms"] for r in ok]),
            "demo_wav": next((r["demo_wav"] for r in ok if r.get("demo_wav")), None),
            "verdict": (f"dead air on {len(dead)}/{len(ok)} turns: nothing above "
                        f"{ABS_FLOOR_DBFS} dBFS in any measured gap"
                        if len(dead) == len(ok) else
                        f"something audible in the gap on {len(ok) - len(dead)}/{len(ok)} turns"),
        }
    return out


def run(system_names, n: int = 20, out=None, device=None, audio_dir="audio/nonverbal") -> dict:
    ad = Path(audio_dir)
    ctrl = positive_control(ad)
    systems, info, player = open_systems(system_names, device=device)
    try:
        rows = collect(systems, n, ad)
    finally:
        close_systems(systems, player)
    res = base_result(DIMENSION, info, {
        "n_per_system": n,
        "detector": f"frame-RMS over {FRAME_MS}ms frames, absolute floor {ABS_FLOOR_DBFS} dBFS "
                    f"(the same floor harness/audio.py uses), inside the gap window",
        "positive_control": ctrl,
        "score": score(rows),
        "rows": [{k: v for k, v in r.items() if k != "out_rec"} for r in rows],
    })
    write_result(out or "results/nonverbal.json", res)
    return res


def report(res) -> str:
    c = res.get("positive_control", {})
    L = []
    if c.get("valid"):
        L.append(f"positive control PASSES: the detector fires on all of "
                 f"{', '.join(c['detector_fires_on'])} and stays silent on cue=none. "
                 f"A zero below is a real zero.")
    else:
        L.append(f"positive control DID NOT PASS ({c.get('reason', c)}) -- treat every "
                 f"zero below as unverified.")
    L.append("")
    L.append(f"{'system':<20} {'n':>3} {'filled frac':>12} {'peak dBFS':>11} "
             f"{'dead air':>10}  verdict")
    for name, s in res["score"].items():
        if not s.get("n"):
            L.append(f"{name:<20} {'-':>3}  not-measured: {s.get('reason')}")
            continue
        L.append(f"{name:<20} {s['n']:>3} {s['filled_fraction']['median']:>12} "
                 f"{s['peak_dbfs']['median']:>11} {s['n_dead_air']:>4}/{s['n']:<5} "
                 f"{s['verdict']}")
    return "\n".join(L)


def demo():
    """Self-check: the detector must find a sound placed in the gap and must
    report an empty gap as empty."""
    a = audio_mod()

    def rec_with(gap_content):
        # 200 ms silence | 800 ms gap content | 500 ms of speech
        t = np.arange(a.samples(500.0)) / a.SR
        return np.concatenate([
            np.zeros(a.samples(200.0), np.float32),
            gap_content,
            (0.3 * np.sin(2 * np.pi * 200 * t)).astype(np.float32),
            np.zeros(a.samples(200.0), np.float32)])

    empty = analyse_gap(rec_with(np.zeros(a.samples(800.0), np.float32)), 800.0, 500.0)
    assert empty["status"] == "ok", empty
    assert empty["filled_fraction"] == 0.0 and empty["verdict"] == "dead air", empty
    assert empty["n_events"] == 0, empty

    # a 200 ms quiet blip 150 ms into the gap: a filled pause sits 20-30 dB under
    # speech, so the detector must catch something well below the reply's level
    blip = np.zeros(a.samples(800.0), np.float32)
    tt = np.arange(a.samples(200.0)) / a.SR
    blip[a.samples(150.0) : a.samples(150.0) + len(tt)] = (
        0.02 * np.sin(2 * np.pi * 130 * tt)).astype(np.float32)
    filled = analyse_gap(rec_with(blip), 800.0, 500.0)
    assert filled["status"] == "ok", filled
    assert filled["filled_fraction"] > 0.15, filled
    assert filled["n_events"] == 1, filled
    assert filled["verdict"] != "dead air", filled
    assert filled["peak_dbfs"] > empty["peak_dbfs"] + 20, (filled, empty)

    s = score([{"system": "x", **empty}, {"system": "x", **empty}])
    assert s["x"]["n_dead_air"] == 2 and s["x"]["dead_air_rate"] == 1.0, s
    assert "dead air on 2/2" in s["x"]["verdict"], s
    # and the anchor itself: with a cue in the gap, anchoring on the first
    # sound would put the window in the wrong place entirely
    assert not filled["window_clipped_by_recording_start"], filled
    assert abs(filled["window_ms"] - (800.0 - GUARD_MS)) < 20, filled
    # the guard must keep the reply out: an empty gap is exactly zero hot
    # frames, not "one frame of the reply leaked in"
    assert empty["n_frames_above_floor"] == 0, empty
    assert empty["peak_dbfs"] < -100, empty
    print(f"nonverbal scorer self-check OK  (empty gap -> dead air, quiet blip at "
          f"{filled['peak_dbfs']} dBFS detected as {filled['n_events']} event, "
          f"window {filled['window_ms']:.0f}ms anchored off the reply's end)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--systems", nargs="+", default=["cascade-serial"])
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--out", default="results/nonverbal.json")
    ap.add_argument("--audio-dir", default="audio/nonverbal")
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
