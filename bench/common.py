"""Shared spine: the gap definition, run provenance, and the stats everything reports.

THE GAP DEFINITION, stated once and used by every scorer in this repo:

    gap = offset of user speech -> onset of agent speech,
          both silence-trimmed, both located by frame-RMS voice-activity
          segmentation with merge_gap_ms=30, min_len_ms=20.

That is verbatim the definition in
``~/Desktop/Playground/aliveness-threshold/harness/exchange.py`` (by Abhay,
same machine, same evening). We do not reimplement it -- we import
``harness.audio.segments`` from that repo so the two cannot drift. Any scorer
here that measured a gap its own way would make cross-system numbers
meaningless, which is the one thing a benchmark cannot afford.

PROVENANCE. Every result file carries the git SHA of *this* repo and of the
sibling repo whose code was imported, plus loadavg at the start and end of the
run. That last one is not decoration. See README: the sibling's unchanged
baseline measured 1452 ms at 22:27 and 807 ms at 22:47 the same evening, and
loadavg did not explain it. A number without the machine state around it is
not reproducible here.
"""

from __future__ import annotations

import json
import os
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# The sibling repo we import the gap definition and the cascade agent from.
# Override with VAB_ALIVENESS_DIR if it ever moves.
ALIVENESS = Path(
    os.environ.get("VAB_ALIVENESS_DIR", Path.home() / "Desktop/Playground/aliveness-threshold")
)

if str(ALIVENESS) not in sys.path:
    sys.path.insert(0, str(ALIVENESS))

# Same params harness/exchange.py::measure_exchange uses. Do not change these
# without changing them there, or the two sides stop measuring the same thing.
SEG_KW = {"merge_gap_ms": 30.0, "min_len_ms": 20.0}


def audio_mod():
    """harness.audio from the sibling repo. Imported lazily so that a scorer
    which needs no audio still runs on a machine where the sibling is absent."""
    from harness import audio  # noqa: PLC0415

    return audio


def speech_bounds_ms(x):
    """(onset, offset) of speech in a mono float32 array, in ms, or None.

    This is THE measurement. Both ends of every gap in this repo come from
    here."""
    segs = audio_mod().segments(x, **SEG_KW)
    if not segs:
        return None
    return float(segs[0][0]), float(segs[-1][1])


def reply_bounds_ms(rec, reply_audio_ms=None):
    """(onset, offset) of the agent's REPLY inside a recorded output timeline.

    Not the same thing as the first sound in the recording, and the difference
    is the whole of dimension 5. If a system puts a breath or an "uh" in the
    gap, that cue is speech to the VAD, so ``speech_bounds_ms(rec)[0]`` returns
    the cue's onset and every gap window computed from it is wrong -- exactly
    on the systems that would have scored well.

    So the reply is anchored from its END, which no gap cue can precede:

        offset = last speech in the recording  (the reply's last sample)
        onset  = offset - reply_audio_ms       (the length the agent synthesised)

    Falls back to the first speech segment when the reply length is unknown,
    and says so, rather than silently using the fragile anchor.
    """
    b = speech_bounds_ms(rec)
    if b is None:
        return None
    first_on, offset = b
    if reply_audio_ms is None:
        return first_on, offset, "first-segment anchor (reply length unknown)"
    onset = offset - float(reply_audio_ms)
    # The reply cannot start before the recording does, and if the anchored
    # onset lands before the first detected sound the recording simply had
    # nothing else in it -- use the first segment then.
    if onset < first_on - 1.0:
        onset = max(0.0, onset)
    return onset, offset, "anchored from the reply's end using reply_audio_ms"


def git_sha(d: Path) -> str:
    try:
        r = subprocess.run(
            ["git", "-C", str(d), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=10,
        )
        sha = r.stdout.strip()
        dirty = subprocess.run(
            ["git", "-C", str(d), "status", "--porcelain"],
            capture_output=True, text=True, timeout=10,
        ).stdout.strip()
        return f"{sha}{'-dirty' if dirty else ''}" if sha else "unknown"
    except Exception:
        return "unknown"


def provenance() -> dict:
    return {
        "run_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "host": platform.node(),
        "platform": f"{platform.system()} {platform.release()} {platform.machine()}",
        "python": platform.python_version(),
        "bench_sha": git_sha(ROOT),
        "aliveness_sha": git_sha(ALIVENESS),
        "loadavg_start": [round(v, 2) for v in os.getloadavg()],
    }


def stats(v) -> dict:
    """Median and IQR, never a bare mean. n travels with the number, always."""
    v = sorted(float(a) for a in v if a is not None)
    if not v:
        return {"n": 0}
    if len(v) > 3:
        q = statistics.quantiles(v, n=4)
        p25, p75 = q[0], q[2]
    else:
        p25, p75 = v[0], v[-1]
    return {
        "n": len(v),
        "median": round(statistics.median(v), 1),
        "p25": round(p25, 1),
        "p75": round(p75, 1),
        "iqr": round(p75 - p25, 1),
        "min": round(v[0], 1),
        "max": round(v[-1], 1),
        "mean": round(statistics.fmean(v), 1),
        "sd": round(statistics.stdev(v), 1) if len(v) > 1 else 0.0,
    }


def spearman(a, b) -> dict:
    """Spearman rho with a permutation p-value.

    Permutation rather than a t-approximation because n is ~20-40 per system
    and ties are common in the difficulty ranks. 10k shuffles, fixed seed, so
    the p-value is reproducible.
    """
    import numpy as np  # noqa: PLC0415

    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = ~(np.isnan(a) | np.isnan(b))
    a, b = a[ok], b[ok]
    n = len(a)
    if n < 4 or np.all(a == a[0]) or np.all(b == b[0]):
        return {"n": n, "rho": None, "p": None, "note": "too few points or no variance"}

    def rank(v):
        o = np.argsort(v, kind="mergesort")
        r = np.empty(len(v), float)
        r[o] = np.arange(len(v), dtype=float)
        # average ranks within ties, or rho is biased by the tie ordering
        for u in np.unique(v):
            m = v == u
            r[m] = r[m].mean()
        return r

    ra, rb = rank(a), rank(b)
    rho = float(np.corrcoef(ra, rb)[0, 1])
    rng = np.random.default_rng(0)
    null = np.array([np.corrcoef(rng.permutation(ra), rb)[0, 1] for _ in range(10000)])
    p = float((np.abs(null) >= abs(rho) - 1e-12).mean())
    return {"n": n, "rho": round(rho, 3), "p": round(p, 4), "p_method": "10k permutations, seed 0"}


def write_result(path, obj) -> Path:
    """Write a result JSON, stamping loadavg at write time as well as at start."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(obj, dict):
        obj.setdefault("loadavg_end", [round(v, 2) for v in os.getloadavg()])
    path.write_text(json.dumps(obj, indent=2, default=str))
    return path


def demo():
    """Self-check: the gap definition round-trips, and the stats do not lie."""
    import numpy as np  # noqa: PLC0415

    a = audio_mod()
    # 500 ms silence, 400 ms tone, 800 ms silence, 300 ms tone, 200 ms silence
    def tone(ms, f=200.0):
        t = np.arange(a.samples(ms)) / a.SR
        return (0.3 * np.sin(2 * np.pi * f * t)).astype(np.float32)

    def sil(ms):
        return np.zeros(a.samples(ms), np.float32)

    x = np.concatenate([sil(500), tone(400), sil(800), tone(300), sil(200)])
    on, off = speech_bounds_ms(x)
    assert abs(on - 500) < 15, on
    assert abs(off - 2000) < 15, off
    # the gap between the two utterances, measured the one way this repo measures
    segs = a.segments(x, **SEG_KW)
    assert len(segs) == 2, segs
    gap = segs[1][0] - segs[0][1]
    assert abs(gap - 800) < 15, gap

    s = stats([1, 2, 3, 4, 100])
    assert s["n"] == 5 and s["median"] == 3 and s["max"] == 100

    r = spearman([1, 2, 3, 4, 5, 6, 7, 8], [1, 2, 3, 4, 5, 6, 7, 8])
    assert r["rho"] == 1.0 and r["p"] < 0.01, r
    r0 = spearman([1, 2, 3, 4, 5, 6, 7, 8], [3, 1, 4, 1, 5, 9, 2, 6])
    assert abs(r0["rho"]) < 0.9, r0
    print(f"common self-check OK  (gap {gap:.1f}ms measured vs 800 nominal, "
          f"rho monotone {r['rho']}, p {r['p']})")


if __name__ == "__main__":
    demo()
