#!/usr/bin/env python3
"""Run the whole benchmark, or one dimension of it.

    ./run_bench.py all      --systems cascade-serial cascade-fast cascade-fast-tiny
    ./run_bench.py gap      --systems cascade-serial --n 25
    ./run_bench.py selfcheck                # every scorer's own check, no agent needed
    ./run_bench.py aggregate                # fold results/ into one table + figures

Each dimension is also runnable on its own:  python bench/gap.py --help

ADDING A SYSTEM. Write an adapter in systems/ exposing:

    .name                      str
    .meta()                    -> dict, whatever you want recorded
    .open() / .close()
    .render_prompt(text)       -> float32 mono @ 22050 Hz, with a trailing tail
    .turn(prompt_audio, label, record) -> dict with at least:
         gap_ms            user speech offset -> agent speech onset, ms
         reply             the text it said
         reply_audio_ms    length of the reply audio
         speech_offset_ms  user speech offset, ms from the input stream's start
         out_rec, out_rec_t0   recorded output samples + their perf_counter t0
    .synth(text)               -> the agent's voice saying text

then register it in bench/runner.py::open_systems. Nothing else changes; the
five scorers were written against that contract, not against the cascade.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

DIMENSIONS = ["gap", "interrupt", "silence", "prosody", "nonverbal"]
DEFAULT_SYSTEMS = ["cascade-serial", "cascade-fast", "cascade-fast-tiny"]


def run_dimension(dim: str, systems, n: int, results="results") -> dict:
    mod = __import__(f"bench.{dim}", fromlist=["run", "report"])
    out = f"{results}/{dim}.json"
    print(f"\n{'=' * 78}\n{dim}  ->  {out}\n{'=' * 78}", flush=True)
    t0 = time.perf_counter()
    try:
        res = mod.run(systems, n=n, out=out)
        print()
        print(mod.report(res))
        return {"dimension": dim, "status": "ok",
                "elapsed_s": round(time.perf_counter() - t0, 1), "out": out}
    except Exception as e:  # noqa: BLE001 -- a dimension that will not run is a result
        print(f"\n{dim} FAILED: {type(e).__name__}: {e}", flush=True)
        return {"dimension": dim, "status": "failed",
                "reason": f"{type(e).__name__}: {e}",
                "elapsed_s": round(time.perf_counter() - t0, 1)}


def selfcheck() -> int:
    """Every scorer's own check. No audio device, no models, no agent."""
    mods = ["bench.common", "bench.prompts", "bench.gap", "bench.interrupt",
            "bench.silence", "bench.prosody", "bench.nonverbal", "bench.aggregate",
            "bench.figures", "systems.moshi"]
    bad = []
    for m in mods:
        try:
            mod = __import__(m, fromlist=["demo"])
            mod.demo()
        except Exception as e:  # noqa: BLE001
            print(f"  {m}: FAILED {type(e).__name__}: {e}")
            bad.append(m)
    print()
    print(f"{len(mods) - len(bad)}/{len(mods)} self-checks passed"
          + (f"; failed: {', '.join(bad)}" if bad else ""))
    return 1 if bad else 0


def aggregate(results="results", figures="figures") -> int:
    from bench import aggregate as A, figures as F
    agg = A.build(results)
    Path(results).mkdir(parents=True, exist_ok=True)
    Path(f"{results}/aggregate.json").write_text(json.dumps(agg, indent=2, default=str))
    print(A.render(agg))
    made = F.build(results, figures)
    print()
    for k, v in made.items():
        print(f"figures {k}: {v}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["all", "selfcheck", "aggregate", *DIMENSIONS])
    ap.add_argument("--systems", nargs="+", default=DEFAULT_SYSTEMS)
    ap.add_argument("--n", type=int, default=25,
                    help="turns per system per dimension (the benchmark asks for >=20)")
    ap.add_argument("--results", default="results")
    ap.add_argument("--figures", default="figures")
    a = ap.parse_args()

    if a.cmd == "selfcheck":
        return selfcheck()
    if a.cmd == "aggregate":
        return aggregate(a.results, a.figures)

    if a.n < 20:
        print(f"warning: n={a.n} is below the benchmark's own n>=20 rule; "
              f"the medians will be reported but treat them as indicative", flush=True)

    dims = DIMENSIONS if a.cmd == "all" else [a.cmd]
    log = [run_dimension(d, a.systems, a.n, a.results) for d in dims]
    print(f"\n{'=' * 78}")
    for r in log:
        print(f"  {r['dimension']:<12} {r['status']:<8} {r['elapsed_s']:>7.1f}s "
              f"{r.get('reason', '')}")
    if a.cmd == "all":
        print()
        aggregate(a.results, a.figures)
    return 0


if __name__ == "__main__":
    sys.exit(main())
