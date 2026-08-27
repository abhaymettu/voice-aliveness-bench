"""Opening systems, and the interleaving rule every dimension runs under.

INTERLEAVING IS NOT A STYLE CHOICE. The sibling repo re-ran its own unchanged
code twenty minutes apart the same evening and got 1452 ms, then 807 ms -- a
1.8x swing with no code change, not explained by loadavg (16.79 then, 16.97
now). Whatever the machine was doing is not captured by any variable either
harness records.

The consequence for a *benchmark* is severe: measuring system A in one block
and system B in the next block confounds system with wall-clock time, and on
this machine that confound is bigger than most effects worth reporting. So
every run here round-robins the systems turn by turn, all systems stay loaded
for the whole session, and every turn carries its own loadavg and timestamp.
A block design would have been easier and would have produced numbers that
mean nothing.

All systems also share one output device stream, so a difference between them
is never a difference in which audio path they were given.
"""

from __future__ import annotations

import os
import time

from bench.common import provenance


def open_systems(names, device=None, tts_backend: str = "auto"):
    """Open every named system on one shared output stream.

    Returns (systems, info, player). Caller must call ``close_systems``.
    A system that fails to open is NOT silently dropped -- it comes back in
    ``info[name]["error"]`` so the results file records it as not-measured with
    a reason, which is the only honest way to report a system that would not
    run.
    """
    from systems.cascade import CONFIGS as CASCADE_CONFIGS, Cascade, RecordingPlayer

    player = RecordingPlayer(device)
    systems, info = [], {}
    for n in names:
        try:
            # "<config>-say" is the same config speaking through macOS `say`
            # instead of piper. A TTS backend swap is a different system under
            # test: it changes the voice, the synthesis latency and everything
            # dimension 4 measures, while leaving the ASR and LM stages alone.
            base, backend = n, tts_backend
            if n.endswith("-say"):
                base, backend = n[: -len("-say")], "say"
            if base.startswith("moshi"):
                from systems.moshi import MoshiSystem, build as moshi_build
                moshi_build()  # refuses loudly if the weights are not runnable
                s = MoshiSystem(n)
            elif base in CASCADE_CONFIGS:
                s = Cascade(base, device=device, tts_backend=backend, player=player)
                s.name = n  # report under the name that was asked for
            else:
                raise ValueError(f"no adapter registered for system {n!r}")
            opened = s.open()
            systems.append(s)
            info[n] = {**s.meta(), **opened}
        except Exception as e:  # noqa: BLE001 -- a system that will not run is a result
            info[n] = {
                "name": n,
                "status": "not-measured",
                "error": f"{type(e).__name__}: {e}",
            }
    return systems, info, player


def close_systems(systems, player) -> None:
    for s in systems:
        try:
            s.close()
        except Exception:  # noqa: BLE001, S110
            pass
    try:
        player.close()
    except Exception:  # noqa: BLE001, S110
        pass


def interleave(systems, items):
    """Yield (system, item) round-robining systems within each item.

    Item k is run on every system before item k+1 starts, so any drift in the
    machine over the session hits all systems alike instead of landing on
    whichever one happened to run late.
    """
    for it in items:
        for s in systems:
            yield s, it


def turn_stamp() -> dict:
    """Per-turn machine state. Cheap, and the one thing that would have made
    the sibling's 1.8x mystery diagnosable after the fact."""
    return {
        "t_wall": time.strftime("%H:%M:%S"),
        "loadavg": [round(v, 2) for v in os.getloadavg()],
    }


def base_result(dimension: str, systems_info: dict, extra: dict | None = None) -> dict:
    """The header every results file in this repo starts with."""
    return {
        "dimension": dimension,
        "provenance": provenance(),
        "design": {
            "order": "systems interleaved turn-by-turn, never blocked",
            "why": "this machine's own speed drifts within a session by more than "
                   "the effects being measured; see README",
            "output": "all systems share one output device stream",
        },
        "systems": systems_info,
        **(extra or {}),
    }
