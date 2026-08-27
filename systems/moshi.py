"""Adapter slot for Moshi (Kyutai full-duplex speech-to-speech, MLX build).

STATUS: NOT-MEASURED. This file exists to record *why*, and to make the reason
checkable rather than remembered.

Moshi is the one system in reach tonight that is architecturally capable of
scoring above zero on dimensions 2 and 5 -- it is full-duplex, it listens while
it speaks, and it emits a continuous audio stream rather than a synthesised
utterance, so it has something to put in a gap. Every cascade configuration in
this benchmark is structurally incapable of both. Its absence is therefore the
single biggest hole in these results, and the README says so.

Weights are being pulled by a sibling effort at ~/Desktop/Playground/
fullduplex-voice. This adapter deliberately does NOT start a second download:
bandwidth was the binding constraint for the whole session, and a duplicate
4.8 GB pull would have starved the run that was actually producing numbers.

``status()`` reports what is on disk right now. When a runnable wrapper exists
next to those weights, the driver goes here -- ``turn()`` needs to accept the
same ``(prompt_audio) -> {gap_ms, reply, out_rec}`` contract the cascade adapter
honours, and the barge-in probe in bench/interrupt.py will then work unchanged,
because it was written against a duplex system from the start.

NOTHING IS ESTIMATED. There is no published-claims fallback here and there must
not be one. A system that did not run on this machine has no row in the
leaderboard.
"""

from __future__ import annotations

from pathlib import Path

FULLDUPLEX = Path.home() / "Desktop/Playground/fullduplex-voice"
HF_CACHE = Path.home() / ".cache/huggingface/hub"
REPO = "models--kyutai--moshiko-mlx-q4"

NAME = "moshi-mlx-q4"


def status() -> dict:
    """What is actually on disk, measured now. No claims, no inference."""
    d = HF_CACHE / REPO
    blobs = sorted((d / "blobs").glob("*")) if (d / "blobs").is_dir() else []
    complete = [b for b in blobs if b.is_file() and not b.name.endswith(".incomplete")]
    partial = [b for b in blobs if b.name.endswith(".incomplete")]
    total = sum(b.stat().st_size for b in blobs if b.is_file())

    # a wrapper is whatever the sibling effort exposes; we import, never vendor
    wrappers = sorted(p.name for p in FULLDUPLEX.glob("*.py")) if FULLDUPLEX.is_dir() else []
    has_runner = any("moshi" in w for w in wrappers)

    reason = None
    if not d.is_dir():
        reason = "weights not present on this machine"
    elif partial:
        reason = (f"weight download incomplete: {len(partial)} file(s) still "
                  f"partial, {total / 1e9:.2f} GB fetched of ~5.2 GB")
    elif not has_runner:
        reason = ("weights are on disk but no Moshi wrapper exists in "
                  f"{FULLDUPLEX} yet, so there is nothing to drive")

    return {
        "name": NAME,
        "kind": "local full-duplex speech-to-speech",
        "status": "not-measured" if reason else "weights-ready",
        "reason": reason,
        "weights_dir": str(d),
        "bytes_on_disk": total,
        "n_complete_files": len(complete),
        "n_incomplete_files": len(partial),
        "wrapper_dir": str(FULLDUPLEX),
        "wrapper_files": wrappers,
        "note": "no second download was started; bandwidth was the binding constraint "
                "and the sibling pull was already in flight",
        "why_it_matters": "the only system in reach that is architecturally able to "
                          "score above zero on interruption handling and non-verbal "
                          "presence; every cascade config here cannot, by construction",
    }


def build(*_a, **_kw):
    s = status()
    raise NotImplementedError(
        f"moshi is {s['status']}: {s['reason']}. No driver is written yet, and no "
        f"number for this system may be estimated, inferred or quoted from "
        f"published claims.")


def demo():
    """Self-check: status() must describe disk, and build() must refuse rather
    than return anything a scorer could mistake for a measurement."""
    s = status()
    assert s["name"] == NAME
    assert s["status"] in ("not-measured", "weights-ready")
    if s["status"] == "not-measured":
        assert s["reason"], "not-measured must always carry a reason"
    try:
        build()
    except NotImplementedError as e:
        assert "estimated" in str(e)
    else:
        raise AssertionError("build() must refuse while there is no driver")
    print(f"moshi adapter self-check OK  status={s['status']} "
          f"({s['bytes_on_disk'] / 1e9:.2f} GB on disk, "
          f"{s['n_incomplete_files']} incomplete)  reason={s['reason']}")


if __name__ == "__main__":
    demo()
