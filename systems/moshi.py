"""Adapter for Moshi (Kyutai full-duplex speech-to-speech, MLX build).

NOT VENDORED. The model wrapper is imported from a sibling effort at
``~/Desktop/Playground/fullduplex-voice/moshi_run.py`` (its ``Moshi`` class and
``run_turn``), which is not modified by this repo. Attribution: the streaming
wrapper, the Mimi frame handling and the two-clock reporting are that repo's
work; this file adapts them to this benchmark's system contract so the same
five scorers can be pointed at them.

WHY THIS SYSTEM MATTERS. Every cascade configuration in this benchmark is
half-duplex and synthesises a whole utterance before speaking, so it is
structurally incapable of scoring above zero on dimension 2 (interruption) or
dimension 5 (non-verbal presence). Moshi listens while it speaks and emits a
continuous 80 ms audio frame every step whether or not it has anything to say.
It is the only system in reach that *could* fill a gap or yield to a barge-in.

TWO CLOCKS, AND THEY ARE NOT THE SAME NUMBER.

- **stream time** -- ``frame_index * 80 ms``. This is the architectural
  latency: how far into the conversation the reply lands. It is what is
  comparable to the cascade's gap, which ran in real time, and it is what
  ``gap_ms`` reports.
- **wall time** -- what the machine actually took. Reported separately as
  ``rtf`` (real-time factor). **If rtf > 1.0 the model cannot hold a live
  conversation on this hardware at all**, no matter how good its architectural
  latency looks. That is a first-class result and is never folded into the gap.

WHERE THE CONTRACT STRAINS, stated rather than hidden. This benchmark's other
systems have an unambiguous "reply": one synthesised utterance. Moshi emits a
stream. ``reply_audio_ms`` here is first-output-speech-onset to
last-output-speech-offset, which means dimension 5's reply anchor lands on the
first sound Moshi makes. If Moshi breathes before it speaks, that breath is
counted as the start of the reply rather than as gap filler, which makes this
adapter's dimension-5 number CONSERVATIVE for Moshi -- it can only understate
gap filling, never overstate it. Read it that way.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bench.common import speech_bounds_ms  # noqa: E402

FULLDUPLEX = Path.home() / "Desktop/Playground/fullduplex-voice"
MODELS = FULLDUPLEX / "models"
SR_MOSHI = 24000
FRAME = 1920  # 80 ms at 24 kHz -- Mimi's frame, not a choice

NAME = "moshi-mlx-q4"

# bytes the q4 checkpoint must reach before it is loadable
EXPECTED_MODEL_BYTES = 4805545317
REQUIRED = ("model.q4.safetensors", "mimi.safetensors", "tokenizer_spm_32k_3.model")


def status() -> dict:
    """What is actually on disk, measured now. No claims, no inference."""
    files, missing, partial = {}, [], []
    for f in REQUIRED:
        p = MODELS / f
        if not p.is_file():
            missing.append(f)
            continue
        n = p.stat().st_size
        files[f] = n
        if f == "model.q4.safetensors" and n < EXPECTED_MODEL_BYTES:
            partial.append(f"{f} at {n}/{EXPECTED_MODEL_BYTES} bytes "
                           f"({100.0 * n / EXPECTED_MODEL_BYTES:.1f}%)")

    wrapper = FULLDUPLEX / "moshi_run.py"
    have_wrapper = wrapper.is_file()

    reason = None
    if missing:
        reason = f"weight files not present: {', '.join(missing)}"
    elif partial:
        reason = f"weight download incomplete: {'; '.join(partial)}"
    elif not have_wrapper:
        reason = f"no Moshi wrapper at {wrapper}"

    return {
        "name": NAME,
        "kind": "local full-duplex speech-to-speech",
        "status": "not-measured" if reason else "ready",
        "reason": reason,
        "weights_dir": str(MODELS),
        "files": files,
        "wrapper": str(wrapper) if have_wrapper else None,
        "note": "no second download was started; bandwidth was the binding constraint "
                "and a sibling pull was already in flight",
        "why_it_matters": "the only system in reach that is architecturally able to "
                          "score above zero on interruption handling and non-verbal "
                          "presence; every cascade config here cannot, by construction",
    }


def _resample(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out:
        return x.astype(np.float32)
    n = int(round(len(x) * sr_out / sr_in))
    return np.interp(np.linspace(0, len(x) - 1, n), np.arange(len(x)),
                     x.astype(np.float64)).astype(np.float32)


class MoshiSystem:
    """Moshi as a system under test, on this benchmark's contract."""

    kind = "local full-duplex speech-to-speech"
    duplex = True

    def __init__(self, name: str = NAME, device=None, tts_backend: str = "auto",
                 player=None, tail_ms: float = 4000.0):
        self.name = name
        self.tail_ms = tail_ms
        self.player = player  # unused: Moshi is measured on its own emitted stream
        self._m = None
        self._prompt_voice = None
        self._load_ms = None
        self._mod = None

    def open(self) -> dict:
        st = status()
        if st["status"] != "ready":
            raise RuntimeError(f"moshi is not runnable: {st['reason']}")
        if str(FULLDUPLEX) not in sys.path:
            sys.path.insert(0, str(FULLDUPLEX))
        import moshi_run  # noqa: PLC0415  -- the sibling's wrapper, unmodified
        self._mod = moshi_run
        t0 = time.perf_counter()
        self._m = moshi_run.Moshi(quantized=4)
        load_ms = (time.perf_counter() - t0) * 1000.0
        self._load_ms = load_ms

        from live import loop as L  # noqa: PLC0415
        # prompts use the SAME piper voice every other system is prompted with,
        # so the input side is identical across the leaderboard
        self._prompt_voice = L.pick_voice("auto")
        return {
            "model_load_ms": round(load_ms, 1),
            "model": "kyutai/moshiko-mlx-q4",
            "frame_ms": 1000.0 * FRAME / SR_MOSHI,
            "sample_rate_native": SR_MOSHI,
            "prompt_tts": {"backend": self._prompt_voice.backend,
                           "voice": self._prompt_voice.name},
            "warmup": "moshi_run.Moshi calls model.warmup() at construction",
        }

    def close(self) -> None:
        self._m = None

    def meta(self) -> dict:
        return {
            "name": self.name,
            "kind": self.kind,
            "description": "Kyutai Moshi, 4-bit MLX, full-duplex; one 80ms frame in, "
                           "one 80ms frame out, no endpointer and no turn structure",
            "duplex": self.duplex,
            "source_repo": str(FULLDUPLEX),
            "source_note": "wrapper imported, not vendored; that repo is unmodified",
            "clock_note": "gap_ms is STREAM time (frames x 80ms). rtf is wall time and "
                          "is never folded into the gap.",
        }

    def render_prompt(self, text: str, lead_ms=300.0, tail_ms=900.0) -> np.ndarray:
        """At this benchmark's 22050 Hz, like every other adapter. Resampling to
        Moshi's 24 kHz happens inside turn(), so the prompt a scorer holds is
        identical to the one the cascade was given."""
        from harness import audio  # noqa: PLC0415
        p = self._prompt_voice.synth(text)
        return np.concatenate([
            np.zeros(audio.samples(lead_ms), np.float32), p,
            np.zeros(audio.samples(tail_ms), np.float32)])

    def synth(self, text: str) -> np.ndarray:
        raise NotImplementedError(
            "Moshi has no separable TTS stage -- it emits audio directly, so the "
            "prosody dimension's context test (which drives a voice stage with a "
            "fixed sentence) does not apply to it. Not-measured, not zero.")

    def turn(self, prompt_audio: np.ndarray, label: str = "", record: bool = True) -> dict:
        from harness import audio  # noqa: PLC0415

        x24 = _resample(prompt_audio, audio.SR, SR_MOSHI)
        t0 = time.perf_counter()
        xin, y, step_ms, text = self._mod.run_turn(
            self._m, x24, lead_ms=0.0, tail_ms=self.tail_ms)
        wall_ms = (time.perf_counter() - t0) * 1000.0

        # back to this repo's rate so the ONE gap definition applies unchanged
        xb = _resample(xin, SR_MOSHI, audio.SR)
        yb = _resample(y, SR_MOSHI, audio.SR)

        bi = speech_bounds_ms(xb)
        bo = speech_bounds_ms(yb)
        if bi is None:
            raise RuntimeError("no speech found in the prompt handed to moshi")
        user_off = bi[1]
        gap_ms = reply_ms = None
        if bo is not None:
            gap_ms = bo[0] - user_off
            reply_ms = bo[1] - bo[0]

        stream_ms = 1000.0 * len(xin) / SR_MOSHI
        return {
            "system": self.name,
            "label": label,
            "reply": text,
            "transcript": None,  # moshi has no ASR stage to read a transcript from
            "gap_ms": gap_ms,
            "ttfa_ms": gap_ms,
            "acoustic_gap_ms": gap_ms,
            "reply_audio_ms": reply_ms,
            "speech_offset_ms": user_off,
            "user_speech_ms": bi[1] - bi[0],
            # input and output streams are frame-aligned, so the output stream's
            # t=0 is the input stream's t=0 and every scorer's clock arithmetic
            # lands without a correction term
            "out_rec": yb if record else None,
            "out_rec_t0": 0.0 if record else None,
            "stream_ms": round(stream_ms, 1),
            "wall_ms": round(wall_ms, 1),
            # the number that decides whether this can hold a live conversation
            "rtf": round(wall_ms / stream_ms, 3) if stream_ms else None,
            "step_ms_median": round(float(np.median(step_ms)), 2) if step_ms else None,
            "n_frames": len(step_ms),
            "speculated": False,
            "truncated": False,
            "wer": None,
            "stage_ms": {},
            "work_ms": {},
            "lm_tokens": len(str(text).split()),
        }


def build(name: str = NAME, **kw) -> MoshiSystem:
    st = status()
    if st["status"] != "ready":
        raise NotImplementedError(
            f"moshi is {st['status']}: {st['reason']}. No number for this system may be "
            f"estimated, inferred, or quoted from published claims.")
    return MoshiSystem(name, **kw)


def demo():
    """Self-check: status() describes disk truthfully, and build() refuses rather
    than returning anything a scorer could mistake for a measurement."""
    s = status()
    assert s["name"] == NAME
    assert s["status"] in ("not-measured", "ready")
    if s["status"] == "not-measured":
        assert s["reason"], "not-measured must always carry a reason"
        try:
            build()
        except NotImplementedError as e:
            assert "estimated" in str(e)
        else:
            raise AssertionError("build() must refuse while moshi is not runnable")
    n = s["files"].get("model.q4.safetensors", 0)
    print(f"moshi adapter self-check OK  status={s['status']} "
          f"({n / 1e9:.2f} GB of {EXPECTED_MODEL_BYTES / 1e9:.2f} GB checkpoint) "
          f"reason={s['reason']}")


if __name__ == "__main__":
    demo()
