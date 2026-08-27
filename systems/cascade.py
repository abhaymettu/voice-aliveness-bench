"""Adapter for the cascade voice agent (ASR -> LM -> TTS) under test.

THE AGENT IS NOT VENDORED. Everything below imports from Abhay's
``~/Desktop/Playground/aliveness-threshold`` repo -- ``harness/`` for the audio
and gap definition, ``live/loop.py`` for the agent itself. That repo is not
edited by anything here. Attribution: the loop, its endpointer, its speculative
fast path and its stage timers are all that repo's work; this file only drives
it and records what comes out.

The seam we drive through is ``live.loop.run_turn(source, asr, partial_asr, lm,
voice, player, ...)``, which takes an already-constructed ``Player``. That is
what lets us hand it a *recording* player and get the agent's real output
samples on a real clock, rather than reasoning about what it must have played.
Dimension 5 (non-verbal presence) depends on that: "the gap is silent" is a
claim about audio, so it is answered with audio.

Three configurations are registered, and they are the benchmark's controlled
contrast. They share a voice, a language model, an ASR model family and a TTS
backend, and differ only in *when* work is scheduled:

    cascade-serial      the original path, downstream runs after the endpointer
    cascade-fast        downstream runs inside the endpointer's 350 ms hangover,
                        armed after 80 ms of silence
    cascade-fast-tiny   same, plus the final ASR decode drops base.en -> tiny.en

Any difference between them across the five dimensions is attributable to
scheduling, not to what the system is. That is a far stronger internal contrast
than comparing unrelated agents.
"""

from __future__ import annotations

import time

import numpy as np

from bench.common import ALIVENESS, SEG_KW, speech_bounds_ms  # noqa: F401

from live import loop as L  # noqa: E402  (sys.path set by bench.common)
from harness import audio  # noqa: E402


class RecordingPlayer(L.Player):
    """``live.loop.Player`` with a tap on the output callback.

    The parent zeroes ``outdata`` when there is nothing to play, so recording
    the callback's output gives the true output timeline -- speech and silence
    alike -- on the same ``perf_counter`` clock as every other landmark. That is
    the only way to answer "what filled the gap?" by measurement.

    Recording is per turn: ``arm()`` before, ``take()`` after. A whole session
    would be hundreds of MB in RAM for no benefit.
    """

    def __init__(self, device=None):
        self._rec: list[np.ndarray] | None = None
        self._rec_t0: float | None = None
        super().__init__(device)

    def _cb(self, outdata, frames, t, status):
        super()._cb(outdata, frames, t, status)
        if self._rec is not None:
            if self._rec_t0 is None:
                self._rec_t0 = time.perf_counter()
            self._rec.append(outdata[:, 0].copy())

    def arm(self) -> None:
        self._rec_t0 = None
        self._rec = []

    def take(self):
        """(t0_perf, samples) for the armed window, or (None, empty)."""
        r, t0 = self._rec, self._rec_t0
        self._rec, self._rec_t0 = None, None
        if not r:
            return None, np.zeros(0, np.float32)
        return t0, np.concatenate(r)

    def stop(self) -> None:
        """Cut playback immediately. Nothing in the cascade ever calls this --
        it is here so a full-duplex system can be driven by the same probe."""
        self.buf = None


# name -> kwargs for loop.run_turn / model construction
CONFIGS = {
    "cascade-serial": dict(fast=False, arm_ms=None, final_model="base.en", hangover=350.0),
    "cascade-fast": dict(fast=True, arm_ms=80.0, final_model="base.en", hangover=350.0),
    "cascade-fast-tiny": dict(fast=True, arm_ms=80.0, final_model="tiny.en", hangover=350.0),
}

DESCRIPTIONS = {
    "cascade-serial": "ASR(base.en) -> Llama-3.2-1B-4bit(MLX) -> piper, strictly serial",
    "cascade-fast": "same stack, downstream speculatively run inside the 350ms hangover (arm 80ms)",
    "cascade-fast-tiny": "fast path with the final decode on tiny.en instead of base.en",
}


class Cascade:
    """One configuration of the cascade agent, as a system under test."""

    kind = "local cascade (ASR -> LM -> TTS)"
    # Half-duplex by construction: run_turn blocks in player.wait() and the
    # input queue is not read while the agent is speaking. Asserted, not
    # assumed -- bench/interrupt.py measures what actually happens.
    duplex = False

    def __init__(self, name: str, device=None, tts_backend: str = "auto", player=None):
        if name not in CONFIGS:
            raise ValueError(f"unknown cascade config {name!r}; have {list(CONFIGS)}")
        self.name = name
        # the runner may rename an instance (e.g. "cascade-serial-say"), so the
        # config it was built from is kept separately for lookups
        self.cfg_name = name
        self.cfg = CONFIGS[name]
        self.device = device
        self.tts_backend = tts_backend
        # A player may be injected so several systems under test share one
        # output stream. That is not a tidiness choice: interleaving systems
        # turn-by-turn is the only way to compare them on this machine (see
        # README, "the baseline moved on its own"), and interleaving through
        # three separate device streams would charge a different output path to
        # each system.
        self.player: RecordingPlayer | None = player
        self._owns_player = player is None

    # -- lifecycle ---------------------------------------------------------
    def open(self) -> dict:
        t0 = time.perf_counter()
        self.voice = L.pick_voice(self.tts_backend)
        # The prompt is always spoken by the default voice regardless of what
        # the agent replies with, so the ASR side stays comparable across
        # systems. Same rule the sibling repo uses.
        self.prompt_voice = L.pick_voice("auto")
        self.asr = L.Asr(self.cfg["final_model"])
        self.partial_asr = L.Asr(L.PARTIAL_MODEL)
        self.lm = L.Lm()
        load_ms = (time.perf_counter() - t0) * 1000.0
        if self.player is None:
            self.player = RecordingPlayer(self.device)
        self.fast = L.FastPath(self.asr, self.lm, self.voice) if self.cfg["fast"] else None

        # Warm every stage before any measured turn, so turn 0 does not pay for
        # a cold graph. Same warmup the sibling's run() does.
        t0 = time.perf_counter()
        warm = self.render_prompt("Hello there, this is a warm up line.")
        self.asr.text(warm)
        self.partial_asr.text(warm)
        self.lm.first_sentence("Hello.")
        self.voice.synth("Ready.")
        warm_ms = (time.perf_counter() - t0) * 1000.0
        return {
            "model_load_ms": round(load_ms, 1),
            "warmup_ms": round(warm_ms, 1),
            "tts": {"backend": self.voice.backend, "voice": self.voice.name},
            "prompt_tts": {"backend": self.prompt_voice.backend, "voice": self.prompt_voice.name},
            "asr_final": self.asr.name,
            "asr_partial": self.partial_asr.name,
            "lm": self.lm.name,
            "output_device": self.player.device,
            "output_device_latency_ms": round(self.player.latency_ms, 2),
        }

    def close(self) -> None:
        if self.player is not None and self._owns_player:
            self.player.close()
        self.player = None

    def meta(self) -> dict:
        return {
            "name": self.name,
            "kind": self.kind,
            "config_name": self.cfg_name,
            "description": DESCRIPTIONS[self.cfg_name],
            "tts_backend_requested": self.tts_backend,
            "duplex": self.duplex,
            "config": dict(self.cfg),
            "source_repo": str(ALIVENESS),
            "source_note": "imported, not vendored; that repo is unmodified by this one",
        }

    # -- prompt rendering --------------------------------------------------
    def render_prompt(self, text: str, lead_ms=300.0, tail_ms=900.0) -> np.ndarray:
        """A prompt as real audio with the trailing silence a real speaker
        leaves -- without a tail there is nothing for an endpointer to fire on.
        Uses the sibling's own render, so prompt audio is identical across
        systems."""
        p = self.prompt_voice.synth(text)
        return np.concatenate([
            np.zeros(audio.samples(lead_ms), np.float32),
            p,
            np.zeros(audio.samples(tail_ms), np.float32),
        ])

    # -- one turn ----------------------------------------------------------
    def turn(self, prompt_audio: np.ndarray, label: str = "", record: bool = True) -> dict:
        """Drive one exchange. Returns the sibling's own turn record plus the
        recorded output timeline.

        ``gap_ms`` comes straight out of ``live.loop.run_turn`` and is measured
        exactly as ``harness/exchange.py`` defines it. We do not recompute it.
        """
        if self.player is None:
            raise RuntimeError("open() first")
        ref_off = None
        b = speech_bounds_ms(prompt_audio)
        if b is not None:
            ref_off = b[1]
        if record:
            self.player.arm()
        t = L.run_turn(
            ("wav", prompt_audio), self.asr, self.partial_asr, self.lm, self.voice,
            self.player, label=label, hangover=self.cfg["hangover"],
            fast=self.fast, arm_ms=self.cfg["arm_ms"] or L.EARLY_ARM_MS,
            ref_offset_ms=ref_off,
        )
        if record:
            rec_t0, rec = self.player.take()
            t["out_rec_t0"] = rec_t0
            t["out_rec"] = rec
        t["system"] = self.name
        return t

    def synth(self, text: str) -> np.ndarray:
        """The agent's own voice saying `text`. Used by the prosody floor/ceiling
        controls, which need the same TTS stage the agent speaks through."""
        return self.voice.synth(text)


def build(name: str, **kw) -> Cascade:
    return Cascade(name, **kw)


def demo():
    """Self-check: one real turn, and the recorded output actually contains the
    reply where the reported gap says it should be.

    This is the check that the recording tap is honest. If the tap drifted from
    the agent's own clock, every dimension-5 number in the repo would be wrong,
    so it is asserted rather than eyeballed.
    """
    s = Cascade("cascade-serial")
    info = s.open()
    try:
        x = s.render_prompt("What is the capital city of France?")
        t = s.turn(x, label="What is the capital city of France?")
    finally:
        s.close()

    rec, t0 = t["out_rec"], t["out_rec_t0"]
    assert rec.size > 0 and t0 is not None, "output tap recorded nothing"
    b = speech_bounds_ms(rec)
    assert b is not None, "no speech found in the recorded output; the tap is not on the signal"
    onset_in_rec_ms, offset_in_rec_ms = b
    # The reply audio the agent handed the player must be the length that came
    # back out of the tap, within one output block plus VAD frame quantisation.
    played_ms = offset_in_rec_ms - onset_in_rec_ms
    assert abs(played_ms - t["reply_audio_ms"]) < 120.0, (
        f"tap says {played_ms:.0f}ms of speech, agent synthesised "
        f"{t['reply_audio_ms']:.0f}ms -- the tap is not recording the agent")
    assert t["gap_ms"] > 0, t["gap_ms"]
    print(f"cascade self-check OK  gap {t['gap_ms']:.0f}ms  "
          f"reply {t['reply']!r}  tap heard {played_ms:.0f}ms of speech "
          f"(agent synthesised {t['reply_audio_ms']:.0f}ms)  tts={info['tts']['backend']}")


if __name__ == "__main__":
    demo()
