# Voice Aliveness Bench

**Everyone building a voice agent argues about whether it "feels alive." Nobody measures it.
This is the measure.**

Five dimensions, defined operationally, scored separately, run on local systems on one
laptop. Every number below came from a run that actually happened on this machine, and the
per-turn JSON that produced it is in `results/`.

---

## The headline finding

**The gap does not track how hard the question is. It tracks how long the answer is.**

A person takes longer to answer *"what is the meaning of life"* than *"what is two plus two."*
None of the three agents measured here do. The raw correlation between response gap and
question difficulty looks positive — and on the fastest configuration it even reaches
significance — but every prompt was matched to 7–9 words, and once **reply length** is
partialled out the correlation collapses to nothing on all three:

| system | gap vs difficulty (ρ) | p | gap vs reply length (ρ) | ρ with reply length partialled out |
|---|---|---|---|---|
| cascade-serial | 0.180 | 0.390 | 0.454 | **−0.065** |
| cascade-fast | 0.298 | 0.146 | 0.612 | **−0.032** |
| cascade-fast-tiny | 0.482 | **0.017** | 0.854 | **0.195** |

n = 25 turns per system. Spearman ρ, p from 10,000 permutations (seed 0).

The apparent adaptation is an artefact of synthesising the whole utterance before speaking:
harder question → wordier answer → longer TTS → longer gap. Nothing in any of these systems
models difficulty at all. The difference between tier 5 and tier 1 is 68–106 ms on a
400–820 ms gap.

**And four of the five dimensions are floored at zero for every system measured.** Nothing
yields to an interruption, nothing speaks into silence, no voice carries any conversational
state, and no gap contains a single sample above the noise floor. Those are real results, not
missing measurements — each one has a positive control proving the detector would have fired
if there had been anything to find.

---

## The five dimensions

Each is a module in `bench/`, independently runnable, with its own `--selfcheck`.

### 1. Turn-taking timing — `bench/gap.py`

`gap_ms` = **user speech offset → agent speech onset, both silence-trimmed.** Located by
frame-RMS voice-activity segmentation (`merge_gap_ms=30`, `min_len_ms=20`).

Reports speed (median + IQR, never a bare mean) and, separately, **adaptation**: Spearman ρ
between gap and an a-priori difficulty tier (1 trivial … 5 open-ended), the effect in
milliseconds, and two controls — the correlation against prompt *audio duration* (the length
confound, killed by design and checked by measurement) and against reply length (the
mechanism confound, removed by partial correlation).

### 2. Interruption handling — `bench/interrupt.py`

One input stream carries `[prompt] [silence] [barge-in] [silence]`, with the barge-in aimed
to land ~350 ms into the reply using the system's own pilot median gap. Measured from the
**recorded output of the real device callback**: does output speech cease, how many ms after
the barge-in, did the reply merely run to its natural end, and did the barge-in reach the
transcript at all. Turns whose barge-in lands outside the reply are discarded with the
reason, never counted as a non-response.

### 3. Silence behaviour — `bench/silence.py`

Say nothing, for 8 s, in two situations: cold (never spoken to) and post-reply (the realistic
one). Outcome is one of `spoke | returned | error | hung`, with the time to it and any audio
the tap heard. A re-prompt is audible, so it is answered with audio. "Does it hang?" runs
behind a join deadline, so the hang test cannot itself hang the harness.

### 4. Prosodic responsiveness — `bench/prosody.py`

Feature extraction is **imported, not reimplemented**, from `expressive-tts-audit/features.py`
(Praat via parselmouth for F0 and voice quality, librosa for energy, pausing, spectral shape).
Three things are measured, and only the second settles the question:

1. **Content range** across five affectively distinct prompts — descriptive only. Different
   words have different F0 whatever the speaker feels.
2. **Context sensitivity** (decisive): the same fixed sentence through the agent's own voice
   stage after a joyful exchange and after a grieving one. Sample-identical means the voice
   carries no conversational state and responsiveness to context is *exactly* zero.
3. **Ceiling** (scale check): macOS `say` under explicit prosody tags, measured by the same
   code. Not a system under test — it exists so a zero reads as a zero and not a broken
   extractor.

### 5. Non-verbal presence — `bench/nonverbal.py`

Does anything fill the gap — breath, filled pause, backchannel, dead air? The gap window is
located inside the recorded output, anchored from the **reply's end** (anchoring on the first
sound would break on exactly the systems that fill gaps: a cue is speech to the VAD). Then
`filled_fraction`, `peak_dbfs`, and the count of audible events.

**Positive control:** the same detector over gaps rendered with each of the four cues from
`harness/cues.py`. It fires on all four — catching a breath at −33.2 dBFS — and stays silent
on the no-cue case. The systems read −240 dBFS, which is literal digital silence.

---

## Results

| system | gap median [IQR] | ρ (difficulty, reply length out) | barge-in yields | speaks into silence | prosody vs context | gaps filled |
|---|---|---|---|---|---|---|
| cascade-serial | 822 ms [782–930] | −0.065 | 0/17 | never | none | 0/20 |
| cascade-fast | 550 ms [496–694] | −0.032 | 0/16 | never | none | 0/20 |
| cascade-fast-tiny | 402 ms [371–561] | 0.195 | 0/13 | never | none | 0/20 |

Full per-turn records in `results/*.json`; the aggregated table in `results/aggregate.json`;
figures in `figures/` (light and dark); the leaderboard with playable audio in
`web/index.html`.

**There is no composite "aliveness score", on purpose.** It would be dominated by whichever
dimension has the widest numeric range, it would hide that four of five are floored at zero,
and it would let a system buy a better headline by getting faster at the one thing that is
easy to get faster at. If you want one number, weight the columns yourself — and say plainly
that the weights are your choice, not a finding.

### What the three systems are

They are the same cascade agent (`aliveness-threshold/live/loop.py`, imported not vendored)
under three schedules. They share a voice, a language model, an ASR family and a TTS backend,
and differ **only in when work happens**:

- `cascade-serial` — downstream runs after the endpointer fires. The original path.
- `cascade-fast` — the final decode, LM and TTS run speculatively *inside* the endpointer's
  350 ms hangover, armed after 80 ms of silence.
- `cascade-fast-tiny` — same, with the final ASR decode on `tiny.en` instead of `base.en`.

That makes them a clean controlled contrast: any difference across the five dimensions is
attributable to scheduling, not to what the system is. **The result is that scheduling buys
2.0× on the gap and changes nothing else.** Speed did not cost interruptibility either —
the fast path commits to a reply after 80 ms of silence, which we expected might make barge-in
worse, and it did not, because none of the three was ever interruptible to begin with.

---

## Two independent cross-checks

Both were run by a separate effort in `~/Desktop/Playground/fullduplex-voice` against the
same agent on the same machine, and both were re-tested here against this repo's own data
rather than taken on trust.

### Barge-in: converges on the behaviour, differs 2x on the milliseconds

| | this harness | fullduplex-voice `cascade_bargein.py` |
|---|---|---|
| stopped early | **0 / 73** | **0 / 20** |
| barge-in reached the transcript | **0 / 73** | not measured (mechanism argued from source) |
| median stop latency | 1992 ms [511–2785] | 1173 ms [1060–1264] |

**Every categorical result agrees. The millisecond number does not, and it should not.**
`stop_latency_ms` on a system that never yields is just whatever was left of the reply, so
it is a property of the probe and the reply length, not of the agent. They interrupt at a
fixed 900 ms over a five-prompt set; this probe aims 350 ms into the reply (measured median
361 ms) over 25 prompts whose replies run to a median of 2445 ms.
`stop_latency ≈ reply_audio_ms − barge_offset` predicts 2084 ms here against 1992 ms
measured. Two probes, one behaviour, two numbers — which is why the verdict rests on the
categorical result and not on the number.

Their mechanism claim was verified here in the source: `live/loop.py:360` allocates
`q = queue.Queue()` *inside* `capture()` (line 346), so the queue is rebuilt every turn and
speech arriving during playback is discarded rather than buffered. This harness measures the
consequence independently, through the transcript: 0 of 73 landed barge-ins left any trace.

### The gap has a closed form, and it replicates

`gap ≈ max(HANGOVER, ARM + W_total)` — for the serial path, `HANGOVER + W_total` — where
`W_total` is summed per-stage work (ASR final + LM to first token + LM to end of sentence +
TTS). Fitted there on 8 runs of a five-prompt set. Tested here on 25 prompts across five
difficulty tiers, a different session:

| system | measured gap | predicted | median residual |
|---|---|---|---|
| cascade-serial | 822 ms | 802 ms | **+24.9 ms** [+14/+41] |
| cascade-fast | 550 ms | 534 ms | **+16.1 ms** [+7/+23] |
| cascade-fast-tiny | 402 ms | 393 ms | **+17.8 ms** [+9/+22] |

Their reported median residual was +15.3 ms. The residual stays a small positive constant —
playback dispatch plus VAD frame quantisation — and never goes negative, which independently
confirms the seriality the model assumes: the stages do not overlap.

**This closed form explains the headline finding.** `W_total` contains TTS synthesis and
LM-to-end-of-sentence time, and both grow with the length of the reply. The gap is a fixed
floor plus work that scales with how much there is to say. There is no term in it for how
hard the question was, which is exactly what the partial correlation shows empirically.
Computed by `bench/gap.py::floor_model` and reported in `results/gap.json` under
`floor_model`.

---

## Running it

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv numpy soundfile scipy librosa praat-parselmouth \
    matplotlib sounddevice faster-whisper mlx-lm piper-tts

./run_bench.py selfcheck                       # every scorer's own check, no agent needed
./run_bench.py all --systems cascade-serial cascade-fast cascade-fast-tiny --n 25
./run_bench.py gap --systems cascade-serial    # one dimension
./run_bench.py aggregate                       # fold results/ into the table + figures
python bench/gap.py --help                     # each scorer is standalone
```

The cascade adapter imports the agent from `~/Desktop/Playground/aliveness-threshold`
(override with `VAB_ALIVENESS_DIR`) and the prosody extractor from
`~/Desktop/Playground/expressive-tts-audit`. **Neither repo is modified by this one.**

### Adding a system

Write an adapter in `systems/` exposing `.name`, `.meta()`, `.open()/.close()`,
`.render_prompt(text)`, `.synth(text)`, and

```python
.turn(prompt_audio, label, record) -> {
    "gap_ms", "reply", "reply_audio_ms", "speech_offset_ms",
    "out_rec", "out_rec_t0",       # recorded output + its perf_counter t0
}
```

then register it in `bench/runner.py::open_systems`. The five scorers were written against
that contract, not against the cascade — the barge-in probe in particular was written for a
duplex system from the start.

---

## Two rules that make this a benchmark and not a demo

**One gap definition, imported not reimplemented.** Every scorer here gets its speech
boundaries from `harness.audio.segments` in the sibling `aliveness-threshold` repo, with the
same parameters that repo's `exchange.py` uses. If this repo had its own copy, the two would
drift and every cross-system number would quietly stop meaning anything.

**Systems are interleaved turn by turn, never run in blocks.** The sibling repo re-ran its own
unchanged code twenty minutes apart the same evening and measured 1452 ms, then 807 ms — a
1.8× swing that loadavg did not explain (16.79 then, 16.97 now). Whatever the machine was
doing is not captured by any variable either harness records. On this machine, a block design
would confound *system* with *wall-clock time* by more than most effects worth reporting. So
every run round-robins the systems, all systems stay loaded for the session, all share one
output device stream, and every turn carries its own timestamp and loadavg.

---

## Honest limitations

- **No commercial system was measured, and none is estimated.** No API keys were available
  for this work, so every system under test had to run on one laptop. GPT Realtime, Gemini
  Live, Siri and Alexa are absent. Their published latency claims are not in this repo and
  must not be added to the table — a system that did not run on this machine has no row.
- **The three systems are not independent.** One agent, three schedules. A clean controlled
  contrast, but not a survey of the field. Every "0/n" below should be read as "this
  architecture cannot do this", not "local voice agents cannot do this".
- **Human raters: n = 0.** Nothing here is validated against what a listener actually feels.
  The five dimensions are motivated by conversation analysis, not by ratings collected in
  this repo. No LLM judge stands in for a person anywhere; if one were ever added it would be
  labelled as an LLM judge everywhere it appeared.
- **The test talker is a TTS voice, and none of these numbers are validated on human
  speech.** Every prompt here is piper-rendered, whose longest internal pause measures 65 ms.
  What that does *not* mean, and an earlier draft of this file got wrong: a separate effort
  (`~/Desktop/Playground/expressive-s2s`) ran 24 held-out CREMA-D human actor clips through
  this same loop and found `--fast --arm 80` produced **3/24 false endpoints against 3/24 on
  the serial control** — two of them the same clips. Arming added zero truncation, because
  arming is not committing: a stale snapshot is thrown away and the turn falls back to the
  serial path. Aggressive arming costs wasted compute, not cut-off talkers.
  The real caveat is the endpointer itself, armed or not: it truncates about **1 in 8 real
  human turns** (3/24), a failure mode this TTS-only prompt set cannot see at all — it
  records 0 false endpoints on every system.
- **Difficulty tiers are an a-priori design variable**, assigned before any system ran. They
  are a design choice, not a measurement of difficulty.
- **Moshi (Kyutai full-duplex, MLX) is NOT-MEASURED.** Its weight download did not finish
  during the session (3.9 GB of ~5.2 GB fetched; bandwidth was the binding constraint and a
  second download was deliberately not started). It is the one system in reach that is
  architecturally capable of scoring above zero on dimensions 2 and 5 — it listens while it
  speaks and emits a continuous stream — so **its absence is the single biggest hole in these
  results.** See `systems/moshi.py`, which records the reason and refuses to return a number.
- **TTS backend variants were not run as separate systems.** All three configurations used
  the piper backend; the macOS `say` fallback appears only as the prosody ceiling reference.
- The prosody dimension reports `not-measured` when librosa and parselmouth are unavailable,
  rather than falling back to a weaker extractor whose numbers would not be comparable.

---

## Provenance

Apple M4 Pro, 24 GB, macOS, Python 3.12. Every result file records the git SHA of this repo
and of the imported agent repo, the loadavg at the start and end of the run, and a
per-turn timestamp and loadavg. Weights and bulk audio are gitignored; a small demo set is
committed under `audio/`.

---

## Environment note (documented fallback)

The night this was built, bandwidth was saturated by a sibling repo's 5 GB weight download.
This repo's own `.venv` finished installing the analysis stack (numpy, scipy, soundfile,
librosa, praat-parselmouth) but **`mlx-metal`, `faster-whisper`, `sounddevice` and `piper-tts`
did not finish downloading into it**.

Rather than block, the documented fallback was used: runs execute on the sibling
`aliveness-threshold` venv's interpreter — which already had the agent stack — with this
repo's `.venv/lib/python3.12/site-packages` on `PYTHONPATH` for the two extraction
libraries. Both are CPython 3.12, macOS arm64, so the ABI matches.

```bash
A=~/Desktop/Playground/aliveness-threshold/.venv/bin/python
PYTHONPATH=.:$PWD/.venv/lib/python3.12/site-packages $A run_bench.py all
```

Every results file records `provenance.executable` and `provenance.sys_path_head`, so which
interpreter produced a given number is checkable rather than remembered. Once `.venv`
completes, `./run_bench.py` works directly with no `PYTHONPATH` and the fallback can be
dropped.
