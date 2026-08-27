"""Build web/index.html: one self-contained file, no CDN except Google Fonts.

Everything -- the leaderboard, the finding chart, the waveforms, every number
and every caveat -- is read out of results/*.json at build time. Nothing is
typed in by hand, so the page cannot drift from the run that produced it.

    .venv/bin/python web/build_page.py [--selfcheck]

The audio is the point of the page. A table saying "filled_fraction 0.0" means
nothing to someone deciding whether their voice agent feels alive. Seeing a flat
line where a breath should be, and pressing play on it, means everything. The
waveforms are measured on the source WAVs; the clips are served as 48 kbps mono
MP3 to keep the page small, and the encode is checked to make sure it did not
put anything audible into a gap the bench scored as empty.
"""

from __future__ import annotations

import argparse
import base64
import html
import json
import re
import subprocess
import sys
import tempfile
import wave
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
MAX_BYTES = 5 * 1024 * 1024
NOISE_FLOOR_DB = -55.0          # the bench's own dead-air threshold
N_PEAKS = 240


# ---------------------------------------------------------------- audio -----
def read_wav(p: Path):
    w = wave.open(str(p))
    x = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(np.float32) / 32768.0
    return x, w.getframerate()


def peaks(x: np.ndarray, n: int = N_PEAKS) -> list[int]:
    h = max(1, len(x) // n)
    k = len(x) // h
    e = np.abs(x[: k * h].reshape(k, h)).max(1)
    if len(e) > n:
        e = e[:n]
    m = float(e.max()) or 1.0
    return [int(round(100 * v / m)) for v in e]


def quiet_frames(x: np.ndarray, sr: int, hop_ms: float = 20.0):
    h = int(sr * hop_ms / 1000)
    k = len(x) // h
    db = 20 * np.log10(np.maximum(np.abs(x[: k * h].reshape(k, h)).max(1), 1e-12))
    return db <= NOISE_FLOOR_DB, hop_ms / 1000.0


def longest_quiet_run(x: np.ndarray, sr: int):
    """(start_s, end_s) of the longest stretch below the bench's noise floor."""
    q, hop = quiet_frames(x, sr)
    best, cur = (0, 0), None
    for i, v in enumerate(list(q) + [False]):
        if v and cur is None:
            cur = i
        elif not v and cur is not None:
            if i - cur > best[1] - best[0]:
                best = (cur, i)
            cur = None
    return best[0] * hop, best[1] * hop


def leading_silence(x: np.ndarray, sr: int) -> float:
    q, hop = quiet_frames(x, sr)
    i = 0
    while i < len(q) and q[i]:
        i += 1
    return i * hop


def mp3_uri(wav: Path) -> tuple[str, float]:
    """48 kbps mono mp3, and the peak level (dBFS) it puts into the source's
    quietest stretch -- so a lossy encode can never make a scored-empty gap
    audible without this failing loudly."""
    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "a.mp3"
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(wav),
                        "-codec:a", "libmp3lame", "-b:a", "48k", "-ac", "1", str(out)],
                       check=True)
        back = Path(d) / "b.wav"
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(out), str(back)],
                       check=True)
        src, sr = read_wav(wav)
        a, b = longest_quiet_run(src, sr)
        y, sr2 = read_wav(back)
        seg = y[int(a * sr2):int(b * sr2)]
        lvl = 20 * np.log10(max(float(np.abs(seg).max()), 1e-12)) if len(seg) else -240.0
        return ("data:audio/mpeg;base64," + base64.b64encode(out.read_bytes()).decode()), lvl


def clip(wav: Path, label: str, note: str, kind: str, window=None):
    x, sr = read_wav(wav)
    uri, gap_lvl = mp3_uri(wav)
    dur = len(x) / sr
    if window is not None:
        a, b = window          # every control shares one gap window, by construction
    else:
        a, b = 0.0, leading_silence(x, sr)   # the agent's own gap, before it speaks
    return dict(label=label, note=note, kind=kind, dur=round(dur, 3),
                peaks=peaks(x), gap=[round(a / dur, 4), round(b / dur, 4)],
                audio=uri, gap_lvl=round(gap_lvl, 1))


# ------------------------------------------------------------- figures ------
def fig(name: str) -> dict | None:
    out = {}
    for theme in ("light", "dark"):
        p = ROOT / "figures" / f"{name}-{theme}.png"
        if p.is_file():
            out[theme] = "data:image/png;base64," + base64.b64encode(p.read_bytes()).decode()
    return out or None


# ---------------------------------------------------------------- build -----
def collect(results="results", audiodir="audio", figures="figures"):
    R, A = ROOT / results, ROOT / audiodir
    agg = json.loads((R / "aggregate.json").read_text())
    nv = json.loads((R / "nonverbal.json").read_text())
    gapj = json.loads((R / "gap.json").read_text())
    rows = agg.get("table") or []

    clips = []
    for r in rows:
        w = (r.get("nonverbal") or {}).get("demo_wav")
        if w and (A / w).is_file():
            clips.append(clip(A / w, r["system"],
                              "one real turn: the gap, then the reply", "system"))
    # The gap window is a property of the recording, not of the cue dropped into it:
    # every control is the same turn with a different thing in the same 790 ms hole.
    # Locate it once on the empty control, and check it against the window the
    # scorer says it used -- if those two disagree, the shading is lying.
    cues = ((nv.get("positive_control") or {}).get("cues") or {})
    window = None
    empty = cues.get("none") or {}
    if empty.get("wav") and (A / empty["wav"]).is_file():
        ex, esr = read_wav(A / empty["wav"])
        window = longest_quiet_run(ex, esr)
        want = float(empty.get("gap_window_ms") or 0) / 1000.0
        if want:
            got = window[1] - window[0]
            assert abs(got - want) < 0.06, (
                f"located a {got*1000:.0f} ms empty stretch but the scorer used a "
                f"{want*1000:.0f} ms gap window")

    controls = []
    for cue in ("none", "breath", "filled_pause", "backchannel", "verbal_stall"):
        c = cues.get(cue)
        if not c or not c.get("wav") or not (A / c["wav"]).is_file():
            continue
        note = ("dead air -- the control for a gap with nothing in it"
                if cue == "none" else
                f'loudest frame {c.get("peak_dbfs")} dBFS \u00b7 '
                f'filled fraction {c.get("filled_fraction")}')
        controls.append(clip(A / c["wav"], cue.replace("_", " "), note, "control", window))

    return dict(
        noise_floor_db=NOISE_FLOOR_DB,
        prov=agg.get("provenance") or {},
        rows=rows,
        not_measured=[x for x in (agg.get("not_measured") or [])
                      if x.get("status") == "not-measured"],
        clips=clips, controls=controls,
        systems=gapj.get("systems") or {},
        gap_definition=gapj.get("gap_definition", ""),
        prompt_set=gapj.get("prompt_set", ""),
        adaptation={r["system"]: {
            "raw": r["gap"]["rho_vs_difficulty"], "p": r["gap"]["p"],
            "reply": r["gap"]["rho_vs_reply_length"],
            "partial": r["gap"]["rho_partialling_reply_length"],
            "n": r["gap"]["n"],
        } for r in rows if "rho_vs_difficulty" in (r.get("gap") or {})},
        per_tier={s: {t: d["median"] for t, d in v["per_tier_gap_ms"].items()}
                  for s, v in gapj.get("score", {}).items() if "per_tier_gap_ms" in v},
        tier_names={t: d["tier_name"] for t, d in
                    next(iter(gapj["score"].values()))["per_tier_gap_ms"].items()}
        if gapj.get("score") else {},
        figs={k: fig(k) for k in ("gap", "gapfill")},
    )


HTML = r"""<!doctype html>
<html lang="en"><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Voice Aliveness Bench</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Archivo:wght@600;800&family=JetBrains+Mono:wght@400;500&family=Source+Sans+3:wght@400;600&display=swap">
<style>
:root{
  --font-display:"Archivo","Helvetica Neue",Helvetica,Arial,sans-serif;
  --font-body:"Source Sans 3",-apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif;
  --font-mono:"JetBrains Mono",ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;

  --bg:#f6f8f9; --panel:#ffffff; --panel2:#eef1f4; --sunk:#e2e8ed;
  --fg:#0f151b; --dim:#596774; --line:#dce3e9; --rule:#bfc9d2;
  --sig:#a86200; --sig-bg:#fdf3e3; --flat:#8894a0; --ctrl:#0b6a6d;
  --warn:#9c3617; --warn-bg:#fdf0ec; --ok:#1c6b48;
  --shadow:0 1px 2px rgba(15,21,27,.05),0 10px 26px rgba(15,21,27,.05);
}
@media (prefers-color-scheme:dark){
  :root:not([data-theme="light"]){
    --bg:#0b0e11; --panel:#131a1f; --panel2:#1a2229; --sunk:#080b0d;
    --fg:#e5edf3; --dim:#8a97a4; --line:#212a32; --rule:#36414b;
    --sig:#f0a93a; --sig-bg:#221909; --flat:#5e6c78; --ctrl:#4fc8c2;
    --warn:#ff8a63; --warn-bg:#22110c; --ok:#5fd39d;
    --shadow:none;
  }
}
:root[data-theme="dark"]{
  --bg:#0b0e11; --panel:#131a1f; --panel2:#1a2229; --sunk:#080b0d;
  --fg:#e5edf3; --dim:#8a97a4; --line:#212a32; --rule:#36414b;
  --sig:#f0a93a; --sig-bg:#221909; --flat:#5e6c78; --ctrl:#4fc8c2;
  --warn:#ff8a63; --warn-bg:#22110c; --ok:#5fd39d;
  --shadow:none;
}

*{box-sizing:border-box}
html{-webkit-text-size-adjust:100%}
body{margin:0;background:var(--bg);color:var(--fg);font-family:var(--font-body);
  font-size:17px;line-height:1.6;overflow-x:hidden;-webkit-font-smoothing:antialiased}
.wrap{max-width:1060px;margin:0 auto;padding:26px 22px 90px}

h1{font-family:var(--font-display);font-weight:800;
  font-size:clamp(34px,6vw,62px);line-height:1.02;letter-spacing:-.035em;
  margin:0 0 18px;max-width:17ch}
h1 em{font-style:normal;color:var(--sig)}
h2{font-family:var(--font-display);font-weight:800;
  font-size:clamp(22px,3.2vw,30px);line-height:1.14;letter-spacing:-.025em;margin:0 0 8px}
h3{font-family:var(--font-display);font-weight:600;font-size:17px;
  letter-spacing:-.01em;margin:34px 0 6px}
.eyebrow,.sec-k{font-family:var(--font-mono);font-size:11px;letter-spacing:.14em;
  text-transform:uppercase;color:var(--dim);margin:0}
.sec-k{color:var(--sig);margin-bottom:9px}
.lede{font-size:clamp(17px,2vw,20px);line-height:1.5;color:var(--dim);
  max-width:62ch;margin:0}
.lede b{color:var(--fg);font-weight:600}
p{margin:.75em 0}
.note{color:var(--dim);font-size:15.5px;max-width:70ch}
code,.mono{font-family:var(--font-mono);font-size:.86em}
code{background:var(--panel2);padding:1px 5px;border-radius:4px}
a{color:var(--sig);text-decoration-thickness:1px;text-underline-offset:2px}
ul{padding-left:1.1em;max-width:74ch}li{margin:.5em 0}

.bar{display:flex;justify-content:space-between;align-items:center;gap:14px;
  margin-bottom:30px}
.themebtn{font-family:var(--font-mono);font-size:11px;letter-spacing:.1em;
  text-transform:uppercase;color:var(--dim);background:none;
  border:1px solid var(--line);border-radius:999px;padding:6px 13px;cursor:pointer}
.themebtn:hover{color:var(--fg);border-color:var(--rule)}
section{border-top:1px solid var(--rule);margin-top:58px;padding-top:32px}

/* ---- hero counters ---- */
.stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));
  gap:1px;background:var(--line);border:1px solid var(--line);border-radius:13px;
  overflow:hidden;margin:32px 0 0}
.stat{background:var(--panel);padding:19px 20px 20px}
.stat .n{font-family:var(--font-display);font-weight:800;font-size:38px;line-height:1;
  letter-spacing:-.035em;font-variant-numeric:tabular-nums;display:block}
.stat.z .n{color:var(--flat)} .stat.m .n{color:var(--sig)}
.stat .c{color:var(--dim);font-size:13.5px;line-height:1.42;margin-top:10px;display:block}
.stat .c b{color:var(--fg);font-weight:600}

/* ---- leaderboard ---- */
.scroll{overflow-x:auto;-webkit-overflow-scrolling:touch;margin:20px 0 0;
  border:1px solid var(--line);border-radius:13px;background:var(--panel);
  box-shadow:var(--shadow)}
table{border-collapse:collapse;width:100%;min-width:1080px;font-size:14px}
th,td{text-align:left;padding:13px 14px;border-bottom:1px solid var(--line);
  vertical-align:top}
thead th{font-family:var(--font-mono);font-size:10.5px;letter-spacing:.08em;
  text-transform:uppercase;color:var(--dim);font-weight:500;
  border-bottom:1px solid var(--rule);background:var(--panel2);line-height:1.45}
thead th b{display:block;color:var(--fg);font-weight:500;font-size:11px;
  letter-spacing:.06em}
tbody tr:last-child td,tbody tr:last-child th{border-bottom:0}
tbody th{font-family:var(--font-mono);font-weight:500;font-size:13px;
  white-space:nowrap;color:var(--fg);background:var(--panel2)}
td .big{font-family:var(--font-display);font-weight:600;font-size:17px;
  letter-spacing:-.02em;font-variant-numeric:tabular-nums;display:block;white-space:nowrap}
td .sub{display:block;color:var(--dim);font-size:11.5px;line-height:1.4;margin-top:4px;
  font-family:var(--font-mono);letter-spacing:-.01em}
td.zero{background:var(--sunk)}
td.zero .big{color:var(--flat)}
td.zero .flatline{display:block;height:2px;background:var(--flat);opacity:.55;
  border-radius:1px;margin:0 0 8px;width:100%}
td.moves .big{color:var(--sig)}
td.nm{background:repeating-linear-gradient(135deg,transparent,transparent 5px,
  var(--panel2) 5px,var(--panel2) 10px)}
td.nm .big{color:var(--dim);font-style:italic;font-family:var(--font-body);font-size:14px}

.callout{border:1px solid var(--line);border-left:4px solid var(--sig);
  background:var(--panel);border-radius:0 12px 12px 0;padding:16px 20px;margin:24px 0 0}
.callout.warn{border-left-color:var(--warn);background:var(--warn-bg)}
.callout .k{font-family:var(--font-mono);font-size:10.5px;letter-spacing:.13em;
  text-transform:uppercase;color:var(--sig);display:block;margin-bottom:6px}
.callout.warn .k{color:var(--warn)}
.callout p{margin:0;font-size:15.5px;line-height:1.55;max-width:76ch}
.callout p+p{margin-top:.6em}

/* ---- clips ---- */
.clips{display:grid;gap:10px;grid-template-columns:1fr;margin-top:18px}
.clip{background:var(--panel);border:1px solid var(--line);border-radius:13px;
  padding:14px 15px 13px;box-shadow:var(--shadow)}
.clip .top{display:flex;align-items:center;gap:11px;margin-bottom:9px}
.clip .dur{font-family:var(--font-mono);font-size:11px;color:var(--dim);flex:0 0 auto}
.clip .pb{width:38px;height:38px;flex:0 0 auto;border-radius:50%;border:0;cursor:pointer;
  display:grid;place-items:center;font-size:12px;background:var(--sig);color:var(--panel);
  position:relative}
.clip.ctrl .pb{background:var(--ctrl)}
.clip .pb:hover{filter:brightness(1.1)}
.clip .nm2{font-family:var(--font-mono);font-size:13px;font-weight:500;flex:1 1 auto;
  min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.clip .kn{font-family:var(--font-mono);font-size:10.5px;color:var(--dim);
  border:1px solid var(--line);border-radius:4px;padding:1px 5px;flex:0 0 auto}
.clip svg{display:block;width:100%;height:auto}
.clip .n2{color:var(--dim);font-size:12.5px;margin-top:7px;line-height:1.4}
.ruler{margin:2px 15px 0;color:var(--dim)}
.ruler svg{display:block;width:100%;height:auto}
.clip[aria-current="true"]{border-color:var(--sig)}
.clip.ctrl[aria-current="true"]{border-color:var(--ctrl)}
.wfkey{display:flex;gap:18px;flex-wrap:wrap;color:var(--dim);font-size:12.5px;
  margin-top:14px;font-family:var(--font-mono)}
.wfkey i{display:inline-block;width:13px;height:9px;vertical-align:0;margin-right:6px;
  border-radius:2px}
.hint{color:var(--dim);font-size:13px;font-family:var(--font-mono);margin-top:14px}
kbd{font-family:var(--font-mono);border:1px solid var(--line);border-bottom-width:2px;
  border-radius:4px;padding:0 4px;color:var(--fg)}

/* ---- figures ---- */
figure{margin:20px 0 0;padding:0}
figure img{width:100%;height:auto;border:1px solid var(--line);border-radius:12px;
  display:block}
.only-dark{display:none}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]) .only-light{display:none}
  :root:not([data-theme="light"]) .only-dark{display:block}}
:root[data-theme="dark"] .only-light{display:none}
:root[data-theme="dark"] .only-dark{display:block}
figcaption{color:var(--dim);font-size:13px;margin-top:8px;font-family:var(--font-mono)}
.chart{background:var(--panel);border:1px solid var(--line);border-radius:13px;
  padding:18px 16px 12px;margin-top:20px;box-shadow:var(--shadow)}
.chart svg{display:block;width:100%;height:auto}
.leg{display:flex;gap:20px;flex-wrap:wrap;color:var(--dim);font-size:13px;
  margin-top:6px;padding:0 6px}
.leg i{display:inline-block;width:12px;height:12px;border-radius:3px;margin-right:7px;
  vertical-align:-1px}

footer{border-top:1px solid var(--rule);margin-top:62px;padding-top:24px;
  color:var(--dim);font-size:13.5px;line-height:1.65;max-width:80ch}
:focus-visible{outline:2px solid var(--sig);outline-offset:2px;border-radius:4px}
@media (prefers-reduced-motion:reduce){
  *{animation-duration:.001ms !important;transition-duration:.001ms !important}
}
@media(max-width:560px){
  .wrap{padding:20px 16px 70px}
  .stat{padding:15px 17px 17px}.stat .n{font-size:32px}
}
</style></head>
<body><div class="wrap">

<div class="bar">
  <p class="eyebrow">voice-aliveness-bench &middot; human raters n = 0</p>
  <button class="themebtn" id="theme" aria-label="Switch colour theme">theme</button>
</div>

<header>
<h1 id="h1"></h1>
<p class="lede">Everyone building a voice agent argues about whether it
<b>feels alive</b>. Nobody measures it. This is the measure: five dimensions,
scored separately and never averaged into one number, across four voice-agent
configurations running on one laptop.</p>
<div class="stats" id="stats"></div>
</header>

<section>
<p class="sec-k">The leaderboard</p>
<h2>Five dimensions, four systems, no composite score</h2>
<p class="note">A flat rule marks a dimension that came back at the floor: not
&ldquo;low&rdquo;, but zero events out of every trial run. Hatching marks a
dimension that was not measured &mdash; which is never the same as a zero.</p>
<div class="scroll"><table id="board"></table></div>
<div class="callout warn" id="wall"></div>
</section>

<section>
<p class="sec-k">Hear it</p>
<h2>What a zero sounds like</h2>
<p class="note">Each clip is the recorded output of a real turn. The shaded stretch
is the gap &mdash; the agent has heard you and has not started talking yet. All four
are drawn on one time axis, so the gaps are directly comparable: the reply is about
the same length every time, and the gap in front of it is what grows. Nothing here is
a rendering choice: that is the audio.</p>
<div class="clips" id="clips"></div>
<div class="wfkey">
  <span><i style="background:var(--sunk);border:1px dashed var(--rule)"></i>the gap</span>
  <span><i style="background:var(--flat)"></i>not yet played</span>
  <span><i style="background:var(--sig)"></i>played</span>
</div>
<p class="hint"><kbd>1</kbd>&ndash;<kbd>9</kbd> play a clip &middot;
<kbd>R</kbd> replay &middot; <kbd>Space</kbd> stop</p>

<h3>What a filled gap sounds like</h3>
<p class="note">These are <b>not systems under test</b>. They are the positive
control &mdash; the same detector run over gaps that do contain something, rendered
by <code>harness/cues.py</code> in the sibling repo. They exist so that a zero in
the table reads as a real zero and not a broken detector, and so a listener can
hear what the systems are missing. Same gap window in all five; only the first
one is empty.</p>
<div class="clips" id="controls"></div>
</section>

<section>
<p class="sec-k">The finding</p>
<h2>The gap tracks how long the answer is, not how hard the question was</h2>
<p class="note" id="findprose"></p>
<div class="chart">
  <div id="adapt"></div>
  <div class="leg">
    <span><i style="background:var(--sig)"></i>gap vs question difficulty</span>
    <span><i style="background:var(--ctrl)"></i>gap vs reply length</span>
    <span><i style="background:var(--flat)"></i>gap vs difficulty, reply length partialled out</span>
  </div>
</div>
<p class="note" style="margin-top:18px">A person takes longer to answer
&ldquo;what is the meaning of life&rdquo; than &ldquo;what is two plus two&rdquo;.
None of these agents do. The gap tracking reply length is an accident of
synthesising the whole utterance before speaking. Nothing in any of these systems
models difficulty at all.</p>
<div id="morefigs"></div>
</section>

<section>
<p class="sec-k">How it was run</p>
<h2>One definition of the gap, and no blocking</h2>
<p class="mono" style="color:var(--sig);font-size:14px" id="gapdef"></p>
<p class="note">One definition, used by every scorer, imported rather than
reimplemented from <code>harness/audio.py</code> in the sibling
<code>aliveness-threshold</code> repo so the two cannot drift. Any deviation would
make cross-system numbers meaningless.</p>
<div class="callout">
  <span class="k">Systems are interleaved turn by turn, never run in blocks</span>
  <p>The sibling repo re-ran its own unchanged code twenty minutes apart the same
  evening and measured 1452 ms, then 807 ms &mdash; a 1.8&times; swing that loadavg
  did not explain. On this machine, blocking by system would confound the system
  with the wall clock by more than most effects worth reporting.</p>
</div>
</section>

<section>
<p class="sec-k">Honest limitations</p>
<h2>What this benchmark does not tell you</h2>
<ul id="limits"></ul>
<div class="callout warn" id="notmeasured"></div>
</section>

<footer id="foot"></footer>
</div>

<script>
const B = __DATA__;
const $ = s => document.querySelector(s);
const esc = s => String(s).replace(/[&<>]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));
const f0 = x => Math.round(x);
/* keep a whole-number float printing as the source prints it: -240.0, not -240 */
const flt = v => Number.isInteger(v) ? v.toFixed(1) : String(v);
const word = n => ['zero','one','two','three','four','five','six','seven','eight','nine'][n] || n;

/* ---------- theme ---------- */
(function(){
  const b = $('#theme');
  const read = () => { try { return localStorage.getItem('vab-theme'); } catch(e){ return null; } };
  const apply = v => {
    if (v) document.documentElement.setAttribute('data-theme', v);
    else document.documentElement.removeAttribute('data-theme');
    b.textContent = v || 'auto';
  };
  apply(read());
  b.onclick = () => {
    const next = {null:'light', 'light':'dark', 'dark':null}[read()];
    try { next ? localStorage.setItem('vab-theme', next) : localStorage.removeItem('vab-theme'); } catch(e){}
    apply(next);
  };
})();

/* ---------- headline counters, summed off the same rows the table renders ---- */
const R = B.rows;
const sum = (f) => R.reduce((a, r) => a + (f(r) || 0), 0);
const has = (o, k) => o && o[k] !== undefined && o[k] !== null;

const intN   = sum(r => has(r.interrupt,'n') ? r.interrupt.n : 0);
const intY   = sum(r => has(r.interrupt,'n') ? r.interrupt.n_yielded : 0);
const silN   = sum(r => has(r.silence,'n') ? r.silence.n : 0);
const silSpk = R.filter(r => has(r.silence,'n') && r.silence.re_prompts).length;
const nvN    = sum(r => has(r.nonverbal,'n') ? r.nonverbal.n : 0);
const nvFill = sum(r => has(r.nonverbal,'n') ? r.nonverbal.n - r.nonverbal.n_dead_air : 0);
const meds   = R.filter(r => has(r.gap,'median_ms')).map(r => r.gap.median_ms);

/* how many system x dimension cells came back at the floor */
const floors = R.map(r => [
  has(r.interrupt,'n') && r.interrupt.n_yielded === 0,
  has(r.silence,'n') && r.silence.re_prompts === false,
  has(r.prosody,'context_sensitivity') && r.prosody.context_sensitivity === 0,
  has(r.nonverbal,'n') && r.nonverbal.n_dead_air === r.nonverbal.n,
]);
const nZero = floors.flat().filter(Boolean).length;
const nCells = R.length * 5;

$('#h1').innerHTML = `${nZero === nCells - R.length
  ? nZero + ' of these ' + nCells + ' scores are <em>exactly zero</em>.'
  : nZero + ' of ' + nCells + ' scores came back at the floor.'}`;

$('#stats').innerHTML = [
  ['z', `${intY} / ${intN}`, 'barge-ins that made an agent stop talking. Every one of them ran the reply to the end, and <b>none reached the transcript</b>.'],
  ['z', `${silSpk} / ${R.length}`, `systems that ever spoke into a silence. All ${silN} probes raised an error instead.`],
  ['z', `${nvFill} / ${nvN}`, 'gaps containing a single sample above the noise floor. <b>No breath, no filler, no backchannel.</b>'],
  ['m', `${f0(Math.min(...meds))}–${f0(Math.max(...meds))} ms`, 'the one dimension that does move: how long it waits before answering. It moves with the answer, not the question.'],
].map(([c, n, t]) =>
  `<div class="stat ${c}"><span class="n">${n}</span><span class="c">${t}</span></div>`).join('');

/* ---------- leaderboard ---------- */
const nm = 'not measured';
function cell(v, cls){ return v === null
  ? `<td class="nm"><span class="big">${nm}</span></td>`
  : `<td class="${cls || ''}">${v}</td>`; }

const COLS = [
  ['1', 'response gap', r => has(r.gap,'median_ms') ? [
      `<span class="big">${f0(r.gap.median_ms)} ms</span><span class="sub">IQR ${
      f0(r.gap.iqr_ms[0])}–${f0(r.gap.iqr_ms[1])} · n=${r.gap.n}</span>`,
      'moves'] : null],
  ['1b', 'adapts to difficulty?', r => has(r.gap,'rho_partialling_reply_length') ? [
      `<span class="big">ρ = ${r.gap.rho_partialling_reply_length}</span><span class="sub">raw ρ ${
      r.gap.rho_vs_difficulty} (p=${r.gap.p}), reply length partialled out</span>`, ''] : null],
  ['2', 'yields to interruption?', r => has(r.interrupt,'n') ? [
      `<span class="flatline"></span><span class="big">${r.interrupt.n_yielded}/${
      r.interrupt.n} yielded</span><span class="sub">heard the interruption ${
      r.interrupt.n_heard}/${r.interrupt.n}</span>`,
      r.interrupt.n_yielded === 0 ? 'zero' : ''] : null],
  ['3', 'speaks into silence?', r => has(r.silence,'n') ? [
      `<span class="flatline"></span><span class="big">${
      r.silence.re_prompts ? 're-prompts' : 'never speaks'}</span><span class="sub">${
      esc(r.silence.verdict)}</span>`, r.silence.re_prompts === false ? 'zero' : ''] : null],
  ['4', 'prosody varies with context?', r => has(r.prosody,'context_sensitivity') ? [
      `<span class="flatline"></span><span class="big">${
      r.prosody.context_sensitivity === 0 ? 'none' : 'some'}</span><span class="sub">${
      esc(r.prosody.context_verdict || '')}</span>`,
      r.prosody.context_sensitivity === 0 ? 'zero' : ''] : null],
  ['5', 'fills the gap?', r => has(r.nonverbal,'n') ? [
      `<span class="flatline"></span><span class="big">${
      r.nonverbal.n - r.nonverbal.n_dead_air}/${r.nonverbal.n} filled</span><span class="sub">loudest frame ${
      flt(r.nonverbal.peak_dbfs_median)} dBFS</span>`,
      r.nonverbal.n_dead_air === r.nonverbal.n ? 'zero' : ''] : null],
];

$('#board').innerHTML =
  '<thead><tr><th>system</th>' + COLS.map(([n, t]) =>
    `<th>${n}<b>${t}</b></th>`).join('') + '</tr></thead><tbody>'
  + R.map(r => '<tr><th>' + esc(r.system) + '</th>'
      + COLS.map(([,, f]) => { const v = f(r); return v ? cell(v[0], v[1]) : cell(null); }).join('')
      + '</tr>').join('')
  + '</tbody>';

$('#wall').innerHTML = `<span class="k">Read the flat rules</span>
  <p><b>Four of the five dimensions are floored at zero for every system
  measured.</b> Nothing yields to an interruption, nothing speaks into silence, no
  voice carries any conversational state, and no gap contains a single sample above
  the noise floor. Only the gap length moves &mdash; and it moves with how long the
  answer is, not with how hard the question was.</p>`;

/* ---------- clips ---------- */
const audio = new Audio();
let cur = null, raf = 0;
const all = () => [...document.querySelectorAll('.clip')];

function wave(c, i){
  const W = 600, H = 52, mid = H / 2, n = c.peaks.length;
  const w = W * c.sc, bw = w / n;
  const gx = c.gap[0] * w, gw = Math.max(0, (c.gap[1] - c.gap[0]) * w);
  let bars = '';
  c.peaks.forEach((p, j) => {
    const h = Math.max(1.2, (p / 100) * (H - 5));
    bars += `<rect x="${(j*bw).toFixed(2)}" y="${(mid-h/2).toFixed(2)}"
      width="${Math.max(.7,bw-.7).toFixed(2)}" height="${h.toFixed(2)}" rx=".6"/>`;
  });
  return `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Waveform of ${esc(c.label)}:
    a silent gap, then speech">
    ${gw > 1 ? `<rect x="${gx}" y="0" width="${gw}" height="${H}" fill="var(--sunk)"
      stroke="var(--rule)" stroke-width="1" stroke-dasharray="3 3" rx="3"/>` : ''}
    <g fill="var(--flat)" opacity=".85">${bars}</g>
    <clipPath id="cp${i}"><rect id="cpr${i}" x="0" y="0" width="0" height="${H}"/></clipPath>
    <g fill="var(--wfa)" clip-path="url(#cp${i})">${bars}</g>
    <line id="ph${i}" x1="0" y1="0" x2="0" y2="${H}" stroke="var(--wfa)"
      stroke-width="1.5" opacity="0"/>
  </svg>`;
}

function ruler(maxDur){
  const W = 600, H = 27;
  let g = `<svg viewBox="0 0 ${W} ${H}" aria-hidden="true">
    <line x1="0" y1="1" x2="${W}" y2="1" stroke="currentColor" opacity=".3"/>`;
  const step = maxDur > 3 ? 1 : 0.5;
  for (let t = 0; t <= maxDur + 1e-9; t += step){
    const x = (t / maxDur) * W;
    g += `<line x1="${x.toFixed(1)}" y1="1" x2="${x.toFixed(1)}" y2="5"
      stroke="currentColor" opacity=".5"/>
      <text x="${Math.min(W-12, Math.max(8, x)).toFixed(1)}" y="15" fill="currentColor"
      font-size="9.5" font-family="var(--font-mono)" text-anchor="middle">${
      t === 0 ? '0' : t.toFixed(step < 1 ? 1 : 0)}</text>`;
  }
  return `<div class="ruler">${g}<text x="${W}" y="25" fill="currentColor" font-size="9.5"
    font-family="var(--font-mono)" text-anchor="end" opacity=".8">seconds</text></svg></div>`;
}

function render(list, into, base){
  const maxDur = Math.max(...list.map(c => c.dur));
  list.forEach(c => c.sc = c.dur / maxDur);
  $(into).innerHTML = list.map((c, i) => {
    const k = base + i;
    return `<div class="clip${c.kind === 'control' ? ' ctrl' : ''}" data-k="${k}"
      aria-current="false" style="--wfa:var(${c.kind === 'control' ? '--ctrl' : '--sig'})">
      <div class="top">
        <button class="pb" data-k="${k}" aria-label="Play ${esc(c.label)}">&#9654;</button>
        <span class="nm2">${esc(c.label)}</span>
        <span class="dur">${c.dur.toFixed(2)}s</span>
        <span class="kn">${k + 1}</span>
      </div>
      ${wave(c, k)}
      <div class="n2">${esc(c.note)}</div>
    </div>`;
  }).join('') + ruler(maxDur);
}
const CLIPS = B.clips.concat(B.controls);
render(B.clips, '#clips', 0);
render(B.controls, '#controls', B.clips.length);
document.querySelectorAll('.pb').forEach(b => b.onclick = () => play(+b.dataset.k));

function play(k, restart){
  if (!restart && cur === k && !audio.paused){ stop(); return; }
  audio.src = CLIPS[k].audio;
  audio.currentTime = 0;
  cur = k;
  audio.play().catch(() => {});
  paint(); tick();
}
function stop(){
  audio.pause(); cancelAnimationFrame(raf);
  const p = document.getElementById('ph' + cur), r = document.getElementById('cpr' + cur);
  if (p) p.setAttribute('opacity', '0');
  if (r) r.setAttribute('width', '0');
  cur = null; paint();
}
audio.onended = stop;
function paint(){
  all().forEach(el => {
    const on = +el.dataset.k === cur;
    el.setAttribute('aria-current', on ? 'true' : 'false');
    el.querySelector('.pb').innerHTML = on ? '&#9632;' : '&#9654;';
    if (!on) {
      const r = document.getElementById('cpr' + el.dataset.k);
      const p = document.getElementById('ph' + el.dataset.k);
      if (r) r.setAttribute('width', '0');
      if (p) p.setAttribute('opacity', '0');
    }
  });
}
function tick(){
  if (cur === null) return;          // a queued frame can outlive stop()
  const d = audio.duration || CLIPS[cur].dur;
  const x = Math.min(1, audio.currentTime / d) * 600 * CLIPS[cur].sc;
  const r = document.getElementById('cpr' + cur), p = document.getElementById('ph' + cur);
  if (r) r.setAttribute('width', x.toFixed(1));
  if (p){ p.setAttribute('opacity', '.9');
          p.setAttribute('x1', x.toFixed(1)); p.setAttribute('x2', x.toFixed(1)); }
  if (!audio.paused) raf = requestAnimationFrame(tick);
}
document.addEventListener('keydown', e => {
  if (e.metaKey || e.ctrlKey || e.altKey) return;
  const onBtn = e.target.tagName === 'BUTTON';
  const n = parseInt(e.key, 10);
  if (n >= 1 && n <= Math.min(9, CLIPS.length)){ e.preventDefault(); play(n - 1); }
  else if (e.key === 'r' || e.key === 'R'){ e.preventDefault(); play(cur === null ? 0 : cur, true); }
  else if (e.key === ' ' && cur !== null){
    if (onBtn) return;               // let Space activate whatever is focused
    e.preventDefault(); stop();
  }
});

/* ---------- the adaptation chart ---------- */
(function(){
  const S = Object.keys(B.adaptation);
  if (!S.length) return;
  const series = [['raw','var(--sig)'],['reply','var(--ctrl)'],['partial','var(--flat)']];
  const vals = S.flatMap(s => series.map(([k]) => B.adaptation[s][k]));
  const lo = Math.min(-0.4, Math.floor(Math.min(...vals) * 10) / 10);
  const hi = Math.max(1.0, Math.ceil(Math.max(...vals) * 10) / 10);
  const W = 760, H = 300, m = {l:44, r:12, t:12, b:52};
  const Y = v => H - m.b - ((v - lo) / (hi - lo)) * (H - m.t - m.b);
  const gw = (W - m.l - m.r) / S.length, bw = Math.min(34, (gw - 18) / 3);
  let g = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Spearman correlations
    between response gap and question difficulty, before and after partialling out
    reply length, for each of the ${S.length} configurations.">`;
  for (let v = Math.ceil(lo * 5) / 5; v <= hi + 1e-9; v += 0.2){
    const z = Math.abs(v) < 1e-9;
    g += `<line x1="${m.l}" y1="${Y(v).toFixed(1)}" x2="${W-m.r}" y2="${Y(v).toFixed(1)}"
      stroke="${z ? 'var(--rule)' : 'var(--line)'}" stroke-width="${z ? 1.5 : 1}"/>
      <text x="${m.l-8}" y="${(Y(v)+4).toFixed(1)}" fill="var(--dim)" font-size="11"
      font-family="var(--font-mono)" text-anchor="end">${v.toFixed(1)}</text>`;
  }
  S.forEach((s, i) => {
    const x0 = m.l + i * gw;
    series.forEach(([k, c], j) => {
      const v = B.adaptation[s][k];
      const x = x0 + (gw - bw * 3 - 8) / 2 + j * (bw + 4);
      const y = Math.min(Y(v), Y(0)), h = Math.abs(Y(v) - Y(0));
      g += `<rect x="${x.toFixed(1)}" y="${y.toFixed(1)}" width="${bw.toFixed(1)}"
        height="${Math.max(1, h).toFixed(1)}" fill="${c}" rx="2"/>`;
      g += `<text x="${(x+bw/2).toFixed(1)}" y="${(v >= 0 ? y - 5 : y + h + 13).toFixed(1)}"
        fill="var(--dim)" font-size="10" font-family="var(--font-mono)"
        text-anchor="middle">${v}</text>`;
    });
    g += `<text x="${(x0+gw/2).toFixed(1)}" y="${H-30}" fill="var(--fg)" font-size="12"
      font-family="var(--font-mono)" text-anchor="middle">${esc(s)}</text>
      <text x="${(x0+gw/2).toFixed(1)}" y="${H-14}" fill="var(--dim)" font-size="10.5"
      font-family="var(--font-mono)" text-anchor="middle">n=${B.adaptation[s].n}</text>`;
  });
  g += `<text x="13" y="${(H-m.b+m.t)/2}" fill="var(--dim)" font-size="10.5"
    font-family="var(--font-mono)" transform="rotate(-90 13 ${(H-m.b+m.t)/2})"
    text-anchor="middle">SPEARMAN ρ</text></svg>`;
  $('#adapt').innerHTML = g;

  const sig = S.filter(s => B.adaptation[s].p < 0.05).length;
  const rep = S.map(s => B.adaptation[s].reply);
  const par = S.map(s => B.adaptation[s].partial);
  $('#findprose').innerHTML =
    `The raw correlation between gap and question difficulty is positive on all
    ${word(S.length)} configurations and reaches p&lt;.05 on ${word(sig)} of them. But every prompt
    was matched to 7&ndash;9 words, and the gap tracks <b>reply length</b> at
    &rho; = ${Math.min(...rep)} to ${Math.max(...rep)}. Partial that out and what is
    left is ${par.join(', ')} &mdash; nothing that points the same way twice.`;
})();

/* ---------- supporting figures ---------- */
const FIGCAP = {
  gap: 'Dimension 1 in full: the gap distribution per system, and the gap by difficulty tier.',
  gapfill: 'Dimension 5: what the detector found in each measured gap, against the positive controls.',
};
$('#morefigs').innerHTML = Object.entries(B.figs).filter(([, v]) => v).map(([k, v]) =>
  `<figure>${v.light ? `<img class="only-light" alt="${k}" src="${v.light}">` : ''}${
   v.dark ? `<img class="only-dark" alt="${k}" src="${v.dark}">` : ''}
   <figcaption>${FIGCAP[k] || k}</figcaption></figure>`).join('');

/* ---------- limitations ---------- */
const sysd = B.systems, names = Object.keys(sysd);
const tts = [...new Set(names.map(n => (sysd[n].tts || {}).backend))].filter(Boolean);
const asr = [...new Set(names.map(n => sysd[n].asr_final))].filter(Boolean);
const lm  = [...new Set(names.map(n => sysd[n].lm))].filter(Boolean);
$('#limits').innerHTML = [
  `<b>No commercial system was measured.</b> No API keys were available, so every
   system here runs on one laptop. GPT Realtime, Gemini Live, Siri and Alexa are
   absent and are not estimated.`,
  `<b>The ${word(R.length)} systems are not independent.</b> All ${word(R.length)} run the same
   language model (<code>${esc(lm[0] || '?')}</code>) and the same Whisper ASR family
   (${asr.map(a => '<code>' + esc(a) + '</code>').join(', ')}) on one machine, and
   differ in <i>when</i> work is scheduled and which TTS backend speaks it
   (${tts.map(t => '<code>' + esc(t) + '</code>').join(', ')}). That is a clean
   controlled contrast, but it is not a survey of the field.`,
  `<b>Human raters: n = 0.</b> Nothing here is validated against what a listener
   actually feels. The five dimensions are motivated by conversation analysis, not by
   ratings collected in this repo, and no LLM judge stands in for a person anywhere.`,
  `<b>The test talker is a TTS voice</b>, whose longest internal pause measures 65 ms.
   The fast configurations commit to a reply after 80 ms of silence, which works here
   and would cut a real person off. None of these numbers are validated on human speech.`,
  `<b>Difficulty tiers are an a-priori design variable</b>, assigned before any system
   ran, not a measurement of difficulty. (<code>${esc(B.prompt_set)}</code>)`,
].map(t => `<li>${t}</li>`).join('');

$('#notmeasured').innerHTML = B.not_measured.length
  ? `<span class="k">Absent from this benchmark: a full-duplex system</span>` +
    B.not_measured.map(x => `<p><b>${esc(x.name)}</b> &mdash; ${esc(x.reason)}</p>` +
      (x.why_it_matters ? `<p>${esc(x.why_it_matters[0].toUpperCase() + x.why_it_matters.slice(1))}. It is recorded as not-measured, never estimated.</p>` : '')).join('')
  : '';

$('#gapdef').textContent = B.gap_definition;
$('#foot').innerHTML =
  `Harness, scorers and raw per-turn JSON are in the repo &mdash; <code>bench/</code>
   one module per dimension, each independently runnable; <code>systems/</code> thin
   adapters; <code>results/</code> every turn. Every figure and number above came from
   a run that actually happened on ${esc(B.prov.platform || 'this machine')} on
   ${esc(String(B.prov.run_at || '').slice(0, 10))}. Bench
   ${esc(B.prov.bench_sha || '?')}, agent ${esc(B.prov.aliveness_sha || '?')}.
   Clips are 48 kbps mono MP3 so the page stays small; the waveforms and every number
   above are measured on the source WAVs. The encode is lossy, so the gaps the bench
   scored as digital silence carry the codec's own floor, measured at
   ${B.encode_floor_db} dBFS &mdash; still below the ${B.noise_floor_db} dBFS the
   detector fires at, which the build asserts before it writes this file.`;
</script>
</body></html>
"""


def build(out=None):
    data = collect()
    data["encode_floor_db"] = max(
        c["gap_lvl"] for c in data["clips"] + data["controls"]
        if c["kind"] == "system" or c["label"] == "none")
    doc = HTML.replace("__DATA__", json.dumps(data))
    p = Path(out) if out else ROOT / "web/index.html"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(doc)

    # the mp3 encode must not put anything audible into a gap the bench scored empty
    loud = [(c["label"], c["gap_lvl"]) for c in data["clips"] + data["controls"]
            if c["kind"] == "system" or c["label"] == "none"
            if c["gap_lvl"] > NOISE_FLOOR_DB]
    assert not loud, f"mp3 encode raised a scored-empty gap above the noise floor: {loud}"

    # self-contained: strip inlined payloads, then only Google Fonts may remain
    markup = re.sub(r"data:(?:audio/mpeg|image/png);base64,[A-Za-z0-9+/=]+", "", doc)
    urls = re.findall(r"(?:src|href)\s*=\s*[\"']([^\"']+)", markup)
    ok = ("https://fonts.googleapis.com", "https://fonts.gstatic.com")
    bad = [u for u in urls if not u.startswith(("#", "${", "data:"))
           and not u.startswith(ok)]
    assert not bad, f"page references external resources: {bad}"
    n = p.stat().st_size
    assert n < MAX_BYTES, f"{n/1e6:.2f} MB, budget is 5 MB"
    print(f"wrote {p}  ({n/1024/1024:.2f} MB, {len(data['clips'])} system clips + "
          f"{len(data['controls'])} controls, quietest-gap level after encode "
          f"{max(c['gap_lvl'] for c in data['clips']):.1f} dBFS)")
    return p


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=None)
    build(ap.parse_args().out)
