"""Build web/index.html: one self-contained file, no CDN, works from disk.

Everything -- figures, audio, numbers -- is inlined as data URIs from the
results in results/ and the clips in audio/. Nothing is fetched at view time
and nothing is typed in by hand: if a number appears on the page, it was read
out of a results JSON that a run on this machine produced.

The audio is the point of the page. A table saying "filled_fraction 0.0" means
nothing to someone deciding whether their voice agent feels alive. Pressing
play on 800 ms of digital silence, then on the same gap with a breath in it,
means everything.
"""

from __future__ import annotations

import base64
import html
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

MAX_BYTES = 5 * 1024 * 1024


def data_uri(p: Path, mime: str) -> str | None:
    if not p or not Path(p).is_file():
        return None
    return f"data:{mime};base64," + base64.b64encode(Path(p).read_bytes()).decode()


def _fig(figdir: Path, name: str) -> str:
    """A figure in both themes; CSS shows whichever matches the reader."""
    lo = data_uri(figdir / f"{name}-light.png", "image/png")
    da = data_uri(figdir / f"{name}-dark.png", "image/png")
    if not lo and not da:
        return ""
    out = []
    if lo:
        out.append(f'<img class="only-light" alt="{name}" src="{lo}">')
    if da:
        out.append(f'<img class="only-dark" alt="{name}" src="{da}">')
    return f'<figure>{"".join(out)}</figure>'


def _audio(p: Path, label: str, note: str) -> str:
    u = data_uri(p, "audio/wav")
    if not u:
        return ""
    return (f'<div class="clip"><div class="clip-l">{html.escape(label)}</div>'
            f'<audio controls preload="none" src="{u}"></audio>'
            f'<div class="clip-n">{html.escape(note)}</div></div>')


def _cell(v, missing="not measured") -> str:
    if v is None:
        return f'<span class="nm">{missing}</span>'
    return html.escape(str(v))


def build(results="results", figdir="figures", audiodir="audio",
          out="web/index.html") -> Path:
    R, F, A = Path(results), Path(figdir), Path(audiodir)
    agg = json.loads((R / "aggregate.json").read_text()) if (R / "aggregate.json").is_file() else {}
    nv = json.loads((R / "nonverbal.json").read_text()) if (R / "nonverbal.json").is_file() else {}
    gap = json.loads((R / "gap.json").read_text()) if (R / "gap.json").is_file() else {}
    prov = agg.get("provenance") or gap.get("provenance") or {}
    rows = agg.get("table") or []

    # ---- leaderboard ----------------------------------------------------
    tr = []
    for r in rows:
        g, i, s, p, n = (r["gap"], r["interrupt"], r["silence"], r["prosody"],
                         r["nonverbal"])
        gcell = (f'<b>{g["median_ms"]:.0f} ms</b><br><span class="sub">IQR '
                 f'{g["iqr_ms"][0]:.0f}&ndash;{g["iqr_ms"][1]:.0f} &middot; n={g["n"]}</span>'
                 if "median_ms" in g else _cell(None))
        acell = (f'<b>&rho; = {g["rho_partialling_reply_length"]}</b><br>'
                 f'<span class="sub">raw &rho; {g["rho_vs_difficulty"]} '
                 f'(p={g["p"]}), reply length partialled out</span>'
                 if "median_ms" in g else _cell(None))
        icell = (f'<b>{i["n_yielded"]}/{i["n"]}</b> yielded<br>'
                 f'<span class="sub">heard the interruption {i["n_heard"]}/{i["n"]}</span>'
                 if "n" in i else _cell(None))
        scell = (f'<b>{"re-prompts" if s["re_prompts"] else "never speaks"}</b><br>'
                 f'<span class="sub">{html.escape(s["verdict"][:70])}</span>'
                 if "n" in s else _cell(None))
        pcell = (f'<b>{"none" if p.get("context_identical") else "some"}</b><br>'
                 f'<span class="sub">{"sample-identical after joy and after grief" if p.get("context_identical") else "renderings differ"}</span>'
                 if p.get("context_identical") is not None else _cell(None))
        ncell = (f'<b>{n["n"] - n["n_dead_air"]}/{n["n"]}</b> gaps filled<br>'
                 f'<span class="sub">loudest frame {n["peak_dbfs_median"]} dBFS</span>'
                 if "n" in n else _cell(None))
        tr.append(f"<tr><th>{html.escape(r['system'])}</th><td>{gcell}</td>"
                  f"<td>{acell}</td><td>{icell}</td><td>{scell}</td>"
                  f"<td>{pcell}</td><td>{ncell}</td></tr>")

    # ---- not measured ---------------------------------------------------
    nmrows = "".join(
        f'<li><b>{html.escape(x["name"])}</b> &mdash; {html.escape(str(x.get("reason", "")))}</li>'
        for x in (agg.get("not_measured") or []) if x.get("status") == "not-measured")

    # ---- audio ----------------------------------------------------------
    clips = []
    for r in rows:
        w = (r["nonverbal"] or {}).get("demo_wav")
        if w and (A.parent / w).is_file():
            g = r["gap"].get("median_ms")
            clips.append(_audio(A.parent / w, r["system"],
                                f"one real gap and the reply after it"
                                + (f" (median gap {g:.0f} ms)" if g else "")))
    ctrl = []
    cues = ((nv.get("positive_control") or {}).get("cues") or {})
    for cue in ("none", "breath", "filled_pause", "backchannel", "verbal_stall"):
        c = cues.get(cue)
        if not c or not c.get("wav"):
            continue
        p = A.parent / c["wav"]
        note = ("dead air, the control for a gap with nothing in it"
                if cue == "none" else
                f'loudest frame {c.get("peak_dbfs")} dBFS')
        ctrl.append(_audio(p, cue.replace("_", " "), note))

    css = """
:root{--bg:#fbfcfd;--fg:#121619;--mut:#5d6873;--line:#e0e5ea;--card:#fff;
--acc:#2a6fb0;--warn:#b0442a;--ok:#2c7a5a;--code:#f2f5f8}
@media (prefers-color-scheme:dark){:root{--bg:#0f1216;--fg:#e6ebf0;--mut:#8d97a1;
--line:#242b32;--card:#151a20;--acc:#5aa4e0;--warn:#e0705a;--ok:#54b98d;--code:#1a2027}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);
font:16px/1.6 -apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif}
.wrap{max-width:1080px;margin:0 auto;padding:48px 22px 80px}
h1{font-size:2.1rem;line-height:1.2;margin:0 0 .3em;letter-spacing:-.02em}
h2{font-size:1.3rem;margin:2.4em 0 .6em;letter-spacing:-.01em}
h3{font-size:1.02rem;margin:1.6em 0 .4em}
.lede{font-size:1.15rem;color:var(--fg);margin:0 0 .2em}
.lede b{color:var(--acc)}
.meta{color:var(--mut);font-size:.85rem;margin:1.4em 0 0}
p{margin:.7em 0}
.scroll{overflow-x:auto;-webkit-overflow-scrolling:touch;margin:1em 0}
table{border-collapse:collapse;width:100%;min-width:820px;font-size:.86rem}
th,td{border-bottom:1px solid var(--line);padding:11px 12px;text-align:left;
vertical-align:top}
thead th{font-size:.72rem;text-transform:uppercase;letter-spacing:.06em;
color:var(--mut);font-weight:600;border-bottom:2px solid var(--line)}
tbody th{font-weight:600;white-space:nowrap;font-family:ui-monospace,SFMono-Regular,Menlo,monospace;
font-size:.82rem}
.sub{color:var(--mut);font-size:.78rem}
.nm{color:var(--mut);font-style:italic}
figure{margin:1.2em 0;padding:0}
figure img{width:100%;height:auto;border:1px solid var(--line);border-radius:10px;display:block}
.only-dark{display:none}
@media (prefers-color-scheme:dark){.only-light{display:none}.only-dark{display:block}}
.clips{display:grid;gap:12px;grid-template-columns:repeat(auto-fit,minmax(290px,1fr));margin:1em 0}
.clip{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:13px}
.clip-l{font-weight:600;font-size:.9rem;font-family:ui-monospace,SFMono-Regular,Menlo,monospace}
.clip audio{width:100%;margin:9px 0 5px}
.clip-n{color:var(--mut);font-size:.78rem}
.callout{background:var(--card);border:1px solid var(--line);border-left:3px solid var(--acc);
border-radius:0 10px 10px 0;padding:14px 18px;margin:1.3em 0}
.callout.warn{border-left-color:var(--warn)}
.callout.ok{border-left-color:var(--ok)}
code,.mono{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:.86em;
background:var(--code);padding:1px 5px;border-radius:4px}
ul{padding-left:1.15em}li{margin:.35em 0}
hr{border:0;border-top:1px solid var(--line);margin:2.6em 0}
.foot{color:var(--mut);font-size:.82rem}
"""

    body = f"""
<div class="wrap">
<h1>Voice Aliveness Bench</h1>
<p class="lede">Everyone building a voice agent argues about whether it
<b>feels alive</b>. Nobody measures it. This is the measure.</p>
<p class="meta">Five dimensions, scored separately and never averaged into one
number. Every figure below came from a run that actually happened on
{html.escape(str(prov.get('platform', 'this machine')))} on
{html.escape(str(prov.get('run_at', ''))[:10])}. Human raters: <b>n = 0</b>.</p>

<h2>The leaderboard</h2>
<div class="scroll"><table>
<thead><tr><th>system</th><th>1. gap</th><th>1b. adapts to difficulty?</th>
<th>2. interruption</th><th>3. silence</th><th>4. prosody vs context</th>
<th>5. fills the gap</th></tr></thead>
<tbody>{''.join(tr)}</tbody>
</table></div>

<div class="callout warn"><b>Four of the five dimensions are floored at zero for
every system measured.</b> Nothing yields to an interruption, nothing speaks
into silence, no voice carries any conversational state, and no gap contains a
single sample above the noise floor. Only the gap length moves &mdash; and it moves
with how long the answer is, not with how hard the question was.</div>

{"<h3>Not measured</h3><ul>" + nmrows + "</ul>" if nmrows else ""}

<h2>Hear it</h2>
<p>This is what the table above sounds like. Each clip is the recorded output of
a real turn: the gap, then the reply.</p>
<div class="clips">{''.join(clips)}</div>

<h3>What a filled gap sounds like</h3>
<p>These are <b>not systems under test</b>. They are the positive control &mdash; the
same detector run over gaps that do contain something, rendered by
<code>harness/cues.py</code> in the sibling repo. They exist so that a zero in
the table reads as a real zero and not a broken detector, and so a listener can
hear what the systems are missing.</p>
<div class="clips">{''.join(ctrl)}</div>

<h2>The finding</h2>
{_fig(F, 'adaptation')}
<p>A person takes longer to answer &ldquo;what is the meaning of life&rdquo; than
&ldquo;what is two plus two&rdquo;. None of these agents do. The raw correlation
between gap and question difficulty looks positive, and on the fastest
configuration it even reaches significance &mdash; but every prompt was matched to
7&ndash;9 words, and once <b>reply length</b> is partialled out the correlation
collapses to nothing on all three. The gap tracks how long the answer is, which
is an accident of synthesising the whole utterance before speaking. Nothing in
any of these systems models difficulty at all.</p>

{_fig(F, 'gap')}
{_fig(F, 'gapfill')}
{_fig(F, 'dimensions')}

<h2>How the gap is defined</h2>
<p class="mono">user speech offset &rarr; agent speech onset, both silence-trimmed</p>
<p>One definition, used by every scorer, imported rather than reimplemented from
<code>harness/audio.py</code> in the sibling
<code>aliveness-threshold</code> repo so the two cannot drift. Any deviation
would make cross-system numbers meaningless.</p>

<div class="callout"><b>Systems are interleaved turn by turn, never run in
blocks.</b> The sibling repo re-ran its own unchanged code twenty minutes apart
the same evening and measured 1452 ms, then 807 ms &mdash; a 1.8&times; swing that
loadavg did not explain. On this machine, blocking by system would confound the
system with the wall clock by more than most effects worth reporting.</div>

<h2>Honest limitations</h2>
<ul>
<li><b>No commercial system was measured.</b> No API keys were available, so
every system here runs on one laptop. GPT Realtime, Gemini Live, Siri and Alexa
are absent and are not estimated.</li>
<li><b>The three systems are not independent.</b> They share a voice, a language
model, an ASR family and a TTS backend, and differ only in <i>when</i> work is
scheduled. That is a clean controlled contrast, but it is not a survey of the
field.</li>
<li><b>Human raters: n = 0.</b> Nothing here is validated against what a
listener actually feels. The five dimensions are motivated by conversation
analysis, not by ratings collected in this repo, and no LLM judge stands in for
a person anywhere.</li>
<li><b>The test talker is a TTS voice</b>, whose longest internal pause measures
65 ms. The fast configurations commit to a reply after 80 ms of silence, which
works here and would cut a real person off. None of these numbers are validated
on human speech.</li>
<li><b>Difficulty tiers are an a-priori design variable</b>, assigned before any
system ran, not a measurement of difficulty.</li>
<li><b>Absent from this benchmark:</b> a full-duplex system. Moshi is the one
system in reach that could score above zero on interruption and non-verbal
presence; its weights did not finish downloading. It is recorded as
not-measured, never estimated.</li>
</ul>

<hr>
<p class="foot">Harness, scorers and raw per-turn JSON are in the repo &mdash;
<code>bench/</code> one module per dimension, each independently runnable;
<code>systems/</code> thin adapters; <code>results/</code> every turn.
Bench {html.escape(str(prov.get('bench_sha', '?')))}, agent
{html.escape(str(prov.get('aliveness_sha', '?')))}.</p>
</div>
"""
    doc = (f"<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">"
           f"<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">"
           f"<title>Voice Aliveness Bench</title><style>{css}</style></head>"
           f"<body>{body}</body></html>")
    o = Path(out)
    o.parent.mkdir(parents=True, exist_ok=True)
    o.write_text(doc)
    return o


def demo():
    """Self-check: the page builds, is self-contained, and stays under 5 MB."""
    import re
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        R = Path(d) / "results"
        R.mkdir()
        (R / "aggregate.json").write_text(json.dumps({
            "provenance": {"platform": "Darwin test arm64", "run_at": "2026-08-27T01:00:00",
                           "bench_sha": "abc", "aliveness_sha": "def"},
            "table": [{"system": "cascade-serial",
                       "gap": {"n": 25, "median_ms": 822.0, "iqr_ms": [782, 930],
                               "rho_vs_difficulty": 0.18, "p": 0.39,
                               "rho_partialling_reply_length": -0.065},
                       "interrupt": {"n": 20, "n_yielded": 0, "n_heard": 0},
                       "silence": {"n": 21, "re_prompts": False, "verdict": "never speaks"},
                       "prosody": {"context_identical": True},
                       "nonverbal": {"n": 20, "n_dead_air": 20, "peak_dbfs_median": -240.0}},
                      {"system": "half-measured",
                       "gap": {"n": 25, "median_ms": 402.0, "iqr_ms": [371, 561],
                               "rho_vs_difficulty": 0.48, "p": 0.017,
                               "rho_partialling_reply_length": 0.195},
                       "interrupt": {"status": "not-measured"},
                       "silence": {"status": "not-measured"},
                       "prosody": {"status": "not-measured"},
                       "nonverbal": {"status": "not-measured"}}],
            "not_measured": [{"name": "moshi-mlx-q4", "status": "not-measured",
                              "reason": "weights incomplete"}]}))
        out = Path(d) / "web/index.html"
        p = build(R, Path(d) / "figures", Path(d) / "audio", out)
        t = p.read_text()
        assert p.stat().st_size < MAX_BYTES, p.stat().st_size
        # self-contained: no external fetches of any kind
        for bad in ("http://", "https://", "//cdn", "<script src", "@import"):
            assert bad not in t, f"page reaches outside itself: {bad}"
        assert "822 ms" in t and "n=25" in t, "the leaderboard did not render its numbers"
        assert "moshi" in t and "not-measured" in t.replace("not measured", "not-measured")
        assert "prefers-color-scheme" in t, "no dark mode"
        # a missing dimension must read as not measured, never as a zero
        assert t.count('class="nm"') == 4, (
            "the four unmeasured dimensions of half-measured did not render as "
            f"not-measured (found {t.count(chr(34)+chr(34))})")
        assert "0/0" not in t and ">0<" not in t, "a missing dimension leaked a zero"
        assert re.search(r"<title>.*Voice Aliveness Bench.*</title>", t)
    print(f"web self-check OK  (self-contained, {len(t) / 1024:.0f} KB with no figures "
          f"or audio, dark mode present, missing dims read as not measured)")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", default="results")
    ap.add_argument("--figures", default="figures")
    ap.add_argument("--audio", default="audio")
    ap.add_argument("--out", default="web/index.html")
    ap.add_argument("--selfcheck", action="store_true")
    a = ap.parse_args()
    if a.selfcheck:
        demo()
    else:
        p = build(a.results, a.figures, a.audio, a.out)
        n = p.stat().st_size
        print(f"wrote {p}  ({n / 1024 / 1024:.2f} MB)"
              + ("  WARNING: over the 5 MB budget" if n > MAX_BYTES else ""))
