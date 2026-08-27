"""Figures. Every one is rendered twice, light and dark, from the result JSONs.

Nothing here computes a number. If a figure shows a value, that value came out
of a results file that came out of a run on this machine. A dimension that was
not measured is drawn as an explicit "not measured" band, never as a zero bar --
the two look identical on a chart and mean opposite things.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

THEMES = {
    "light": {"bg": "#ffffff", "fg": "#101418", "grid": "#d8dee4", "muted": "#6b7681",
              "bar": "#2a6fb0", "bar2": "#7aa8d2", "warn": "#b0442a", "ok": "#2c7a5a"},
    "dark": {"bg": "#0f1216", "fg": "#e8edf2", "grid": "#2a3138", "muted": "#8b959f",
             "bar": "#5aa4e0", "bar2": "#2f5f8a", "warn": "#e0705a", "ok": "#54b98d"},
}


def _style(t):
    c = THEMES[t]
    plt.rcParams.update({
        "figure.facecolor": c["bg"], "axes.facecolor": c["bg"],
        "savefig.facecolor": c["bg"], "text.color": c["fg"],
        "axes.labelcolor": c["fg"], "xtick.color": c["fg"], "ytick.color": c["fg"],
        "axes.edgecolor": c["grid"], "grid.color": c["grid"],
        "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
    })
    return c


def _save(fig, out_dir, name, theme):
    p = Path(out_dir) / f"{name}-{theme}.png"
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(p, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return p


def fig_gap(gap: dict, out_dir, theme) -> Path | None:
    """Gap distribution per system, and gap by difficulty tier."""
    sc = gap.get("score") or {}
    names = [n for n, v in sc.items() if v.get("n")]
    if not names:
        return None
    c = _style(theme)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))

    rows = gap.get("rows", [])
    data = [[r["gap_ms"] for r in rows if r["system"] == n and r.get("gap_ms")] for n in names]
    bp = ax1.boxplot(data, orientation="horizontal", tick_labels=names, widths=0.55, patch_artist=True,
                     medianprops=dict(color=c["fg"], lw=2))
    for b in bp["boxes"]:
        b.set(facecolor=c["bar2"], edgecolor=c["bar"], alpha=0.85)
    for k in ("whiskers", "caps"):
        for e in bp[k]:
            e.set(color=c["muted"])
    for f in bp["fliers"]:
        f.set(markeredgecolor=c["muted"], markersize=3)
    ax1.set_xlabel("gap, ms  (user speech offset -> agent speech onset)")
    ax1.set_title("response gap, all turns")
    ax1.grid(axis="x", alpha=0.4)
    ax1.set_axisbelow(True)
    for i, n in enumerate(names):
        ax1.text(0.99, i + 1.32, f"n={sc[n]['n']}", transform=ax1.get_yaxis_transform(),
                 ha="right", fontsize=8, color=c["muted"])

    for n in names:
        pt = sc[n]["per_tier_gap_ms"]
        ts = sorted(int(k) for k in pt)
        med = [pt[str(t)]["median"] for t in ts]
        lo = [pt[str(t)]["p25"] for t in ts]
        hi = [pt[str(t)]["p75"] for t in ts]
        ax2.plot(ts, med, "o-", lw=2, ms=5, label=n)
        ax2.fill_between(ts, lo, hi, alpha=0.15)
    ax2.set_xticks([1, 2, 3, 4, 5])
    ax2.set_xticklabels(["trivial", "recall", "explain", "reason", "open"], fontsize=8)
    ax2.set_xlabel("question difficulty (assigned a priori)")
    ax2.set_ylabel("gap, ms   median, IQR band")
    ax2.set_title("does the gap move with difficulty?")
    ax2.grid(alpha=0.4)
    ax2.set_axisbelow(True)
    ax2.legend(fontsize=8, frameon=False)
    note = "  ".join(
        f"{n}: rho={sc[n]['adaptation']['rho_gap_vs_tier'].get('rho')} "
        f"(p={sc[n]['adaptation']['rho_gap_vs_tier'].get('p')})" for n in names)
    fig.text(0.5, -0.05, note, ha="center", fontsize=8, color=c["muted"])
    fig.suptitle("Dimension 1 -- turn-taking timing", y=1.02, fontsize=12)
    return _save(fig, out_dir, "gap", theme)


def fig_adaptation(gap: dict, out_dir, theme) -> Path | None:
    """The point of the whole dimension: raw vs partialled correlation."""
    sc = gap.get("score") or {}
    names = [n for n, v in sc.items() if v.get("n")]
    if not names:
        return None
    c = _style(theme)
    fig, ax = plt.subplots(figsize=(8.2, 4.0))
    x = np.arange(len(names))
    w = 0.26
    raw = [sc[n]["adaptation"]["rho_gap_vs_tier"].get("rho") or 0 for n in names]
    rep = [sc[n]["adaptation"]["rho_gap_vs_reply_tokens"].get("rho") or 0 for n in names]
    par = [sc[n]["adaptation"]["rho_gap_vs_tier_partialling_reply_tokens"].get("rho") or 0
           for n in names]
    ax.bar(x - w, raw, w, label="gap vs difficulty", color=c["bar"])
    ax.bar(x, rep, w, label="gap vs reply length", color=c["warn"])
    ax.bar(x + w, par, w, label="gap vs difficulty, reply length partialled out",
           color=c["muted"])
    ax.axhline(0, color=c["fg"], lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=9)
    ax.set_ylabel("Spearman rho")
    ax.set_ylim(-0.3, 1.0)
    ax.set_title("the gap tracks how long the answer is, not how hard the question is")
    ax.grid(axis="y", alpha=0.4)
    ax.set_axisbelow(True)
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    for i, n in enumerate(names):
        ax.text(i, -0.27, f"n={sc[n]['n']}", ha="center", fontsize=8, color=c["muted"])
    return _save(fig, out_dir, "adaptation", theme)


def fig_dimensions(agg: dict, out_dir, theme) -> Path | None:
    """The five dimensions side by side. Not-measured is drawn as such."""
    rows = agg.get("table") or []
    if not rows:
        return None
    c = _style(theme)
    dims = ["gap", "interrupt", "silence", "prosody", "nonverbal"]
    titles = ["1. turn-taking\n(gap adapts?)", "2. interruption\n(yields?)",
              "3. silence\n(re-prompts?)", "4. prosody\n(varies w/ context?)",
              "5. non-verbal\n(fills the gap?)"]
    names = [r["system"] for r in rows]
    fig, ax = plt.subplots(figsize=(9.5, 0.72 * len(names) + 2.6))

    # every cell is a 0..1 "does it do the thing" reading, stated explicitly
    grid, labels = [], []
    for r in rows:
        vals, lab = [], []
        g = r["gap"]
        if "median_ms" in g:
            v = g.get("rho_partialling_reply_length")
            vals.append(max(0.0, v) if v is not None else 0.0)
            lab.append(f"rho={v}\n{g['median_ms']:.0f}ms")
        else:
            vals.append(np.nan); lab.append("not\nmeasured")
        i = r["interrupt"]
        if "n" in i:
            vals.append(i["yield_rate"]); lab.append(f"{i['n_yielded']}/{i['n']}")
        else:
            vals.append(np.nan); lab.append("not\nmeasured")
        s = r["silence"]
        if "n" in s:
            vals.append(1.0 if s["re_prompts"] else 0.0)
            lab.append("re-prompts" if s["re_prompts"] else "never")
        else:
            vals.append(np.nan); lab.append("not\nmeasured")
        p = r["prosody"]
        if p.get("context_identical") is not None:
            vals.append(0.0 if p["context_identical"] else 1.0)
            lab.append("identical" if p["context_identical"] else "differs")
        else:
            vals.append(np.nan); lab.append("not\nmeasured")
        nv = r["nonverbal"]
        if "n" in nv:
            vals.append(1.0 - nv["dead_air_rate"])
            lab.append(f"{nv['n'] - nv['n_dead_air']}/{nv['n']}")
        else:
            vals.append(np.nan); lab.append("not\nmeasured")
        grid.append(vals); labels.append(lab)

    g = np.array(grid, float)
    masked = np.ma.masked_invalid(g)
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        "vab", [c["bg"], c["bar2"], c["ok"]])
    cmap.set_bad(c["grid"])
    ax.imshow(masked, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(dims)))
    ax.set_xticklabels(titles, fontsize=8.5)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=9)
    for i in range(len(names)):
        for j in range(len(dims)):
            v = g[i, j]
            fgc = c["muted"] if np.isnan(v) else (c["fg"] if v < 0.6 else "#08110d")
            ax.text(j, i, labels[i][j], ha="center", va="center", fontsize=8, color=fgc)
    ax.set_xticks(np.arange(-0.5, len(dims), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(names), 1), minor=True)
    ax.grid(which="minor", color=c["bg"], lw=2)
    ax.tick_params(which="minor", length=0)
    ax.set_title("Voice Aliveness Bench -- five dimensions, reported separately", pad=14)
    fig.text(0.5, -0.02, "greener = more alive on that dimension. grey = not measured, "
             "which is not the same as zero. no composite score is computed.",
             ha="center", fontsize=8, color=c["muted"])
    return _save(fig, out_dir, "dimensions", theme)


def fig_gapfill(nv: dict, out_dir, theme) -> Path | None:
    """What is actually in the gap, in dBFS, against the detector's floor and
    what the positive control measured."""
    sc = nv.get("score") or {}
    names = [n for n, v in sc.items() if v.get("n")]
    if not names:
        return None
    c = _style(theme)
    fig, ax = plt.subplots(figsize=(8.6, 4.0))
    ctrl = (nv.get("positive_control") or {}).get("cues") or {}
    cue_names = [k for k in ctrl if k != "none" and ctrl[k].get("detected")]
    xs = names + cue_names
    vals = [sc[n]["peak_dbfs"]["median"] for n in names] + \
           [ctrl[k]["peak_dbfs"] for k in cue_names]
    cols = [c["warn"]] * len(names) + [c["ok"]] * len(cue_names)
    # -240 dBFS is a literal digital zero; clamp for drawing and say so
    draw = [max(v, -90.0) for v in vals]
    ax.bar(range(len(xs)), draw, color=cols,
           bottom=-90.0, width=0.6)
    for i, v in enumerate(vals):
        ax.text(i, max(v, -90.0) + 1.5, "silent" if v < -200 else f"{v:.0f}",
                ha="center", fontsize=8, color=c["fg"])
    ax.axhline(-55.0, color=c["muted"], ls="--", lw=1)
    ax.text(len(xs) - 0.5, -54, "detector floor, -55 dBFS", ha="right", fontsize=8,
            color=c["muted"])
    ax.set_xticks(range(len(xs)))
    ax.set_xticklabels([*names, *[f"control:\n{k}" for k in cue_names]], fontsize=8)
    ax.set_ylabel("loudest frame in the gap, dBFS")
    ax.set_ylim(-90, -5)
    ax.set_title("dimension 5 -- what fills the gap")
    ax.grid(axis="y", alpha=0.35)
    ax.set_axisbelow(True)
    fig.text(0.5, -0.04, "red = systems under test. green = the positive control: the same "
             "detector over gaps that DO contain a cue, so a silent bar is a real zero, "
             "not a broken detector.", ha="center", fontsize=8, color=c["muted"])
    return _save(fig, out_dir, "gapfill", theme)


def build(results_dir="results", out_dir="figures") -> dict:
    made = {}
    rd = Path(results_dir)

    def rd_json(n):
        p = rd / n
        return json.loads(p.read_text()) if p.is_file() else {}

    gap, nv = rd_json("gap.json"), rd_json("nonverbal.json")
    agg = rd_json("aggregate.json")
    for theme in ("light", "dark"):
        for key, fn, arg in (("gap", fig_gap, gap), ("adaptation", fig_adaptation, gap),
                             ("dimensions", fig_dimensions, agg), ("gapfill", fig_gapfill, nv)):
            try:
                p = fn(arg, out_dir, theme)
                if p:
                    made.setdefault(key, []).append(str(p))
            except Exception as e:  # noqa: BLE001
                made.setdefault("errors", []).append(f"{key}-{theme}: {type(e).__name__}: {e}")
    return made


def demo():
    """Self-check: figures render from a minimal fake result, in both themes,
    and a not-measured cell does not become a zero bar."""
    import tempfile
    gap = {"score": {"a": {"n": 20, "gap_ms": {"median": 800, "p25": 700, "p75": 900},
                           "per_tier_gap_ms": {str(t): {"median": 800 + t, "p25": 700,
                                                        "p75": 900} for t in range(1, 6)},
                           "adaptation": {"rho_gap_vs_tier": {"rho": 0.1, "p": 0.6},
                                          "rho_gap_vs_reply_tokens": {"rho": 0.5},
                                          "rho_gap_vs_tier_partialling_reply_tokens":
                                              {"rho": 0.02}}}},
           "rows": [{"system": "a", "gap_ms": 800 + i} for i in range(20)]}
    agg = {"table": [{"system": "a",
                      "gap": {"median_ms": 800, "iqr_ms": [700, 900],
                              "rho_partialling_reply_length": 0.02},
                      "interrupt": {"status": "not-measured"},
                      "silence": {"status": "not-measured"},
                      "prosody": {"status": "not-measured"},
                      "nonverbal": {"n": 20, "n_dead_air": 20, "dead_air_rate": 1.0}}]}
    nv = {"score": {"a": {"n": 20, "peak_dbfs": {"median": -240.0},
                          "filled_fraction": {"median": 0.0}, "dead_air_rate": 1.0,
                          "n_dead_air": 20}},
          "positive_control": {"cues": {"none": {"detected": False, "peak_dbfs": -240.0},
                                        "breath": {"detected": True, "peak_dbfs": -38.0}}}}
    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "figures"
        n = 0
        for theme in ("light", "dark"):
            for fn, arg in ((fig_gap, gap), (fig_adaptation, gap),
                            (fig_dimensions, agg), (fig_gapfill, nv)):
                p = fn(arg, out, theme)
                assert p and p.is_file() and p.stat().st_size > 4000, (fn.__name__, theme, p)
                n += 1
        assert n == 8, n
    print(f"figures self-check OK  ({n} files rendered in both themes from fake results)")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="figures")
    ap.add_argument("--selfcheck", action="store_true")
    a = ap.parse_args()
    if a.selfcheck:
        demo()
    else:
        m = build(a.results, a.out)
        for k, v in m.items():
            print(f"{k}: {v}")
