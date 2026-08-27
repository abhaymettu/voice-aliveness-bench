"""Fold the five dimension result files into one table.

NO COMPOSITE SCORE. The five dimensions are reported side by side and never
collapsed into a single "aliveness" number. That is a deliberate refusal: the
number would be dominated by whichever dimension happened to have the widest
numeric range, it would hide that four of the five are floored at zero for every
system measured, and it would let a system buy a better headline by getting
faster at the one thing that is easy to get faster at.

If you want one number anyway, weight the columns yourself. The weights would
be your choice, not a finding, and any table that shows them must say so.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

RESULTS = Path("results")

# A system can appear in one dimension's results and not another's, because
# dimensions are run as separate sessions. That is not the same as a system
# that ran and scored zero, and it must never render as one.
NOT_IN_RUN = "this system was not part of that dimension's run"
FILES = {
    "gap": "gap.json",
    "interrupt": "interrupt.json",
    "silence": "silence.json",
    "prosody": "prosody.json",
    "nonverbal": "nonverbal.json",
}


def load(results_dir=RESULTS) -> dict:
    out = {}
    for k, f in FILES.items():
        p = Path(results_dir) / f
        if p.is_file():
            out[k] = json.loads(p.read_text())
        else:
            out[k] = {"status": "not-run", "reason": f"{p} does not exist"}
    return out


def systems_seen(res: dict) -> list[str]:
    s = set()
    for d in res.values():
        s.update((d.get("score") or {}).keys())
    return sorted(s)


def table(res: dict) -> list[dict]:
    """One row per system, one cell per dimension. Cells carry n and spread."""
    rows = []
    for name in systems_seen(res):
        r = {"system": name}

        g = (res.get("gap", {}).get("score") or {}).get(name)
        if g and g.get("n"):
            a = g["adaptation"]
            r["gap"] = {
                "n": g["n"],
                "median_ms": g["gap_ms"]["median"],
                "iqr_ms": [g["gap_ms"]["p25"], g["gap_ms"]["p75"]],
                "rho_vs_difficulty": a["rho_gap_vs_tier"].get("rho"),
                "p": a["rho_gap_vs_tier"].get("p"),
                "rho_partialling_reply_length":
                    a["rho_gap_vs_tier_partialling_reply_tokens"].get("rho"),
                "rho_vs_reply_length": a["rho_gap_vs_reply_tokens"].get("rho"),
                "tier5_minus_tier1_ms": a["tier5_minus_tier1_ms"],
                "false_endpoints": g["endpointing"]["false_endpoints"],
                "wer": g["endpointing"]["mean_wer_vs_prompt"],
            }
        else:
            r["gap"] = {"status": "not-measured", "reason": (g or {}).get("reason", NOT_IN_RUN)}

        i = (res.get("interrupt", {}).get("score") or {}).get(name)
        r["interrupt"] = ({
            "n": i["n"], "n_yielded": i["n_yielded"], "yield_rate": i["yield_rate"],
            "n_heard": i["n_heard_barge_in"],
            "talked_over_median_ms": i["talked_over_ms"]["median"],
            "verdict": i["verdict"],
        } if i and i.get("n") else
            {"status": "not-measured", "reason": (i or {}).get("reason", NOT_IN_RUN)})

        s = (res.get("silence", {}).get("score") or {}).get(name)
        r["silence"] = ({
            "n": s["n"], "re_prompts": s["re_prompts"], "hangs": s["hangs"],
            "verdict": s["verdict"],
            "by_probe": {k: {"outcomes": v["outcomes"],
                             "median_ms": v["time_to_outcome_ms"]["median"]}
                         for k, v in s["by_probe"].items()},
        } if s and s.get("n") else
            {"status": "not-measured", "reason": (s or {}).get("reason", NOT_IN_RUN)})

        p = (res.get("prosody", {}).get("score") or {}).get(name)
        if p and p.get("n"):
            f0 = p["per_feature"].get("f0_range", {})
            ctx = p.get("context_sensitivity") or {}
            r["prosody"] = {
                "n": p["n"],
                "f0_range_median_hz": f0.get("overall", {}).get("median"),
                "f0_range_spread_across_affect_hz": f0.get("between_condition_spread"),
                "context_identical": ctx.get("identical"),
                "context_sensitivity": ctx.get("context_sensitivity"),
            }
        else:
            r["prosody"] = {"status": "not-measured",
                            "reason": (p or {}).get("reason", NOT_IN_RUN)}

        nv = (res.get("nonverbal", {}).get("score") or {}).get(name)
        r["nonverbal"] = ({
            "n": nv["n"], "filled_fraction_median": nv["filled_fraction"]["median"],
            "peak_dbfs_median": nv["peak_dbfs"]["median"],
            "dead_air_rate": nv["dead_air_rate"],
            "n_dead_air": nv["n_dead_air"], "verdict": nv["verdict"],
            "demo_wav": nv.get("demo_wav"),
        } if nv and nv.get("n") else
            {"status": "not-measured", "reason": (nv or {}).get("reason", NOT_IN_RUN)})

        rows.append(r)
    return rows


def not_measured(res: dict) -> list[dict]:
    """Systems that exist but produced no numbers, with the reason each time."""
    out = []
    try:
        from systems.moshi import status as moshi_status
        out.append(moshi_status())
    except Exception as e:  # noqa: BLE001
        out.append({"name": "moshi-mlx-q4", "status": "not-measured",
                    "reason": f"adapter unavailable: {type(e).__name__}: {e}"})
    return out


def build(results_dir=RESULTS) -> dict:
    res = load(results_dir)
    return {
        "note": "five dimensions reported separately; no composite score exists, "
                "on purpose. See the module docstring.",
        "dimensions_run": {k: ("ok" if (v.get("score")) else v.get("status", "not-run"))
                           for k, v in res.items()},
        "provenance": next((v.get("provenance") for v in res.values() if v.get("provenance")),
                           None),
        "table": table(res),
        "not_measured": not_measured(res),
        "positive_controls": {
            "nonverbal": (res.get("nonverbal") or {}).get("positive_control"),
            "prosody_ceiling": (res.get("prosody") or {}).get("ceiling_reference"),
        },
    }


def render(agg: dict) -> str:
    L = []
    L.append(f"{'system':<20} {'gap med [IQR]':>20} {'rho diff':>9} {'rho|len':>8} "
             f"{'barge yields':>13} {'gap filled':>11} {'prosody ctx':>12}")
    L.append("-" * 100)
    for r in agg["table"]:
        g = r["gap"]
        gs = (f"{g['median_ms']:.0f} [{g['iqr_ms'][0]:.0f}-{g['iqr_ms'][1]:.0f}]"
              if "median_ms" in g else "not-measured")
        rho = f"{g.get('rho_vs_difficulty')}" if "median_ms" in g else "-"
        rpl = f"{g.get('rho_partialling_reply_length')}" if "median_ms" in g else "-"
        i = r["interrupt"]
        iv = f"{i['n_yielded']}/{i['n']}" if "n" in i else "not-measured"
        nv = r["nonverbal"]
        nvv = f"{nv['n'] - nv['n_dead_air']}/{nv['n']}" if "n" in nv else "not-measured"
        p = r["prosody"]
        pv = ("none" if p.get("context_identical") else
              "some" if p.get("context_identical") is False else "not-measured")
        L.append(f"{r['system']:<20} {gs:>20} {rho:>9} {rpl:>8} {iv:>13} {nvv:>11} {pv:>12}")
    L.append("")
    for nm in agg["not_measured"]:
        if nm.get("status") == "not-measured":
            L.append(f"not-measured: {nm['name']} -- {nm['reason']}")
    return "\n".join(L)


def demo():
    """Self-check: a missing dimension must surface as not-measured, never as 0."""
    fake = {
        "gap": {"score": {"a": {"n": 20, "gap_ms": {"median": 800.0, "p25": 700.0, "p75": 900.0},
                                "adaptation": {"rho_gap_vs_tier": {"rho": 0.1, "p": 0.6},
                                               "rho_gap_vs_tier_partialling_reply_tokens":
                                                   {"rho": 0.02},
                                               "rho_gap_vs_reply_tokens": {"rho": 0.5},
                                               "tier5_minus_tier1_ms": 40.0},
                                "endpointing": {"false_endpoints": 0,
                                                "mean_wer_vs_prompt": 0.01}}}},
        "interrupt": {"status": "not-run"},
        "silence": {"status": "not-run"},
        "prosody": {"status": "not-run"},
        "nonverbal": {"score": {"a": {"n": 20, "filled_fraction": {"median": 0.0},
                                      "peak_dbfs": {"median": -240.0}, "dead_air_rate": 1.0,
                                      "n_dead_air": 20, "verdict": "dead air on 20/20"}}},
    }
    t = table(fake)
    assert len(t) == 1 and t[0]["system"] == "a"
    assert t[0]["gap"]["median_ms"] == 800.0
    assert t[0]["interrupt"]["status"] == "not-measured", t[0]["interrupt"]
    assert t[0]["prosody"]["status"] == "not-measured"
    assert "0" not in str(t[0]["interrupt"].get("n_yielded", "")), "a missing dim leaked a zero"
    assert t[0]["nonverbal"]["dead_air_rate"] == 1.0
    txt = render({"table": t, "not_measured": []})
    assert "not-measured" in txt
    print("aggregate self-check OK  (a dimension that did not run reports not-measured, "
          "never zero)")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="results/aggregate.json")
    ap.add_argument("--selfcheck", action="store_true")
    a = ap.parse_args()
    if a.selfcheck:
        demo()
    else:
        agg = build(a.results)
        Path(a.out).write_text(json.dumps(agg, indent=2, default=str))
        print(render(agg))
        print(f"\nwrote {a.out}")
