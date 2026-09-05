"""Streamlit demo surface for solution.investigate().

Not part of the graded interface - it imports the investigator and adds no
logic of its own. Its only job is to make the *reasoning* visible: which
documents were retrieved, which corroboration axes fired, and why the
confidence landed where it did.

    cd usecase-2-production-incident-investigator
    streamlit run submissions/aljabri-alam/app.py

Deployable as-is to Streamlit Community Cloud: the corpus root is discovered by
walking up from this file, so it works whether the app sits in this repo or in a
flattened one next to a data/ directory.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import streamlit as st

HERE = Path(__file__).resolve().parent


def _find_corpus_root() -> Path | None:
    """Walk up looking for the directory that holds data/<incident>/query.txt.
    Keeps the app runnable from this repo, from a flattened deployment repo, and
    from any working directory."""
    for base in (HERE, *HERE.parents):
        data = base / "data"
        if data.is_dir() and any(d.joinpath("query.txt").exists()
                                 for d in data.iterdir() if d.is_dir()):
            return base
    return None


ROOT = _find_corpus_root()
sys.path[:0] = [str(HERE)] + ([str(ROOT)] if ROOT else [])

import solution                                    # noqa: E402


def _load_incident(name: str) -> tuple[str, dict]:
    """Prefer the repo's own loader; fall back to reading the folder directly so
    the app does not depend on data/loader.py being present."""
    try:
        from data.loader import load_incident
        return load_incident(name)
    except Exception:
        folder = ROOT / "data" / name
        query = (folder / "query.txt").read_text(encoding="utf-8").strip()
        paths = sorted(folder.glob("*.md")) + sorted(folder.glob("*.csv"))
        return query, {p.name: p.read_text(encoding="utf-8") for p in paths}


INCIDENTS = sorted(p.name for p in (ROOT / "data").iterdir()
                   if p.is_dir() and (p / "query.txt").exists()) if ROOT else []

BADGE = {"STRONG": ("#0F2A22", "#00D492", "strong"),
         "WEAK": ("#2A2413", "#D29922", "weak / hedged"),
         "CONTRADICTED": ("#2A1614", "#F85149", "disconfirming"),
         "ABSENT": ("#161B22", "#6E7681", "no signal")}

AXIS_LABEL = {"LOGS": "Application logs", "DEPLOY": "Deployment history",
              "KNOWN_ISSUE": "Known-issues catalog", "PRECEDENT": "Prior incidents",
              "RUNBOOK": "Runbook", "MECHANISM": "Architecture / API (mechanism)"}

st.set_page_config(page_title="Incident Investigator", page_icon="🔎",
                   layout="wide")

st.title("🔎 Production Incident Investigator")
st.caption("Retrieval + cross-document evidence correlation. Confidence comes "
           "from how many independent sources agree — not from how relevant "
           "the top-ranked document felt.")

if not INCIDENTS:
    st.error("No incident corpus found. This app expects a `data/<incident>/` "
             "directory (each holding `query.txt` plus the incident's `.md` and "
             "`.csv` documents) somewhere at or above its own location.")
    st.stop()

with st.sidebar:
    st.header("Incident")
    incident = st.radio("Corpus", INCIDENTS, format_func=lambda n: n.replace("_", " "))
    default_query, corpus = _load_incident(incident)
    query = st.text_area("Symptom (editable)", default_query, height=170)
    st.caption(f"{len(corpus)} documents loaded")
    for name in corpus:
        st.code(name, language=None)

out = solution.explain(query, corpus)
report = out["report"]
score = report["confidence_score"]

left, right = st.columns([2, 1])
with left:
    st.subheader("Probable root cause")
    st.write(report["root_cause"])
with right:
    st.metric("Confidence", f"{score:.1f} / 100")
    st.progress(min(1.0, score / 100))
    if report["needs_human_review"]:
        st.error("**needs_human_review = True** — evidence too thin to act on")
    else:
        st.success("**needs_human_review = False** — corroborated across sources")
    st.metric("MTTR (minutes)",
              report["mttr_minutes"] if report["mttr_minutes"] is not None
              else "unknown")
    if report["mttr_minutes"] is None:
        st.caption("No MTTR reported: every figure in the corpus comes from a "
                   "source that hedges or contradicts this hypothesis.")

st.divider()
st.subheader("Corroboration axes")
st.caption("Each axis is a different document answering the same question "
           "independently. The score is a function of these, and of nothing else.")
cols = st.columns(6)
for col, axis in zip(cols, solution.AXES):
    state = out["axes"][axis]["state"]
    bg, fg, label = BADGE[state]
    col.markdown(
        f"<div style='background:{bg};border:1px solid {fg};border-radius:8px;"
        f"padding:10px 12px;min-height:118px'>"
        f"<div style='font-size:11px;color:#8B949E;text-transform:uppercase;"
        f"letter-spacing:.5px'>{AXIS_LABEL[axis]}</div>"
        f"<div style='color:{fg};font-weight:700;margin:6px 0'>{label}</div>"
        f"<div style='font-size:11px;color:#8B949E'>"
        f"{out['axes'][axis]['detail'] or '&mdash;'}</div></div>",
        unsafe_allow_html=True)

hypo = out["hypothesis"]
st.markdown(
    f"**Leading hypothesis** — `{hypo['component']}` · "
    f"`{hypo['signature']}` · {hypo['count']}× {hypo['level']}"
    + (f" · derived delays up to {max(m for _i, m in hypo['latency'])} min"
       if hypo["latency"] else ""))
if len(out["alternatives"]) > 1:
    alts = " · ".join(f"{a['component']} ({a['score']})"
                      for a in out["alternatives"][1:])
    st.caption(f"Candidates considered and rejected on evidence: {alts}")

st.divider()
tab_ev, tab_sys, tab_rem, tab_ret, tab_json = st.tabs(
    ["Supporting evidence", "Impacted systems", "Remediation",
     "Retrieval ranking", "answers.json"])

with tab_ev:
    st.caption(f"{len(report['supporting_evidence'])} excerpts from "
               f"{len({e['source'] for e in report['supporting_evidence']})} "
               f"distinct documents. Disconfirming evidence is included on "
               f"purpose — it is why the confidence is what it is.")
    for item in report["supporting_evidence"]:
        with st.expander(item["source"], expanded=True):
            st.write(item["excerpt"])

with tab_sys:
    for system in report["impacted_systems"]:
        st.markdown(f"- {system}")

with tab_rem:
    st.write(report["remediation"])

with tab_ret:
    st.caption("TF-IDF cosine over chunked units (log line / CSV row / markdown "
               "section). Note that the top hit is often the architecture "
               "overview — which is exactly why retrieval alone is not the answer.")
    st.dataframe([{"score": r["score"], "source": r["source"],
                   "unit": r["anchor"], "type": r["doctype"],
                   "text": r["text"]} for r in out["ranking"]],
                 hide_index=True)
    st.caption(f"{len(out['units'])} units ingested · components learned from "
               f"the corpus: {', '.join(out['components'])}")

with tab_json:
    st.code(json.dumps(report, indent=2), language="json")
