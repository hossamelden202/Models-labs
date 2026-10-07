import os

import streamlit as st

from ai_client import AiApiError, AiClient

DEFAULT_FAMILY = "weapons-final-model"


def _rows(items, keys):
    return [{k: (round(i[k], 3) if isinstance(i.get(k), float) else i.get(k)) for k in keys} for i in items]


def show_record(client, rec):
    res = rec["result"]
    analysis = res.get("analysis")
    st.caption(f"Research {rec['id']}  |  family {rec['family_id']}  |  status {rec['status']}")

    for note in res.get("notes", []):
        st.info(note)
    if not analysis:
        st.warning("Nothing to propose for this family.")
        return

    st.subheader("Analysis")
    st.write(analysis["summary"])
    if analysis["weaknesses"]:
        st.table(_rows(analysis["weaknesses"], ["class_name", "metric", "value", "support", "note"]))
    for issue in analysis["data_issues"]:
        st.warning(issue)
    if res["retrieved"]:
        st.write("Retrieved past experiments")
        st.table(_rows(res["retrieved"], ["title", "score"]))

    hyp, prop, val = res["hypothesis"], res["proposal"], res["validation"]
    st.subheader("Hypothesis")
    st.write(hyp["claim"])
    st.caption(
        ("Chosen by the language model" if res["llm_used"] else "Chosen by rule, the language model was unavailable")
        + f"  |  evidence {hyp['evidence_level']}  |  target {hyp['target_metric']}"
    )
    st.write(hyp["rationale"])
    if hyp["evidence"]:
        st.markdown("\n".join(f"- {e}" for e in hyp["evidence"]))
    with st.expander("Why this experiment"):
        st.table([
            {"candidate": r["candidate_id"], "score": r["score"], "why": "; ".join(r["reasons"])}
            for r in res.get("ranking", [])
        ])

    st.subheader("Proposed experiment")
    st.write(f"{prop['name']}: {prop['expected_effect']}")
    st.table(_rows(prop["changes"], ["path", "value"]))
    if val["valid"]:
        st.success(f"Validation passed ({val['experiment_id']})")
    else:
        st.error("Validation failed: " + "; ".join(val["errors"]))

    if rec["status"] == "proposed" and val["valid"]:
        d = rec.get("defaults") or {}
        c1, c2 = st.columns(2)
        model_id = c1.text_input("Model id", d.get("model_id") or "", key=f"m_{rec['id']}")
        dataset_id = c2.text_input("Dataset id", d.get("dataset_id") or "", key=f"d_{rec['id']}")
        if rec.get("control_available"):
            st.caption("A default-settings control already exists for this family and will be used for comparison.")
            control = None
        else:
            control = st.checkbox(
                "Also run a default-settings control first (recommended, roughly doubles the time)",
                value=True, key=f"ct_{rec['id']}",
            )
        a, b = st.columns(2)
        if a.button("Approve and run", key=f"ap_{rec['id']}"):
            try:
                out = client.approve(rec["id"], model_id or None, dataset_id or None, include_control=control)
                st.success(f"Started job {out['job'].get('job_id')}")
                st.rerun()
            except AiApiError as exc:
                st.error(str(exc))
        if b.button("Reject", key=f"rj_{rec['id']}"):
            try:
                client.reject(rec["id"])
                st.rerun()
            except AiApiError as exc:
                st.error(str(exc))

    if rec.get("approval"):
        job = rec.get("job") or {}
        st.write(f"Job {job.get('job_id')}: {job.get('status')}")
        if st.button("Refresh status", key=f"rf_{rec['id']}"):
            st.rerun()
    if rec.get("finding"):
        f = rec["finding"]
        st.subheader("Result")
        st.write(f"{f['verdict']}: {f['hypothesis_status']}")
        st.write(f["summary"])
        if f.get("basis"):
            st.caption(f"Compared against: {f['basis']}")
        for caveat in f["caveats"]:
            st.caption(caveat)


def render(base_url, session=None):
    client = AiClient(base_url, session=session)
    st.header("AI research")

    family = st.text_input("Family id", DEFAULT_FAMILY)
    question = st.text_area("Question", "Why is my weapon detector weak?", height=70)

    if st.button("Run research"):
        try:
            with st.spinner("Analysing, this can take a minute on CPU"):
                st.session_state["ai_rid"] = client.research(family, question)["id"]
        except AiApiError as exc:
            st.error(str(exc))

    try:
        history = client.list_research()
    except AiApiError as exc:
        st.error(str(exc))
        return
    ids = [h["id"] for h in history]
    current = st.session_state.get("ai_rid")
    if ids:
        index = ids.index(current) if current in ids else 0
        label = {h["id"]: f"{h['id']}  ({h['status']})" for h in history}
        chosen = st.selectbox("Past research", ids, index=index, format_func=label.get)
        st.session_state["ai_rid"] = chosen
        try:
            show_record(client, client.get(chosen))
        except AiApiError as exc:
            st.error(str(exc))


def main():
    st.set_page_config(page_title="ModelLab AI research", layout="wide")
    url = st.sidebar.text_input("Backend URL", os.environ.get("MODELLAB_URL", "http://127.0.0.1:8000"))
    render(url)


if __name__ == "__main__":
    main()
