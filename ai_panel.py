import html
import os
from urllib.parse import quote

import streamlit as st

from ai_client import AiApiError, AiClient

CSS = """
<style>
.block-container {max-width: 860px; padding-top: 2.2rem; padding-bottom: 7rem;}
.ml-brand {font-size: 22px; font-weight: 650; letter-spacing: -0.01em;}
.ml-sub {opacity: .65; font-size: 13px; margin-bottom: .6rem;}
.ml-title {font-size: 30px; font-weight: 650; letter-spacing: -0.015em; line-height: 1.2;}
.ml-lead {opacity: .75; font-size: 15px; margin: .4rem 0 1.2rem 0;}
.ml-chip {display: inline-block; padding: 1px 10px; margin: 0 6px 4px 0; border-radius: 999px;
          font-size: 12px; border: 1px solid rgba(128,128,128,.4); opacity: .9;}
.ml-ok {color: #3fb27f; border-color: #3fb27f;}
.ml-warn {color: #d9a441; border-color: #d9a441;}
.ml-bad {color: #e5686a; border-color: #e5686a;}
.ml-label {font-size: 12px; letter-spacing: .06em; text-transform: uppercase; opacity: .6; margin: .9rem 0 .2rem 0;}
div[data-testid="stMetricValue"] {font-size: 26px;}
</style>
"""

def _avatar(text, fill):
    svg = (f"<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 32 32'><rect width='32' height='32' rx='16' fill='{fill}'/>"
           f"<text x='16' y='20.5' font-size='12' font-weight='600' text-anchor='middle' fill='white' "
           f"font-family='sans-serif'>{text}</text></svg>")
    return "data:image/svg+xml;utf8," + quote(svg, safe="/:=,;'<> ")


AVATARS = {"user": _avatar("You", "#4b5563"), "assistant": _avatar("AI", "#5b8def")}

EXAMPLES = ["Why is my detector weak?", "Propose the next experiment", "How are the classes doing?"]


def chip(text, tone=""):
    return f'<span class="ml-chip ml-{tone}">{html.escape(str(text))}</span>'


def label(text):
    st.markdown(f'<div class="ml-label">{html.escape(text)}</div>', unsafe_allow_html=True)


def num(value):
    return f"{value:.3f}" if isinstance(value, (int, float)) else "n/a"


def rows(items, mapping):
    return [{head: (round(i[key], 3) if isinstance(i.get(key), float) else i.get(key)) for head, key in mapping.items()}
            for i in items]


def finding_block(f):
    tone = {"improved": "ok", "worsened": "bad", "failed": "bad"}.get(f["verdict"], "warn")
    label("Result")
    chips = chip(f["verdict"].replace("_", " "), tone) + chip(f["hypothesis_status"])
    if f.get("basis"):
        chips += chip("vs " + f["basis"])
    st.markdown(chips, unsafe_allow_html=True)
    st.write(f["summary"])
    if f.get("trained"):
        st.table([{"Metric": m, f["basis"].capitalize(): num(f["reference"].get(m)), "Trained": num(f["trained"].get(m)),
                   "Change": num(f["delta"].get(m))} for m in ("f1", "precision", "recall", "mean_iou")])
    for caveat in f["caveats"]:
        st.caption(caveat)


def research_card(client, rec, key):
    res = rec["result"]
    a = res.get("analysis")
    with st.container(border=True):
        bm = res.get("baseline_metrics") or {}
        c1, c2, c3 = st.columns(3)
        c1.metric("Precision", num(bm.get("precision")))
        c2.metric("Recall", num(bm.get("recall")))
        c3.metric("F1", num(bm.get("f1")))
        st.caption(f"{bm.get('num_samples', 'n/a')} evaluation samples  |  failure data: {res.get('failure_source')}")

        if not a:
            st.info(" ".join(res.get("notes", [])) or "Nothing to propose.")
            return

        label("Where it fails")
        if a["weaknesses"]:
            st.table(rows(a["weaknesses"][:4], {"Class": "class_name", "Metric": "metric", "Value": "value",
                                                "Support": "support", "Note": "note"}))
        if a["data_issues"]:
            with st.expander(f"Data notes ({len(a['data_issues'])})"):
                for issue in a["data_issues"]:
                    st.write(issue)

        hyp, prop, val = res["hypothesis"], res["proposal"], res["validation"]
        label("Hypothesis")
        st.markdown(f"**{hyp['claim']}**")
        st.write(hyp["rationale"])
        st.markdown(
            chip("evidence " + hyp["evidence_level"], "ok" if hyp["evidence_level"] == "medium" else "")
            + chip("chosen by language model" if res["llm_used"] else "chosen by rule"),
            unsafe_allow_html=True)
        with st.expander("Why this experiment"):
            st.table([{"Option": r["candidate_id"], "Score": r["score"], "Why": "; ".join(r["reasons"])}
                      for r in res.get("ranking", [])])
            if hyp["evidence"]:
                st.markdown("\n".join(f"- {e}" for e in hyp["evidence"]))
            for notes in res.get("notes", []):
                st.caption(notes)

        label("Proposed experiment")
        st.write(f"{prop['name']}: {prop['expected_effect']}")
        st.table([{"Setting": c["path"], "Value": c["value"]} for c in prop["changes"]])
        st.markdown(chip("validation passed", "ok") if val["valid"] else chip("validation failed", "bad"),
                    unsafe_allow_html=True)
        if not val["valid"]:
            st.error("; ".join(val["errors"]))

        status = rec["status"]
        if status == "proposed" and val["valid"]:
            d = rec.get("defaults") or {}
            with st.expander("Run settings"):
                m1, m2 = st.columns(2)
                model_id = m1.text_input("Model id", d.get("model_id") or "", key=f"m_{key}")
                dataset_id = m2.text_input("Dataset id", d.get("dataset_id") or "", key=f"d_{key}")
                if rec.get("control_available"):
                    st.caption("A default-settings control already exists and will be used for comparison.")
                    control = None
                else:
                    control = st.checkbox("Also run a default-settings control first (recommended, about twice the time)",
                                          value=True, key=f"ct_{key}")
            a1, a2 = st.columns(2)
            if a1.button("Approve and run", key=f"ap_{key}", type="primary", use_container_width=True):
                try:
                    client.approve(rec["id"], model_id or None, dataset_id or None, include_control=control)
                    st.rerun()
                except AiApiError as exc:
                    st.error(str(exc))
            if a2.button("Reject", key=f"rj_{key}", use_container_width=True):
                try:
                    client.reject(rec["id"])
                    st.rerun()
                except AiApiError as exc:
                    st.error(str(exc))
        elif status == "rejected":
            st.caption("Rejected. Ask for another experiment and I will propose a different one.")
        elif status == "approved":
            job = rec.get("job") or {}
            label("Run")
            st.write(f"Job {job.get('job_id')}: {job.get('status')}")
            if not rec.get("finding") and st.button("Refresh status", key=f"rf_{key}"):
                st.rerun()
            if rec.get("finding"):
                finding_block(rec["finding"])

        cache_key = f"report_{rec['id']}"
        if st.session_state.get(cache_key):
            st.download_button("Download report (.md)", st.session_state[cache_key], file_name=f"{rec['id']}.md",
                               mime="text/markdown", key=f"dl_{key}")
        elif st.button("Prepare report", key=f"pr_{key}"):
            try:
                st.session_state[cache_key] = client.report(rec["id"])
                st.rerun()
            except AiApiError as exc:
                st.error(str(exc))


def render_message(client, m):
    with st.chat_message(m["role"], avatar=AVATARS.get(m["role"])):
        if m["text"]:
            st.markdown(m["text"])
        if m.get("warning"):
            st.caption(m["warning"])
        if m["kind"] == "report" and m.get("report"):
            st.download_button("Download report (.md)", m["report"], file_name=f"{m['research_id']}.md",
                               mime="text/markdown", key=f"dl_{m['id']}")
            with st.expander("Preview"):
                st.markdown(m["report"])
        if m["kind"] == "research":
            if m.get("research"):
                research_card(client, m["research"], m["id"])
            else:
                st.caption("This research record is no longer available.")


def send(client, chat_id, text):
    try:
        with st.spinner("Thinking..."):
            client.send(chat_id, text)
    except AiApiError as exc:
        st.session_state["flash"] = str(exc)
    st.rerun()


def sidebar(client, slot):
    with slot:
        st.markdown('<div class="ml-brand">ModelLab AI</div><div class="ml-sub">Experiment advisor</div>',
                    unsafe_allow_html=True)
        try:
            options = client.options()
            chats = client.list_chats()
        except AiApiError as exc:
            st.error(str(exc))
            return
        label("New chat")
        families = options["families"]
        if not families:
            st.info("No detection experiment families were found on the backend.")
        else:
            family = st.selectbox("Experiment family", [f["family_id"] for f in families], key="new_family",
                                  format_func=lambda fid: next(f"{f['family_id']} ({f['num_experiments']} runs)"
                                                               for f in families if f["family_id"] == fid))
            scoped = client.options(family)
            audit = st.selectbox("Dataset audit (optional)", ["None"] + [a["audit_id"] for a in scoped["audits"]])
            analysis = st.selectbox("Failure analysis (optional)",
                                    ["Latest for the baseline"] + [a["analysis_id"] for a in scoped["analyses"]])
            if st.button("Start chat", type="primary", use_container_width=True):
                try:
                    chat = client.create_chat(family, None if audit == "None" else audit,
                                              None if analysis.startswith("Latest") else analysis)
                    st.session_state["chat_id"] = chat["id"]
                    st.rerun()
                except AiApiError as exc:
                    st.error(str(exc))
        if chats:
            label("Chats")
            for c in chats[:10]:
                current = c["id"] == st.session_state.get("chat_id")
                title = c["title"] if len(c["title"]) <= 34 else c["title"][:31] + "..."
                if st.button(title, key=f"open_{c['id']}", use_container_width=True,
                             type="primary" if current else "secondary"):
                    st.session_state["chat_id"] = c["id"]
                    st.rerun()


def hero():
    st.markdown('<div class="ml-title">Why is my detector failing, and what should I try next?</div>'
                '<div class="ml-lead">Start a chat from the sidebar. Pick the experiment family, optionally attach a dataset '
                'audit or a failure analysis, then talk to the advisor. It reads your results, proposes one experiment at a '
                'time, and only runs it when you approve.</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    c1.markdown("**Ask**\n\nWhy a class is weak, what an audit shows, what a result means.")
    c2.markdown("**Propose**\n\nOne validated experiment with the evidence behind it.")
    c3.markdown("**Report**\n\nExport the whole analysis as a markdown file.")


def chat_header(client, chat):
    st.markdown('<div class="ml-title">AI research</div>', unsafe_allow_html=True)
    st.markdown(chip("family " + chat["family_id"]) + chip("audit " + (chat.get("audit_id") or "none"))
                + chip("failure analysis " + (chat.get("analysis_id") or "latest")), unsafe_allow_html=True)
    with st.expander("Change context"):
        scoped = client.options(chat["family_id"])
        audits = ["None"] + [a["audit_id"] for a in scoped["audits"]]
        analyses = ["Latest for the baseline"] + [a["analysis_id"] for a in scoped["analyses"]]
        audit = st.selectbox("Dataset audit", audits, key="ctx_audit",
                             index=audits.index(chat["audit_id"]) if chat.get("audit_id") in audits else 0)
        analysis = st.selectbox("Failure analysis", analyses, key="ctx_analysis",
                                index=analyses.index(chat["analysis_id"]) if chat.get("analysis_id") in analyses else 0)
        if st.button("Apply", key="ctx_apply"):
            try:
                client.set_context(chat["id"], None if audit == "None" else audit,
                                   None if analysis.startswith("Latest") else analysis)
                st.rerun()
            except AiApiError as exc:
                st.error(str(exc))


def main_panel(client):
    chat_id = st.session_state.get("chat_id")
    if not chat_id:
        hero()
        return
    try:
        chat = client.get_chat(chat_id)
    except AiApiError as exc:
        st.session_state.pop("chat_id", None)
        st.error(str(exc))
        return

    chat_header(client, chat)
    flash = st.session_state.pop("flash", None)
    if flash:
        st.error(flash)

    if not chat["messages"]:
        examples = list(EXAMPLES)
        if chat.get("audit_id"):
            examples[2] = "What does the dataset audit show?"
        st.markdown('<div class="ml-lead">Try one of these, or type your own question below.</div>', unsafe_allow_html=True)
        cols = st.columns(len(examples))
        for col, text in zip(cols, examples):
            if col.button(text, key=f"ex_{text}", use_container_width=True):
                send(client, chat["id"], text)
    for m in chat["messages"]:
        render_message(client, m)

    prompt = st.chat_input("Ask about your detector, or request an experiment")
    if prompt:
        send(client, chat["id"], prompt)


def render(base_url, session=None, slot=None):
    client = AiClient(base_url, session=session)
    sidebar(client, slot if slot is not None else st.sidebar)
    main_panel(client)


def main():
    st.set_page_config(page_title="ModelLab AI", layout="centered", initial_sidebar_state="expanded")
    st.markdown(CSS, unsafe_allow_html=True)
    with st.sidebar:
        top = st.container()
        bottom = st.container()
    with bottom:
        with st.expander("Connection"):
            url = st.text_input("Backend URL", os.environ.get("MODELLAB_URL", "http://127.0.0.1:8000"))
    render(url, slot=top)


if __name__ == "__main__":
    main()
