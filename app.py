"""Local web chat. Launch with: .venv/bin/python -m streamlit run app.py"""

import asyncio
import platform

import streamlit as st

from chat import ChatError, load_chat_settings
from web_backend import answer_question, chart_files, data_notes, display_answer


STARTERS = [
    ("Latest GDP", "What is the latest reported U.S. GDP?"),
    ("Inflation trends", "How has Core CPI inflation changed over the past five years?"),
    ("Compare & chart", "Compare GDP growth and Core CPI inflation over the past five years and generate a chart."),
]


def render_turn(turn, index):
    with st.chat_message("user"):
        st.markdown(turn["question"])
    with st.chat_message("assistant"):
        if turn.get("error") or "answer" not in turn:
            st.error(turn.get("error") or "The request was interrupted. Retry to continue.")
            return st.button("Retry request", key=f"retry_{index}")
        results = turn.get("tool_results", [])
        st.markdown(display_answer(turn["answer"], results))
        for chart_index, path in enumerate(chart_files(results)):
            if path.suffix.lower() != ".pdf":
                st.image(str(path), caption="Nominal GDP growth and Core CPI inflation", width="stretch")
            mime = {".png": "image/png", ".svg": "image/svg+xml", ".pdf": "application/pdf"}[path.suffix.lower()]
            st.download_button(
                "Download chart", data=path.read_bytes(), file_name=path.name,
                mime=mime, key=f"download_{index}_{chart_index}", on_click="ignore",
            )
        notes = data_notes(results)
        if any(notes.values()):
            with st.expander("Sources & data notes"):
                if notes["periods"]:
                    st.caption("Latest available periods: " + ", ".join(notes["periods"]))
                for source in notes["sources"]:
                    label = "GDP · Bureau of Economic Analysis" if source.endswith("/GDP") else "Core CPI · Bureau of Labor Statistics"
                    st.markdown(f"[{label}]({source})")
                for warning in notes["warnings"]:
                    st.caption(warning)
                st.caption("GDP is nominal, in current dollars. Core CPI excludes food and energy. Correlation does not establish causation.")
    return False


def main():
    st.set_page_config(page_title="Economic Explorer · FRED", page_icon="◈", layout="wide")
    st.markdown("""<style>
        .stMainBlockContainer {max-width: 1080px; padding-top: 3rem;}
        [data-testid="stSidebar"] {border-right: 1px solid #e1e7ed;}
        [data-testid="stChatMessage"] {border: 1px solid #e1e7ed; border-radius: 14px;}
        h1 {letter-spacing: -0.035em;}
    </style>""", unsafe_allow_html=True)
    st.session_state.setdefault("turns", [])
    st.session_state.setdefault("conversation", [])

    setup_error = None
    if platform.system() != "Darwin":
        setup_error = "This local version requires macOS for Keychain sign-in."
    else:
        try:
            load_chat_settings()
        except ChatError as exc:
            setup_error = str(exc)

    with st.sidebar:
        st.markdown("### ◈ FRED ANALYSIS")
        st.caption("A conversation with economic data")
        if st.button("New conversation", width="stretch", type="primary"):
            st.session_state.turns = []
            st.session_state.conversation = []
            st.rerun()
        st.divider()
        st.markdown("**Two datasets. More perspective.**")
        st.markdown("**U.S. GDP**  \nQuarterly · nominal, current dollars")
        st.markdown("**Core CPI**  \nMonthly · excludes food and energy")
        st.caption("Both series are seasonally adjusted. Data may be revised after release.")
        st.divider()
        st.markdown("**Try a follow-up**")
        st.caption('“Now just the past five years.”  \n“Make a chart of that comparison.”')
        st.caption("Conversation stays in this browser session. New conversation clears its context.")

    st.caption("U.S. ECONOMIC DATA / CONVERSATIONAL ANALYSIS")
    st.title("Economic Explorer")
    st.markdown("Ask about GDP and inflation. Explore a period, compare trends, or turn the data into a chart.")
    if setup_error:
        st.warning(setup_error)
        if platform.system() == "Darwin":
            with st.container(border=True):
                st.markdown("**Configure the local app**")
                st.code("cp .env.example .env", language="bash")
                st.markdown("Add an OpenAI API key to `OPEN_AI_API_KEY` in `.env`, then restart the app. FRED sign-in opens when data is first requested.")
                st.caption("Keep .env private. It is excluded from Git.")
        st.chat_input("Complete setup to start a conversation", disabled=True)
        return

    question = None
    retry_index = None
    if not st.session_state.turns:
        st.caption("START WITH A QUESTION")
        for column, (label, prompt) in zip(st.columns(3), STARTERS):
            if column.button(label, width="stretch"):
                question = prompt
        st.info("On first use, FRED may open a sign-in tab and macOS may request Keychain access. Return here after authorization.")

    for index, turn in enumerate(st.session_state.turns):
        if render_turn(turn, index):
            retry_index = index
            question = turn["question"]
    typed = st.chat_input("Ask about GDP, Core CPI, or a period you want to explore…")
    question = typed or question
    if not question:
        return

    if retry_index is None:
        retry_index = len(st.session_state.turns)
        st.session_state.turns.append({"question": question})
        with st.chat_message("user"):
            st.markdown(question)

    with st.chat_message("assistant"):
        authorization = st.empty()

        def on_authorize(url):
            with authorization.container():
                st.info("Complete FRED sign-in in the browser tab, then return here. This request will continue automatically.")
                st.link_button("Open FRED sign-in", url)

        with st.spinner("Checking the data and preparing your answer…"):
            try:
                result = asyncio.run(answer_question(question, st.session_state.conversation, on_authorize))
            except ChatError as exc:
                st.session_state.turns[retry_index] = {"question": question, "error": str(exc)}
            else:
                st.session_state.conversation = result.history
                st.session_state.turns[retry_index] = {
                    "question": question, "answer": result.answer, "tool_results": result.tool_results,
                }
        authorization.empty()
    st.rerun()


if __name__ == "__main__":
    main()
