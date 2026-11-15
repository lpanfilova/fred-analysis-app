"""One web turn with serializable conversation state and fresh async clients."""

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
import re

from openai import AsyncOpenAI

from chat import ChatError, EconomicChat, load_chat_settings
from chat_tools import EconomicTools
from fred_connection import fred_session


OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"


@dataclass
class WebTurn:
    answer: str
    history: list
    tool_results: list


def _display_error(error):
    if isinstance(error, ChatError):
        return str(error)
    if isinstance(error, BaseExceptionGroup):
        for child in error.exceptions:
            if message := _display_error(child):
                return message
    return None


async def answer_question(question, history, on_authorize=None):
    """Reuse history, never async transports, across Streamlit reruns.

    Each MCP task group is entered and exited within this same coroutine.
    A fresh OpenAI client avoids using a connection pool on a closed event loop.
    """
    key, model = load_chat_settings()
    try:
        async with AsyncOpenAI(api_key=key, timeout=60, max_retries=1) as api:
            async with fred_session(on_authorize=on_authorize) as fred:
                chat = EconomicChat(api, EconomicTools(fred), model=model)
                chat.history = deepcopy(history)
                answer = await chat.ask(question)
                return WebTurn(answer, chat.history, chat.last_tool_results)
    except Exception as exc:
        message = _display_error(exc) or (
            "Could not complete the request. Check the connection, allow macOS Keychain access, "
            "and complete FRED sign-in if prompted. Then try again."
        )
        raise ChatError(message) from None


def chart_files(tool_results):
    """Only serve chart files created inside this application's output directory."""
    files = []
    for entry in tool_results:
        value = entry.get("result", {}).get("chart_path")
        if not value:
            continue
        path = Path(value).resolve()
        if path.is_relative_to(OUTPUT_DIR.resolve()) and path.suffix.lower() in {".png", ".svg", ".pdf"} and path.is_file():
            if path not in files:
                files.append(path)
    return files


def display_answer(answer, tool_results):
    """Replace filesystem links with the inline chart and download controls."""
    for entry in tool_results:
        path = entry.get("result", {}).get("chart_path")
        if path:
            answer = re.sub(r"\[[^\]]*\]\(" + re.escape(path) + r"\)", "**Chart shown below.**", answer)
            answer = answer.replace(path, "the chart below")
    return answer


def data_notes(tool_results):
    sources, warnings, periods = [], [], []
    for entry in tool_results:
        report = entry.get("result", {})
        for source in [report.get("source_url"), *report.get("sources", [])]:
            if source and source not in sources:
                sources.append(source)
        for warning in report.get("warnings", []):
            if warning not in warnings:
                warnings.append(warning)
        latest = report.get("latest_available_yoy") or report.get("latest") or report.get("latest_common")
        if latest:
            period = latest.get("observation_period") or latest.get("quarter")
            if period and period not in periods:
                periods.append(period)
    return {"sources": sources, "warnings": warnings, "periods": periods}
