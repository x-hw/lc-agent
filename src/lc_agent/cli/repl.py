import json
import shlex

from langchain.messages import AIMessage, AIMessageChunk, AnyMessage, ToolMessage
from langgraph.errors import GraphRecursionError
from prompt_toolkit import PromptSession
from prompt_toolkit.styles import Style
from rich.console import Console

from lc_agent.core.agent import AgentService, AgentServiceError
from lc_agent.core.session import SessionStore
from lc_agent.core.settings import app_settings

console = Console(markup=False, highlight=False)
_dim_console = Console(markup=False, highlight=False, style="dim")
_styled_console = Console()
_prompt_style = Style.from_dict({"prompt": "ansicyan"})
usage = None


def run_repl(session_id: str) -> None:
    """Run the interactive conversation loop for one session."""
    service = AgentService()
    store = SessionStore(app_settings)
    session = store.get(session_id)
    if session is None:
        raise ValueError(f"Session not found: {session_id}")

    with service.make_agent() as agent:
        history = service.load_history(agent, session_id)
        if history:
            console.rule("history")
            _print_history(history)
            console.print()

        prompt_session: PromptSession[str] = PromptSession(
            style=_prompt_style, history=None
        )
        while True:
            try:
                user_input = prompt_session.prompt("› ")
            except (KeyboardInterrupt, EOFError):
                _styled_console.print(f"Bye! Resume with [cyan]{session_id}[/cyan]")
                return

            text = user_input.strip()
            if not text:
                continue

            command = _parse_slash_command(text)
            if command is not None:
                if _run_slash_command(command, session_id):
                    return
                continue

            try:
                stream = service.stream_turn(agent, session_id, text)
                usage = _print_stream(stream)
            except AgentServiceError as exc:
                _styled_console.print(f"[red]Error:[/red] {exc}")
                continue
            except GraphRecursionError:
                _styled_console.print(
                    "[red]Agent stopped because it reached the recursion limit.[/red]"
                )
                continue
            except Exception as exc:
                _styled_console.print(f"[red]Request failed:[/red] {exc}")
                continue

            if usage is None:
                continue
            store.add_usage(
                session_id,
                input_tokens=usage.input_tokens,
                output_tokens=usage.output_tokens,
                total_tokens=usage.total_tokens,
                last_model=usage.model,
            )
            _print_usage(usage)
            usage = None
            console.print()


def _parse_slash_command(text: str) -> tuple[str, list[str]] | None:
    if not text.startswith("/"):
        return None
    try:
        parts = shlex.split(text)
    except ValueError:
        return ("", [])
    if not parts:
        return None
    return parts[0], parts[1:]


def _run_slash_command(command: tuple[str, list[str]], session_id: str) -> bool:
    name, arguments = command
    if name == "/exit":
        console.print("Bye!")
        return True
    if name == "/tokens":
        session = SessionStore(app_settings).get(session_id)
        if session is None:
            _styled_console.print(f"[red]Session not found:[/red] {session_id}")
            return False
        _print_session_tokens(session)
        return False
    if name == "/new":
        session = SessionStore(app_settings).create(
            title=arguments[0] if arguments else "会话",
            model=app_settings.model,
        )
        _styled_console.print(f"Created session [cyan]{session.id}[/cyan]")
        run_repl(session.id)
        return True
    if name == "/switch":
        if not arguments:
            console.print("Usage: /switch <session-id>")
            return False
        session = SessionStore(app_settings).get(arguments[0])
        if session is None:
            _styled_console.print(f"[red]Session not found:[/red] {arguments[0]}")
            return False
        _styled_console.print(f"Switched to session [cyan]{session.id}[/cyan]")
        run_repl(session.id)
        return True
    console.print("Unknown command. Use /exit, /new, /switch, /tokens.")
    return False


def _print_session_tokens(session) -> None:
    console.print("---- tokens ----")
    console.print(f"input: {session.total_input_tokens}")
    console.print(f"output: {session.total_output_tokens}")
    console.print(f"total: {session.total_tokens}")
    console.print(f"requests: {session.request_count}")
    console.print(f"last model: {session.last_model or session.model}")
    console.print()


def _print_usage(usage) -> None:
    _styled_console.print(
        f"[dim]usage: input={usage.input_tokens} output={usage.output_tokens} "
        f"total={usage.total_tokens} model={usage.model or 'unknown'}[/dim]"
    )


def _print_stream(messages) -> None:
    collected: list[AnyMessage] = []
    last_section = None

    for message in messages:
        collected.append(message)
        if isinstance(message, ToolMessage):
            if last_section is not None:
                _dim_console.print()
            _dim_console.print(f"[tool] {message.name}: {message.text}")
            last_section = "tool"
            continue
        if not isinstance(message, (AIMessageChunk, AIMessage)):
            continue

        reasoning = message.additional_kwargs.get("reasoning_content", "")
        if reasoning:
            if last_section != "reasoning":
                _start_section(
                    last_section is not None, "[ai] thinking: ", last_section
                )
            last_section = "reasoning"
            _dim_console.print(reasoning, end="")

        content = message.text
        if content:
            if last_section != "content":
                _start_section(last_section is not None, "[ai] ", last_section)
            last_section = "content"
            console.print(content, end="")

        tool_call_chunks = getattr(message, "tool_call_chunks", None)
        if tool_call_chunks is None:
            for tool_call in getattr(message, "tool_calls", []):
                arguments = tool_call["args"]
                if isinstance(arguments, dict):
                    arguments = json.dumps(arguments, ensure_ascii=False)
                label = f"[tool-call] {tool_call['name']}: "
                _start_section(last_section is not None, label, last_section)
                _dim_console.print(arguments, end="")
                last_section = "tool-call"
            continue

        for tool_call_chunk in tool_call_chunks:
            name = tool_call_chunk.get("name")
            arguments = tool_call_chunk.get("args")
            if name:
                label = f"[tool-call] {name}: "
                _start_section(last_section is not None, label, last_section)
                last_section = "tool-call"
            if arguments:
                _dim_console.print(arguments, end="")
                last_section = "tool-call"

    if last_section is not None:
        console.print()
    return AgentService.extract_usage(collected)


def _start_section(needs_newline: bool, label: str, last_section: str | None) -> None:
    if needs_newline:
        _dim_console.print()
    if label.startswith("[ai]"):
        console.print(label, end="")
    else:
        _dim_console.print(label, end="")


def _print_history(messages: list[AnyMessage]) -> None:
    for message in messages:
        if isinstance(message, ToolMessage):
            console.print(f"[tool] {message.name}: {message.text}")
        elif message.type == "human":
            console.print(f"[human] {message.text}")
        elif message.type == "ai":
            content = message.text
            if content:
                console.print(f"[ai] {content}")
            for tool_call in message.tool_calls:
                console.print(f"[tool-call] {tool_call['name']}: {tool_call['args']}")
