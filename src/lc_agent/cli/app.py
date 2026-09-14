from typing import Any

import typer

from lc_agent.cli.repl import run_repl
from lc_agent.core.agent import AgentService
from lc_agent.core.session import SessionStore
from lc_agent.core.settings import app_settings

app = typer.Typer(help="lc-agent: a small local coding assistant")


@app.command()
def new(
    title: str = typer.Option("会话", "--title", help="Session title"),
) -> None:
    """Create a session and start a new conversation."""
    store = SessionStore(app_settings)
    session = store.create(title=title, model=app_settings.model)
    typer.echo(f"Created session {session.id}")
    run_repl(session.id)


@app.command("list")
def list_sessions() -> None:
    """List saved sessions."""
    sessions = SessionStore(app_settings).list()
    if not sessions:
        typer.echo("No sessions found.")
        return
    for session in sessions:
        typer.echo(f"{session.id}  {session.title}  requests={session.request_count}")


@app.command()
def resume(session_id: str) -> None:
    """Resume a saved session."""
    store = SessionStore(app_settings)
    session = store.get(session_id)
    if session is None:
        raise typer.BadParameter(f"Session not found: {session_id}")
    run_repl(session.id)


@app.command()
def delete(session_id: str) -> None:
    """Delete a saved session."""
    deleted = SessionStore(app_settings).delete(session_id)
    if not deleted:
        raise typer.BadParameter(f"Session not found: {session_id}")
    typer.echo(f"Deleted session {session_id}")


@app.command("export-last")
def export_last_turn(session_id: str) -> None:
    """Export the last completed turn as JSON (useful for scripted checks)."""
    import json

    from langchain.messages import HumanMessage

    service = AgentService()
    with service.make_agent() as agent:
        messages = service.load_history(agent, session_id)
    turn_messages: list[dict[str, Any]] = []
    found_last_human = False
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            turn_messages.append({"type": "human", "content": message.text()})
            found_last_human = True
            break
        turn_messages.append({"type": message.type, "content": message.text()})
    if not found_last_human:
        raise typer.BadParameter("Session has no completed turn")
    typer.echo(json.dumps(list(reversed(turn_messages)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    app()
