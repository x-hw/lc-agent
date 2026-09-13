from langchain.messages import AIMessage, AIMessageChunk, HumanMessage, ToolMessage

from src.lc_agent.core.agent import make_agent
from src.lc_agent.core.session import Session, SessionStore
from src.lc_agent.core.settings import app_settings


def create_or_resume_a_session(session_id: str = None) -> Session:
    session_store = SessionStore(app_settings)
    if not session_id:
        session = session_store.create(title="会话", model=app_settings.model)
    else:
        session = session_store.get(session_id)
        if not session:
            raise ValueError(f"Session {session_id} does not exist.")
    return session


def print_a_message(msg):
    if msg.type == "human":
        msg: HumanMessage
        print(f"[human]: {msg.content}")
    if msg.type == "ai":
        msg: AIMessage
        reasoning_content = msg.additional_kwargs.get("reasoning_content", "")
        if reasoning_content:
            print(f"[ai]: thinking: {reasoning_content}")
        content = msg.content
        if content:
            print(f"[ai]: content: {content}")
        tool_calls = msg.tool_calls
        if tool_calls:
            print(f"[ai]: tool_calls: {tool_calls}")
    if msg.type == "tool":
        msg: ToolMessage
        print(f"[tool]: tool name: {msg.name}")
        print(f"[tool]: tool result: {msg.content}")


def print_stream(agent, user_input: str, config: dict) -> None:
    printed_reasoning = False
    printed_content = False
    printed_tool_header = False
    for chunk in agent.stream(
        {"messages": [{"role": "user", "content": user_input}]},
        config=config,
        stream_mode=["messages"],
    ):
        if not isinstance(chunk, tuple) or len(chunk) != 2:
            continue

        _mode, data = chunk

        msg, _metadata = data

        if not isinstance(msg, (AIMessageChunk, ToolMessage)):
            continue

        if isinstance(msg, ToolMessage):
            print(f"\n[tool]: result: {msg.name}: {msg.content}", flush=True)
            continue

        reasoning_content = msg.additional_kwargs.get("reasoning_content", "")
        if reasoning_content:
            if not printed_reasoning:
                if printed_content or printed_tool_header:
                    print()
                print("[ai]: thinking: ", end="", flush=True)
                printed_reasoning = True
            print(reasoning_content, end="", flush=True)

        if msg.content:
            if (printed_reasoning or printed_tool_header) and not printed_content:
                print()
            if not printed_content:
                print("[ai]: content: ", end="", flush=True)
                printed_content = True
            print(msg.content, end="", flush=True)

        if msg.tool_call_chunks:
            for tool_call_chunk in msg.tool_call_chunks:
                if tool_call_chunk.get("name"):
                    if printed_reasoning or printed_content:
                        print()
                    print(
                        f"[ai]: tool_call: {tool_call_chunk.get('name')}",
                        end="",
                        flush=True,
                    )
                    printed_tool_header = True
                if tool_call_chunk.get("args"):
                    print(tool_call_chunk.get("args"), end="", flush=True)
                    printed_tool_header = True

    if printed_reasoning or printed_content:
        print()


def main(session_id: str = None):
    session = create_or_resume_a_session(session_id)
    config = {"configurable": {"thread_id": session.id}}

    with make_agent() as agent:
        existing_messages = agent.get_state(config).values.get("messages", [])
        existing_message_count = len(existing_messages)
        if existing_message_count:
            print("----- history -----")
        for msg in existing_messages:
            print_a_message(msg)
        if existing_message_count:
            print("----- end history -----\n")

        while True:
            try:
                user_input = input("> ")
            except KeyboardInterrupt:
                print(f"\nBye! You can resume session by {session.id}")
                return

            print_stream(agent, user_input, config)
            print("--------------------------\n")


if __name__ == "__main__":
    import sys

    session_id = sys.argv[1] if len(sys.argv) == 2 else None
    main(session_id)
