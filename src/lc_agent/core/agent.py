from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass

from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware, ModelRequest
from langchain.messages import AnyMessage, trim_messages
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_openrouter import ChatOpenRouter
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph.state import CompiledStateGraph

from lc_agent.core.settings import AppSettings, app_settings
from lc_agent.tools.file import make_list_directory_tool, make_read_text_file_tool
from lc_agent.tools.search import make_search_tool


@dataclass(frozen=True)
class TurnUsage:
    input_tokens: int
    output_tokens: int
    total_tokens: int
    model: str | None


class AgentServiceError(RuntimeError):
    pass


class TrimHistoryMiddleware(AgentMiddleware):
    """Trim the message history before each model call."""

    def __init__(self, max_tokens: int = app_settings.model_max_tokens):
        self.max_tokens = max_tokens

    def wrap_model_call(self, request: ModelRequest, handler):
        trimmed = trim_messages(
            request.messages,
            max_tokens=self.max_tokens,
            token_counter="approximate",
            strategy="last",
            start_on="human",
        )
        return handler(request.override(messages=trimmed))


class AgentService:
    def __init__(self, settings: AppSettings = app_settings):
        self._settings = settings

    @contextmanager
    def make_agent(self, model: BaseChatModel | None = None):
        chat_model = model or ChatOpenRouter(
            model=self._settings.model,
            api_key=self._settings.model_api_key,
        )
        tools = [
            make_list_directory_tool(self._settings.workspace_root),
            make_read_text_file_tool(self._settings.workspace_root),
        ]
        search_tool = make_search_tool(self._settings.tavily_api_key)
        if search_tool is not None:
            tools.append(search_tool)
        if model is not None and tools:
            try:
                chat_model = model.bind_tools(tools)
            except NotImplementedError:
                pass
        with SqliteSaver.from_conn_string(self._settings.db_path) as checkpointer:
            checkpointer.setup()
            yield create_agent(
                model=chat_model,
                system_prompt=self._settings.system_prompt_file.read_text(),
                tools=tools,
                checkpointer=checkpointer,
                middleware=[TrimHistoryMiddleware()],
            )

    @staticmethod
    def load_history(agent: CompiledStateGraph, session_id: str) -> list[AnyMessage]:
        state = agent.get_state({"configurable": {"thread_id": session_id}})
        return list(state.values.get("messages", []))

    @staticmethod
    def stream_turn(
        agent: CompiledStateGraph,
        session_id: str,
        user_input: str,
    ) -> Iterator[AnyMessage]:
        config = {"configurable": {"thread_id": session_id}}
        events = agent.stream(
            {"messages": [{"role": "user", "content": user_input}]},
            config=config,
            stream_mode="messages",
        )
        for event in events:
            if isinstance(event, tuple) and len(event) == 2:
                message = event[0]
            else:
                message = event
            if isinstance(message, BaseMessage):
                yield message

    @staticmethod
    def extract_usage(messages: Sequence[BaseMessage]) -> TurnUsage:
        input_tokens = 0
        output_tokens = 0
        total_tokens = 0
        model = None
        for message in reversed(messages):
            usage = getattr(message, "usage_metadata", None)
            if usage:
                input_tokens = usage.get("input_tokens", input_tokens)
                output_tokens = usage.get("output_tokens", output_tokens)
                total_tokens = usage.get("total_tokens", total_tokens)
            metadata = getattr(message, "response_metadata", {})
            response_model = metadata.get("model") or metadata.get("model_name")
            if response_model:
                model = response_model
        return TurnUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            model=model,
        )
