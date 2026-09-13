from contextlib import contextmanager

from langchain.agents import create_agent
from langchain_openrouter import ChatOpenRouter
from langgraph.checkpoint.sqlite import SqliteSaver

from src.lc_agent.core.settings import app_settings
from src.lc_agent.tools.file import make_list_directory_tool, make_read_text_file_tool


@contextmanager
def make_agent():
    model = ChatOpenRouter(
        model=app_settings.model,
        api_key=app_settings.model_api_key,
    )
    system_prompt = app_settings.system_prompt_file.read_text()
    with SqliteSaver.from_conn_string(app_settings.db_path) as checkpointer:
        checkpointer.setup()
        agent = create_agent(
            model=model,
            system_prompt=system_prompt,
            tools=[
                make_list_directory_tool(app_settings.workspace_root),
                make_read_text_file_tool(app_settings.workspace_root),
            ],
            checkpointer=checkpointer,
        )
        yield agent
