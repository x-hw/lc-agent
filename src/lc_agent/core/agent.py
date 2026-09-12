from langchain.agents import create_agent
from langchain_openrouter import ChatOpenRouter

from lc_agent.core.settings import app_settings
from lc_agent.tools.file import make_list_directory_tool, make_read_text_file_tool


def make_agent():
    model = ChatOpenRouter(
        model=app_settings.model,
        api_key=app_settings.model_api_key,
    )
    agent = create_agent(
        model=model,
        system_prompt="You are a helpful assistant",
        tools=[
            make_list_directory_tool(app_settings.workspace_root),
            make_read_text_file_tool(app_settings.workspace_root),
        ],
    )
    return agent
