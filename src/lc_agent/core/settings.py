from pathlib import Path

from pydantic import DirectoryPath, FilePath
from pydantic_settings import BaseSettings, SettingsConfigDict


class AppSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="LC_AGENT_",
        env_file=".env",
        env_file_encoding="utf-8",
    )

    model: str
    model_api_key: str
    model_max_tokens: int
    tavily_api_key: str
    system_prompt_file: FilePath = FilePath(
        Path(__file__).with_name("system_prompt.md")
    )
    db_path: Path
    workspace_root: DirectoryPath


app_settings = AppSettings()  # type: ignore[call-arg]
