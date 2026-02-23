from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import SecretStr
from typing import Optional

class Settings(BaseSettings):

    # MCP servers
    mcp_travel_port: int = 8001
    mcp_productivity_port: int = 8002
    mcp_comms_port: int = 8003

    # External APIs
    github_token: Optional[SecretStr] = None
    notion_token: Optional[SecretStr] = None
    amadeus_client_id: Optional[SecretStr] = None
    amadeus_client_secret: Optional[SecretStr] = None

    # Goggle Calender
    google_credentials_path: Optional[str] = None

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False
    )

settings = Settings()
