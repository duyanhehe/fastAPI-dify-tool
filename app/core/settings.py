import enum
from pathlib import Path
from tempfile import gettempdir

from pydantic_settings import BaseSettings, SettingsConfigDict
from yarl import URL

TEMP_DIR = Path(gettempdir())


class LogLevel(str, enum.Enum):
    """Possible log levels."""

    NOTSET = "NOTSET"
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    FATAL = "FATAL"


class Settings(BaseSettings):
    """
    Application settings.

    These parameters can be configured
    with environment variables.
    """

    title: str = "Fast API App"
    version: str = "0.0.1"
    description: str = ""
    domain: str = ""
    host: str = "127.0.0.1"
    port: int = 8000
    # quantity of workers for uvicorn
    workers_count: int = 1
    # Enable uvicorn reloading
    reload: bool = False

    # Current environment
    environment: str = "dev"

    log_level: LogLevel = LogLevel.INFO
    # Variables for the database
    db_host: str = "localhost"
    db_host_name: str = "localhost"
    db_port: int = 5432
    db_user: str = "app"
    db_pass: str = "app"
    db_base: str = "admin"
    db_echo: bool = False

    # Path to the directory with media
    media_dir: str = "media"
    gemini_key: str = ""

    @property
    def db_url(self) -> URL:
        """
        Assemble database URL from settings.

        :return: database URL.
        """
        return URL.build(
            scheme="postgresql+asyncpg",
            host=self.db_host,
            port=self.db_port,
            user=self.db_user,
            password=self.db_pass,
            path=f"/{self.db_base}",
        )

    @property
    def media_dir_static(self) -> Path:
        """
        Get path to the directory with media files.

        :return: path to the directory.
        """
        static_dir = Path(self.media_dir)
        # create directory if not exists
        static_dir.mkdir(parents=True, exist_ok=True)
        return static_dir

    @property
    def media_base_url(self) -> str:
        """
        Get base URL for media files.

        :return: base URL.
        """
        if self.domain:
            return f"{self.domain}/static/media"
        return f"http://{self.host}:{self.port}/static/media"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="APP_",
        env_file_encoding="utf-8",
    )


settings = Settings()
