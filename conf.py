from pydantic_settings import (
    BaseSettings,
    EnvSettingsSource,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
)

class Settings(BaseSettings):
    mle_bench_absolute_path: str = "/default/path"

settings = Settings()