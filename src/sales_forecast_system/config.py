from pydantic import BaseModel, Field


class DatabaseConfig(BaseModel):
    driver: str = Field(default="sqlite")
    dsn: str = Field(default="sqlite:///sales.db")
    pool_size: int = Field(default=5)


class ForecastConfig(BaseModel):
    horizon_days: int = Field(default=30)
    seasonality_period: int = Field(default=7)
    moving_average_window: int = Field(default=14)
    enable_ensemble: bool = Field(default=True)


class AppConfig(BaseModel):
    env: str = Field(default="dev")
    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8080)
    debug: bool = Field(default=True)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    forecast: ForecastConfig = Field(default_factory=ForecastConfig)


def load_config() -> AppConfig:
    return AppConfig()
