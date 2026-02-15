from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    FRONTEND_ORIGIN: str = "http://localhost:3000"
    STORAGE_DIR: str = "./storage"
    MODAL_ENABLED: bool = True

    DEFAULT_USER_ID: str = "default"

    S3_ENABLED: bool = False
    S3_ENDPOINT: str = ""
    S3_BUCKET: str = "tensorrag"
    S3_ACCESS_KEY: str = ""
    S3_SECRET_KEY: str = ""
    S3_REGION: str = "us-east-1"

    model_config = {"env_file": ".env"}


settings = Settings()
