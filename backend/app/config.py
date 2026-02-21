from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    FRONTEND_ORIGIN: str = "http://localhost:3000"
    STORAGE_DIR: str = "./storage"
    MODAL_ENABLED: bool = True

    DEFAULT_USER_ID: str = "default"

    # S3/R2 Storage Configuration (R2 compatible via S3 API)
    S3_ENABLED: bool = False
    S3_ENDPOINT: str = ""  # Use R2 endpoint: https://<account-id>.r2.cloudflarestorage.com
    S3_BUCKET: str = "tensorrag"
    S3_ACCESS_KEY: str = ""
    S3_SECRET_KEY: str = ""
    S3_REGION: str = "us-east-1"  # Use "auto" for R2
    
    # IAM Authentication
    IAM_VALIDATION_ENDPOINT: str = "https://iam.api.poridhi.io/auth/validate-token"

    model_config = {"env_file": ".env"}


settings = Settings()
