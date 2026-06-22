"""Storage settings: which backend, plus the S3 connection details.

One active backend, selected by ``STORAGE_BACKEND`` (only ``s3`` today). The S3
fields keep their established env-var names so existing deployments are
unaffected. All optional: blank credentials/region/endpoint fall back to boto3's
default chain (IAM role / ~/.aws / real env). UPPERCASE field = env var.
"""
from pydantic import Field
from pydantic_settings import BaseSettings


class StorageConfig(BaseSettings):
    """Active storage backend + S3 connection. Read via the ``app_config`` singleton."""

    STORAGE_BACKEND: str = Field(default="s3", description="Active storage backend id (only 's3' is registered today).")

    BUCKET_NAME: str = Field(default="bedrock-kb-data", description="S3 bucket name (provisioned manually).")
    AWS_ACCESS_KEY_ID: str = Field(default="", description="S3 access key (blank => boto3 default credential chain).")
    AWS_SECRET_ACCESS_KEY: str = Field(default="", description="S3 secret key (blank => boto3 default credential chain).")
    AWS_REGION: str = Field(default="", description="S3 region, e.g. ap-southeast-1 (blank => boto3 default).")
    AWS_S3_ENDPOINT_URL: str = Field(default="", description="Custom S3 endpoint for MinIO/non-AWS (blank => real AWS S3).")
