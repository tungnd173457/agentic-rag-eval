"""Google Drive connector settings: enable flag + service-account credentials.

Folded into ``AppConfig``; read via ``app_config``. The fields map 1:1 to the
service-account JSON. The scanner is OFF unless ``GDRIVE_ENABLED`` is true.
A future SharePoint connector gets its own ``SharePointConfig`` — not this class.
"""
from pydantic import Field
from pydantic_settings import BaseSettings


class GDriveConfig(BaseSettings):
    GDRIVE_ENABLED: bool = Field(default=False, description="Enable the Google Drive folder scanner.")
    GDRIVE_FOLDER_ID: str = Field(default="", description="Drive folder id to watch (the <ID> in .../folders/<ID>).")
    GDRIVE_PROJECT_ID: str = Field(default="", description="Service account: project_id.")
    GDRIVE_CLIENT_EMAIL: str = Field(default="", description="Service account: client_email — the watched folder MUST be shared with this address (Viewer).")
    GDRIVE_PRIVATE_KEY_ID: str = Field(default="", description="Service account: private_key_id.")
    GDRIVE_PRIVATE_KEY: str = Field(default="", description="Service account: private_key (one-line with literal \\n escapes; the connector restores newlines).")
    GDRIVE_TOKEN_URI: str = Field(default="https://oauth2.googleapis.com/token", description="Service account: token_uri.")
