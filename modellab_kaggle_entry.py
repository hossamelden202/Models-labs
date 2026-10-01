
from pathlib import Path

from modellab.server.app import create_app
from modellab.server.settings import ServerSettings

settings = ServerSettings(
    workspace=Path('/kaggle/working/ml_workspace'),
    token=None,
    max_workers=1,
    allow_local_paths=True,
    cors_origins=("*",),
)

app = create_app(settings)
