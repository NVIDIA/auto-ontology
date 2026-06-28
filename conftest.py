"""Root conftest: set required env vars before any test module is imported."""

import os

os.environ.setdefault("NVIDIA_API_KEY", "test-dummy-key")
os.environ.setdefault("EMBED_API_KEY", "test-dummy-key")
