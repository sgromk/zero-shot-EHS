"""
Vertex AI + GCS client initialization — fully lazy.

Nothing is initialized at import time. The first call to get_model() or
the first use of vertex_model / gcs_bucket triggers vertexai.init().

Import `vertex_model` and `gcs_bucket` from here instead of
initializing in every module.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from config.settings import config

if TYPE_CHECKING:
    from vertexai.generative_models import GenerativeModel


# ── Lazy Vertex AI init ────────────────────────────────────────────────────────

_vertexai_initialized = False


def _ensure_init() -> None:
    global _vertexai_initialized
    if not _vertexai_initialized:
        import vertexai
        vertexai.init(project=config.gcp_project_id, location=config.gcp_location)
        _vertexai_initialized = True


# ── Lazy default model ─────────────────────────────────────────────────────────
# Many pipeline modules do `from pipeline.client import vertex_model` and then
# use it as a fallback: `_model = model or vertex_model`.
# This proxy defers all initialization until the object is actually used.

class _LazyModel:
    """Proxy for the default GenerativeModel. Initializes on first attribute access."""
    _instance: GenerativeModel | None = None

    def _get(self) -> GenerativeModel:
        if self._instance is None:
            _ensure_init()
            from vertexai.generative_models import GenerativeModel
            self._instance = GenerativeModel(config.vertex_model)
        return self._instance

    def __getattr__(self, name: str):
        return getattr(self._get(), name)

    def __bool__(self) -> bool:
        return True  # so `model or vertex_model` works correctly


vertex_model = _LazyModel()


# ── Shared GCS client + bucket (lazy) ─────────────────────────────────────────

_gcs_client = None
_gcs_bucket_instance = None


def _get_gcs_bucket():
    global _gcs_client, _gcs_bucket_instance
    if _gcs_bucket_instance is None:
        from google.cloud import storage
        _gcs_client = storage.Client(project=config.gcp_project_id)
        _gcs_bucket_instance = _gcs_client.bucket(config.gcs_bucket_name)
    return _gcs_bucket_instance


class _LazyBucket:
    def __getattr__(self, name):
        return getattr(_get_gcs_bucket(), name)


gcs_bucket = _LazyBucket()


# ── Public helper ──────────────────────────────────────────────────────────────

def get_model(model_name: str | None = None) -> GenerativeModel:
    """Return a GenerativeModel instance. Defaults to the configured model."""
    from vertexai.generative_models import GenerativeModel
    _ensure_init()
    if model_name and model_name != config.vertex_model:
        return GenerativeModel(model_name)
    return vertex_model._get()
