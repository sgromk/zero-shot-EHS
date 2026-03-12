"""
Vertex AI + GCS client initialization.

Import `vertex_model` and `gcs_bucket` from here instead of
initializing in every module.
"""

import vertexai
from google.cloud import storage
from vertexai.generative_models import GenerativeModel

from config.settings import config

# ── Initialize Vertex AI ───────────────────────────────────────────────────────
vertexai.init(project=config.gcp_project_id, location=config.gcp_location)

# ── Shared model instance ──────────────────────────────────────────────────────
vertex_model = GenerativeModel(config.vertex_model)

# ── Shared GCS client + bucket (lazy — only created when first accessed) ───────
_gcs_client = None
_gcs_bucket = None


def _get_gcs_bucket():
    global _gcs_client, _gcs_bucket
    if _gcs_bucket is None:
        _gcs_client = storage.Client(project=config.gcp_project_id)
        _gcs_bucket = _gcs_client.bucket(config.gcs_bucket_name)
    return _gcs_bucket


# Backward-compatible name used by ingestion.py
class _LazyBucket:
    def __getattr__(self, name):
        return getattr(_get_gcs_bucket(), name)


gcs_bucket = _LazyBucket()


def get_model(model_name: str | None = None) -> GenerativeModel:
    """Return a GenerativeModel instance. Defaults to the configured model."""
    if model_name and model_name != config.vertex_model:
        return GenerativeModel(model_name)
    return vertex_model
