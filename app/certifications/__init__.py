"""Certification package public API."""

from pathlib import Path
from typing import Optional

from .models import Certification
from .formatter import CertificationFormatter
from .registry import CertificationRegistry

_DEFAULT_REGISTRY: Optional[CertificationRegistry] = None


def get_registry(data_dir: Optional[Path] = None) -> CertificationRegistry:
    """Return a shared certification registry instance.

    Args:
        data_dir: Optional override for the certification definitions directory.
            Defaults to ``<project_root>/data/certifications``.
    """

    global _DEFAULT_REGISTRY
    if _DEFAULT_REGISTRY is None or data_dir is not None:
        root = data_dir or Path(__file__).resolve().parents[2] / "data" / "certifications"
        _DEFAULT_REGISTRY = CertificationRegistry(root)
    return _DEFAULT_REGISTRY


__all__ = [
    "Certification",
    "CertificationRegistry",
    "CertificationFormatter",
    "get_registry",
]
