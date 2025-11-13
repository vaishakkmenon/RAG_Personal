"""Certification registry implementation.

This module is responsible for loading certification metadata from the
``data/certifications`` directory and providing lookup helpers that the
rest of the application (prompt generation, validation, discovery
scripts) can leverage.
"""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set
import logging
import re

import yaml

from .models import Certification

LOGGER = logging.getLogger(__name__)


class CertificationRegistry:
    """In-memory registry backed by YAML certification definitions."""

    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)
        self.certifications: Dict[str, Certification] = {}
        self._issuers: Set[str] = set()
        self._domains: Set[str] = set()

        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.reload()

    # ------------------------------------------------------------------
    # Loading & persistence
    # ------------------------------------------------------------------
    def reload(self) -> None:
        """Reload certification definitions from disk."""

        self.certifications.clear()
        self._issuers.clear()
        self._domains.clear()

        for yaml_file in self._definition_files():
            try:
                with open(yaml_file, "r", encoding="utf-8") as handle:
                    data = yaml.safe_load(handle) or {}
            except (OSError, yaml.YAMLError) as exc:
                LOGGER.warning("Failed to load certification file %s: %s", yaml_file, exc)
                continue

            for cert_id, cert_payload in (data.get("certifications") or {}).items():
                try:
                    certification = Certification.from_dict(cert_id, cert_payload)
                except Exception as exc:  # noqa: BLE001 - surface bad entries gracefully
                    LOGGER.warning(
                        "Skipping invalid certification %s in %s: %s", cert_id, yaml_file, exc
                    )
                    continue
                self._register(certification, persist=False)

    def save(self, certification: Certification) -> None:
        """Persist a certification definition back to disk."""

        issuer_dir = self.data_dir / _issuer_to_dirname(certification.issuer)
        issuer_dir.mkdir(parents=True, exist_ok=True)

        cert_file = issuer_dir / f"{certification.id}.yaml"
        payload = {"certifications": {certification.id: certification.to_dict()}}
        with open(cert_file, "w", encoding="utf-8") as handle:
            yaml.dump(payload, handle, default_flow_style=False, sort_keys=False, allow_unicode=True)

    # ------------------------------------------------------------------
    # Mutation helpers
    # ------------------------------------------------------------------
    def add(self, certification: Certification, persist: bool = True) -> None:
        """Add a certification to the registry."""

        self._register(certification, persist=persist)

    def update(self, cert_id: str, **fields) -> Optional[Certification]:
        """Update an existing certification, returning the updated object."""

        certification = self.certifications.get(cert_id)
        if not certification:
            return None

        cert_dict = certification.to_dict()
        cert_dict.update(fields)
        updated = Certification.from_dict(cert_id, cert_dict)
        self._register(updated, persist=True)
        return updated

    def remove(self, cert_id: str) -> bool:
        """Remove a certification from the registry and disk."""

        certification = self.certifications.pop(cert_id, None)
        if not certification:
            return False

        issuer_dir = self.data_dir / _issuer_to_dirname(certification.issuer)
        cert_file = issuer_dir / f"{cert_id}.yaml"
        if cert_file.exists():
            try:
                cert_file.unlink()
            except OSError as exc:
                LOGGER.warning("Failed to delete certification file %s: %s", cert_file, exc)

        # rebuild indexes to ensure consistency
        self.reload()
        return True

    # ------------------------------------------------------------------
    # Lookup helpers
    # ------------------------------------------------------------------
    def find(self, text: str) -> Optional[Certification]:
        """Find a certification by id, name, or alias."""

        query = text.strip().lower()
        if not query:
            return None

        # direct id match
        if query in self.certifications:
            return self.certifications[query]

        for certification in self.certifications.values():
            if certification.matches(query):
                return certification
        return None

    def find_by_issuer(self, issuer: str) -> List[Certification]:
        """Return all certifications issued by the given organization."""

        target = issuer.strip().lower()
        return [
            cert
            for cert in self.certifications.values()
            if cert.issuer.lower() == target
        ]

    def find_by_domain(self, domain: str) -> List[Certification]:
        """Return all certifications that belong to a given domain."""

        target = domain.strip().lower()
        return [
            cert
            for cert in self.certifications.values()
            if any(d.lower() == target for d in cert.domains)
        ]

    @property
    def issuers(self) -> List[str]:
        return sorted(self._issuers)

    @property
    def domains(self) -> List[str]:
        return sorted(self._domains)

    def __contains__(self, cert_id: str) -> bool:  # pragma: no cover - convenience helper
        return cert_id in self.certifications

    def __len__(self) -> int:  # pragma: no cover - convenience helper
        return len(self.certifications)

    def __iter__(self):  # pragma: no cover - convenience helper
        return iter(self.certifications.values())

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _register(self, certification: Certification, persist: bool) -> None:
        self.certifications[certification.id.lower()] = certification
        self._issuers.add(certification.issuer)
        self._domains.update(certification.domains)

        if persist:
            self.save(certification)

    def _definition_files(self) -> Iterable[Path]:
        return self.data_dir.glob("**/*.yaml")


def _issuer_to_dirname(issuer: str) -> str:
    """Convert an issuer name into a stable directory name."""

    return re.sub(r"[^a-z0-9]+", "_", issuer.lower()).strip("_") or "unknown"
