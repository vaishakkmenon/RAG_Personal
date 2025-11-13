"""Domain models for professional certifications.

This module defines data structures used by the certification registry
and discovery tooling. The goal is to provide a canonical representation
of certification metadata that can be easily serialized to and from
YAML definitions stored under ``data/certifications``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any, Dict, Iterable, List, Optional


@dataclass(slots=True)
class Certification:
    """Represents a professional certification entry."""

    id: str
    official_name: str
    issuer: str
    domains: List[str] = field(default_factory=list)
    aliases: List[str] = field(default_factory=list)
    technologies: List[str] = field(default_factory=list)
    categories: List[str] = field(default_factory=list)
    related_skills: List[str] = field(default_factory=list)
    description: str = ""
    url: str = ""
    earned_date: Optional[date] = None
    expiry_date: Optional[date] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Normalize identifiers and ensure uniqueness for list fields
        self.id = self.id.strip()
        self.official_name = self.official_name.strip()
        self.issuer = self.issuer.strip()
        self.domains = _dedupe_preserve(self.domains)
        self.aliases = _dedupe_preserve(self.aliases)
        self.technologies = _dedupe_preserve(self.technologies)
        self.categories = _dedupe_preserve(self.categories)
        self.related_skills = _dedupe_preserve(self.related_skills)

    @classmethod
    def from_dict(cls, cert_id: str, data: Dict[str, Any]) -> "Certification":
        """Create a certification from a dictionary payload."""

        payload = {**data, "id": cert_id}
        # Convert string dates when present
        earned = payload.get("earned_date")
        if isinstance(earned, str):
            payload["earned_date"] = date.fromisoformat(earned)
        expiry = payload.get("expiry_date")
        if isinstance(expiry, str):
            payload["expiry_date"] = date.fromisoformat(expiry)
        return cls(**payload)

    def to_dict(self, include_dates: bool = True) -> Dict[str, Any]:
        """Serialize the certification back to a dictionary."""

        result: Dict[str, Any] = {
            "official_name": self.official_name,
            "issuer": self.issuer,
            "domains": list(self.domains),
            "aliases": list(self.aliases),
            "technologies": list(self.technologies),
            "categories": list(self.categories),
            "related_skills": list(self.related_skills),
        }
        if self.description:
            result["description"] = self.description
        if self.url:
            result["url"] = self.url
        if self.metadata:
            result["metadata"] = dict(self.metadata)
        if include_dates:
            if self.earned_date:
                result["earned_date"] = self.earned_date.isoformat()
            if self.expiry_date:
                result["expiry_date"] = self.expiry_date.isoformat()
        return result


def _dedupe_preserve(values: Iterable[str]) -> List[str]:
    """Remove duplicates from an iterable while preserving order."""

    seen: set[str] = set()
    result: List[str] = []
    for value in values:
        trimmed = value.strip()
        if trimmed and trimmed.lower() not in seen:
            seen.add(trimmed.lower())
            result.append(trimmed)
    return result
