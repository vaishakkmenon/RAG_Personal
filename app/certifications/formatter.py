"""Utilities for consistent certification response formatting."""
from __future__ import annotations

from datetime import date
from typing import Iterable, Mapping, MutableMapping, Optional, Sequence

from .models import Certification


class CertificationFormatter:
    """Format certification objects for downstream presentation layers."""

    def __init__(self, date_format: str = "%B %Y") -> None:
        self.date_format = date_format

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def format_single(
        self,
        cert: Certification,
        *,
        include_description: bool = True,
        include_metadata: bool = True,
    ) -> str:
        """Return a Markdown representation of a single certification."""

        lines = [f"**{cert.official_name}**", f"Issuer: {cert.issuer}"]

        if cert.aliases:
            lines.append(f"Also known as: {', '.join(cert.aliases)}")

        if cert.domains:
            lines.append(f"Domains: {', '.join(cert.domains)}")

        date_block = self._format_date_block(cert)
        if date_block:
            lines.append(date_block)

        if cert.url:
            lines.append(f"More info: {cert.url}")

        if include_description and cert.description:
            lines.append("")
            lines.append(cert.description.strip())

        if include_metadata and cert.metadata:
            metadata_block = self._format_metadata(cert.metadata)
            if metadata_block:
                lines.append("")
                lines.append(metadata_block)

        return "\n".join(lines)

    def format_multiple(
        self,
        certs: Sequence[Certification],
        *,
        group_by: Optional[str] = None,
        include_description: bool = False,
        include_metadata: bool = True,
    ) -> str:
        """Return a Markdown section for multiple certifications."""

        if not certs:
            return "No certifications found."

        if group_by == "issuer":
            return self._format_grouped(
                certs,
                key_fn=lambda c: c.issuer,
                heading_label="Issuer",
                include_description=include_description,
                include_metadata=include_metadata,
            )

        if group_by == "domain":
            return self._format_grouped(
                certs,
                key_fn=lambda c: c.domains or ["Other"],
                heading_label="Domain",
                include_description=include_description,
                include_metadata=include_metadata,
            )

        return "\n\n".join(
            self.format_single(
                cert,
                include_description=include_description,
                include_metadata=include_metadata,
            )
            for cert in certs
        )

    def format_date_range(
        self,
        start_date: Optional[date],
        end_date: Optional[date],
        *,
        fallback: str = "Present",
    ) -> str:
        """Return a human-readable date range."""

        start = self._format_date(start_date)
        end = self._format_date(end_date) if end_date else fallback

        if start and end:
            return f"{start} – {end}"

        return start or end or ""

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _format_grouped(
        self,
        certs: Sequence[Certification],
        *,
        key_fn,
        heading_label: str,
        include_description: bool,
        include_metadata: bool,
    ) -> str:
        chunks = []

        grouped: MutableMapping[str, list[Certification]] = {}
        for cert in certs:
            keys = key_fn(cert)
            if isinstance(keys, str):
                keys = [keys]
            for key in keys:
                grouped.setdefault(key, []).append(cert)

        for group_name in sorted(grouped):
            chunks.append(f"### {heading_label}: {group_name}")
            for cert in sorted(grouped[group_name], key=lambda c: c.official_name.lower()):
                chunks.append(
                    f"- {cert.official_name}"
                    + (f" ({cert.issuer})" if heading_label != "Issuer" else "")
                )
                detail = self.format_single(
                    cert,
                    include_description=include_description,
                    include_metadata=include_metadata,
                )
                detail_lines = [f"  {line}" for line in detail.splitlines()]
                chunks.extend(detail_lines)

            chunks.append("")

        return "\n".join(chunks).strip()

    def _format_date_block(self, cert: Certification) -> str:
        earned = self._format_date(cert.earned_date)
        expiry = self._format_date(cert.expiry_date)

        if not earned and not expiry:
            return ""

        parts: list[str] = []
        if earned:
            parts.append(f"Earned: {earned}")

        if expiry:
            status = "Expires" if cert.expiry_date and cert.expiry_date >= date.today() else "Expired"
            parts.append(f"{status}: {expiry}")

        return " | ".join(parts)

    def _format_date(self, value: Optional[date]) -> str:
        if not value:
            return ""
        return value.strftime(self.date_format)

    def _format_metadata(self, metadata: Mapping[str, object]) -> str:
        items = []
        for key in sorted(metadata):
            value = metadata[key]
            normalized = self._normalize_metadata_value(value)
            if normalized:
                items.append(f"{key.title()}: {normalized}")
        return "Metadata\n" + "\n".join(f"- {item}" for item in items) if items else ""

    def _normalize_metadata_value(self, value: object) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            return value.strip()
        if isinstance(value, date):
            return self._format_date(value)
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            normalized = [self._normalize_metadata_value(v) for v in value]
            normalized = [v for v in normalized if v]
            return ", ".join(normalized)
        if isinstance(value, Mapping):
            parts = []
            for k in sorted(value):
                normalized = self._normalize_metadata_value(value[k])
                if normalized:
                    parts.append(f"{k.title()}: {normalized}")
            return "; ".join(parts)
        return str(value)
