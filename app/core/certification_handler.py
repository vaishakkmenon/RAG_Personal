"""
Certification handler for Personal RAG system.

Handles certification-specific queries, formatting, and metadata extraction.
"""

import logging
from datetime import date, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

if TYPE_CHECKING:  # pragma: no cover - import for typing only
    from ..certifications import Certification, CertificationRegistry

logger = logging.getLogger(__name__)


class CertificationHandler:
    """Handler for certification-related operations."""

    def __init__(self, registry: Optional["CertificationRegistry"] = None):
        """Initialize certification handler.

        Args:
            registry: Optional certification registry
        """
        self.registry = registry

    @staticmethod
    def extract_cert_id_from_source(source: str) -> Optional[str]:
        """Derive certification identifier from a chunk source path.

        Args:
            source: File path or source string

        Returns:
            Certification ID if found, None otherwise
        """
        if not source:
            return None

        try:
            filename = Path(source).name
        except (OSError, ValueError):
            return None

        if not filename.lower().startswith("certificate--"):
            return None

        parts = filename.split("--")
        if len(parts) < 3:
            return None

        slug = parts[1].strip().lower()
        return slug or None

    @staticmethod
    def parse_date(value: Any) -> Optional[date]:
        """Parse various date formats into a date object.

        Args:
            value: Date value (date, datetime, str, or None)

        Returns:
            Parsed date or None
        """
        if isinstance(value, date):
            return value
        if isinstance(value, datetime):
            return value.date()
        if isinstance(value, str):
            text = value.strip()
            if not text:
                return None
            try:
                return datetime.fromisoformat(text).date()
            except ValueError:
                for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%B %d, %Y", "%b %d, %Y"):
                    try:
                        return datetime.strptime(text, fmt).date()
                    except ValueError:
                        continue
        return None

    @staticmethod
    def harvest_cert_metadata(target: Dict[str, Any], source: Dict[str, Any]) -> None:
        """Extract certification metadata from source dict into target dict.

        Args:
            target: Target dict to populate
            source: Source dict containing metadata
        """
        date_fields = (
            ("earned_date", "earned_date"),
            ("earned", "earned_date"),
            ("date_earned", "earned_date"),
            ("expires", "expiry_date"),
            ("expiry", "expiry_date"),
            ("expiry_date", "expiry_date"),
        )

        for raw_key, dest_key in date_fields:
            if dest_key in target:
                continue
            value = source.get(raw_key)
            if not value:
                continue
            parsed = CertificationHandler.parse_date(value)
            if parsed:
                target[dest_key] = parsed

        if "status" not in target and source.get("status"):
            target["status"] = str(source["status"])

    def collect_certifications(
        self,
        params: Dict[str, Any],
        chunks: List[dict],
        question: Optional[str] = None,
    ) -> Tuple[List["Certification"], Dict[str, Dict[str, Any]]]:
        """Gather certification objects with metadata overrides.

        Args:
            params: Query parameters that may contain cert_id
            chunks: Retrieved chunks that may contain cert metadata
            question: Optional user question for matching

        Returns:
            Tuple of (certifications, metadata_overrides)
        """
        if not self.registry:
            return [], {}

        seen: set[str] = set()
        collected: List["Certification"] = []
        overrides: Dict[str, Dict[str, Any]] = {}

        def add_cert_by_id(
            candidate: Optional[str], source_metadata: Optional[Dict[str, Any]] = None
        ) -> None:
            if not candidate:
                return
            cert = self.registry.certifications.get(str(candidate).lower())
            if not cert:
                return
            key = cert.id.lower()
            if key not in seen:
                seen.add(key)
                collected.append(cert)
            if source_metadata:
                bucket = overrides.setdefault(key, {})
                self.harvest_cert_metadata(bucket, source_metadata)

        # Add cert from params
        add_cert_by_id(params.get("cert_id"))

        # Add certs from chunks
        for chunk in chunks:
            metadata = chunk.get("metadata") or {}
            meta_source = metadata.get("source") or chunk.get("source")

            candidate = metadata.get("cert_id") or metadata.get("certification_id")
            if candidate:
                add_cert_by_id(str(candidate), metadata)
                continue

            parsed = (
                self.extract_cert_id_from_source(str(meta_source))
                if meta_source
                else None
            )
            if parsed:
                add_cert_by_id(parsed, metadata)

        # Match by question content
        question_lower = (question or "").strip().lower()
        if question_lower:
            name_matches: set[str] = set()
            domain_matches: set[str] = set()

            for cert in self.registry.certifications.values():
                names = [cert.official_name, *cert.aliases, cert.id]
                if any(
                    name and name.strip() and name.lower() in question_lower
                    for name in names
                ):
                    add_cert_by_id(cert.id)
                    name_matches.add(cert.id.lower())
                    continue

                domains = getattr(cert, "domains", [])
                if any(
                    domain and domain.strip() and domain.lower() in question_lower
                    for domain in domains
                ):
                    add_cert_by_id(cert.id)
                    domain_matches.add(cert.id.lower())

            # Filter to most specific matches
            if name_matches:
                collected = [cert for cert in collected if cert.id.lower() in name_matches]
            elif domain_matches:
                collected = [
                    cert for cert in collected if cert.id.lower() in domain_matches
                ]

        # Fallback: include all certs if nothing specific found
        if not collected:
            if params.get("cert_id"):
                add_cert_by_id(params.get("cert_id"))
            else:
                for cert in self.registry.certifications.values():
                    add_cert_by_id(cert.id)

        collected.sort(key=lambda cert: cert.official_name.lower())
        return collected, overrides

    @staticmethod
    def format_date_phrase(value: Optional[date]) -> Optional[str]:
        """Format a date as both ISO and human-readable.

        Args:
            value: Date to format

        Returns:
            Formatted string like "2024-06-26 (June 26, 2024)" or None
        """
        if not value:
            return None
        return f"{value.isoformat()} ({value.strftime('%B %d, %Y')})"

    @staticmethod
    def determine_status(
        cert: "Certification", override: Dict[str, Any]
    ) -> Optional[str]:
        """Determine certification status based on expiry date.

        Args:
            cert: Certification object
            override: Metadata overrides

        Returns:
            Status string like "Status: Active" or "Status: Expired"
        """
        status = override.get("status")
        if status:
            return status

        expiry_date: Optional[date] = override.get("expiry_date") or cert.expiry_date
        if isinstance(expiry_date, str):
            parsed = CertificationHandler.parse_date(expiry_date)
            if parsed:
                override["expiry_date"] = parsed
                expiry_date = parsed
            else:
                return None

        if not isinstance(expiry_date, date):
            return None

        return "Status: Active" if expiry_date >= date.today() else "Status: Expired"

    @staticmethod
    def display_name(cert: "Certification") -> str:
        """Get display name for certification (includes short alias if available).

        Args:
            cert: Certification object

        Returns:
            Display name like "AWS Certified Cloud Practitioner (CCP)"
        """
        alias = next((alias for alias in cert.aliases if alias and len(alias) <= 5), None)
        if alias:
            return f"{cert.official_name} ({alias})"
        return cert.official_name

    def format_certification_answer(
        self,
        certs: List["Certification"],
        overrides: Dict[str, Dict[str, Any]],
        question: Optional[str],
    ) -> str:
        """Produce a deterministic certification answer block tailored to the query.

        Args:
            certs: List of certifications to format
            overrides: Metadata overrides for each cert
            question: The user's question

        Returns:
            Formatted answer string
        """
        if not certs:
            return "I couldn't find any certifications in your records."

        question_lower = (question or "").strip().lower()
        multiple = len(certs) > 1

        # Single certification response
        if not multiple:
            return self._format_single_certification(
                certs[0], overrides.get(certs[0].id.lower(), {}), question_lower
            )

        # Multiple certifications response
        return self._format_multiple_certifications(certs, overrides, question_lower)

    def _format_single_certification(
        self,
        cert: "Certification",
        override: Dict[str, Any],
        question_lower: str,
    ) -> str:
        """Format answer for a single certification.

        Args:
            cert: Certification object
            override: Metadata overrides
            question_lower: Lowercased question

        Returns:
            Formatted answer
        """
        # Parse dates
        earned_date = override.get("earned_date") or cert.earned_date
        if isinstance(earned_date, str):
            parsed = self.parse_date(earned_date)
            if parsed:
                override["earned_date"] = parsed
                earned_date = parsed

        expiry_date = override.get("expiry_date") or cert.expiry_date
        if isinstance(expiry_date, str):
            parsed = self.parse_date(expiry_date)
            if parsed:
                override["expiry_date"] = parsed
                expiry_date = parsed

        earned_phrase = self.format_date_phrase(earned_date)
        expiry_phrase = self.format_date_phrase(expiry_date)
        status_text = self.determine_status(cert, override)
        name = self.display_name(cert)

        lines: List[str] = []

        # Tailor response based on question type
        if "do i have" in question_lower or "have any" in question_lower:
            sentence = f"Yes — you hold the **{name}** certification ({cert.issuer})."
            if earned_phrase or expiry_phrase:
                details = []
                if earned_phrase:
                    details.append(f"Earned: {earned_phrase}")
                if expiry_phrase:
                    details.append(f"Expires: {expiry_phrase}")
                sentence += " " + "; ".join(details) + "."
            lines.append(sentence)

        elif "when" in question_lower and (
            "earn" in question_lower or "get" in question_lower
        ):
            if earned_phrase:
                lines.append(
                    f"You earned the **{name}** certification ({cert.issuer}) on {earned_phrase}."
                )
            else:
                lines.append(f"You hold the **{name}** certification ({cert.issuer}).")
            if expiry_phrase:
                lines.append(f"It expires on {expiry_phrase}.")

        elif "expire" in question_lower:
            if expiry_phrase:
                lines.append(
                    f"Your **{name}** certification ({cert.issuer}) expires on {expiry_phrase}."
                )
            else:
                lines.append(
                    f"Your **{name}** certification ({cert.issuer}) is currently active."
                )
            if status_text:
                lines.append(status_text)

        else:
            # Default comprehensive answer
            sentence = f"Your **{name}** certification ({cert.issuer})"
            detail_parts = []
            if earned_phrase:
                detail_parts.append(f"Earned: {earned_phrase}")
            if expiry_phrase:
                detail_parts.append(f"Expires: {expiry_phrase}")
            if status_text:
                detail_parts.append(status_text)
            if detail_parts:
                sentence += " — " + "; ".join(detail_parts)
            lines.append(sentence)

        return "\n".join(lines)

    def _format_multiple_certifications(
        self,
        certs: List["Certification"],
        overrides: Dict[str, Dict[str, Any]],
        question_lower: str,
    ) -> str:
        """Format answer for multiple certifications.

        Args:
            certs: List of certifications
            overrides: Metadata overrides
            question_lower: Lowercased question

        Returns:
            Formatted answer
        """
        lines = []
        if "what certifications" in question_lower or "list" in question_lower:
            lines.append("You hold the following certifications:")

        for cert in certs:
            override = overrides.get(cert.id.lower(), {})

            # Parse dates
            earned_date = override.get("earned_date") or cert.earned_date
            if isinstance(earned_date, str):
                parsed = self.parse_date(earned_date)
                if parsed:
                    override["earned_date"] = parsed
                    earned_date = parsed

            expiry_date = override.get("expiry_date") or cert.expiry_date
            if isinstance(expiry_date, str):
                parsed = self.parse_date(expiry_date)
                if parsed:
                    override["expiry_date"] = parsed
                    expiry_date = parsed

            earned_phrase = self.format_date_phrase(earned_date)
            expiry_phrase = self.format_date_phrase(expiry_date)
            status_text = self.determine_status(cert, override)
            name = self.display_name(cert)

            detail_parts: List[str] = []
            if earned_phrase:
                detail_parts.append(f"Earned: {earned_phrase}")
            if expiry_phrase:
                detail_parts.append(f"Expires: {expiry_phrase}")
            if status_text:
                detail_parts.append(status_text)

            detail_text = " — " + "; ".join(detail_parts) if detail_parts else ""
            lines.append(f"- **{name}** ({cert.issuer}){detail_text}")

        return "\n".join(lines)


__all__ = ["CertificationHandler"]
