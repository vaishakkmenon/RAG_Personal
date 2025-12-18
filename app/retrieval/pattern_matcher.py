"""
Pattern Matcher - Individual pattern matching and rewriting logic

Handles:
- Regex-based matching
- Keyword-based matching
- Entity extraction
- Rewrite transformations (category expansion, list expansion, synonym expansion)
"""

import re
import logging
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class MatchResult:
    """Result of pattern matching."""

    matched: bool
    score: float = 0.0
    extracted_entities: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


class PatternMatcher:
    """Handles matching and rewriting for a single pattern."""

    def __init__(self, pattern_config: Dict):
        """
        Initialize pattern matcher.

        Args:
            pattern_config: Pattern configuration dict from YAML
        """
        self.name = pattern_config["name"]
        self.priority = pattern_config["priority"]
        self.description = pattern_config.get("description", "")
        self.matching_config = pattern_config["matching"]
        self.rewrite_config = pattern_config["rewrite_strategy"]

        # Pre-compile regex patterns for performance
        self.compiled_patterns: List[Dict] = []
        if self.matching_config["type"] == "regex_list":
            for rule in self.matching_config.get("rules", []):
                try:
                    self.compiled_patterns.append(
                        {
                            "pattern": re.compile(rule["pattern"], re.IGNORECASE),
                            "extract_groups": rule.get("extract_groups", []),
                        }
                    )
                except re.error as e:
                    logger.error(
                        f"Failed to compile pattern '{rule['pattern']}' in {self.name}: {e}"
                    )

    def match(self, query: str) -> MatchResult:
        """
        Check if query matches this pattern.

        Args:
            query: User query string

        Returns:
            MatchResult with matched status and extracted entities
        """
        match_type = self.matching_config["type"]

        if match_type == "regex_list":
            return self._match_regex_list(query)
        elif match_type == "keyword_presence":
            return self._match_keyword_presence(query)
        else:
            logger.warning(
                f"Unknown matching type '{match_type}' in pattern {self.name}"
            )
            return MatchResult(matched=False)

    def _match_regex_list(self, query: str) -> MatchResult:
        """
        Match query against list of regex patterns.

        Args:
            query: User query string

        Returns:
            MatchResult (first match wins)
        """
        for compiled in self.compiled_patterns:
            match = compiled["pattern"].search(query)
            if match:
                # Extract entities from capture groups
                entities = {}
                for idx in compiled.get("extract_groups", []):
                    if idx <= len(match.groups()):
                        entity_value = match.group(idx).strip()
                        entities[f"entity_{idx}"] = entity_value

                logger.debug(f"Pattern '{self.name}' matched query: {query[:50]}...")

                return MatchResult(matched=True, score=1.0, extracted_entities=entities)

        return MatchResult(matched=False)

    def _match_keyword_presence(self, query: str) -> MatchResult:
        """
        Match query based on keyword presence.

        Args:
            query: User query string

        Returns:
            MatchResult with score based on keyword match rate
        """
        keywords = self.matching_config.get("keywords", [])
        if not keywords:
            return MatchResult(matched=False)

        query_lower = query.lower()
        matched_keywords = [kw for kw in keywords if kw.lower() in query_lower]

        score = len(matched_keywords) / len(keywords) if keywords else 0.0
        min_score = self.matching_config.get("min_score", 0.5)

        if score >= min_score:
            logger.debug(
                f"Pattern '{self.name}' matched {len(matched_keywords)}/{len(keywords)} keywords"
            )
            return MatchResult(
                matched=True,
                score=score,
                metadata={"matched_keywords": matched_keywords},
            )

        return MatchResult(matched=False)

    def apply_rewrite(self, query: str, match_result: MatchResult) -> Tuple[str, Dict]:
        """
        Apply rewrite transformation based on strategy type.

        Args:
            query: Original query
            match_result: MatchResult from pattern matching

        Returns:
            (rewritten_query, metadata_dict) tuple
        """
        strategy_type = self.rewrite_config["type"]

        if strategy_type == "category_expansion":
            return self._apply_category_expansion(query, match_result)
        elif strategy_type == "list_expansion":
            return self._apply_list_expansion(query, match_result)
        elif strategy_type == "synonym_expansion":
            return self._apply_synonym_expansion(query, match_result)
        elif strategy_type == "prefix_injection":
            return self._apply_prefix_injection(query, match_result)
        elif strategy_type == "course_code_injection":
            return self._apply_course_code_injection(query, match_result)
        elif strategy_type == "passthrough":
            return query, {"hint": "Passthrough (no rewrite)"}
        else:
            logger.warning(
                f"Unknown rewrite strategy '{strategy_type}' in pattern {self.name}"
            )
            return query, {}

    def _apply_category_expansion(
        self, query: str, match_result: MatchResult
    ) -> Tuple[str, Dict]:
        """
        Expand specific entity to category search.

        Example:
            "Do I have a PhD?" -> "education degrees academic qualifications"

        Args:
            query: Original query
            match_result: MatchResult with extracted entities

        Returns:
            (category_query, metadata) tuple
        """
        entity_map = self.rewrite_config.get("entity_to_category", {})
        entities = match_result.extracted_entities or {}

        # Try to find matching entity in map
        for entity_key, entity_value in entities.items():
            entity_lower = entity_value.lower()

            for pattern, category in entity_map.items():
                if re.search(pattern, entity_lower, re.IGNORECASE):
                    logger.info(
                        f"Category expansion: '{entity_value}' -> '{category}' "
                        f"(pattern: {self.name})"
                    )
                    return category, {
                        "hint": f"Searching for category instead of specific entity: {entity_value}"
                    }

        # No matching entity found, return original
        logger.debug(f"No category mapping found for entities: {entities}")
        return query, {}

    def _apply_list_expansion(
        self, query: str, match_result: MatchResult
    ) -> Tuple[str, Dict]:
        """
        Expand to list-based query.

        Example:
            "Did I take CS 350?" -> "List all courses classes taken enrolled"

        Args:
            query: Original query
            match_result: MatchResult

        Returns:
            (expanded_query, metadata) tuple
        """
        template = self.rewrite_config.get("template", query)

        # Replace {entity} placeholder if present
        entities = match_result.extracted_entities or {}
        expanded = template

        for entity_key, entity_value in entities.items():
            expanded = expanded.replace("{entity}", entity_value)

        logger.info(
            f"List expansion: '{query[:50]}...' -> '{expanded}' (pattern: {self.name})"
        )

        return expanded, {"hint": "List-based expansion for comprehensive retrieval"}

    def _apply_synonym_expansion(
        self, query: str, match_result: MatchResult
    ) -> Tuple[str, Dict]:
        """
        Add synonyms to query for better coverage.

        Example:
            "What AI courses?" -> "What AI artificial intelligence machine learning courses?"

        Args:
            query: Original query
            match_result: MatchResult

        Returns:
            (expanded_query, metadata) tuple
        """
        synonyms = self.rewrite_config.get("synonyms", {})
        expanded = query
        added_synonyms = []

        # Sort terms by length (longest first) to match specific terms like "AI/ML" before "AI"
        sorted_terms = sorted(synonyms.keys(), key=len, reverse=True)

        for term in sorted_terms:
            synonym_list = synonyms[term]
            # Check if term appears in query (case-insensitive)
            if term.lower() in query.lower():
                # Add synonyms after the term
                synonym_str = " ".join(synonym_list)
                # Find the term in original query (preserve case)
                term_pattern = re.compile(re.escape(term), re.IGNORECASE)
                match = term_pattern.search(expanded)

                if match:
                    # Insert synonyms after the matched term
                    insert_pos = match.end()
                    expanded = (
                        expanded[:insert_pos]
                        + " "
                        + synonym_str
                        + expanded[insert_pos:]
                    )
                    added_synonyms.extend(synonym_list)
                    break  # Only expand first matching term (now the longest one)

        if added_synonyms:
            logger.info(
                f"Synonym expansion: added {len(added_synonyms)} synonyms "
                f"(pattern: {self.name})"
            )

        return expanded, {"synonyms_added": added_synonyms}

    def _apply_prefix_injection(
        self, query: str, match_result: MatchResult
    ) -> Tuple[str, Dict]:
        """
        Prepend prefix keywords to query for better document targeting.

        Example:
            Query: "What AI courses did I take?"
            Prefix: "transcript"
            Result: "transcript What AI courses did I take?"

        This helps target specific document types (e.g., transcripts for course queries)
        without expanding abbreviations that might match unrelated documents.

        Args:
            query: Original query
            match_result: MatchResult

        Returns:
            (prefixed_query, metadata) tuple
        """
        prefix = self.rewrite_config.get("prefix", "")

        if not prefix:
            logger.warning(
                f"No prefix configured for prefix_injection pattern {self.name}"
            )
            return query, {}

        # Prepend the prefix
        prefixed_query = f"{prefix} {query}"

        logger.info(
            f"Prefix injection: prepended '{prefix}' to query "
            f"(pattern: {self.name})"
        )

        return prefixed_query, {"prefix_added": prefix}

    def _apply_course_code_injection(
        self, query: str, match_result: MatchResult
    ) -> Tuple[str, Dict]:
        """
        Inject specific course codes based on topic keywords.

        This maps semantic topic keywords (AI, ML) to specific course codes
        that are easier to match exactly in the transcript documents.

        Example:
            Query: "What AI/ML courses did I take?"
            Mappings: {"AI/ML": ["CS 660", "CS 662", "CS 667"]}
            Result: "transcript CS 660 CS 662 CS 667 What AI/ML courses did I take?"

        Args:
            query: Original query
            match_result: MatchResult (may contain extracted entities)

        Returns:
            (injected_query, metadata) tuple
        """
        mappings = self.rewrite_config.get("course_mappings", {})
        prefix = self.rewrite_config.get("prefix", "transcript")

        if not mappings:
            logger.warning(f"No course_mappings configured for {self.name}")
            return query, {}

        # Find which mapping applies (longest match first)
        sorted_keys = sorted(mappings.keys(), key=len, reverse=True)
        matched_key = None
        course_codes = []

        for key in sorted_keys:
            if key.lower() in query.lower():
                matched_key = key
                course_codes = mappings[key]
                break

        if not course_codes:
            # No mapping found, just prepend prefix
            return f"{prefix} {query}", {"prefix_added": prefix, "courses_added": []}

        # Build the injected query: "transcript CS 660 CS 662 CS 667 <original query>"
        codes_str = " ".join(course_codes)
        injected_query = f"{prefix} {codes_str} {query}"

        logger.info(
            f"Course code injection: added {course_codes} for '{matched_key}' "
            f"(pattern: {self.name})"
        )

        return injected_query, {
            "prefix_added": prefix,
            "courses_added": course_codes,
            "matched_topic": matched_key,
        }

    def __repr__(self) -> str:
        """String representation for debugging."""
        return f"PatternMatcher(name='{self.name}', priority={self.priority})"
