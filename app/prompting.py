from typing import List, Dict, Optional, Tuple
import re
from dataclasses import dataclass

@dataclass
class PromptConfig:
    """Configuration for prompt construction and validation."""
    min_question_length: int = 3
    max_context_length: int = 4000
    system_prompt: str = """You are an AI assistant that provides factual answers based on the provided context.

GUIDELINES:
1. Answer ONLY using information from the provided context.
2. If the context doesn't contain the answer, say "I don't know. It isn't mentioned in the provided documents."
3. Be concise and factual.
4. Use bullet points for lists.
5. Always include source references when available.
6. If a question is ambiguous or unclear, ask for clarification.
7. For numerical questions, provide exact figures from the context."""
    
    # Common ambiguous phrases and patterns
    ambiguous_phrases = [
        "tell me about", "what about", "what do you know about",
        "background", "history", "experience", "skills", "explain",
        "how to", "help with", "what's the deal with"
    ]
    
    # Question words that often indicate vagueness without specifics
    vague_question_words = ["how", "what", "why", "when", "where", "who"]

    clarification_cues = (
        "clarify",
        "which specific",
        "could you clarify",
        "please specify",
        "what aspect",
    )

    refusal_cues = (
        "i don't know",
        "i do not know",
        "not mentioned",
        "couldn't find",
        "could not find",
        "cannot find",
        "not available",
        "no relevant information",
        "i'm not sure",
        "unable to find",
        "insufficient information",
    )


@dataclass
class PromptResult:
    status: str
    prompt: str = ""
    message: str = ""
    context: str = ""


def _format_context(chunks: List[Dict[str, str]]) -> str:
    parts: List[str] = []
    for chunk in chunks:
        source = chunk.get("source", "unknown")
        text = (chunk.get("text") or "").strip()
        if text:
            parts.append(f"[Source: {source}]\n{text}")
    return "\n\n".join(parts)


def _clarification_examples(question: str) -> Tuple[str, List[str]]:
    stripped = (question or "").strip().lower()

    default_domains = [
        "your education",
        "your work experience",
        "your certifications",
        "your skills",
    ]

    if "gpa" in stripped:
        return (
            "your academic performance",
            [
                "your undergraduate GPA",
                "your graduate GPA",
                "your overall GPA",
            ],
        )
    if "experience" in stripped:
        return (
            "your experience",
            [
                "your professional roles",
                "projects you've completed",
                "your technical skills",
            ],
        )
    if any(word in stripped for word in ("background", "qualifications")):
        return (
            "your background and qualifications",
            [
                "your education",
                "your work experience",
                "your certifications",
                "projects that used Kubernetes",
                "certifications like the CKA",
                "work experience involving Kubernetes",
            ],
        )
    if "history" in stripped:
        return (
            "your history",
            [
                "your education history",
                "your work history",
                "key milestones",
            ],
        )
    if "kubernetes" in stripped:
        return (
            "Kubernetes",
            [
                "projects that used Kubernetes",
                "certifications like the CKA",
                "work experience involving Kubernetes",
            ],
        )

    return ("your profile", default_domains)


def build_clarification_message(question: str, config: PromptConfig) -> str:
    topic, domains = _clarification_examples(question)

    core_domains = [
        "education",
        "work experience",
        "certifications",
        "skills",
        "qualifications",
    ]
    domain_set = list(dict.fromkeys(domains + [f"your {word}" for word in core_domains]))

    if len(domain_set) == 1:
        examples_text = domain_set[0]
    elif len(domain_set) == 2:
        examples_text = " or ".join(domain_set)
    else:
        examples_text = ", ".join(domain_set[:-1]) + ", or " + domain_set[-1]

    return (
        f"Your question seems a bit broad. Could you clarify which specific details you're looking for about {topic}? "
        f"For example, I can share information about {examples_text}. "
        "Typical areas I can cover include education, work experience, certifications, skills, and other qualifications. "
        "Please let me know which detail to focus on so I can reference the right documents."
    )


class PromptBuilder:
    """Handles construction and validation of prompts with safety guards."""
    
    def __init__(self, config: Optional[PromptConfig] = None):
        self.config = config or PromptConfig()
        
    def is_ambiguous(self, question: str) -> Tuple[bool, str]:
        """Check if a question is ambiguous or too vague."""
        question = (question or "").strip()
        
        # Basic validation
        if not question:
            return True, "Question cannot be empty"
            
        if len(question.split()) < self.config.min_question_length:
            return True, "Question is too short"
            
        # Check for ambiguous phrases
        lower_question = question.lower()
        for phrase in self.config.ambiguous_phrases:
            if phrase in lower_question:
                return True, f"Question contains ambiguous phrase: {phrase}"
                
        # Check for vague questions without specifics
        first_word = lower_question.split()[0]
        if (
            first_word in self.config.vague_question_words
            and len(question.split()) < 5
        ):
            return True, "Question is too vague, please be more specific"
            
        return False, ""
        
    def build_prompt(
        self,
        question: str,
        context_chunks: List[Dict[str, str]],
        **kwargs
    ) -> PromptResult:
        """Build a structured prompt with validation."""
        stripped_question = (question or "").strip()
        ambiguous, reason = self.is_ambiguous(stripped_question)
        if ambiguous:
            return PromptResult(
                status="ambiguous",
                message=build_clarification_message(stripped_question, self.config),
            )

        context = _format_context(context_chunks)
        if not context:
            return PromptResult(
                status="no_context",
                message="No relevant context found to answer the question.",
            )

        prompt = """{system_prompt}

CONTEXT:
{context}

QUESTION: {question}

ANSWER:""".format(
            system_prompt=self.config.system_prompt,
            context=context,
            question=stripped_question,
        )

        return PromptResult(status="success", prompt=prompt, context=context)

    def is_refusal(self, answer: str) -> bool:
        text = (answer or "").strip().lower()
        if not text:
            return True
        if any(cue in text for cue in self.config.refusal_cues):
            return True
        if not any(ch.isalnum() for ch in text):
            return True
        return False

    def needs_clarification(self, answer: str) -> bool:
        text = (answer or "").strip().lower()
        if not text:
            return False
        return not any(cue in text for cue in self.config.clarification_cues)


def create_default_prompt_builder() -> PromptBuilder:
    """Create a pre-configured prompt builder with default settings."""
    return PromptBuilder()
