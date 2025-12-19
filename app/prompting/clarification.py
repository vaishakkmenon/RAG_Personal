"""
Clarification - Ambiguous query handling

Generates clarification prompts for ambiguous or vague questions.
"""

from typing import List, Tuple


def _clarification_examples(question: str) -> Tuple[str, List[str]]:
    """Generate clarification examples based on question content.

    Args:
        question: The user's question

    Returns:
        Tuple of (topic, clarification_options)
    """
    stripped = (question or "").strip().lower()

    default_domains = [
        "your education",
        "your work experience",
        "your certifications",
        "your skills",
    ]

    if any(
        word in stripped
        for word in ("certification", "certifications", "cert", "certs")
    ):
        return (
            "your certifications",
            [
                "specific certification details (name, date, expiration)",
                "which certifications you hold",
                "skills validated by each certification",
            ],
        )
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
            "your background",
            [
                "your education (degrees, GPA, coursework)",
                "your work experience (roles, companies, responsibilities)",
                "your certifications (CKA, AWS certifications)",
                "your technical skills and projects",
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


def build_clarification_message(question: str, config) -> str:
    """Generate a clarification message for ambiguous questions.

    Args:
        question: The user's question
        config: PromptConfig instance

    Returns:
        Clarification message string
    """
    topic, domains = _clarification_examples(question)

    # For domain-specific queries (3 or fewer options), use simpler message
    if len(domains) <= 3:
        if len(domains) == 1:
            examples_text = domains[0]
        elif len(domains) == 2:
            examples_text = " or ".join(domains)
        else:
            examples_text = ", ".join(domains[:-1]) + ", or " + domains[-1]

        return (
            f"Could you clarify what you'd like to know about {topic}? "
            f"For example, I can provide information about {examples_text}."
        )

    # For broad queries, include more context
    core_domains = [
        "education",
        "work experience",
        "certifications",
        "skills",
        "qualifications",
    ]
    domain_set = list(
        dict.fromkeys(domains + [f"your {word}" for word in core_domains])
    )

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
