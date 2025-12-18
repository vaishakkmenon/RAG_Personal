"""
LLM-based Test Validator

Uses an LLM to semantically validate RAG answers instead of simple keyword matching.
Supports both Ollama (local) and Groq (cloud) providers.
"""

import os
from typing import Dict, Any, List, Optional
import ollama
from groq import Groq


class LLMValidator:
    """Validates RAG answers using LLM-based semantic evaluation"""

    def __init__(
        self,
        provider: str = "ollama",
        model: str = None,
        groq_api_key: str = None,
        keep_alive: str = "10m",
    ):
        """Initialize validator with specified provider and model

        Args:
            provider: "ollama" or "groq"
            model: Model name (defaults based on provider)
            groq_api_key: Groq API key (required if provider='groq')
            keep_alive: How long to keep Ollama model loaded (e.g., "10m", "1h")
        """
        self.provider = provider.lower()
        self.keep_alive = keep_alive

        # Set default models based on provider
        if model is None:
            if self.provider == "groq":
                self.model = "llama-3.1-8b-instant"
            else:
                self.model = (
                    "llama3.1:8b"  # Updated to use 8B model for better validation
                )
        else:
            self.model = model

        # Initialize clients
        if self.provider == "groq":
            api_key = (
                groq_api_key
                or os.getenv("LLM_GROQ_API_KEY")
                or os.getenv("GROQ_API_KEY")
            )
            if not api_key:
                raise ValueError(
                    "Groq API key required. Set LLM_GROQ_API_KEY env var or pass groq_api_key parameter"
                )
            self.groq_client = Groq(api_key=api_key)
            self.ollama_client = None
            print(f"Initialized validator with Groq: {self.model}")
        else:
            self.ollama_client = ollama.Client()
            self.groq_client = None
            print(f"Initialized validator with Ollama: {self.model}")
            # Pre-load the model to avoid loading delays during tests
            self._warm_up_model()

    def _warm_up_model(self):
        """Pre-load the Ollama model to avoid delays during first validation"""
        if self.provider != "ollama" or not self.ollama_client:
            return

        try:
            # Send a minimal request to load the model into memory
            self.ollama_client.generate(
                model=self.model,
                prompt="test",
                options={"temperature": 0.0},
                keep_alive=self.keep_alive,
            )
            print(f"Ollama model {self.model} warmed up")
        except Exception as e:
            print(f"Warning: Failed to warm up model: {e}")
            # If warm-up fails, continue anyway (model will load on first real request)

    def _generate(self, prompt: str, temperature: float = 0.0) -> str:
        """Generate response using configured provider"""

        if self.provider == "groq":
            response = self.groq_client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=500,
            )
            return response.choices[0].message.content
        else:
            response = self.ollama_client.generate(
                model=self.model,
                prompt=prompt,
                options={
                    "temperature": temperature,
                    "num_predict": 500,
                },
                keep_alive=self.keep_alive,
            )
            return response["response"]

    def validate_answer(
        self,
        question: str,
        answer: str,
        expected_content: List[str],
        category: str,
        context: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Validate an answer using LLM semantic evaluation

        Args:
            question: The original question asked
            answer: The RAG system's answer
            expected_content: List of content that should be in the answer
            category: Test category (e.g., "certifications", "impossible")
            context: Optional context/sources used to generate answer

        Returns:
            Dict with validation results:
            {
                "is_correct": bool,
                "correctness_score": float (0-1),
                "issues": List[str],
                "explanation": str
            }
        """

        # Build validation prompt
        validation_prompt = self._build_validation_prompt(
            question, answer, expected_content, category
        )

        # Get LLM evaluation
        try:
            result_text = self._generate(validation_prompt, temperature=0.0)

            # Parse the structured response
            return self._parse_validation_response(result_text)

        except Exception as e:
            return {
                "is_correct": False,
                "correctness_score": 0.0,
                "issues": [f"Validation error: {str(e)}"],
                "explanation": "Failed to validate answer",
            }

    def _build_validation_prompt(
        self, question: str, answer: str, expected_content: List[str], category: str
    ) -> str:
        """Build the validation prompt for the LLM"""

        prompt = f"""You are a strict evaluator checking if an AI answer is correct.

QUESTION: {question}
CATEGORY: {category}

ANSWER PROVIDED:
{answer}

EXPECTED CONTENT (must be present):
{chr(10).join(f"- {item}" for item in expected_content)}

EVALUATION CRITERIA:
1. COMPLETENESS: Does answer include ALL expected content? Missing any = lower score
2. ACCURACY: Is information factually correct? Wrong info = fail
3. PRECISION: No irrelevant information? Extra wrong items = lower score
4. DIRECTNESS: Does it directly answer the question?

SCORING GUIDE:
- 1.0: Perfect - all expected content, accurate, no issues
- 0.8: Good - all expected content present, minor formatting issues only
- 0.5: Partial - missing some expected content OR has inaccuracies
- 0.0: Wrong - missing most expected content OR factually incorrect OR refuses to answer when it should

IMPORTANT:
- If answer says "I don't know" but question is answerable, score = 0.0
- If answer contains wrong information (wrong dates, names, etc), score <= 0.5
- Be strict: small omissions matter

Output format (EXACTLY):
CORRECT: [YES if score >= 0.8, else NO]
SCORE: [0.0 to 1.0]
ISSUES: [list problems, or "none"]
EXPLANATION: [1-2 sentences why]

Evaluation:"""

        return prompt

    def _parse_validation_response(self, response_text: str) -> Dict[str, Any]:
        """Parse the LLM's structured validation response"""

        lines = response_text.strip().split("\n")
        result = {
            "is_correct": False,
            "correctness_score": 0.0,
            "issues": [],
            "explanation": "",
        }

        for line in lines:
            line = line.strip()

            if line.startswith("CORRECT:"):
                value = line.split(":", 1)[1].strip().upper()
                result["is_correct"] = value == "YES"

            elif line.startswith("SCORE:"):
                try:
                    score_str = line.split(":", 1)[1].strip()
                    result["correctness_score"] = float(score_str)
                except ValueError:
                    result["correctness_score"] = 0.0

            elif line.startswith("ISSUES:"):
                issues_str = line.split(":", 1)[1].strip()
                if issues_str.lower() != "none":
                    result["issues"] = [i.strip() for i in issues_str.split(",")]

            elif line.startswith("EXPLANATION:"):
                result["explanation"] = line.split(":", 1)[1].strip()

        return result

    def validate_impossible_question(
        self, question: str, answer: str, grounded: bool
    ) -> Dict[str, Any]:
        """Special validation for impossible questions

        Impossible questions should:
        1. Express uncertainty ("I don't know", etc.)
        2. Not provide false/made-up information
        3. Ideally not be grounded (but this is checked separately)
        """

        issues = []
        answer_lower = answer.lower()

        # Check for uncertainty expression (REQUIRED)
        uncertainty_phrases = [
            "don't know",
            "do not know",
            "not mentioned",
            "not available",
            "can't find",
            "cannot find",
            "isn't mentioned",
            "could you clarify",
            "unclear",
            "no information",
        ]

        has_uncertainty = any(phrase in answer_lower for phrase in uncertainty_phrases)

        if not has_uncertainty:
            issues.append("Impossible question did not express uncertainty")

        # Check for false information (making up answers)
        # If answer is long and doesn't express uncertainty, likely made up info
        if not has_uncertainty and len(answer) > 50:
            issues.append(
                "Appears to provide made-up information for impossible question"
            )

        # Grounding check is less strict (warn but don't fail)
        if grounded and not has_uncertainty:
            issues.append("Answer is grounded and provides info (should be uncertain)")

        # Calculate score
        if len(issues) == 0:
            score = 1.0
        elif has_uncertainty:
            # Has uncertainty but might be grounded - partial credit
            score = 0.8
        else:
            # No uncertainty = wrong
            score = 0.0

        return {
            "is_correct": score >= 0.8,
            "correctness_score": score,
            "issues": issues,
            "explanation": "Impossible question handled correctly"
            if not issues
            else f"Issues: {', '.join(issues)}",
        }


# Example usage
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test LLM Validator")
    parser.add_argument("--provider", choices=["ollama", "groq"], default="ollama")
    parser.add_argument("--model", help="Model name (optional, uses defaults)")
    args = parser.parse_args()

    validator = LLMValidator(provider=args.provider, model=args.model)

    # Test case 1: Correct answer
    result1 = validator.validate_answer(
        question="What certifications do I have?",
        answer="• Certified Kubernetes Administrator (CKA)\n• AWS Certified Cloud Practitioner\n• AWS Certified AI Practitioner",
        expected_content=["CKA", "AWS", "Cloud Practitioner", "AI Practitioner"],
        category="certifications",
    )
    print("Test 1 - Correct answer:", result1)

    # Test case 2: Incorrect answer (includes wrong cert)
    result2 = validator.validate_answer(
        question="Do I have any Kubernetes certifications?",
        answer="Yes.\n- Certified Kubernetes Administrator (CKA)\n- AWS Certified Cloud Practitioner (CCP)",
        expected_content=["CKA"],
        category="certifications",
    )
    print(
        "\nTest 2 - Incorrect answer (includes AWS CCP for Kubernetes question):",
        result2,
    )

    # Test case 3: Impossible question
    result3 = validator.validate_impossible_question(
        question="Where was I born?",
        answer="I don't know. It isn't mentioned in the provided documents.",
        grounded=False,
    )
    print("\nTest 3 - Impossible question:", result3)
