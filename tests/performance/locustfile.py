import uuid
from locust import HttpUser, task, between
import os
import urllib3

# Disable SSL warnings for local testing
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Load API Key from env (REQUIRED)
API_KEY = os.getenv("API_KEY")
if not API_KEY:
    raise ValueError(
        "API_KEY environment variable must be set for performance testing."
    )


class RAGUser(HttpUser):
    # Wait between 1 and 5 seconds between tasks (think time)
    wait_time = between(1, 5)

    def on_start(self):
        """Called when a User starts running."""
        # Disable SSL verification for local testing
        self.client.verify = False
        self.client.headers.update({"X-API-Key": API_KEY})
        # Generate a unique session ID for this simulated user
        self.session_id = f"loadtest_{uuid.uuid4()}"

    @task(3)
    def ask_simple_question(self):
        """Standard short query."""
        self.client.post(
            "/chat", json={"question": "What is RAG?", "session_id": self.session_id}
        )

    @task(1)
    def ask_complex_question(self):
        """More complex query to stress retrieval."""
        self.client.post(
            "/chat",
            json={
                "question": "Explain the architecture of this system and how it handles citations.",
                "session_id": self.session_id,
            },
        )

    @task(1)
    def ask_creative_question(self):
        """Creative task."""
        self.client.post(
            "/chat",
            json={
                "question": "Write a short poem about vector databases.",
                "session_id": self.session_id,
            },
        )
