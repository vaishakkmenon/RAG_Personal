#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Interactive Multi-Turn Conversation Tester

Usage:
    python scripts/manual_tests/interactive_chat.py

Commands:
    - Type your questions to chat
    - 'reset' - Start a new conversation (new session)
    - 'history' - Show conversation history
    - 'session' - Show session ID
    - 'quit' or 'exit' - Exit the tester
"""

import requests
from datetime import datetime

API_URL = "http://localhost:8000/chat"
API_KEY = "dev-key-1"


class ConversationTester:
    """Interactive multi-turn conversation tester"""

    def __init__(self):
        self.session_id = None
        self.turn_count = 0
        self.history = []

    def chat(self, question):
        """Send a chat request"""
        self.turn_count += 1

        payload = {"question": question}
        if self.session_id:
            payload["session_id"] = self.session_id

        print(f"\n{'='*80}")
        print(f"[Turn {self.turn_count}] You: {question}")
        print("=" * 80)

        try:
            response = requests.post(
                API_URL, json=payload, headers={"X-API-Key": API_KEY}, timeout=30
            )

            if response.status_code != 200:
                print(f"[ERROR] HTTP {response.status_code}")
                print(response.text)
                return None

            data = response.json()

            # Store session ID from first response
            if not self.session_id:
                self.session_id = data.get("session_id")
                print(f"[Session] New session created: {self.session_id}")

            # Display response
            answer = data.get("answer", "")
            print(f"\n[Assistant]\n{answer}")

            # Show metadata
            print("\n[Metadata]")
            print(f"  Grounded: {data.get('grounded', False)}")
            print(f"  Sources: {len(data.get('sources', []))}")
            print(f"  Session: {data.get('session_id', 'N/A')}")

            # Show ambiguity info if present
            if data.get("ambiguity"):
                amb = data["ambiguity"]
                if amb.get("is_ambiguous"):
                    print(
                        f"  [!] Ambiguity detected (score: {amb.get('score', 0):.2f})"
                    )
                    if amb.get("clarification_requested"):
                        print("  [!] Clarification was requested")

            # Add to history
            self.history.append(
                {
                    "turn": self.turn_count,
                    "question": question,
                    "answer": answer,
                    "grounded": data.get("grounded", False),
                    "timestamp": datetime.now().isoformat(),
                }
            )

            return data

        except requests.exceptions.ConnectionError:
            print("[ERROR] Could not connect to API server at http://localhost:8000")
            print("       Make sure the server is running: docker-compose up -d")
            return None
        except requests.exceptions.Timeout:
            print("[ERROR] Request timed out")
            return None
        except Exception as e:
            print(f"[ERROR] {type(e).__name__}: {e}")
            return None

    def reset(self):
        """Reset conversation (new session)"""
        old_session = self.session_id
        self.session_id = None
        self.turn_count = 0
        print(f"\n[Session] Reset - old session: {old_session}")
        print("[Session] Next message will start a new session")

    def show_history(self):
        """Show conversation history"""
        if not self.history:
            print("\n[History] No conversation history yet")
            return

        print(f"\n{'='*80}")
        print(f"CONVERSATION HISTORY ({len(self.history)} turns)")
        print(f"Session: {self.session_id}")
        print("=" * 80)

        for item in self.history:
            print(f"\n[Turn {item['turn']}] {item['question']}")
            print(f"  Answer: {item['answer'][:100]}...")
            print(f"  Grounded: {item['grounded']}")
            print(f"  Time: {item['timestamp']}")

    def show_session(self):
        """Show current session info"""
        print("\n[Session Info]")
        print(f"  Session ID: {self.session_id or 'No active session'}")
        print(f"  Turn count: {self.turn_count}")
        print(f"  History: {len(self.history)} turns")


def print_help():
    """Print help message"""
    print("\n" + "=" * 80)
    print("COMMANDS")
    print("=" * 80)
    print("  Just type your question to chat")
    print("  'reset'    - Start a new conversation")
    print("  'history'  - Show conversation history")
    print("  'session'  - Show session information")
    print("  'help'     - Show this help message")
    print("  'quit'     - Exit the tester")
    print("=" * 80)


def main():
    tester = ConversationTester()

    print("\n" + "=" * 80)
    print("INTERACTIVE MULTI-TURN CONVERSATION TESTER")
    print("=" * 80)
    print("\nType 'help' for commands or just ask questions")
    print("Examples:")
    print("  - What certifications do I have?")
    print("  - Expiration?")
    print("  - Which one is newest?")

    while True:
        try:
            question = input("\n> ").strip()

            if not question:
                continue

            # Handle commands
            if question.lower() in ["quit", "exit", "q"]:
                print("\n[Goodbye] Thanks for testing!")
                break

            elif question.lower() == "reset":
                tester.reset()
                continue

            elif question.lower() == "history":
                tester.show_history()
                continue

            elif question.lower() == "session":
                tester.show_session()
                continue

            elif question.lower() == "help":
                print_help()
                continue

            # Send chat message
            tester.chat(question)

        except KeyboardInterrupt:
            print("\n\n[Interrupted] Use 'quit' to exit")
            continue
        except EOFError:
            print("\n[Goodbye] Thanks for testing!")
            break


if __name__ == "__main__":
    main()
