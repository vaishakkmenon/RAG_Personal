#!/usr/bin/env python3
"""
Realistic Load Testing Script for RAG API

Designed to work within Groq's rate limits:
- 28 requests/minute (~0.47 req/sec)
- Tests realistic production scenarios
- Measures latency, throughput, success rate

Requires: pip install httpx
"""

import asyncio
import httpx
import time
import statistics
import sys
from typing import List, Dict
from datetime import datetime

# Configuration
API_URL = "http://localhost:8000"
API_KEY = "d5160646e4199a5d88ea3626a3795e4139eef33adb29a56568b4b52bcbe703d5"

# Test questions simulating realistic user queries
TEST_QUESTIONS = [
    "What is your educational background?",
    "Tell me about your work experience",
    "What programming languages do you know?",
    "Do you have any certifications?",
    "What AI/ML projects have you worked on?",
    "Tell me about your experience with Python",
    "What is your GPA?",
    "Have you worked with Kubernetes?",
    "What databases have you used?",
    "Describe your experience with cloud platforms",
]


async def send_request(
    client: httpx.AsyncClient,
    session_id: str,
    question: str,
    request_num: int
) -> Dict:
    """Send a single chat request and measure performance."""
    start = time.time()
    try:
        response = await client.post(
            f"{API_URL}/chat",
            json={"question": question, "session_id": session_id},
            headers={"X-API-Key": API_KEY},
            timeout=30.0,
        )
        latency = time.time() - start

        return {
            "success": response.status_code == 200,
            "latency": latency,
            "status": response.status_code,
            "request_num": request_num,
            "question": question[:50],  # Truncate for display
        }
    except Exception as e:
        latency = time.time() - start
        return {
            "success": False,
            "latency": latency,
            "error": str(e),
            "request_num": request_num,
            "question": question[:50],
        }


async def scenario_1_sequential_baseline(client: httpx.AsyncClient):
    """
    Scenario 1: Sequential Baseline
    - 1 user sending requests sequentially
    - 10 requests over ~60 seconds (respects rate limit)
    - Measures baseline latency without concurrency
    """
    print("\n" + "="*70)
    print("SCENARIO 1: Sequential Baseline (1 user, 10 requests)")
    print("="*70)

    results = []
    session_id = "load-test-sequential"

    start_time = time.time()

    for i in range(10):
        question = TEST_QUESTIONS[i % len(TEST_QUESTIONS)]
        print(f"  [{i+1}/10] Sending: {question[:50]}...")

        result = await send_request(client, session_id, question, i+1)
        results.append(result)

        if result["success"]:
            print(f"    OK Success ({result['latency']:.2f}s)")
        else:
            print(f"    X Failed: {result.get('error', 'Unknown')}")

        # Wait 3 seconds between requests to stay well within rate limit
        if i < 9:  # Don't wait after last request
            await asyncio.sleep(3)

    total_time = time.time() - start_time
    print_results("Sequential Baseline", results, total_time)
    return results


async def scenario_2_low_concurrency(client: httpx.AsyncClient):
    """
    Scenario 2: Low Concurrency
    - 2 concurrent users
    - 5 requests per user = 10 total requests
    - Requests spread over 30 seconds
    - Tests realistic multi-user scenario within rate limits
    """
    print("\n" + "="*70)
    print("SCENARIO 2: Low Concurrency (2 users, 5 requests each)")
    print("="*70)

    async def user_session(user_id: int, num_requests: int):
        """Simulate one user's session"""
        session_id = f"load-test-user-{user_id}"
        user_results = []

        for i in range(num_requests):
            question = TEST_QUESTIONS[(user_id * num_requests + i) % len(TEST_QUESTIONS)]
            result = await send_request(client, session_id, question, i+1)
            user_results.append(result)

            # Wait 6 seconds between requests (10 req/min = well within limit)
            if i < num_requests - 1:
                await asyncio.sleep(6)

        return user_results

    start_time = time.time()

    # Run 2 users concurrently
    user_tasks = [
        user_session(user_id=0, num_requests=5),
        user_session(user_id=1, num_requests=5),
    ]

    user_results_list = await asyncio.gather(*user_tasks)
    results = [r for user_results in user_results_list for r in user_results]

    total_time = time.time() - start_time
    print_results("Low Concurrency", results, total_time)
    return results


async def scenario_3_burst_test(client: httpx.AsyncClient):
    """
    Scenario 3: Burst Test
    - Send 5 requests as fast as possible
    - Tests caching, queueing, rate limiter behavior
    - Will likely hit rate limiter - expected behavior
    """
    print("\n" + "="*70)
    print("SCENARIO 3: Burst Test (5 rapid requests)")
    print("="*70)
    print("  Note: This will likely hit rate limiter - testing queue behavior")

    session_id = "load-test-burst"
    tasks = []

    # Send 5 requests concurrently
    for i in range(5):
        question = TEST_QUESTIONS[i % len(TEST_QUESTIONS)]
        tasks.append(send_request(client, session_id, question, i+1))

    start_time = time.time()
    results = await asyncio.gather(*tasks)
    total_time = time.time() - start_time

    print_results("Burst Test", results, total_time)
    return results


def print_results(scenario_name: str, results: List[Dict], total_time: float):
    """Print formatted test results."""
    successes = [r for r in results if r["success"]]
    failures = [r for r in results if not r["success"]]
    latencies = [r["latency"] for r in results]

    print(f"\n{scenario_name} Results:")
    print("-" * 70)
    print(f"  Total requests:  {len(results)}")
    print(f"  Successful:      {len(successes)} ({len(successes)/len(results)*100:.1f}%)")
    print(f"  Failed:          {len(failures)} ({len(failures)/len(results)*100:.1f}%)")

    if successes:
        success_latencies = [r["latency"] for r in successes]
        print(f"\n  Performance:")
        print(f"    Total time:    {total_time:.2f}s")
        print(f"    Requests/sec:  {len(results)/total_time:.2f}")

        print(f"\n  Latency (successful requests):")
        print(f"    Mean:          {statistics.mean(success_latencies):.2f}s")
        print(f"    Median:        {statistics.median(success_latencies):.2f}s")
        if len(success_latencies) > 1:
            sorted_lat = sorted(success_latencies)
            p95_idx = int(len(sorted_lat) * 0.95)
            p99_idx = int(len(sorted_lat) * 0.99)
            print(f"    P95:           {sorted_lat[p95_idx]:.2f}s")
            if len(sorted_lat) > 10:
                print(f"    P99:           {sorted_lat[p99_idx]:.2f}s")
        print(f"    Min:           {min(success_latencies):.2f}s")
        print(f"    Max:           {max(success_latencies):.2f}s")

    if failures:
        print(f"\n  Failures (first 3):")
        for f in failures[:3]:
            error = f.get('error', f.get('status', 'Unknown'))
            print(f"    - Request {f['request_num']}: {error}")


async def run_all_scenarios():
    """Run all load test scenarios."""
    print("\n" + "="*70)
    print("RAG API Load Testing - Realistic Scenarios")
    print(f"Target: {API_URL}")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)

    async with httpx.AsyncClient() as client:
        # Test API availability
        try:
            response = await client.get(f"{API_URL}/health", timeout=5.0)
            if response.status_code != 200:
                print(f"\nX API is not healthy (status: {response.status_code})")
                print("Please ensure the API is running: docker compose up")
                return
        except Exception as e:
            print(f"\nX Cannot connect to API at {API_URL}")
            print(f"Error: {e}")
            print("\nPlease ensure the API is running: docker compose up")
            return

        print("OK API is healthy and responding")

        # Run scenarios
        all_results = []

        # Scenario 1: Sequential baseline
        results_1 = await scenario_1_sequential_baseline(client)
        all_results.extend(results_1)

        await asyncio.sleep(5)  # Brief pause between scenarios

        # Scenario 2: Low concurrency
        results_2 = await scenario_2_low_concurrency(client)
        all_results.extend(results_2)

        await asyncio.sleep(5)  # Brief pause between scenarios

        # Scenario 3: Burst test
        results_3 = await scenario_3_burst_test(client)
        all_results.extend(results_3)

        # Overall summary
        print("\n" + "="*70)
        print("OVERALL SUMMARY - All Scenarios")
        print("="*70)

        total_requests = len(all_results)
        total_successes = len([r for r in all_results if r["success"]])
        success_rate = (total_successes / total_requests * 100) if total_requests > 0 else 0

        print(f"  Total requests:      {total_requests}")
        print(f"  Total successful:    {total_successes}")
        print(f"  Overall success rate: {success_rate:.1f}%")

        # Performance targets check
        print(f"\n  Performance Target Validation:")

        successful = [r for r in all_results if r["success"]]
        if successful:
            latencies = [r["latency"] for r in successful]
            mean_lat = statistics.mean(latencies)
            sorted_lat = sorted(latencies)
            p95_lat = sorted_lat[int(len(sorted_lat) * 0.95)]

            # Check against targets
            mean_ok = "OK" if mean_lat < 2.0 else "X"
            p95_ok = "OK" if p95_lat < 5.0 else "X"
            success_ok = "OK" if success_rate > 99.0 else "X"

            print(f"    {mean_ok} Mean latency <2s:     {mean_lat:.2f}s")
            print(f"    {p95_ok} P95 latency <5s:      {p95_lat:.2f}s")
            print(f"    {success_ok} Success rate >99%:  {success_rate:.1f}%")

        print("\n" + "="*70)
        print("Load testing complete!")
        print("="*70 + "\n")


if __name__ == "__main__":
    # Check if API key is set
    if API_KEY == "your-dev-api-key-here":
        print("\nX Error: Please set your API_KEY in the script")
        print("   Edit scripts/load_test.py and replace API_KEY value")
        sys.exit(1)

    try:
        asyncio.run(run_all_scenarios())
    except KeyboardInterrupt:
        print("\n\nLoad testing interrupted by user")
    except Exception as e:
        print(f"\nX Error during load testing: {e}")
        import traceback
        traceback.print_exc()
