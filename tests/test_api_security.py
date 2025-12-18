from fastapi.testclient import TestClient
from app.main import app
from app.settings import settings

client = TestClient(app)


def test_security_headers():
    """Test that security headers are present in responses."""
    # Authenticated request to get 200 OK (or whatever status)
    # The health endpoint might be protected depending on router config, usually it's open but tests showed 401.
    # We'll use the API key just in case.
    # Authenticated request to get 200 OK
    headers_dict = {"X-API-Key": settings.api_key}
    response = client.get("/health", headers=headers_dict)

    # Even if 404, headers should be present. But 401 might not reach middleware if raised early?
    # Actually BaseHTTPMiddleware wraps everything.
    # But let's aim for a known valid response.

    headers = response.headers
    assert headers["X-Content-Type-Options"] == "nosniff"
    assert headers["X-Frame-Options"] == "DENY"
    assert headers["X-XSS-Protection"] == "1; mode=block"
    assert headers["Referrer-Policy"] == "strict-origin-when-cross-origin"


def test_cors_valid_origin():
    """Test CORS with a valid origin."""
    # Mock settings.api.cors_origins to include a specific origin
    # However settings are already loaded. We can check against default allowed origins.
    # defaults include http://localhost:3000

    origin = "http://localhost:3000"
    headers_dict = {"X-API-Key": settings.api_key}
    headers = {
        "Origin": origin,
        "Access-Control-Request-Method": "POST",
        "Access-Control-Request-Headers": "Content-Type, X-API-Key",
        **headers_dict,
    }

    response = client.options("/chat", headers=headers)
    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == origin


def test_cors_invalid_origin():
    """Test CORS with an invalid origin."""
    origin = "http://evil.com"
    headers_dict = {"X-API-Key": settings.api_key}
    headers = {
        "Origin": origin,
        "Access-Control-Request-Method": "POST",
        "Access-Control-Request-Headers": "Content-Type, X-API-Key",
        **headers_dict,
    }

    # TestClient doesn't enforce CORS blocking (browser does), but it receives the headers.
    # If origin is not allowed, Access-Control-Allow-Origin should NOT be present
    # OR it should not match the requested origin.

    response = client.options("/chat", headers=headers)

    # Default behavior of CORSMiddleware: if origin not allowed, it doesn't send ACAO header
    assert "access-control-allow-origin" not in response.headers


def test_cors_deploy_preview_regex():
    """Test CORS with a deploy preview regex match."""
    origin = "https://deploy-preview-123.vaishakmenon.com"
    headers_dict = {"X-API-Key": settings.api_key}
    headers = {
        "Origin": origin,
        "Access-Control-Request-Method": "POST",
        **headers_dict,
    }

    response = client.options("/chat", headers=headers)
    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == origin


def test_xss_payload_handling():
    """Test that XSS payloads in responses are handled safely."""
    xss_payloads = [
        "<script>alert('xss')</script>",
        "<img src=x onerror=alert('xss')>",
        "javascript:alert('xss')",
        "<svg onload=alert('xss')>",
    ]

    for payload in xss_payloads:
        response = client.post(
            "/chat",
            json={"question": payload, "session_id": "xss-test"},
            headers={"X-API-Key": settings.api_key},
        )

        # Response should either succeed or be caught by prompt guard
        assert response.status_code in [
            200,
            400,
            403,
        ], f"Unexpected status {response.status_code} for XSS payload"

        # If successful, response should not contain unescaped HTML
        if response.status_code == 200:
            response_text = response.json().get("response", "")
            # Response should not contain raw script tags or event handlers
            assert (
                "<script>" not in response_text.lower()
            ), "Response contains unescaped <script> tag"
            assert (
                "onerror=" not in response_text.lower()
            ), "Response contains unescaped event handler"


def test_oversized_request_rejected():
    """Test that excessively large requests are rejected."""
    # Create a 100KB message (exceeds typical limits)
    huge_message = "x" * 100000

    response = client.post(
        "/chat",
        json={"question": huge_message, "session_id": "size-test"},
        headers={"X-API-Key": settings.api_key},
    )

    # Should be rejected with 413 (Payload Too Large) or 422 (Validation Error)
    assert response.status_code in [
        413,
        422,
    ], f"Oversized request got status {response.status_code}, expected 413 or 422"


def test_rapid_fire_rate_limiting(monkeypatch):
    """Test that rapid-fire requests trigger rate limiting."""
    # Temporarily set a lower rate limit for testing (15 queries per hour)
    monkeypatch.setattr(settings.session, "queries_per_hour", 15)

    session_id = "rapid-fire-test"

    # Send 20 requests (more than the 15 allowed)
    responses = []
    for i in range(20):
        response = client.post(
            "/chat",
            json={"question": f"Test question {i}", "session_id": session_id},
            headers={"X-API-Key": settings.api_key},
        )
        responses.append(response)

    # Count successful and rate-limited responses
    successful = [r for r in responses if r.status_code == 200]
    rate_limited = [r for r in responses if r.status_code == 429]

    # Expect rate limiting to kick in after 15 requests
    assert len(rate_limited) > 0, (
        f"Expected rate limiting after 15 requests, but all {len(responses)} succeeded. "
        f"Got {len(successful)} successful, {len(rate_limited)} rate limited"
    )
