
import os
import pytest
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
        **headers_dict
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
        **headers_dict
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
        **headers_dict
    }
    
    response = client.options("/chat", headers=headers)
    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == origin
