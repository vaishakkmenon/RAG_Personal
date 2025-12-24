import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi.testclient import TestClient
from app.main import app


def verify_security():
    print("🛡️  Verifying Endpoint Security...")
    client = TestClient(app)

    # 1. Verify Public Access (Should Succeed)
    print("\n🌍 Testing Public Endpoints:")

    # Health
    resp = client.get("/health")
    if resp.status_code == 200:
        print("   ✅ /health is accessible (200 OK)")
    else:
        print(f"   ❌ /health failed ({resp.status_code})")

    # 2. Verify Protected Access (Should Fail without token)
    print("\n🔒 Testing Protected Endpoints (No Auth):")

    protected_routes = [
        ("GET", "/debug/search?q=test"),
        ("DELETE", "/admin/chromadb"),
        ("POST", "/ingest"),  # POST requires body usually, but 401 should happen first
    ]

    for method, route in protected_routes:
        if method == "GET":
            resp = client.get(route)
        elif method == "DELETE":
            resp = client.delete(route)
        elif method == "POST":
            resp = client.post(route, json={"paths": []})

        if resp.status_code == 401:
            print(f"   ✅ {route} blocked (401 Unauthorized)")
        else:
            print(f"   ❌ {route} allowed or wrong error ({resp.status_code})")

    # 3. Verify Login & Authenticated Access
    print("\n🔑 Testing Authenticated Access:")

    username = os.getenv("ADMIN_USER", "admin")
    password = os.getenv("ADMIN_PASSWORD")

    if not password:
        # Try sourcing from .env directly roughly if not in env
        print("   ⚠️  ADMIN_PASSWORD not in env, skipping authenticated test.")
        return

    login_resp = client.post(
        "/auth/token", data={"username": username, "password": password}
    )
    if login_resp.status_code != 200:
        print(f"   ❌ Login failed ({login_resp.status_code})")
        return

    token = login_resp.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    # Test debug route with token
    resp = client.get("/debug/search?q=test", headers=headers)
    if resp.status_code == 200:
        print("   ✅ Authenticated access to /debug/search successful")
    else:
        print(f"   ❌ Authenticated access failed ({resp.status_code})")


if __name__ == "__main__":
    verify_security()
