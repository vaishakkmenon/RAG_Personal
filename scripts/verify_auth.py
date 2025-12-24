import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi.testclient import TestClient
from app.main import app

# Mock environment variables if needed, but we are running against the app instance directly
# which will use the loaded .env or defaults.
# We need to make sure we use the VALID credentials.


def verify_login():
    print("🔐 Verifying Authentication Flow...")

    client = TestClient(app)

    # Get credentials (same logic as create_admin.py)
    username = os.getenv("ADMIN_USER", "admin")
    password = os.getenv("ADMIN_PASSWORD")

    if not password:
        print("❌ ADMIN_PASSWORD not set in environment. Cannot verify login.")
        return

    print(f"DTO: Attempting login for '{username}'...")

    response = client.post(
        "/auth/token", data={"username": username, "password": password}
    )

    if response.status_code == 200:
        token = response.json().get("access_token")
        if token:
            print("✅ Login Successful! JWT Token received.")
            print(f"   Token: {token[:15]}...")

            # Verify protected route
            print("🛡️  Testing protected route /auth/users/me...")
            headers = {"Authorization": f"Bearer {token}"}
            me_response = client.get("/auth/users/me", headers=headers)

            if me_response.status_code == 200:
                user_data = me_response.json()
                print(f"✅ Protected route accessed. User: {user_data['username']}")
            else:
                print(
                    f"❌ Failed to access protected route. Status: {me_response.status_code}"
                )
                print(me_response.json())
        else:
            print("❌ Login succeeded but no token returned.")
    else:
        print(f"❌ Login Failed. Status: {response.status_code}")
        print(response.json())


if __name__ == "__main__":
    verify_login()
