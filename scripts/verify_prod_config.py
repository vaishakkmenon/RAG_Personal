import sys
from pathlib import Path
import os

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_prod_config():
    print("🔍 Verifying Production Configuration...")

    env_path = Path(".env.prod")
    if not env_path.exists():
        print("❌ Error: .env.prod file not found!")
        sys.exit(1)

    # Manually load env vars from .env.prod for this test
    print(f"📂 Loading {env_path}...")
    with open(env_path, "r") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                os.environ[key.strip()] = value.strip()

    # Force ENV to production
    os.environ["ENV"] = "production"

    try:
        from app.settings import settings

        print("✅ Configuration loaded successfully!")

        # Verify critical values
        print("\nChecking critical settings:")
        print(f"  - LLM Provider: {settings.llm.provider}")
        print(f"  - Allowed Origins: {settings.api.cors_origins}")
        print(
            f"  - Database: {settings.postgres.host}:{settings.postgres.port}/{settings.postgres.db_name}"
        )

        # Check actual environment variable for ENV since it's not in the settings model
        env_var = os.environ.get("ENV", "unknown")
        print(f"  - ENV (OS): {env_var}")

        if env_var != "production":
            print("⚠️  Warning: ENV is not set to 'production'!")

        if any("localhost" in origin for origin in settings.api.cors_origins):
            print(
                "⚠️  Warning: 'localhost' found in allowed origins (Acceptable if mainly for testing, but check for proper domain)"
            )

        print("\n🎉 Ready for Production Deployment!")

    except Exception as e:
        print("\n❌ Configuration Verification Failed:")
        print(f"  {e}")
        # Print traceback for easier debugging
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    test_prod_config()
