import sys
from pathlib import Path
import os

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.database import SessionLocal, engine, Base
from app.models.users import User
from app.core.auth import get_password_hash


def create_admin_user():
    print("🚀 Initializing Admin User...")

    # Create tables if they don't exist
    Base.metadata.create_all(bind=engine)

    db = SessionLocal()
    try:
        # Check if admin already exists
        # Use os.getenv directly as these are not in the main Settings model
        admin_username = os.getenv("ADMIN_USER", "admin")

        existing_user = db.query(User).filter(User.username == admin_username).first()

        # Create new admin
        admin_password = os.getenv("ADMIN_PASSWORD")
        if not admin_password:
            print("❌ Error: ADMIN_PASSWORD not set.")
            sys.exit(1)

        hashed_pwd = get_password_hash(admin_password)

        if existing_user:
            print(f"🔄 Admin user '{admin_username}' exists. Updating password...")
            existing_user.hashed_password = hashed_pwd
            db.commit()
            print(f"✅ Password updated for user: {admin_username}")
            return

        print(f"👤 Creating admin user: {admin_username}")
        new_user = User(
            username=admin_username,
            email=f"{admin_username}@example.com",  # Placeholder
            hashed_password=hashed_pwd,
            is_active=True,
            is_superuser=True,
        )

        db.add(new_user)
        db.commit()
        db.refresh(new_user)
        print(f"✅ Successfully created admin user: {new_user.username}")

    except Exception as e:
        print(f"❌ Error creating admin user: {e}")
        db.rollback()
    finally:
        db.close()


if __name__ == "__main__":
    create_admin_user()
