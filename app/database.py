"""
Database connection configuration using SQLAlchemy (Sync).
"""

import logging
from typing import Generator

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, declarative_base, Session

from app.settings import settings

logger = logging.getLogger(__name__)

# Construct connection URL
# Use psycopg2 binary driver
SQLALCHEMY_DATABASE_URL = (
    f"postgresql+psycopg2://{settings.postgres.user}:{settings.postgres.password}"
    f"@{settings.postgres.host}:{settings.postgres.port}/{settings.postgres.db_name}"
)

# Create engine
# pool_pre_ping=True handles stale connections
engine = create_engine(
    SQLALCHEMY_DATABASE_URL, pool_pre_ping=True, pool_size=5, max_overflow=10
)

# Session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Declarative base
Base = declarative_base()


def get_db() -> Generator[Session, None, None]:
    """Dependency for getting DB session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def check_db_connectivity() -> bool:
    """Check if database is reachable."""
    try:
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
        return True
    except Exception as e:
        logger.error(f"Database connection failed: {e}")
        return False
