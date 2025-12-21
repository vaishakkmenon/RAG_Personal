"""
Database initialization script.
Run this before starting the application to ensure tables exist.
"""

import logging
from app.database import engine, Base

# Import all SQL models here to ensure they are registered with Base metadata
from app.models.sql_feedback import Feedback  # noqa

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def init_db():
    logger.info("Initializing database...")
    try:
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables created successfully!")
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        # Re-raise to stop startup on critical DB failure
        raise e


if __name__ == "__main__":
    init_db()
