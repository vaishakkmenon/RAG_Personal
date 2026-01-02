"""
Alembic environment configuration.
Reads database URL from environment and configures migrations.
"""

import sys
from logging.config import fileConfig
from pathlib import Path

from sqlalchemy import engine_from_config, pool

from alembic import context

# Add parent directory to path so we can import app modules
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import our database configuration and models
from app.database import Base, SQLALCHEMY_DATABASE_URL

# Import ALL models here to ensure they are registered with Base.metadata
# This is required for autogenerate to detect them
from app.models.users import User  # noqa: F401
from app.models.sql_feedback import Feedback  # noqa: F401

# This is the Alembic Config object
config = context.config

# Override sqlalchemy.url with our actual database URL from settings
config.set_main_option("sqlalchemy.url", SQLALCHEMY_DATABASE_URL)

# Interpret the config file for Python logging
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Target metadata for 'autogenerate' support
target_metadata = Base.metadata


def run_migrations_offline() -> None:
    """
    Run migrations in 'offline' mode.

    This configures the context with just a URL and not an Engine,
    though an Engine is acceptable here as well. By skipping the Engine
    creation we don't even need a DBAPI to be available.

    Calls to context.execute() here emit the given string to the
    script output.
    """
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,  # Detect column type changes
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """
    Run migrations in 'online' mode.

    In this scenario we need to create an Engine and associate
    a connection with the context.
    """
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            compare_type=True,  # Detect column type changes
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
