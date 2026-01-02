"""Initial migration - Create users and feedback tables

Revision ID: 001
Revises:
Create Date: 2026-01-02

This migration creates the initial database schema from existing models.
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "001_initial"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Create users and users_feedback tables."""
    # Create users table
    op.create_table(
        "users",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("username", sa.String(), nullable=False),
        sa.Column("email", sa.String(), nullable=False),
        sa.Column("hashed_password", sa.String(), nullable=False),
        sa.Column("is_active", sa.Boolean(), default=True),
        sa.Column("is_superuser", sa.Boolean(), default=False),
        sa.Column("created_at", sa.DateTime(), default=sa.func.now()),
    )
    op.create_index("ix_users_username", "users", ["username"], unique=True)
    op.create_index("ix_users_email", "users", ["email"], unique=True)

    # Create users_feedback table
    op.create_table(
        "users_feedback",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("session_id", sa.String(), nullable=False),
        sa.Column("message_id", sa.String(), nullable=False),
        sa.Column("thumbs_up", sa.Boolean(), nullable=False),
        sa.Column("comment", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(), default=sa.func.now()),
    )
    op.create_index("ix_users_feedback_session_id", "users_feedback", ["session_id"])
    op.create_index("ix_users_feedback_message_id", "users_feedback", ["message_id"])


def downgrade() -> None:
    """Drop users and users_feedback tables."""
    op.drop_index("ix_users_feedback_message_id", table_name="users_feedback")
    op.drop_index("ix_users_feedback_session_id", table_name="users_feedback")
    op.drop_table("users_feedback")

    op.drop_index("ix_users_email", table_name="users")
    op.drop_index("ix_users_username", table_name="users")
    op.drop_table("users")
