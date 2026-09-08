"""create candidate evidence table

Revision ID: 20260824_000004
Revises: 20260816_000003
Create Date: 2026-08-24 00:00:00.000000
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import UUID

revision = "20260824_000004"
down_revision = "20260816_000003"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "candidate_evidence",
        sa.Column(
            "id",
            UUID(as_uuid=True),
            primary_key=True,
            nullable=False,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column(
            "profile_id",
            UUID(as_uuid=True),
            sa.ForeignKey(
                "candidate_profiles.id",
                ondelete="CASCADE",
            ),
            nullable=False,
        ),
        sa.Column(
            "evidence_type",
            sa.String(),
            nullable=False,
        ),
        sa.Column(
            "title",
            sa.String(),
            nullable=False,
        ),
        sa.Column(
            "content",
            sa.Text(),
            nullable=False,
        ),
        sa.Column(
            "source_type",
            sa.String(),
            nullable=False,
        ),
        sa.Column(
            "source_id",
            UUID(as_uuid=True),
            nullable=True,
        ),
        sa.Column(
            "metadata_json",
            sa.Text(),
            nullable=True,
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
    )

    op.create_index(
        "ix_candidate_evidence_profile_id",
        "candidate_evidence",
        ["profile_id"],
    )

    op.create_index(
        "ix_candidate_evidence_evidence_type",
        "candidate_evidence",
        ["evidence_type"],
    )


def downgrade() -> None:
    op.drop_index(
        "ix_candidate_evidence_evidence_type",
        table_name="candidate_evidence",
    )

    op.drop_index(
        "ix_candidate_evidence_profile_id",
        table_name="candidate_evidence",
    )

    op.drop_table("candidate_evidence")