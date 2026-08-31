"""create tailored resumes table

Revision ID: 20260830_000005
Revises: 239c004b8aac
Create Date: 2026-08-30
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "20260830_000005"
down_revision = "239c004b8aac"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "tailored_resumes",
        sa.Column(
            "id",
            sa.UUID(),
            nullable=False,
        ),
        sa.Column(
            "job_id",
            sa.UUID(),
            nullable=False,
        ),
        sa.Column(
            "profile_id",
            sa.UUID(),
            nullable=False,
        ),
        sa.Column(
            "status",
            sa.String(),
            server_default="generated",
            nullable=False,
        ),
        sa.Column(
            "structured_content",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
        ),
        sa.Column(
            "generator_provider",
            sa.String(),
            nullable=False,
        ),
        sa.Column(
            "generator_model",
            sa.String(),
            nullable=False,
        ),
        sa.Column(
            "generation_metadata",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("'{}'::jsonb"),
            nullable=False,
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "status IN ('generated')",
            name="ck_tailored_resumes_status",
        ),
        sa.ForeignKeyConstraint(
            ["job_id"],
            ["jobs.id"],
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["profile_id"],
            ["candidate_profiles.id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id"),
    )

    op.create_index(
        "ix_tailored_resumes_job_id",
        "tailored_resumes",
        ["job_id"],
        unique=False,
    )

    op.create_index(
        "ix_tailored_resumes_profile_id",
        "tailored_resumes",
        ["profile_id"],
        unique=False,
    )

    op.create_index(
        "ix_tailored_resumes_job_profile_created",
        "tailored_resumes",
        [
            "job_id",
            "profile_id",
            "created_at",
        ],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index(
        "ix_tailored_resumes_job_profile_created",
        table_name="tailored_resumes",
    )

    op.drop_index(
        "ix_tailored_resumes_profile_id",
        table_name="tailored_resumes",
    )

    op.drop_index(
        "ix_tailored_resumes_job_id",
        table_name="tailored_resumes",
    )

    op.drop_table("tailored_resumes")
