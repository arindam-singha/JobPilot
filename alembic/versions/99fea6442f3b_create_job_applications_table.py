"""create job applications table

Revision ID: 99fea6442f3b
Revises: 20260830_000005
Create Date: 2026-09-08
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision = "99fea6442f3b"
down_revision = "20260830_000005"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "job_applications",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            nullable=False,
        ),
        sa.Column(
            "job_id",
            postgresql.UUID(as_uuid=True),
            nullable=False,
        ),
        sa.Column(
            "profile_id",
            postgresql.UUID(as_uuid=True),
            nullable=False,
        ),
        sa.Column(
            "status",
            sa.String(),
            server_default="draft",
            nullable=False,
        ),
        sa.Column(
            "resume_snapshot",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=True,
        ),
        sa.Column(
            "cover_letter_snapshot",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=True,
        ),
        sa.Column(
            "application_url",
            sa.String(),
            nullable=True,
        ),
        sa.Column(
            "notes",
            sa.Text(),
            nullable=True,
        ),
        sa.Column(
            "applied_at",
            sa.DateTime(timezone=True),
            nullable=True,
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "status IN ("
            "'draft', "
            "'ready_to_apply', "
            "'applied', "
            "'screening', "
            "'interview', "
            "'offer', "
            "'rejected', "
            "'withdrawn'"
            ")",
            name="ck_job_applications_status",
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
        sa.UniqueConstraint(
            "job_id",
            "profile_id",
            name="uq_job_applications_job_profile",
        ),
    )

    op.create_index(
        "ix_job_applications_job_id",
        "job_applications",
        ["job_id"],
        unique=False,
    )
    op.create_index(
        "ix_job_applications_profile_id",
        "job_applications",
        ["profile_id"],
        unique=False,
    )
    op.create_index(
        "ix_job_applications_profile_status",
        "job_applications",
        ["profile_id", "status"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index(
        "ix_job_applications_profile_status",
        table_name="job_applications",
    )
    op.drop_index(
        "ix_job_applications_profile_id",
        table_name="job_applications",
    )
    op.drop_index(
        "ix_job_applications_job_id",
        table_name="job_applications",
    )
    op.drop_table("job_applications")