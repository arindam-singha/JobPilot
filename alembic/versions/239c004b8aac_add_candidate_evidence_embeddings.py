"""add candidate evidence embeddings

Revision ID: 239c004b8aac
Revises: 8fe0e5d78fe7
Create Date: 2026-08-28 18:40:20.202603
"""

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector

revision = "239c004b8aac"
down_revision = "8fe0e5d78fe7"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    # op.add_column(
    #     "candidate_evidence",
    #     sa.Column(
    #         "embedding",
    #         Vector(768),
    #         nullable=True,
    #     ),
    # )

    op.add_column(
        "candidate_evidence",
        sa.Column(
            "embedding",
            Vector(768),
            nullable=True,
        ),
    )

    op.add_column(
        "candidate_evidence",
        sa.Column(
            "embedding_provider",
            sa.String(),
            nullable=True,
        ),
    )

    op.add_column(
        "candidate_evidence",
        sa.Column(
            "embedding_model",
            sa.String(),
            nullable=True,
        ),
    )

    op.add_column(
        "candidate_evidence",
        sa.Column(
            "embedded_at",
            sa.DateTime(timezone=True),
            nullable=True,
        ),
    )

    op.execute(
        """
        CREATE INDEX ix_candidate_evidence_embedding_hnsw
        ON candidate_evidence
        USING hnsw (embedding vector_cosine_ops)
        """
    )


def downgrade() -> None:
    op.execute(
        """
        DROP INDEX IF EXISTS ix_candidate_evidence_embedding_hnsw
        """
    )

    op.drop_column(
        "candidate_evidence",
        "embedded_at",
    )

    op.drop_column(
        "candidate_evidence",
        "embedding_model",
    )

    op.drop_column(
        "candidate_evidence",
        "embedding_provider",
    )

    op.drop_column(
        "candidate_evidence",
        "embedding",
    )