"""case failure diagnosis fields

Revision ID: d92b6c1f4e77
Revises: c4a17f2e9b03
Create Date: 2026-08-27 07:05:00.000000

"""
from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = 'd92b6c1f4e77'
down_revision: str | None = 'c4a17f2e9b03'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column('recovery_cases', sa.Column('failure_source', sa.String(length=32), nullable=True))
    op.add_column(
        'recovery_cases', sa.Column('failure_reason_code', sa.String(length=64), nullable=True)
    )


def downgrade() -> None:
    op.drop_column('recovery_cases', 'failure_reason_code')
    op.drop_column('recovery_cases', 'failure_source')
