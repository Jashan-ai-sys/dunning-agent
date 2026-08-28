"""case next eligible at

Revision ID: a7c5e214d80b
Revises: f18d472ac903
Create Date: 2026-08-27 08:45:00.000000

"""
from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = 'a7c5e214d80b'
down_revision: str | None = 'f18d472ac903'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        'recovery_cases', sa.Column('next_eligible_at', sa.DateTime(timezone=True), nullable=True)
    )
    op.create_index(
        op.f('ix_recovery_cases_next_eligible_at'),
        'recovery_cases',
        ['next_eligible_at'],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index(op.f('ix_recovery_cases_next_eligible_at'), table_name='recovery_cases')
    op.drop_column('recovery_cases', 'next_eligible_at')
