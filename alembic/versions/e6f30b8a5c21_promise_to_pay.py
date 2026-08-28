"""promise to pay

Revision ID: e6f30b8a5c21
Revises: d92b6c1f4e77
Create Date: 2026-08-27 07:40:00.000000

"""
from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = 'e6f30b8a5c21'
down_revision: str | None = 'd92b6c1f4e77'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        'recovery_cases', sa.Column('promised_at', sa.DateTime(timezone=True), nullable=True)
    )
    op.add_column(
        'recovery_cases', sa.Column('promise_due_at', sa.DateTime(timezone=True), nullable=True)
    )


def downgrade() -> None:
    op.drop_column('recovery_cases', 'promise_due_at')
    op.drop_column('recovery_cases', 'promised_at')
