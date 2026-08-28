"""dead letter bounds

Revision ID: b3e91f6d5a24
Revises: a7c5e214d80b
Create Date: 2026-08-27 09:30:00.000000

"""
from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = 'b3e91f6d5a24'
down_revision: str | None = 'a7c5e214d80b'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        'webhook_events',
        sa.Column('attempt_count', sa.Integer(), server_default='0', nullable=False),
    )
    op.add_column(
        'webhook_events', sa.Column('dead_at', sa.DateTime(timezone=True), nullable=True)
    )
    op.add_column(
        'recovery_cases',
        sa.Column('delivery_failures', sa.Integer(), server_default='0', nullable=False),
    )


def downgrade() -> None:
    op.drop_column('recovery_cases', 'delivery_failures')
    op.drop_column('webhook_events', 'dead_at')
    op.drop_column('webhook_events', 'attempt_count')
