"""customer phone is wrong

Revision ID: f18d472ac903
Revises: e6f30b8a5c21
Create Date: 2026-08-27 08:20:00.000000

"""
from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = 'f18d472ac903'
down_revision: str | None = 'e6f30b8a5c21'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        'customers',
        sa.Column('phone_is_wrong', sa.Boolean(), server_default='false', nullable=False),
    )


def downgrade() -> None:
    op.drop_column('customers', 'phone_is_wrong')
