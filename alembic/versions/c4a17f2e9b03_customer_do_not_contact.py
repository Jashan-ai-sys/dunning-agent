"""customer do not contact

Revision ID: c4a17f2e9b03
Revises: bf75cb885a95
Create Date: 2026-08-27 06:40:00.000000

"""
from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = 'c4a17f2e9b03'
down_revision: str | None = 'bf75cb885a95'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        'customers',
        sa.Column('do_not_contact', sa.Boolean(), server_default='false', nullable=False),
    )
    op.add_column('customers', sa.Column('do_not_contact_reason', sa.String(length=64), nullable=True))
    op.add_column(
        'customers',
        sa.Column('do_not_contact_at', sa.DateTime(timezone=True), nullable=True),
    )


def downgrade() -> None:
    op.drop_column('customers', 'do_not_contact_at')
    op.drop_column('customers', 'do_not_contact_reason')
    op.drop_column('customers', 'do_not_contact')
