"""checkout cases

Revision ID: d70c8b3f912e
Revises: c58a207e4b91
Create Date: 2026-08-27 11:00:00.000000

"""
from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = 'd70c8b3f912e'
down_revision: str | None = 'c58a207e4b91'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        'recovery_cases', sa.Column('razorpay_order_id', sa.String(length=64), nullable=True)
    )
    op.create_index(
        op.f('ix_recovery_cases_razorpay_order_id'),
        'recovery_cases',
        ['razorpay_order_id'],
        unique=False,
    )
    op.add_column(
        'payments', sa.Column('razorpay_order_id', sa.String(length=64), nullable=True)
    )
    op.create_index(
        op.f('ix_payments_razorpay_order_id'), 'payments', ['razorpay_order_id'], unique=False
    )
    # No backfill: the order id was never stored on either table, so there is
    # nothing on disk to copy from. Existing rows stay null and only new
    # payments carry it.


def downgrade() -> None:
    op.drop_index(op.f('ix_payments_razorpay_order_id'), table_name='payments')
    op.drop_column('payments', 'razorpay_order_id')
    op.drop_index(op.f('ix_recovery_cases_razorpay_order_id'), table_name='recovery_cases')
    op.drop_column('recovery_cases', 'razorpay_order_id')
