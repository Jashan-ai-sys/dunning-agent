"""customer last contacted at

Revision ID: b8e4f1a27c60
Revises: d70c8b3f912e
Create Date: 2026-08-28 15:30:00.000000

"""
from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = 'b8e4f1a27c60'
down_revision: str | None = 'd70c8b3f912e'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        'customers', sa.Column('last_contacted_at', sa.DateTime(timezone=True), nullable=True)
    )
    # Backfill from the calls we have already placed, so the cooldown does not
    # start from zero for customers we spoke to before the column existed.
    op.execute(
        """
        UPDATE customers AS c
           SET last_contacted_at = latest.at
          FROM (
                SELECT rc.razorpay_customer_id AS cid, MAX(vc.created_at) AS at
                  FROM voice_calls vc
                  JOIN recovery_cases rc ON rc.id = vc.recovery_case_id
                 WHERE rc.razorpay_customer_id IS NOT NULL
                 GROUP BY rc.razorpay_customer_id
               ) AS latest
         WHERE c.razorpay_customer_id = latest.cid
        """
    )


def downgrade() -> None:
    op.drop_column('customers', 'last_contacted_at')
