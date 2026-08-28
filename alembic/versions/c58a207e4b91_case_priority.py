"""case priority

Revision ID: c58a207e4b91
Revises: b3e91f6d5a24
Create Date: 2026-08-27 10:15:00.000000

"""
from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = 'c58a207e4b91'
down_revision: str | None = 'b3e91f6d5a24'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column('recovery_cases', sa.Column('failure_step', sa.String(length=64), nullable=True))
    op.add_column(
        'recovery_cases',
        sa.Column('priority_tier', sa.Integer(), server_default='4', nullable=False),
    )
    op.create_index(
        op.f('ix_recovery_cases_priority_tier'), 'recovery_cases', ['priority_tier'], unique=False
    )

    # Backfill from the payments already stored: the failure fields were always
    # in the webhook payload, they simply had nowhere to go on the case.
    op.execute(
        """
        UPDATE recovery_cases AS c
           SET failure_step = p.error_step
          FROM payments AS p
         WHERE p.razorpay_payment_id = c.razorpay_payment_id
           AND c.failure_step IS NULL
        """
    )

    # One-time snapshot of app.priority.tier_from for rows that predate the
    # column. Deliberately duplicated here rather than imported: a migration
    # must keep doing what it did the day it was written, even after the Python
    # rules move on. app.priority stays the source of truth for new cases.
    op.execute(
        """
        UPDATE recovery_cases
           SET priority_tier = CASE
                 WHEN failure_reason_code IN (
                        'card_expired', 'expired_card', 'card_blocked', 'invalid_card',
                        'incorrect_card_details', 'mandate_revoked', 'mandate_cancelled')
                   THEN 1
                 WHEN failure_step IN ('payment_authentication', 'payment_authorization')
                   THEN 2
                 WHEN failure_reason_code IN ('insufficient_funds', 'invalid_otp', 'incorrect_otp')
                   THEN 2
                 WHEN failure_source = 'customer' THEN 2
                 ELSE 4
               END
        """
    )


def downgrade() -> None:
    op.drop_index(op.f('ix_recovery_cases_priority_tier'), table_name='recovery_cases')
    op.drop_column('recovery_cases', 'priority_tier')
    op.drop_column('recovery_cases', 'failure_step')
