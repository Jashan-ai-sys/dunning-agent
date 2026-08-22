"""Builders for Razorpay webhook payloads, shaped like the real ones."""

from typing import Any


def payment_entity(
    payment_id: str = "pay_FAIL1",
    *,
    status: str = "failed",
    amount: int = 49900,
    invoice_id: str | None = "inv_1",
    notes: dict | None = None,
) -> dict[str, Any]:
    entity = {
        "id": payment_id,
        "entity": "payment",
        "amount": amount,
        "currency": "INR",
        "status": status,
        "order_id": "order_1",
        "invoice_id": invoice_id,
        "international": False,
        "method": "card",
        "amount_refunded": 0,
        "captured": status == "captured",
        "description": "Monthly subscription",
        "email": "a@example.com",
        "contact": "+919000000000",
        "notes": notes or {},
        "created_at": 1700000000,
    }
    if status == "failed":
        entity.update(
            {
                "error_code": "BAD_REQUEST_ERROR",
                "error_description": "Your card has insufficient funds.",
                "error_source": "issuer",
                "error_step": "payment_authorization",
                "error_reason": "insufficient_funds",
            }
        )
    return entity


def subscription_entity(
    subscription_id: str = "sub_1",
    *,
    status: str = "pending",
    customer_id: str = "cust_1",
    auth_attempts: int = 1,
) -> dict[str, Any]:
    return {
        "id": subscription_id,
        "entity": "subscription",
        "plan_id": "plan_1",
        "customer_id": customer_id,
        "status": status,
        "auth_attempts": auth_attempts,
        "paid_count": 3,
        "total_count": 12,
    }


def invoice_entity(
    invoice_id: str = "inv_1",
    *,
    subscription_id: str | None = "sub_1",
    customer_id: str = "cust_1",
) -> dict[str, Any]:
    return {
        "id": invoice_id,
        "entity": "invoice",
        "subscription_id": subscription_id,
        "customer_id": customer_id,
        "amount": 49900,
        "currency": "INR",
        "status": "issued",
    }


def customer_entity(customer_id: str = "cust_1") -> dict[str, Any]:
    return {
        "id": customer_id,
        "entity": "customer",
        "name": "Asha Rao",
        "email": "asha@example.com",
        "contact": "+919000000000",
    }


def event(event_name: str, payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "entity": "event",
        "account_id": "acc_test",
        "event": event_name,
        "contains": list(payload.keys()),
        "payload": payload,
        "created_at": 1700000000,
    }


def payment_failed_event(**kwargs) -> dict[str, Any]:
    return event("payment.failed", {"payment": {"entity": payment_entity(**kwargs)}})


def payment_captured_event(payment_id: str = "pay_OK1", **kwargs) -> dict[str, Any]:
    return event(
        "payment.captured",
        {"payment": {"entity": payment_entity(payment_id, status="captured", **kwargs)}},
    )


def subscription_pending_event(**kwargs) -> dict[str, Any]:
    return event(
        "subscription.pending",
        {"subscription": {"entity": subscription_entity(status="pending", **kwargs)}},
    )


def subscription_halted_event(**kwargs) -> dict[str, Any]:
    return event(
        "subscription.halted",
        {"subscription": {"entity": subscription_entity(status="halted", **kwargs)}},
    )


def subscription_charged_event(
    payment_id: str = "pay_OK1", subscription_id: str = "sub_1"
) -> dict[str, Any]:
    return event(
        "subscription.charged",
        {
            "subscription": {
                "entity": subscription_entity(subscription_id, status="active")
            },
            "payment": {"entity": payment_entity(payment_id, status="captured")},
        },
    )
