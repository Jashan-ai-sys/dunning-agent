"""Pure unit tests for webhook signature verification -- no database needed."""

from app.razorpay.signature import compute_signature, verify_webhook_signature

SECRET = "test_webhook_secret"
BODY = b'{"event":"payment.failed","payload":{}}'


def test_accepts_matching_signature():
    assert verify_webhook_signature(BODY, compute_signature(BODY, SECRET), SECRET)


def test_rejects_tampered_body():
    signature = compute_signature(BODY, SECRET)
    assert not verify_webhook_signature(BODY + b" ", signature, SECRET)


def test_rejects_wrong_secret():
    assert not verify_webhook_signature(BODY, compute_signature(BODY, "other"), SECRET)


def test_rejects_missing_signature():
    assert not verify_webhook_signature(BODY, None, SECRET)
    assert not verify_webhook_signature(BODY, "", SECRET)


def test_rejects_unconfigured_secret():
    """An empty secret must never validate -- otherwise a misconfigured deploy
    would accept unsigned traffic."""
    assert not verify_webhook_signature(BODY, compute_signature(BODY, ""), "")


def test_signature_covers_exact_bytes():
    """Re-serialising the JSON changes the bytes and must invalidate the hash."""
    reserialised = b'{"event": "payment.failed", "payload": {}}'
    assert compute_signature(BODY, SECRET) != compute_signature(reserialised, SECRET)
