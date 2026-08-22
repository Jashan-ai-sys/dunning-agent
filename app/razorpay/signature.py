import hashlib
import hmac


def compute_signature(raw_body: bytes, secret: str) -> str:
    """HMAC-SHA256 of the *raw* request body, hex-encoded."""
    return hmac.new(secret.encode("utf-8"), raw_body, hashlib.sha256).hexdigest()


def verify_webhook_signature(raw_body: bytes, signature: str | None, secret: str) -> bool:
    """Validate the X-Razorpay-Signature header.

    Razorpay signs the exact bytes it sent, so the body must never be parsed and
    re-serialised before hashing. Comparison is constant-time.
    """
    if not signature or not secret:
        return False
    return hmac.compare_digest(compute_signature(raw_body, secret), signature)
