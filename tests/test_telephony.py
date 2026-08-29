"""TwilioChannel guards, TwiML shape and channel selection. No network involved."""

from xml.etree import ElementTree

import pytest

from app.channels import LoggingChannel, build_channel
from app.config import Settings
from app.models import Customer, RecoveryCase
from app.voice.pipecat_agent import telephony_body, telephony_provider
from app.voice.telephony import MAX_TWIML_CHARS, TwilioChannel, stream_parameters

CONFIGURED = {
    "twilio_account_sid": "ACtest",
    "twilio_auth_token": "token",
    "twilio_from_number": "+15005550006",
    "twilio_stream_url": "wss://example.ngrok.app/ws",
    "company_name": "Acme",
}


def configure(monkeypatch, **overrides) -> Settings:
    settings = Settings(**{**CONFIGURED, **overrides})
    monkeypatch.setattr("app.voice.telephony.get_settings", lambda: settings)
    return settings


def make_case(**kwargs) -> RecoveryCase:
    defaults = {
        "id": 7,
        "razorpay_payment_id": "pay_1",
        "original_amount": 249_900,
        "failure_reason": "card expired",
    }
    return RecoveryCase(**{**defaults, **kwargs})


def make_customer(**kwargs) -> Customer:
    defaults = {
        "razorpay_customer_id": "cust_1",
        "name": "Asha Rao",
        "phone": "+919000000000",
        "preferred_language": "hi",
    }
    return Customer(**{**defaults, **kwargs})


def parameters_from(markup: str) -> dict[str, str]:
    stream = ElementTree.fromstring(markup).find("./Connect/Stream")
    return {node.get("name"): node.get("value") for node in stream.findall("Parameter")}


def test_refuses_to_construct_without_credentials(monkeypatch):
    configure(monkeypatch, twilio_account_sid="", twilio_auth_token="")
    with pytest.raises(RuntimeError, match="Twilio is not configured"):
        TwilioChannel()


def test_refuses_a_stream_url_twilio_cannot_reach(monkeypatch):
    configure(monkeypatch, twilio_stream_url="ws://localhost:7860/ws")
    with pytest.raises(RuntimeError, match="wss://"):
        TwilioChannel()


def test_twiml_connects_the_call_rather_than_forking_it(monkeypatch):
    """<Start> would let the agent hear the customer but never answer."""
    configure(monkeypatch)
    channel = TwilioChannel()
    markup = channel.twiml(channel._body(make_case(), make_customer()))

    root = ElementTree.fromstring(markup)
    assert root.find("./Connect/Stream") is not None
    assert root.find("./Start") is None
    assert root.find("./Connect/Stream").get("url") == CONFIGURED["twilio_stream_url"]


def test_twiml_carries_the_case_the_agent_needs(monkeypatch):
    configure(monkeypatch)
    channel = TwilioChannel()
    parameters = parameters_from(channel.twiml(channel._body(make_case(), make_customer())))

    assert parameters["recovery_case_id"] == "7"
    assert parameters["customer_name"] == "Asha Rao"
    assert parameters["amount_paise"] == "249900"
    assert parameters["preferred_language"] == "hi"
    # Rendered for speech before it ever crosses the wire, not reconstructed.
    assert "रुपये" in parameters["amount_spoken"]


def test_twiml_escapes_a_name_that_would_break_the_xml(monkeypatch):
    configure(monkeypatch)
    channel = TwilioChannel()
    customer = make_customer(name='Ram & "Sons" <Traders>')

    parameters = parameters_from(channel.twiml(channel._body(make_case(), customer)))
    assert parameters["customer_name"] == 'Ram & "Sons" <Traders>'


def test_twiml_refuses_to_exceed_twilios_limit(monkeypatch):
    """Caught here so the failure names the cause, not as Twilio error 32018."""
    configure(monkeypatch)
    channel = TwilioChannel()
    customer = make_customer(name="न" * MAX_TWIML_CHARS)

    with pytest.raises(ValueError, match="over Twilio's"):
        channel.twiml(channel._body(make_case(), customer))


def test_a_false_flag_is_omitted_rather_than_stringified():
    """"False" is a non-empty string, so sending it would arrive as True."""
    assert stream_parameters({"subscription_halted": False}) == {}
    assert stream_parameters({"subscription_halted": True}) == {"subscription_halted": "1"}
    assert stream_parameters({"phone": None}) == {}


@pytest.mark.asyncio
async def test_refuses_to_dial_a_customer_with_no_number(monkeypatch):
    configure(monkeypatch)
    with pytest.raises(RuntimeError, match="no phone number"):
        await TwilioChannel().initiate(make_case(), make_customer(phone=None))


def test_build_channel_prefers_twilio_over_an_unusable_livekit(monkeypatch):
    """LiveKit without a SIP trunk cannot reach a phone; Twilio can."""
    settings = Settings(
        **CONFIGURED,
        livekit_url="wss://x",
        livekit_api_key="k",
        livekit_api_secret="s",
        livekit_sip_trunk_id="",
    )
    monkeypatch.setattr("app.channels.get_settings", lambda: settings)
    monkeypatch.setattr("app.voice.telephony.get_settings", lambda: settings)

    assert build_channel().name == "twilio"


def test_build_channel_falls_back_to_logging_rather_than_crashing(monkeypatch):
    """An unconfigured deployment should still work cases and say nobody was called.

    Every field is blanked explicitly rather than relying on a bare
    ``Settings()``: that reads the developer's own ``.env``, so once real Twilio
    keys are sitting there this test starts asserting the opposite of its name.
    """
    unconfigured = Settings(
        twilio_account_sid="",
        twilio_auth_token="",
        twilio_from_number="",
        twilio_stream_url="",
        livekit_url="",
        livekit_api_key="",
        livekit_api_secret="",
        livekit_sip_trunk_id="",
    )
    monkeypatch.setattr("app.channels.get_settings", lambda: unconfigured)
    assert isinstance(build_channel(), LoggingChannel)


# --- reading the case back off the carrier's handshake --------------------


class FakeRunnerArgs:
    """Only what `bot()` reads. `call_data` is what create_transport attaches."""

    def __init__(self, *, body=None, call_data=None, transport_type=None):
        self.body = body
        if call_data is not None:
            self.call_data = call_data
        if transport_type is not None:
            self.transport_type = transport_type


def test_the_case_is_read_from_the_carriers_handshake():
    """Twilio's customParameters land on `call_data`, never on `body`.

    Reading the wrong one fails silently: the call connects and the agent
    talks, but to the sample customer about the sample amount.
    """
    args = FakeRunnerArgs(call_data={"body": {"recovery_case_id": "7", "customer_name": "Asha"}})
    assert telephony_body(args) == {"recovery_case_id": "7", "customer_name": "Asha"}


@pytest.mark.parametrize(
    "args",
    [
        FakeRunnerArgs(),
        FakeRunnerArgs(call_data={}),
        FakeRunnerArgs(call_data={"body": {}}),
        FakeRunnerArgs(call_data={"call_id": "CA1"}),
    ],
)
def test_a_handshake_without_a_case_yields_none_so_the_caller_can_fall_back(args):
    assert telephony_body(args) is None


def test_a_malformed_handshake_does_not_kill_the_call():
    """Degrade to the sample rather than hanging up on someone who answered."""

    class Hostile:
        def get(self, _key, _default=None):
            raise RuntimeError("malformed")

    assert telephony_body(FakeRunnerArgs(call_data=Hostile())) is None


def test_the_provider_recorded_is_the_one_that_carried_the_call():
    """The audit trail is the evidence a customer was contacted."""
    assert telephony_provider(FakeRunnerArgs(transport_type="twilio")) == "twilio"
    assert telephony_provider(FakeRunnerArgs(transport_type="plivo")) == "plivo"
    # Unknown beats claiming LiveKit carried a call it did not.
    assert telephony_provider(FakeRunnerArgs()) == "pipecat"


# --- Answering machine detection -------------------------------------------


def test_amd_is_requested_when_a_callback_url_is_configured():
    channel = TwilioChannel()
    channel._settings = Settings(
        twilio_account_sid="AC1", twilio_auth_token="t", twilio_from_number="+1",
        twilio_stream_url="wss://x/ws", machine_detection_enabled=True,
        twilio_amd_callback_url="https://example.invalid/webhooks/twilio/amd",
    )
    amd = channel._machine_detection()
    assert amd["MachineDetection"] == "DetectMessageEnd"
    assert amd["AsyncAmd"] == "true", "sync AMD is dead air for a human who answered"
    assert amd["AsyncAmdStatusCallback"] == "https://example.invalid/webhooks/twilio/amd"


def test_amd_is_off_without_a_callback_url():
    """Detection with nowhere to deliver the verdict changes nothing about the
    call and only costs Twilio's AMD fee."""
    channel = TwilioChannel()
    channel._settings = Settings(
        twilio_account_sid="AC1", twilio_auth_token="t", twilio_from_number="+1",
        twilio_stream_url="wss://x/ws", machine_detection_enabled=True,
        twilio_amd_callback_url="",
    )
    assert channel._machine_detection() == {}


def test_amd_can_be_switched_off_outright():
    channel = TwilioChannel()
    channel._settings = Settings(
        twilio_account_sid="AC1", twilio_auth_token="t", twilio_from_number="+1",
        twilio_stream_url="wss://x/ws", machine_detection_enabled=False,
        twilio_amd_callback_url="https://example.invalid/amd",
    )
    assert channel._machine_detection() == {}


# --- sending a text is not placing a call -----------------------------------


def test_sending_a_text_does_not_need_a_stream_url():
    """The bug this separates.

    TwilioChannel refuses to construct without twilio_stream_url -- the address
    Twilio streams call audio to. That is right for placing a call and
    irrelevant to sending one SMS. The voice service receives calls, so it has
    no stream URL, and building a channel to send the mandate link threw before
    it reached the API: the customer was told a link was coming and nothing
    was sent.
    """
    sms_only = Settings(
        twilio_account_sid="AC1", twilio_auth_token="t", twilio_from_number="+1",
        twilio_stream_url="",
    )
    assert sms_only.twilio_sms_configured is True
    assert sms_only.twilio_configured is False, "still cannot place a call"


def test_placing_a_call_still_needs_everything():
    full = Settings(
        twilio_account_sid="AC1", twilio_auth_token="t", twilio_from_number="+1",
        twilio_stream_url="wss://x/ws",
    )
    assert full.twilio_configured is True
    assert full.twilio_sms_configured is True


@pytest.mark.parametrize(
    "missing", ["twilio_account_sid", "twilio_auth_token", "twilio_from_number"]
)
def test_an_incomplete_sender_cannot_send(missing):
    kwargs = {
        "twilio_account_sid": "AC1",
        "twilio_auth_token": "t",
        "twilio_from_number": "+1",
        missing: "",
    }
    assert Settings(**kwargs).twilio_sms_configured is False


async def test_an_unconfigured_sender_returns_none_rather_than_raising(monkeypatch):
    """A failed SMS must not break a call that is going well."""
    from app.voice import telephony

    monkeypatch.setattr(telephony, "get_settings", lambda: Settings())
    assert await telephony.send_sms("+919000000000", "hi") is None
