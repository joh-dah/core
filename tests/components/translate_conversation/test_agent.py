from homeassistant.components.translate_conversation import TranslateConversationAgent
from homeassistant.components.conversation import agent
import asyncio
from homeassistant.setup import async_setup_component
import pytest
from tests.common import MockConfigEntry


class input_class:
    def __init__(self, text):
        self.language = ""
        self.text = text


a = TranslateConversationAgent()


def test_language_detect():
    global a
    assert a.use_language_detect(input_class("vad är ditt namn")) == "sv"


def test_support_language():
    global a
    assert a.supported_languages == "*"


def test_type():
    global a
    assert isinstance(a, agent.AbstractConversationAgent)


def test_translate():
    global a
    assert a.use_translate("Hello", "en", "sv")[0] == "Hej Hej"


def test_all():
    global a
    i = input_class("vad är ditt namn")
    main = asyncio.run(a.async_process(i))
    assert isinstance(main.response.as_dict()["speech"]["plain"]["speech"][0], str)


@pytest.fixture
def mock_config_entry(hass):
    """Mock a config entry."""
    entry = MockConfigEntry(
        domain="translate_conversation",
    )
    entry.add_to_hass(hass)
    return entry


async def test_mock_init_component(hass, mock_config_entry):
    """Initialize integration."""
    global a
    a = None
    assert await async_setup_component(hass, "translate_conversation", {}) == True
    await hass.async_block_till_done()
