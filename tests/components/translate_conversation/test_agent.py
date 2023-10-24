import asyncio

from homeassistant.components.translate_conversation import TranslateConversationAgent


class input_class:
    def __init__(self, language, text):
        self.language = language
        self.text = text


def test_agent():
    a = TranslateConversationAgent()
    i = input_class("en", "hi")
    b = asyncio.run(a.async_process(i))
    assert b.response.as_dict()["speech"]["plain"]["speech"] == "Test response"
