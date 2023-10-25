from homeassistant.components.translate_conversation import TranslateConversationAgent


class input_class:
    def __init__(self, language, text):
        self.language = language
        self.text = text


def test_language_detect():
    a = TranslateConversationAgent(skip_rwkv=True, skip_translate=True)
    assert a.use_language_detect(input_class("", "vad är ditt namn")) == "sv"
