from homeassistant.components.translate_conversation import TranslateConversationAgent


class input_class:
    def __init__(self, language, text):
        self.language = language
        self.text = text


def test_language_detect():
    a = TranslateConversationAgent(skip_rwkv=True, skip_translate=True)
    assert a.use_language_detect(input_class("", "vad är ditt namn")) == "sv"


def test_support_language():
    a = TranslateConversationAgent(
        skip_rwkv=True, skip_translate=True, skip_language_detect=True
    )
    assert a.supported_languages == "*"
