from homeassistant.components import conversation
from homeassistant.components.conversation import agent
from homeassistant.const import MATCH_ALL
from homeassistant.core import HomeAssistant
from homeassistant.helpers import intent
from spacy.language import Language
from spacy_language_detection import LanguageDetector
import spacy
from rwkv.model import RWKV
from rwkv.utils import PIPELINE, PIPELINE_ARGS
import threading
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
import os
import requests

OFFLINE = True


async def async_setup_entry(hass: HomeAssistant, entry):
    conversation.async_set_agent(hass, entry, TranslateConversationAgent())
    return True


def _spacy_get_lang_detector(nlp, name):
    return LanguageDetector(seed=42)


class TranslateConversationAgent(agent.AbstractConversationAgent):
    def __init__(
        self,
        skip_language_detect=False,
        skip_rwkv=False,
        skip_translate=False,
    ) -> None:
        super().__init__()
        self.model = None
        self.pipeline = None
        if OFFLINE:
            t1, t2, t3 = None, None, None
            if skip_language_detect == False and OFFLINE == True:
                t1 = threading.Thread(target=self.init_language_detect)
                t1.start()
            if skip_rwkv == False and OFFLINE == True:
                t2 = threading.Thread(target=self.load_rwkv)
                t2.start()
            if skip_translate == False and OFFLINE == True:
                t3 = threading.Thread(target=self.init_translate)
                t3.start()

            if t1 != None:
                t1.join()
            if t2 != None:
                t2.join()
            if t3 != None:
                t3.join()

    def load_rwkv(self):
        save_path = "homeassistant/components/translate_conversation/RWKV-t-World-1.5B-v1-20231021-ctx4096.pth"
        if os.path.basename(os.getcwd()) == "translate_conversation":
            save_path = "../../../homeassistant/components/translate_conversation/RWKV-t-World-1.5B-v1-20231021-ctx4096.pth"
        if not os.path.exists(save_path):
            print("start download rwkv")
            response = requests.get(
                "https://huggingface.co/BlinkDL/temp/resolve/main/RWKV-5-World-1.5B-v2-OnlyForTest_85%25_trained-20231021-ctx4096.pth",
                stream=True,
            )
            if response.status_code == 200:
                with open(save_path, "wb") as fd:
                    for chunk in response.iter_content(chunk_size=128):
                        fd.write(chunk)
        self.model = RWKV(model=save_path, strategy="cpu bf16")
        self.pipeline = PIPELINE(self.model, "rwkv_vocab_v20230424")

    def init_language_detect(self):
        if not spacy.util.is_package("en_core_web_sm"):
            spacy.cli.download("en_core_web_sm")

    def init_translate(self):
        print("Init translate")
        self.translate_model = M2M100ForConditionalGeneration.from_pretrained(
            "facebook/m2m100_418M"
        )
        self.tokenizer = M2M100Tokenizer.from_pretrained("facebook/m2m100_418M")
        print("Init translate done")

    @property
    def supported_languages(self):
        return MATCH_ALL

    async def async_process(self, user_input):
        """Process a sentence."""
        if OFFLINE:
            # first: detect language
            language = self.use_language_detect(user_input)
            # language detect model has bugs, it recognize "hi", "hello", "hej" as Dutch
            if user_input.text in ["hi", "Hi", "hello", "Hello"]:
                language = "en"
            elif user_input.text in ["Hej", "hej"]:
                language = "sv"
            print(language)
            if language[:2] == "zh":
                language = "zh"
            # second: if it is in English, answer user question
            if language == "en":
                response = intent.IntentResponse(language=user_input.language)
                response.async_set_speech(self.use_llm(user_input.text))
                return agent.ConversationResult(conversation_id=None, response=response)
            else:
                # 1. translate to English
                translatedQustion = self.use_translate(user_input.text, language, "en")[
                    0
                ]
                print(translatedQustion)
                # 2. get answer
                answer = self.use_llm(translatedQustion)
                # 3. translate back
                translatedAnswer = self.use_translate(answer, "en", language)
                # 4. send to Home Assistant
                response = intent.IntentResponse(language=user_input.language)
                response.async_set_speech(translatedAnswer)
                return agent.ConversationResult(conversation_id=None, response=response)
        else:
            import openai

            openai.api_key = "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"
            res = await openai.ChatCompletion.acreate(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "user", "content": user_input.text},
                ],
            )
            response = intent.IntentResponse(language=user_input.language)
            response.async_set_speech(res["choices"][0]["message"]["content"])
            return agent.ConversationResult(conversation_id=None, response=response)

    def use_llm(self, question):
        prompt = f"""Question: hi

Assistant: Hi. I am your assistant and I will provide expert full response in full details. Please feel free to ask any question and I will always answer it.

Question: {question}

Answer:"""
        args = PIPELINE_ARGS(
            temperature=0.1,
            token_ban=[0],
            token_stop=[261],
        )

        def my_print(s):
            print(s, end="", flush=True)  # only debug

        return self.pipeline.generate(
            prompt, token_count=200, args=args, callback=my_print
        )

    def use_translate(self, input_text, src_lang, des_lang):
        self.tokenizer.src_lang = src_lang
        encoded_zh = self.tokenizer(input_text, return_tensors="pt")
        generated_tokens = self.translate_model.generate(
            **encoded_zh, forced_bos_token_id=self.tokenizer.get_lang_id(des_lang)
        )
        return self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

    def use_language_detect(self, user_input):
        nlp_model = spacy.load("en_core_web_sm")
        Language.factory("language_detector", func=_spacy_get_lang_detector)
        nlp_model.add_pipe("language_detector", last=True)
        doc = nlp_model(user_input.text)
        language_dict = doc._.language
        # print(language_dict)
        language = (
            language_dict.get("language")
            or user_input.language
            or self.hass.config.language
        )
        return language
