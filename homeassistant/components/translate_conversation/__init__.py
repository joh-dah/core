from homeassistant.components import conversation
from homeassistant.components.conversation import agent
from homeassistant.const import MATCH_ALL
from homeassistant.core import HomeAssistant
from homeassistant.helpers import intent


async def async_setup_entry(hass: HomeAssistant, entry):
    conversation.async_set_agent(hass, entry, TranslateConversationAgent())
    return True


class TranslateConversationAgent(agent.AbstractConversationAgent):
    @property
    def supported_languages(self):
        return MATCH_ALL

    async def async_process(self, user_input):
        """Process a sentence."""
        response = intent.IntentResponse(language=user_input.language)
        response.async_set_speech("Test response")
        return agent.ConversationResult(conversation_id=None, response=response)
