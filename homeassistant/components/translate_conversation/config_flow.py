from homeassistant import config_entries


class ExampleConfigFlow(config_entries.ConfigFlow, domain="translate_conversation"):
    async def async_step_user(self, user_input):
        return self.async_create_entry(title="Translate Conversation", data={})
