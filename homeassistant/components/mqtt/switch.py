"""Support for MQTT switches."""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

import voluptuous as vol

from homeassistant.components import switch
from homeassistant.components.switch import DEVICE_CLASSES_SCHEMA, SwitchEntity
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import (
    CONF_DEVICE_CLASS,
    CONF_NAME,
    CONF_OPTIMISTIC,
    CONF_PAYLOAD_OFF,
    CONF_PAYLOAD_ON,
    CONF_VALUE_TEMPLATE,
    STATE_ON,
)
from homeassistant.core import HomeAssistant, callback
import homeassistant.helpers.config_validation as cv
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.restore_state import RestoreEntity
from homeassistant.helpers.service_info.mqtt import ReceivePayloadType
from homeassistant.helpers.typing import ConfigType

from . import subscription
from .config import MQTT_RW_SCHEMA
from .const import (
    CONF_COMMAND_TOPIC,
    CONF_ENCODING,
    CONF_QOS,
    CONF_RETAIN,
    CONF_STATE_TOPIC,
    PAYLOAD_NONE,
)
from .debug_info import log_messages
from .mixins import (
    MQTT_ENTITY_COMMON_SCHEMA,
    MqttEntity,
    async_mqtt_entry_helper,
    write_state_on_attr_change,
)
from .models import MqttValueTemplate, ReceiveMessage

DEFAULT_NAME = "MQTT Switch"
DEFAULT_PAYLOAD_ON = "ON"
DEFAULT_PAYLOAD_OFF = "OFF"
CONF_STATE_ON = "state_on"
CONF_STATE_OFF = "state_off"

PLATFORM_SCHEMA_MODERN = MQTT_RW_SCHEMA.extend(
    {
        vol.Optional(CONF_NAME): vol.Any(cv.string, None),
        vol.Optional(CONF_PAYLOAD_OFF, default=DEFAULT_PAYLOAD_OFF): cv.string,
        vol.Optional(CONF_PAYLOAD_ON, default=DEFAULT_PAYLOAD_ON): cv.string,
        vol.Optional(CONF_STATE_OFF): cv.string,
        vol.Optional(CONF_STATE_ON): cv.string,
        vol.Optional(CONF_VALUE_TEMPLATE): cv.template,
        vol.Optional(CONF_DEVICE_CLASS): vol.Any(DEVICE_CLASSES_SCHEMA, None),
    }
).extend(MQTT_ENTITY_COMMON_SCHEMA.schema)

DISCOVERY_SCHEMA = PLATFORM_SCHEMA_MODERN.extend({}, extra=vol.REMOVE_EXTRA)


async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up MQTT switch through YAML and through MQTT discovery."""
    await async_mqtt_entry_helper(
        hass,
        config_entry,
        MqttSwitch,
        switch.DOMAIN,
        async_add_entities,
        DISCOVERY_SCHEMA,
        PLATFORM_SCHEMA_MODERN,
    )


class MqttSwitch(MqttEntity, SwitchEntity, RestoreEntity):
    """Representation of a switch that can be toggled using MQTT."""

    _default_name = DEFAULT_NAME
    _entity_id_format = switch.ENTITY_ID_FORMAT

    _optimistic: bool
    _state_on: str
    _state_off: str
    _value_template: Callable[[ReceivePayloadType], ReceivePayloadType]

    @staticmethod
    def config_schema() -> vol.Schema:
        """Return the config schema."""
        return DISCOVERY_SCHEMA

    def _setup_from_config(self, config: ConfigType) -> None:
        """(Re)Setup the entity."""
        self._attr_device_class = config.get(CONF_DEVICE_CLASS)

        state_on: str | None = config.get(CONF_STATE_ON)
        self._state_on = state_on if state_on else config[CONF_PAYLOAD_ON]

        state_off: str | None = config.get(CONF_STATE_OFF)
        self._state_off = state_off if state_off else config[CONF_PAYLOAD_OFF]

        self._optimistic = (
            config[CONF_OPTIMISTIC] or config.get(CONF_STATE_TOPIC) is None
        )
        self._attr_assumed_state = bool(self._optimistic)

        self._value_template = MqttValueTemplate(
            self._config.get(CONF_VALUE_TEMPLATE), entity=self
        ).async_render_with_possible_json_value

    def _prepare_subscribe_topics(self) -> None:
        """(Re)Subscribe to topics."""

        @callback
        @log_messages(self.hass, self.entity_id)
        @write_state_on_attr_change(self, {"_attr_is_on"})
        def state_message_received(msg: ReceiveMessage) -> None:
            """Handle new MQTT state messages."""
            payload = self._value_template(msg.payload)
            if payload == self._state_on:
                self._attr_is_on = True
            elif payload == self._state_off:
                self._attr_is_on = False
            elif payload == PAYLOAD_NONE:
                self._attr_is_on = None

        if self._config.get(CONF_STATE_TOPIC) is None:
            # Force into optimistic mode.
            self._optimistic = True
        else:
            self._sub_state = subscription.async_prepare_subscribe_topics(
                self.hass,
                self._sub_state,
                {
                    CONF_STATE_TOPIC: {
                        "topic": self._config.get(CONF_STATE_TOPIC),
                        "msg_callback": state_message_received,
                        "qos": self._config[CONF_QOS],
                        "encoding": self._config[CONF_ENCODING] or None,
                    }
                },
            )

    async def _subscribe_topics(self) -> None:
        """(Re)Subscribe to topics."""
        await subscription.async_subscribe_topics(self.hass, self._sub_state)

        if self._optimistic and (last_state := await self.async_get_last_state()):
            self._attr_is_on = last_state.state == STATE_ON

    async def async_turn_on(self, **kwargs: Any) -> None:
        """Turn the device on.

        This method is a coroutine.
        """
        await self.async_publish(
            self._config[CONF_COMMAND_TOPIC],
            self._config[CONF_PAYLOAD_ON],
            self._config[CONF_QOS],
            self._config[CONF_RETAIN],
            self._config[CONF_ENCODING],
        )
        if self._optimistic:
            # Optimistically assume that switch has changed state.
            self._attr_is_on = True
            self.async_write_ha_state()

    async def async_turn_off(self, **kwargs: Any) -> None:
        """Turn the device off.

        This method is a coroutine.
        """
        await self.async_publish(
            self._config[CONF_COMMAND_TOPIC],
            self._config[CONF_PAYLOAD_OFF],
            self._config[CONF_QOS],
            self._config[CONF_RETAIN],
            self._config[CONF_ENCODING],
        )
        if self._optimistic:
            # Optimistically assume that switch has changed state.
            self._attr_is_on = False
            self.async_write_ha_state()
