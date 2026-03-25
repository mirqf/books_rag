from __future__ import annotations

import base64
import os

try:
    import config as local_config
except ImportError:
    local_config = None


def get_setting(name: str, default: str = "") -> str:
    env_value = os.getenv(name)
    if isinstance(env_value, str) and env_value.strip():
        return env_value.strip()
    if local_config is not None:
        config_value = getattr(local_config, name, default)
        if isinstance(config_value, str) and config_value.strip():
            return config_value.strip()
    return default


def get_secret(name: str, default: str = "") -> str:
    direct = get_setting(name, "")
    if direct:
        return direct

    encoded = get_setting(f"{name}_B64", "")
    if not encoded:
        return default
    try:
        return base64.b64decode(encoded.encode("utf-8")).decode("utf-8").strip()
    except Exception as exc:
        raise RuntimeError(f"Failed to decode base64 secret {name}_B64") from exc
