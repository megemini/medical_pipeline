import platform
import sys
import threading
import time
from os import PathLike
from pathlib import Path
from typing import NamedTuple, Optional


def device_widget(default="AUTO", exclude=None, added=None, description="Device:"):
    import openvino as ov
    import ipywidgets as widgets

    core = ov.Core()

    supported_devices = core.available_devices + ["AUTO"]
    exclude = exclude or []
    if exclude:
        for ex_device in exclude:
            if ex_device in supported_devices:
                supported_devices.remove(ex_device)

    added = added or []
    if added:
        for add_device in added:
            if add_device not in supported_devices:
                supported_devices.append(add_device)

    device = widgets.Dropdown(
        options=supported_devices,
        value=default,
        description=description,
        disabled=False,
    )
    return device


def collect_telemetry(*args, **kwargs):
    """No-op telemetry placeholder."""
    pass
