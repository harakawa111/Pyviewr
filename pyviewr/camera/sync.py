"""GigE Action Command helpers for dual-camera sync."""

from __future__ import annotations

from typing import Any


ACTION_DEVICE_KEY = 0x46594E43  # "FYNC"
ACTION_GROUP_KEY = 0x1
ACTION_GROUP_MASK = 0xFFFFFFFF


def _set(camera: Any, name: str, value: Any) -> None:
    node = getattr(camera, name, None)
    if node is None:
        raise RuntimeError(f"Camera feature not available: {name}")
    node.SetValue(value)


def configure_action_trigger(camera: Any) -> None:
    """Configure a camera to fire FrameStart on Action1."""
    _set(camera, "TriggerSelector", "FrameStart")
    _set(camera, "TriggerMode", "On")
    _set(camera, "TriggerSource", "Action1")

    # SFNC 2.x (ace 2) exposes ActionSelector.
    if getattr(camera, "ActionSelector", None) is not None:
        try:
            camera.ActionSelector.SetValue(1)
        except Exception:
            pass

    _set(camera, "ActionDeviceKey", ACTION_DEVICE_KEY)
    _set(camera, "ActionGroupKey", ACTION_GROUP_KEY)
    _set(camera, "ActionGroupMask", ACTION_GROUP_MASK)


def configure_free_run(camera: Any) -> None:
    """Disable FrameStart trigger for continuous free-run preview."""
    _set(camera, "TriggerSelector", "FrameStart")
    _set(camera, "TriggerMode", "Off")


def issue_action_command(tl_factory: Any, address: str = "255.255.255.255") -> None:
    """Broadcast one Action Command to all configured cameras on the subnet."""
    from pypylon import pylon

    # Prefer BaslerGigE device class constant when available.
    tl_type = getattr(pylon, "BaslerGigEDeviceClass", "BaslerGigE")
    gige_tl = tl_factory.CreateTl(tl_type)
    try:
        # pypylon 26.x+: IssueActionCommand was split into NoWait / Wait.
        # Older builds still expose IssueActionCommand.
        if hasattr(gige_tl, "IssueActionCommandNoWait"):
            gige_tl.IssueActionCommandNoWait(
                ACTION_DEVICE_KEY,
                ACTION_GROUP_KEY,
                ACTION_GROUP_MASK,
                address,
            )
        elif hasattr(gige_tl, "IssueActionCommand"):
            gige_tl.IssueActionCommand(
                ACTION_DEVICE_KEY,
                ACTION_GROUP_KEY,
                ACTION_GROUP_MASK,
                address,
            )
        else:
            raise RuntimeError(
                "GigE TL has no Action Command API "
                "(need IssueActionCommandNoWait or IssueActionCommand)"
            )
    finally:
        tl_factory.ReleaseTl(gige_tl)
