"""Camera enumeration, grab loops, still/video capture."""

from __future__ import annotations

import fcntl
import ipaddress
import socket
import struct
import subprocess
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

import numpy as np

from pyviewr import config as app_config
from pyviewr.camera import sync as sync_mod
from pyviewr.io.still import save_still
from pyviewr.io.video import DEFAULT_VIDEO_FPS, VideoRecorder
from pyviewr.processing.enhance import EnhanceParams, apply as apply_enhance

try:
    from pypylon import genicam, pylon
except ImportError:  # pragma: no cover - runtime dependency
    pylon = None  # type: ignore[assignment]
    genicam = None  # type: ignore[assignment]


FrameCallback = Callable[[int, np.ndarray], None]
ErrorCallback = Callable[[str], None]
DeviceRemovedCallback = Callable[[int], None]

# GenICam auto-function node that must be "Off" before manual writes.
_AUTO_NODE = {"ExposureTime": "ExposureAuto", "Gain": "GainAuto"}
_FPS_VALUE_NODES = ("AcquisitionFrameRate", "AcquisitionFrameRateAbs")
TARGET_FPS = 67.0
_PREVIEW_FPS = 10.0
# Exposure headroom inside one frame period (µs) and allowed shortfall of
# ResultingFrameRate versus the requested rate.
_EXPOSURE_MARGIN_US = 100.0
_FPS_TOLERANCE = 0.995
_BASLER_OUI = "00:30:53"
_SIOCGIFADDR = 0x8915
_SIOCGIFNETMASK = 0x891B


@dataclass(frozen=True)
class DeviceInfo:
    index: int
    model: str
    serial: str
    display_name: str


@dataclass(frozen=True)
class RecordingSummary:
    slot: int
    frames: int
    fps: float
    # Times capture waited for the disk spool (disk too slow).
    disk_waits: int
    # Frames the camera sent but the host missed (BlockID gaps); None if the
    # camera does not report BlockID.
    camera_dropped: int | None
    # Frames that arrived with missing packets (network).
    incomplete: int
    # Frames discarded because no pylon buffer was free (host too slow);
    # None if the stream grabber does not report it.
    buffer_underruns: int | None
    packet_size: int | None


@dataclass(frozen=True)
class FeatureInfo:
    name: str
    value: float
    minimum: float
    maximum: float


def _require_pylon() -> None:
    if pylon is None:
        raise RuntimeError(
            "pypylon is not installed, or Basler pylon Software Suite is missing. "
            "Install pylon 7.1+ then: pip install pypylon"
        )


def _node(cam: object, name: str):
    """Return a GenICam node, or None if it does not exist.

    pypylon InstantCamera.__getattr__ raises LogicalErrorException for missing
    nodes instead of AttributeError, so getattr(..., None) is not safe.
    """
    try:
        return cam.GetNodeMap().GetNode(name)
    except Exception:
        return None


def _is_physically_removed_message(message: str) -> bool:
    return "physically removed" in message.lower()


_UNDERRUN_STAT = "Statistic_Buffer_Underrun_Count"


def _stream_stat(cam: object, name: str) -> int | None:
    try:
        return int(cam.GetStreamGrabberNodeMap().GetNode(name).GetValue())
    except Exception:
        return None


def _block_id(result) -> int | None:
    try:
        return int(result.BlockID)
    except Exception:
        return None


def array_from_grab(result) -> np.ndarray:
    """Convert a pylon grab result to a contiguous uint8 grayscale image.

    GrabResult.Array is a view into pylon memory.  Copy before Release()
    so later preview / spool code does not use a dangling buffer.
    """
    img = result.Array
    if img.ndim == 3:
        # Rare for this UV mono camera; take first channel.
        img = img[:, :, 0]
    if img.dtype == np.uint8:
        return np.array(img, dtype=np.uint8, copy=True, order="C")
    # Mono10/12/16 -> display/save as uint8 by simple scale.
    info = np.iinfo(img.dtype)
    if info.max <= 0:
        return np.zeros(img.shape, dtype=np.uint8)
    scaled = (img.astype(np.float32) * (255.0 / float(info.max))).clip(0, 255)
    return np.ascontiguousarray(scaled.astype(np.uint8))


def _norm_mac(mac: str) -> str:
    hex_digits = "".join(c for c in mac.upper() if c in "0123456789ABCDEF")
    if len(hex_digits) != 12:
        return mac.strip().upper()
    return ":".join(hex_digits[i : i + 2] for i in range(0, 12, 2))


def _mac_oui(mac: str) -> str:
    return _norm_mac(mac)[:8]


def _iface_ipv4(name: str) -> ipaddress.IPv4Interface | None:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        ifreq = struct.pack("256s", name.encode("utf-8")[:15])
        addr = socket.inet_ntoa(fcntl.ioctl(sock, _SIOCGIFADDR, ifreq)[20:24])
        mask = socket.inet_ntoa(fcntl.ioctl(sock, _SIOCGIFNETMASK, ifreq)[20:24])
        prefix = ipaddress.IPv4Network(f"0.0.0.0/{mask}").prefixlen
        return ipaddress.IPv4Interface(f"{addr}/{prefix}")
    except OSError:
        pass
    finally:
        sock.close()
    try:
        out = subprocess.check_output(
            ["ip", "-4", "-o", "addr", "show", "dev", name],
            text=True,
            timeout=1,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    for part in out.split():
        if "/" in part and part[0].isdigit():
            try:
                return ipaddress.IPv4Interface(part)
            except ValueError:
                continue
    return None


def _default_route_nics() -> set[str]:
    nics: set[str] = set()
    try:
        lines = Path("/proc/net/route").read_text(encoding="utf-8").splitlines()[1:]
    except OSError:
        return nics
    for line in lines:
        parts = line.split()
        if len(parts) >= 2 and parts[1] == "00000000":
            nics.add(parts[0])
    return nics


def _nic_mac(name: str) -> str:
    path = Path("/sys/class/net") / name / "address"
    try:
        return _norm_mac(path.read_text(encoding="utf-8").strip())
    except OSError:
        return ""


def _basler_arp_neighbors() -> list[tuple[str, str, str]]:
    """Return (ip, mac, nic) for Basler MACs in the ARP table."""
    path = Path("/proc/net/arp")
    try:
        lines = path.read_text(encoding="utf-8").splitlines()[1:]
    except OSError:
        return []
    out: list[tuple[str, str, str]] = []
    for line in lines:
        parts = line.split()
        if len(parts) < 6:
            continue
        ip, _hw, flags, mac, _mask, nic = parts[:6]
        if flags == "0x0" or mac in ("00:00:00:00:00:00", "*"):
            continue
        if _mac_oui(mac) != _BASLER_OUI:
            continue
        out.append((ip, _norm_mac(mac), nic))
    return out


def _device_str(device: object, getter: str) -> str:
    available = getattr(device, getter.replace("Get", "Is") + "Available", None)
    try:
        if callable(available) and not available():
            return ""
        fn = getattr(device, getter)
        return str(fn() or "")
    except Exception:
        return ""


class CameraManager:
    """Owns InstantCamera instances and background grab threads."""

    def __init__(
        self,
        on_frame: FrameCallback | None = None,
        on_error: ErrorCallback | None = None,
        on_device_removed: DeviceRemovedCallback | None = None,
    ) -> None:
        _require_pylon()
        self._on_frame = on_frame
        self._on_error = on_error
        self._on_device_removed = on_device_removed
        self._tl_factory = pylon.TlFactory.GetInstance()
        self._cameras: list[pylon.InstantCamera] = []
        self._slots: list[int] = []
        self._threads: list[threading.Thread] = []
        self._stop = threading.Event()
        self._device_removed = threading.Event()
        self._grabbing = False
        self._recording = False
        self._recorders: dict[int, VideoRecorder] = {}
        self._lock = threading.Lock()
        self._last_preview_at: dict[int, float] = {}
        self._last_block_id: dict[int, int] = {}
        self._dropped: dict[int, int | None] = {}
        self._incomplete: dict[int, int] = {}
        self._underruns_at_start: dict[int, int | None] = {}
        self._enhance_params = EnhanceParams()

    def set_enhance_params(self, params: EnhanceParams) -> None:
        """Use the same software enhance for Still / Record as the preview."""
        self._enhance_params = params

    def _frame_for_save(self, frame: np.ndarray) -> np.ndarray:
        """Apply enhance for still images and post-processed video output."""
        try:
            return apply_enhance(frame, self._enhance_params)
        except Exception:
            return frame

    # ----- discovery / lifecycle -------------------------------------------------

    def enumerate_devices(self) -> list[DeviceInfo]:
        _require_pylon()
        if not self._cameras:
            self._recover_gige_subnet()
        devices = self._tl_factory.EnumerateDevices()
        out: list[DeviceInfo] = []
        for i, di in enumerate(devices):
            model = di.GetModelName() or "Unknown"
            serial = di.GetSerialNumber() or ""
            name = di.GetFriendlyName() or f"{model} ({serial})"
            out.append(DeviceInfo(i, model, serial, name))
        return out

    def _camera_nics(self) -> list[str]:
        """NICs dedicated to Basler cameras (OUI 00:30:53) that have link."""
        nics = [
            name
            for name in self._wired_nics_with_carrier()
            if _mac_oui(_nic_mac(name)) == _BASLER_OUI
        ]
        if nics:
            return nics
        skip = _default_route_nics()
        return [
            name for name in self._wired_nics_with_carrier() if name not in skip
        ]

    def _pick_force_ip(
        self, iface: ipaddress.IPv4Interface, used: set[str], preferred: str | None
    ) -> str | None:
        used.add(str(iface.ip))
        net = iface.network

        def _ok(addr: ipaddress.IPv4Address) -> bool:
            s = str(addr)
            return (
                addr in net
                and addr != net.network_address
                and addr != net.broadcast_address
                and s not in used
            )

        if preferred:
            try:
                cand = ipaddress.IPv4Address(preferred)
            except ValueError:
                cand = None
            if cand is not None and _ok(cand):
                return str(cand)
            try:
                host = int(preferred.rsplit(".", 1)[1])
            except ValueError:
                host = None
            if host is not None:
                cand = ipaddress.IPv4Address(int(net.network_address) + host)
                if _ok(cand):
                    return str(cand)
        three = ipaddress.IPv4Address(int(net.network_address) + 3)
        if _ok(three):
            return str(three)
        for host_ip in net.hosts():
            if _ok(host_ip):
                return str(host_ip)
        return None

    def _recover_gige_subnet(self) -> None:
        """ForceIP cameras that are on the wire but not on the NIC subnet.

        pylon EnumerateDevices hides GigE cameras whose IP is not in the
        adapter subnet.  This lab's 5GigE NIC is 192.168.10.2/24 while the
        camera often stays on 192.168.5.3 / 192.168.9.3.
        """
        nics = self._camera_nics()
        if not nics or pylon is None:
            return
        nic_ifaces: dict[str, ipaddress.IPv4Interface] = {}
        for nic in nics:
            iface = _iface_ipv4(nic)
            if iface is not None:
                nic_ifaces[nic] = iface
        if not nic_ifaces:
            return

        gige = None
        try:
            gige = self._tl_factory.CreateTl("BaslerGigE")
        except Exception:
            return
        try:
            try:
                all_devs = list(gige.EnumerateAllDevices())
            except Exception:
                all_devs = []
            found: dict[str, tuple[str, str | None, str | None]] = {}
            # mac -> (nic, current_ip, mask)
            for di in all_devs:
                mac = _norm_mac(_device_str(di, "GetMacAddress"))
                if not mac:
                    continue
                ip = _device_str(di, "GetIpAddress")
                nic = self._match_device_nic(di, nic_ifaces)
                found[mac] = (nic, ip or None, _device_str(di, "GetSubnetMask") or None)
            for ip, mac, nic in _basler_arp_neighbors():
                if mac not in found and nic in nic_ifaces:
                    found[mac] = (nic, ip, None)

            used = {str(iface.ip) for iface in nic_ifaces.values()}
            changed = False
            for mac, (nic, current_ip, _mask) in found.items():
                iface = nic_ifaces.get(nic) or next(iter(nic_ifaces.values()))
                if current_ip:
                    try:
                        if ipaddress.IPv4Address(current_ip) in iface.network:
                            continue
                    except ValueError:
                        pass
                new_ip = self._pick_force_ip(iface, used, current_ip)
                if new_ip is None:
                    continue
                used.add(new_ip)
                mask = str(iface.network.netmask)
                try:
                    gige.ForceIp(mac, new_ip, mask, "0.0.0.0")
                    changed = True
                except Exception:
                    try:
                        gige.ForceIp(
                            mac.replace(":", ""), new_ip, mask, "0.0.0.0"
                        )
                        changed = True
                    except Exception:
                        pass
            if changed:
                time.sleep(0.4)
        finally:
            try:
                self._tl_factory.ReleaseTl(gige)
            except Exception:
                pass

    def _match_device_nic(
        self,
        device: object,
        nic_ifaces: dict[str, ipaddress.IPv4Interface],
    ) -> str:
        iface_addr = _device_str(device, "GetInterface")
        for nic, iface in nic_ifaces.items():
            if iface_addr == nic or iface_addr == str(iface.ip):
                return nic
        return next(iter(nic_ifaces))

    @staticmethod
    def _rp_filter_nics() -> list[str]:
        """Return interface names whose IPv4 rp_filter is not Off."""
        conf = Path("/proc/sys/net/ipv4/conf")
        if not conf.is_dir():
            return []
        blocked: list[str] = []
        for name in ("all", "default"):
            path = conf / name / "rp_filter"
            try:
                if int(path.read_text().strip()) != 0:
                    blocked.append(name)
            except (OSError, ValueError):
                pass
        for path in sorted(conf.glob("*/rp_filter")):
            nic = path.parent.name
            if nic in ("all", "default", "lo"):
                continue
            try:
                if int(path.read_text().strip()) != 0:
                    blocked.append(nic)
            except (OSError, ValueError):
                pass
        return blocked

    @staticmethod
    def _wired_nics_with_carrier() -> list[str]:
        """Ethernet interfaces that currently have link, skipping Wi-Fi."""
        sys_net = Path("/sys/class/net")
        if not sys_net.is_dir():
            return []
        nics: list[str] = []
        for nic_dir in sorted(sys_net.iterdir()):
            if nic_dir.name == "lo" or (nic_dir / "wireless").exists():
                continue
            try:
                if (nic_dir / "type").read_text().strip() != "1":
                    continue
                if (nic_dir / "carrier").read_text().strip() != "1":
                    continue
            except OSError:
                continue
            nics.append(nic_dir.name)
        return nics

    @classmethod
    def gige_unreachable_hint(cls, *, detailed: bool = False) -> str | None:
        """Explain common Linux GigE discovery failures without spawning pylon tools."""
        arp = _basler_arp_neighbors()
        blocked = cls._rp_filter_nics()
        wired = cls._wired_nics_with_carrier()
        if arp:
            ip, mac, nic = arp[0]
            iface = _iface_ipv4(nic)
            nic_ip = str(iface.ip) if iface else "?"
            extra = f" (+{len(arp) - 1} more)" if len(arp) > 1 else ""
            if not detailed:
                return (
                    f"Basler {mac} is on {nic} at {ip}{extra}, "
                    f"but the NIC is {nic_ip} (different subnet)"
                )
            return (
                f"Basler camera {mac} answers on {nic} at {ip}{extra}, "
                f"but adapter {nic} is {nic_ip}.\n\n"
                "Pyviewr should ForceIP the camera onto the NIC subnet on Refresh. "
                "If it still does not appear, set the NIC to the camera subnet "
                f"or the camera to {iface.network if iface else 'the NIC subnet'}."
            )
        if blocked:
            nics = [n for n in blocked if n not in ("all", "default")] or wired
            nic = nics[0] if nics else "enp9s0"
            if not detailed:
                return "rp_filter is dropping GigE replies"
            commands = [
                "sudo sysctl -w net.ipv4.conf.all.rp_filter=0",
                "sudo sysctl -w net.ipv4.conf.default.rp_filter=0",
                f"sudo sysctl -w net.ipv4.conf.{nic}.rp_filter=0",
            ]
            for extra in nics[1:]:
                commands.append(
                    f"sudo sysctl -w net.ipv4.conf.{extra}.rp_filter=0"
                )
            return (
                "Linux rp_filter is dropping GigE replies.\n\n"
                "Run:\n"
                + "\n".join(commands)
                + "\n\nThen power-cycle the camera and click Refresh."
            )
        if not detailed:
            return None
        if wired:
            return (
                "pylon found no devices. Wired link is up on "
                + ", ".join(wired)
                + ". Check camera power, subnet, and Jumbo frames."
            )
        return "pylon found no devices. Check power, cable, and NIC link."

    @property
    def camera_count(self) -> int:
        return len(self._cameras)

    @property
    def is_open(self) -> bool:
        return bool(self._cameras)

    @property
    def is_grabbing(self) -> bool:
        return self._grabbing

    @property
    def is_recording(self) -> bool:
        return self._recording

    def open(self, indices: list[int] | None = None) -> list[DeviceInfo]:
        """Open selected devices (default: first two, or all if fewer)."""
        self.close()
        devices = self._tl_factory.EnumerateDevices()
        if not devices:
            raise RuntimeError("No cameras found.")

        if indices is None:
            indices = list(range(min(2, len(devices))))
        if not indices:
            raise RuntimeError("No camera indices selected.")
        for idx in indices:
            if idx < 0 or idx >= len(devices):
                raise RuntimeError(f"Invalid camera index: {idx}")

        opened: list[DeviceInfo] = []
        try:
            for local_i, idx in enumerate(indices):
                di = devices[idx]
                cam = pylon.InstantCamera(self._tl_factory.CreateDevice(di))
                cam.Open()
                # Prefer Mono8 when available for light preview/recording.
                try:
                    cam.PixelFormat.SetValue("Mono8")
                except (genicam.GenericException, AttributeError):
                    pass
                sync_mod.configure_free_run(cam)
                self._configure_frame_rate(cam, TARGET_FPS)
                self._configure_packet_size(cam)
                self._cameras.append(cam)
                opened.append(
                    DeviceInfo(
                        local_i,
                        di.GetModelName() or "Unknown",
                        di.GetSerialNumber() or "",
                        di.GetFriendlyName() or di.GetModelName() or f"cam{local_i}",
                    )
                )
        except Exception:
            self.close()
            raise
        return opened

    def _configure_packet_size(self, cam: object) -> None:
        """Match GevSCPSPacketSize to the MTU of the NIC the camera is on.

        Jumbo packets cut the per-packet load ~6x at 2840x2840 / 67 fps.  The
        camera default (1500) is kept if the NIC cannot be determined, since a
        packet larger than the NIC MTU would stop all frames from arriving.
        """
        node = _node(cam, "GevSCPSPacketSize")
        ip_node = _node(cam, "GevCurrentIPAddress")
        if node is None or ip_node is None:
            return
        try:
            cam_ip = ipaddress.IPv4Address(int(ip_node.GetValue()))
        except Exception:
            return
        mtu = None
        for nic in self._wired_nics_with_carrier():
            iface = _iface_ipv4(nic)
            if iface is not None and cam_ip in iface.network:
                try:
                    mtu = int((Path("/sys/class/net") / nic / "mtu").read_text())
                except (OSError, ValueError):
                    return
                break
        if mtu is None:
            return
        try:
            inc = max(int(node.GetInc()), 1)
            size = min(mtu, int(node.GetMax()))
            size = max(size - size % inc, int(node.GetMin()))
            node.SetValue(size)
        except Exception as exc:
            if self._on_error is not None:
                self._on_error(f"Packet size not set: {exc}")

    def _resolve_slot_order(
        self, devices: list[DeviceInfo]
    ) -> tuple[list[tuple[int, int]], dict[str, str]]:
        """Pair each device with its UVL(0) / UVR(1) slot.

        Matches devices to their persisted serial -> label identity first.
        Any device with an unrecognized serial fills the next open slot (UVL
        before UVR), so the *first* camera ever seen becomes UVL and the
        second becomes UVR — after that, identity sticks regardless of USB
        enumeration order or whether the other camera is connected at all.
        A slot with no matching or fallback device is simply absent, so a
        lone UVR camera keeps slot 1 rather than sliding into slot 0.
        Devices beyond the two known slots get their own trailing slots.

        Returns a list of (device_index, slot_index) pairs, ordered by slot,
        plus any newly-learned serial -> label identity to persist.
        """
        identity = app_config.load_camera_identity()
        by_label: dict[str, int] = {}
        unknown: list[int] = []
        for i, d in enumerate(devices):
            label = identity.get(d.serial)
            if label in app_config.SLOT_LABELS and label not in by_label:
                by_label[label] = i
            else:
                unknown.append(i)

        updates: dict[str, str] = {}
        ordered: list[tuple[int, int]] = []
        for slot, label in enumerate(app_config.SLOT_LABELS):
            if label in by_label:
                ordered.append((by_label[label], slot))
            elif unknown:
                idx = unknown.pop(0)
                ordered.append((idx, slot))
                if devices[idx].serial:
                    updates[devices[idx].serial] = label
        extra_slot = len(app_config.SLOT_LABELS)
        for idx in unknown:
            ordered.append((idx, extra_slot))
            extra_slot += 1
        return ordered, updates

    def open_identified(self, max_devices: int = 2) -> list[DeviceInfo]:
        """Open up to max_devices cameras in a stable UVL / UVR slot order.

        Slot 0 is always the camera identified as UVL, slot 1 always UVR,
        keyed by serial number rather than USB enumeration order — so a lone
        UVR camera shows up as "Camera 1", not "Camera 0", and which physical
        camera lands where no longer depends on connection order.
        """
        devices = self.enumerate_devices()
        if not devices:
            raise RuntimeError("No cameras found.")
        pairs, updates = self._resolve_slot_order(devices)
        for serial, label in updates.items():
            app_config.save_camera_identity(serial, label)
        pairs = pairs[:max_devices]
        opened = self.open([device_index for device_index, _ in pairs])
        self._slots = [slot for _, slot in pairs]
        return opened

    def _slot(self, index: int) -> int:
        """Translate an internal (dense) camera index to its UVL/UVR slot."""
        if 0 <= index < len(self._slots):
            return self._slots[index]
        return index

    def close(self) -> None:
        self.stop_grabbing()
        self.stop_recording()
        for cam in self._cameras:
            self._dispose_camera(cam)
        self._cameras.clear()
        self._slots.clear()
        self._device_removed.clear()

    @staticmethod
    def _dispose_camera(cam: "pylon.InstantCamera") -> None:
        """Stop / close / destroy a camera, including after physical removal."""
        try:
            if cam.IsGrabbing():
                cam.StopGrabbing()
        except Exception:
            pass
        try:
            if cam.IsOpen():
                cam.Close()
        except Exception:
            pass
        try:
            # Required after physical removal so the TL can rediscover the device.
            cam.DestroyDevice()
        except Exception:
            pass

    # ----- camera features (frame rate / exposure / gain / gamma) -----------------

    @staticmethod
    def _frame_rate_node(cam: object):
        for name in _FPS_VALUE_NODES:
            node = _node(cam, name)
            if node is not None:
                return node
        return None

    @staticmethod
    def _enable_frame_rate(cam: object) -> None:
        node = _node(cam, "AcquisitionFrameRateEnable")
        if node is None:
            return
        try:
            if not node.GetValue():
                node.SetValue(True)
        except Exception:
            pass

    def _configure_frame_rate(self, cam: object, fps: float) -> None:
        """Limit free-run acquisition to fps (requires ExposureTime ≤ 1/fps)."""
        self._enable_frame_rate(cam)
        node = self._frame_rate_node(cam)
        if node is None:
            return
        clamped = min(max(float(fps), float(node.GetMin())), float(node.GetMax()))
        node.SetValue(clamped)

    def get_feature_info(self, name: str) -> FeatureInfo | None:
        """Read value + range of a float feature from the first camera."""
        if not self._cameras:
            return None
        node = (
            self._frame_rate_node(self._cameras[0])
            if name == "AcquisitionFrameRate"
            else _node(self._cameras[0], name)
        )
        if node is None:
            return None
        try:
            return FeatureInfo(
                name=name,
                value=float(node.GetValue()),
                minimum=float(node.GetMin()),
                maximum=float(node.GetMax()),
            )
        except Exception:
            return None

    def _target_fps(self, cam: object) -> float | None:
        node = self._frame_rate_node(cam)
        if node is None:
            return None
        try:
            return float(node.GetValue())
        except Exception:
            return None

    def _exposure_cap(self, cam: object) -> float | None:
        """Longest ExposureTime (µs) that fits one frame period at the set rate."""
        fps = self._target_fps(cam)
        if fps is None or fps <= 0:
            return None
        return 1_000_000.0 / fps - _EXPOSURE_MARGIN_US

    @staticmethod
    def _resulting_fps(cam: object) -> float | None:
        for name in ("ResultingFrameRate", "ResultingFrameRateAbs"):
            node = _node(cam, name)
            if node is None:
                continue
            try:
                return float(node.GetValue())
            except Exception:
                continue
        return None

    def maximize_exposure_for_frame_rate(self) -> bool:
        """Set ExposureTime on every camera to the longest value that still
        reaches the set frame rate.

        Returns False when a camera cannot reach the rate (ROI / readout /
        bandwidth limit); its exposure is then left at one frame period.
        """
        reachable = True
        for cam in self._cameras:
            target = self._target_fps(cam)
            node = _node(cam, "ExposureTime")
            if target is None or node is None:
                continue
            auto_node = _node(cam, "ExposureAuto")
            if auto_node is not None:
                try:
                    if auto_node.GetValue() != "Off":
                        auto_node.SetValue("Off")
                except Exception:
                    pass
            minimum = float(node.GetMin())
            cap = self._exposure_cap(cam)
            if cap is None:
                continue
            exposure = min(max(cap, minimum), float(node.GetMax()))
            node.SetValue(exposure)
            # The frame period also includes sensor overhead the cap does not
            # know about, so follow ResultingFrameRate down.  Stop (and keep
            # the brighter exposure) once shortening no longer helps: the
            # limit is then readout / bandwidth, not exposure.
            resulting = self._resulting_fps(cam)
            for _ in range(10):
                if resulting is None or resulting >= target * _FPS_TOLERANCE:
                    break
                if exposure <= minimum:
                    reachable = False
                    break
                shorter = max(exposure * resulting / target * 0.98, minimum)
                node.SetValue(shorter)
                improved = self._resulting_fps(cam)
                if improved is None or improved <= resulting * 1.001:
                    node.SetValue(exposure)
                    reachable = False
                    break
                exposure, resulting = shorter, improved
            else:
                reachable = False
        return reachable

    def get_resulting_fps(self) -> float | None:
        """Read the camera's achievable frame rate under current settings.

        Follows AcquisitionFrameRate (67 by default) plus ExposureTime / ROI /
        bandwidth limits.
        """
        if not self._cameras:
            return None
        return self._resulting_fps(self._cameras[0])

    def set_feature(self, name: str, value: float) -> None:
        """Set a float feature on all open cameras (clamped to valid range).

        Disables the matching auto function (ExposureAuto/GainAuto) first so
        the manual value sticks. Safe to call while grabbing.
        """
        if not self._cameras:
            raise RuntimeError("No cameras open.")
        errors: list[str] = []
        for i, cam in enumerate(self._cameras):
            try:
                auto_name = _AUTO_NODE.get(name)
                if auto_name is not None:
                    auto_node = _node(cam, auto_name)
                    if auto_node is not None:
                        try:
                            if auto_node.GetValue() != "Off":
                                auto_node.SetValue("Off")
                        except Exception:
                            pass
                if name == "AcquisitionFrameRate":
                    self._enable_frame_rate(cam)
                    node = self._frame_rate_node(cam)
                else:
                    node = _node(cam, name)
                if node is None:
                    raise RuntimeError(f"feature not available: {name}")
                clamped = min(max(value, float(node.GetMin())), float(node.GetMax()))
                node.SetValue(clamped)
            except Exception as exc:
                errors.append(f"cam{i}: {exc}")
        if errors:
            raise RuntimeError(f"Set {name} failed — " + "; ".join(errors))

    # ----- grabbing --------------------------------------------------------------

    def start_grabbing(self) -> None:
        if not self._cameras:
            raise RuntimeError("No cameras open.")
        if self._grabbing:
            return
        self._device_removed.clear()
        self._last_preview_at.clear()
        # BlockID may restart with a new grab session.
        self._last_block_id.clear()
        for cam in self._cameras:
            sync_mod.configure_free_run(cam)
            if not cam.IsGrabbing():
                strategy = (
                    pylon.GrabStrategy_OneByOne
                    if self._recording
                    else pylon.GrabStrategy_LatestImageOnly
                )
                cam.StartGrabbing(strategy)
        self._stop.clear()
        self._grabbing = True
        self._threads = []
        for i, cam in enumerate(self._cameras):
            t = threading.Thread(
                target=self._grab_loop,
                args=(i, cam),
                name=f"pyviewr-grab-{i}",
                daemon=True,
            )
            t.start()
            self._threads.append(t)

    def stop_grabbing(self) -> None:
        self._stop.set()
        for t in self._threads:
            t.join(timeout=2.0)
        self._threads.clear()
        self._last_preview_at.clear()
        for cam in self._cameras:
            try:
                if cam.IsGrabbing():
                    cam.StopGrabbing()
            except Exception:
                pass
        self._grabbing = False

    def _camera_device_removed(self, cam: "pylon.InstantCamera") -> bool:
        try:
            return bool(cam.IsCameraDeviceRemoved())
        except Exception:
            return False

    def _notify_device_removed(self, index: int) -> None:
        """Stop all grab loops and notify once that a camera vanished."""
        if self._device_removed.is_set():
            return
        self._device_removed.set()
        self._stop.set()
        if self._on_device_removed is not None:
            self._on_device_removed(index)
        elif self._on_error is not None:
            self._on_error(
                f"Camera {index}: device physically removed — disconnecting"
            )

    def _grab_loop(self, index: int, cam: "pylon.InstantCamera") -> None:
        while not self._stop.is_set():
            try:
                if self._camera_device_removed(cam):
                    self._notify_device_removed(self._slot(index))
                    return
                if not cam.IsGrabbing():
                    time.sleep(0.01)
                    continue
                grab = cam.RetrieveResult(200, pylon.TimeoutHandling_Return)
                with grab:
                    if not grab.GrabSucceeded():
                        if self._recording:
                            self._incomplete[index] = (
                                self._incomplete.get(index, 0) + 1
                            )
                        continue
                    frame = array_from_grab(grab)
                    block_id = _block_id(grab)
                with self._lock:
                    rec = self._recorders.get(index)
                if rec is not None:
                    self._count_dropped(index, block_id)
                # Preview is intentionally throttled.  During recording the
                # acquisition threads must not flood the Qt event queue with
                # one signal per camera frame.  Emit it before the disk
                # write, which can block for a while under backpressure —
                # otherwise a slow disk would stall this camera's preview,
                # and (since the write used to happen under the lock shared
                # with the other camera's grab thread) the other camera's
                # preview too.
                now = time.monotonic()
                last_preview = self._last_preview_at.get(index, 0.0)
                if (
                    self._on_frame is not None
                    and now - last_preview >= 1.0 / _PREVIEW_FPS
                ):
                    self._last_preview_at[index] = now
                    self._on_frame(self._slot(index), frame)
                if rec is not None:
                    rec.write(frame)
            except Exception as exc:
                # Physical unplug / link loss: InstantCamera stays open but is dead.
                if self._camera_device_removed(cam) or _is_physically_removed_message(
                    str(exc)
                ):
                    self._notify_device_removed(self._slot(index))
                    return
                if self._on_error is not None:
                    self._on_error(f"Camera {self._slot(index)}: {exc}")
                time.sleep(0.05)

    def _count_dropped(self, index: int, block_id: int | None) -> None:
        if block_id is None or block_id == 0:
            self._dropped[index] = None
            return
        last = self._last_block_id.get(index)
        self._last_block_id[index] = block_id
        dropped = self._dropped.get(index)
        # A smaller id means the counter wrapped; skip that interval.
        if last is not None and dropped is not None and block_id > last:
            self._dropped[index] = dropped + block_id - last - 1

    # ----- still -----------------------------------------------------------------

    def capture_still(self, save_dir: Path) -> list[Path]:
        """Capture one frame per open camera. Sync via Action Command when 2+ cams."""
        if not self._cameras:
            raise RuntimeError("No cameras open.")

        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        paths: list[Path] = []

        was_grabbing = self._grabbing
        if was_grabbing:
            self.stop_grabbing()

        try:
            if len(self._cameras) >= 2:
                paths = self._capture_still_action(save_dir, stamp)
            else:
                paths = self._capture_still_software(save_dir, stamp)
        finally:
            if was_grabbing:
                self.start_grabbing()
        return paths

    def _capture_still_software(self, save_dir: Path, stamp: str) -> list[Path]:
        paths: list[Path] = []
        cam = self._cameras[0]
        sync_mod.configure_free_run(cam)
        cam.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
        try:
            grab = cam.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
            with grab:
                if not grab.GrabSucceeded():
                    raise RuntimeError("Still grab failed.")
                frame = array_from_grab(grab)
            path = save_dir / f"cam{self._slot(0)}_{stamp}.png"
            save_still(path, self._frame_for_save(frame))
            paths.append(path)
        finally:
            if cam.IsGrabbing():
                cam.StopGrabbing()
        return paths

    def _capture_still_action(self, save_dir: Path, stamp: str) -> list[Path]:
        paths: list[Path] = []
        for cam in self._cameras:
            sync_mod.configure_action_trigger(cam)
            cam.StartGrabbing(pylon.GrabStrategy_OneByOne)

        try:
            # Arm all cameras, then fire one broadcast action command.
            time.sleep(0.05)
            sync_mod.issue_action_command(self._tl_factory)

            frames: list[np.ndarray] = []
            for i, cam in enumerate(self._cameras):
                grab = cam.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
                with grab:
                    if not grab.GrabSucceeded():
                        raise RuntimeError(
                            f"Still grab failed on camera {self._slot(i)}."
                        )
                    frames.append(array_from_grab(grab))

            for i, frame in enumerate(frames):
                path = save_dir / f"cam{self._slot(i)}_{stamp}.png"
                save_still(path, self._frame_for_save(frame))
                paths.append(path)
        finally:
            for cam in self._cameras:
                try:
                    if cam.IsGrabbing():
                        cam.StopGrabbing()
                except Exception:
                    pass
                try:
                    sync_mod.configure_free_run(cam)
                except Exception:
                    pass
        return paths

    # ----- video -----------------------------------------------------------------

    def start_recording(
        self,
        save_dir: Path,
        fps: float | None = None,
        lossless: bool = False,
    ) -> list[Path]:
        if not self._cameras:
            raise RuntimeError("No cameras open.")
        if self._recording:
            raise RuntimeError("Already recording.")
        if fps is None:
            fps = self.get_resulting_fps() or DEFAULT_VIDEO_FPS

        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]

        # Probe size from a quick grab if not already grabbing.
        sizes: list[tuple[int, int]] = []
        if not self._grabbing:
            self.start_grabbing()
            time.sleep(0.2)

        for i, cam in enumerate(self._cameras):
            w = int(cam.Width.GetValue())
            h = int(cam.Height.GetValue())
            sizes.append((w, h))

        ext = "mkv" if lossless else "mp4"
        paths: list[Path] = []
        with self._lock:
            self._recorders.clear()
            self._dropped = {i: 0 for i in range(len(sizes))}
            self._incomplete = {i: 0 for i in range(len(sizes))}
            for i, (w, h) in enumerate(sizes):
                path = save_dir / f"cam{self._slot(i)}_{stamp}.{ext}"
                self._recorders[i] = VideoRecorder(
                    path,
                    (w, h),
                    fps=fps,
                    frame_processor=self._frame_for_save,
                    lossless=lossless,
                    on_complete=lambda path, index=i: self._video_postprocess_done(
                        index, path
                    ),
                    on_error=lambda message, index=i: self._video_postprocess_error(
                        index, message
                    ),
                )
                paths.append(path)
            self._recording = True

        # Align start for multi-cam with a single action pulse then free-run.
        if len(self._cameras) >= 2:
            self._restart_free_run_aligned()
        else:
            # Re-open the single-camera grab stream with OneByOne.  The
            # connected preview normally uses LatestImageOnly.
            self.stop_grabbing()
            self.start_grabbing()
        self._underruns_at_start = {
            i: _stream_stat(cam, _UNDERRUN_STAT)
            for i, cam in enumerate(self._cameras)
        }
        return paths

    def _restart_free_run_aligned(self) -> None:
        """Stop, arm action once, then continue free-run recording."""
        was = self._grabbing
        if was:
            self.stop_grabbing()
        for cam in self._cameras:
            sync_mod.configure_action_trigger(cam)
            cam.StartGrabbing(pylon.GrabStrategy_OneByOne)
        try:
            time.sleep(0.05)
            sync_mod.issue_action_command(self._tl_factory)
            for idx, cam in enumerate(self._cameras):
                grab = cam.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
                with grab:
                    if grab.GrabSucceeded():
                        frame = array_from_grab(grab)
                        with self._lock:
                            rec = self._recorders.get(idx)
                        if rec is not None:
                            rec.write(frame)
        finally:
            for cam in self._cameras:
                try:
                    if cam.IsGrabbing():
                        cam.StopGrabbing()
                except Exception:
                    pass
                sync_mod.configure_free_run(cam)
            self.start_grabbing()

    def _video_postprocess_done(self, index: int, path: Path) -> None:
        if self._on_error is not None:
            self._on_error(f"Camera {self._slot(index)}: video saved: {path.name}")

    def _video_postprocess_error(self, index: int, message: str) -> None:
        if self._on_error is not None:
            self._on_error(
                f"Camera {self._slot(index)}: video post-processing failed: {message}"
            )

    def stop_recording(self) -> list[RecordingSummary]:
        with self._lock:
            recording = self._recording or bool(self._recorders)
        if not recording:
            return []

        underruns: dict[int, int | None] = {}
        for i, cam in enumerate(self._cameras):
            end = _stream_stat(cam, _UNDERRUN_STAT)
            start = self._underruns_at_start.get(i)
            if end is None or start is None:
                underruns[i] = None
            else:
                # Counter may restart with a new grab session.
                underruns[i] = end - start if end >= start else end
        packet_sizes: dict[int, int | None] = {}
        for i, cam in enumerate(self._cameras):
            node = _node(cam, "GevSCPSPacketSize")
            try:
                packet_sizes[i] = int(node.GetValue()) if node is not None else None
            except Exception:
                packet_sizes[i] = None

        was_grabbing = self._grabbing
        if was_grabbing:
            # Stop retrieval first so every frame already accepted by the
            # recorder can drain to the raw spool before post-processing.
            self.stop_grabbing()

        with self._lock:
            recorders = list(self._recorders.items())
            ended_at = time.monotonic()
            self._recorders.clear()
            self._recording = False

        first_error: Exception | None = None
        summaries: list[RecordingSummary] = []
        for index, rec in recorders:
            try:
                rec.close(ended_at)
                summaries.append(
                    RecordingSummary(
                        slot=self._slot(index),
                        frames=rec.frame_count,
                        fps=rec.measured_fps(ended_at),
                        disk_waits=rec.backpressure_events,
                        camera_dropped=self._dropped.get(index),
                        incomplete=self._incomplete.get(index, 0),
                        buffer_underruns=underruns.get(index),
                        packet_size=packet_sizes.get(index),
                    )
                )
            except Exception as exc:
                if first_error is None:
                    first_error = exc

        if was_grabbing and self._cameras:
            self.start_grabbing()
        if first_error is not None:
            raise first_error
        return summaries
