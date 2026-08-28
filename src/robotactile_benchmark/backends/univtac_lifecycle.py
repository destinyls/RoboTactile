"""Fail-closed Isaac lifecycle checks and terminal-signal tracing."""

from __future__ import annotations

import importlib
import os
import signal
from collections.abc import Callable, Sequence
from contextlib import suppress
from dataclasses import dataclass
from types import FrameType, ModuleType
from typing import Optional, Tuple, cast

from robotactile_benchmark.backends.univtac_contracts import (
    UniVTACContractError,
)

HANG_DETECTOR_SETTING_PATH = "/app/hangDetector/enabled"
RUNTIME_SIGNAL_MARKER = "ROBOTACTILE_RUNTIME_SIGNAL"

SignalCallback = Callable[[int, Optional[FrameType]], object]
SignalGetter = Callable[[int], object]
SignalSetter = Callable[[int, SignalCallback], object]
ByteWriter = Callable[[int, bytes], int]
ModuleImporter = Callable[[str], ModuleType]


@dataclass(frozen=True)
class SignalTraceRegistration:
    """One installed wrapper and the AppLauncher handler it delegates to."""

    signum: int
    signal_name: str
    original_handler: SignalCallback
    tracing_handler: SignalCallback


_TERMINAL_SIGNAL_SPECS: Tuple[Tuple[int, str], ...] = (
    (signal.SIGTERM, "SIGTERM"),
    (signal.SIGABRT, "SIGABRT"),
    (signal.SIGSEGV, "SIGSEGV"),
)


def require_hang_detector_disabled(
    import_module: ModuleImporter = importlib.import_module,
) -> None:
    """Require Kit's effective hang-detector setting to be exactly ``False``."""

    carb = import_module("carb")
    settings_module = getattr(carb, "settings", None)
    get_settings = getattr(settings_module, "get_settings", None)
    if not callable(get_settings):
        raise UniVTACContractError("carb.settings.get_settings is unavailable")
    settings = get_settings()
    get_setting = getattr(settings, "get", None)
    if not callable(get_setting):
        raise UniVTACContractError("Isaac runtime settings.get is unavailable")
    value = get_setting(HANG_DETECTOR_SETTING_PATH)
    if value is not False:
        raise UniVTACContractError(
            "Isaac runtime /app/hangDetector/enabled must be exactly false; "
            f"observed {value!r}"
        )


def _signal_marker(signal_name: str, signum: int) -> bytes:
    return (
        f"{RUNTIME_SIGNAL_MARKER} name={signal_name} signum={signum} "
        "handler=AppLauncher\n"
    ).encode("ascii")


def _build_tracing_handler(
    *,
    signal_name: str,
    signum: int,
    original_handler: SignalCallback,
    write_bytes: ByteWriter,
) -> SignalCallback:
    marker = _signal_marker(signal_name, signum)

    def tracing_handler(received_signum: int, frame: Optional[FrameType]) -> object:
        with suppress(OSError):
            write_bytes(2, marker)
        # Evidence emission must never prevent AppLauncher from handling a signal.
        return original_handler(received_signum, frame)

    return tracing_handler


def install_runtime_signal_tracing(
    *,
    signal_specs: Sequence[Tuple[int, str]] = _TERMINAL_SIGNAL_SPECS,
    get_handler: Optional[SignalGetter] = None,
    set_handler: Optional[SignalSetter] = None,
    write_bytes: ByteWriter = os.write,
) -> Tuple[SignalTraceRegistration, ...]:
    """Wrap AppLauncher's terminal handlers without changing their semantics.

    Injectable signal operations keep unit tests isolated from process-global
    signal state. Production callers use ``signal.getsignal``/``signal.signal``.
    """

    resolved_getter = get_handler or cast(SignalGetter, signal.getsignal)
    resolved_setter = set_handler or cast(SignalSetter, signal.signal)
    originals: list[Tuple[int, str, SignalCallback]] = []
    for signum, signal_name in signal_specs:
        handler = resolved_getter(signum)
        if not callable(handler):
            raise UniVTACContractError(
                f"AppLauncher did not register a callable {signal_name} handler"
            )
        originals.append((signum, signal_name, cast(SignalCallback, handler)))

    registrations: list[SignalTraceRegistration] = []
    try:
        for signum, signal_name, original_handler in originals:
            tracing_handler = _build_tracing_handler(
                signal_name=signal_name,
                signum=signum,
                original_handler=original_handler,
                write_bytes=write_bytes,
            )
            resolved_setter(signum, tracing_handler)
            registrations.append(
                SignalTraceRegistration(
                    signum=signum,
                    signal_name=signal_name,
                    original_handler=original_handler,
                    tracing_handler=tracing_handler,
                )
            )
    except Exception as error:
        for registration in reversed(registrations):
            with suppress(Exception):
                resolved_setter(
                    registration.signum,
                    registration.original_handler,
                )
        raise UniVTACContractError(
            "unable to install Isaac runtime signal tracing"
        ) from error
    return tuple(registrations)
