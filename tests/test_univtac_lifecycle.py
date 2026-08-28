from __future__ import annotations

import signal
import types
import unittest
from types import FrameType
from typing import Optional

from robotactile_benchmark.backends.univtac_contracts import UniVTACContractError
from robotactile_benchmark.backends.univtac_lifecycle import (
    HANG_DETECTOR_SETTING_PATH,
    install_runtime_signal_tracing,
    require_hang_detector_disabled,
)


class UniVTACLifecycleTests(unittest.TestCase):
    def test_hang_detector_must_be_exact_false(self) -> None:
        paths: list[str] = []
        settings = types.SimpleNamespace(get=lambda path: paths.append(path) or False)
        carb = types.SimpleNamespace(
            settings=types.SimpleNamespace(get_settings=lambda: settings)
        )

        require_hang_detector_disabled(lambda name: carb)

        self.assertEqual(paths, [HANG_DETECTOR_SETTING_PATH])

    def test_hang_detector_rejects_true_missing_and_integer_zero(self) -> None:
        for value in (True, None, 0):
            with self.subTest(value=value):
                settings = types.SimpleNamespace(
                    get=lambda path, observed=value: observed
                )
                carb = types.SimpleNamespace(
                    settings=types.SimpleNamespace(
                        get_settings=lambda current=settings: current
                    )
                )
                with self.assertRaisesRegex(
                    UniVTACContractError,
                    "must be exactly false",
                ):
                    require_hang_detector_disabled(lambda name, module=carb: module)

    def test_signal_wrappers_emit_marker_before_original_handler(self) -> None:
        events: list[object] = []
        handlers: dict[int, object] = {}
        specs = ((signal.SIGTERM, "SIGTERM"), (signal.SIGABRT, "SIGABRT"))

        def original_handler(
            received_signum: int,
            frame: Optional[FrameType],
        ) -> str:
            events.append(("original", received_signum, frame))
            return "handled"

        handlers.update({signum: original_handler for signum, _ in specs})

        def set_handler(signum: int, handler: object) -> object:
            previous = handlers[signum]
            handlers[signum] = handler
            return previous

        registrations = install_runtime_signal_tracing(
            signal_specs=specs,
            get_handler=lambda signum: handlers[signum],
            set_handler=set_handler,  # type: ignore[arg-type]
            write_bytes=lambda fd, payload: (
                events.append(("write", fd, payload)) or len(payload)
            ),
        )

        result = registrations[0].tracing_handler(signal.SIGTERM, None)

        self.assertEqual(result, "handled")
        self.assertEqual(
            events,
            [
                (
                    "write",
                    2,
                    b"ROBOTACTILE_RUNTIME_SIGNAL name=SIGTERM signum=15 "
                    b"handler=AppLauncher\n",
                ),
                ("original", signal.SIGTERM, None),
            ],
        )
        self.assertIs(registrations[0].original_handler, original_handler)
        self.assertIs(handlers[signal.SIGTERM], registrations[0].tracing_handler)

    def test_signal_marker_write_failure_preserves_original_semantics(self) -> None:
        events: list[int] = []

        def original_handler(signum: int, frame: Optional[FrameType]) -> int:
            events.append(signum)
            return 7

        def failed_write(fd: int, payload: bytes) -> int:
            raise OSError("stderr unavailable")

        registrations = install_runtime_signal_tracing(
            signal_specs=((signal.SIGTERM, "SIGTERM"),),
            get_handler=lambda signum: original_handler,
            set_handler=lambda signum, handler: original_handler,
            write_bytes=failed_write,
        )

        self.assertEqual(
            registrations[0].tracing_handler(signal.SIGTERM, None),
            7,
        )
        self.assertEqual(events, [signal.SIGTERM])

    def test_non_callable_handler_fails_before_any_installation(self) -> None:
        installed: list[int] = []

        with self.assertRaisesRegex(
            UniVTACContractError,
            "callable SIGTERM handler",
        ):
            install_runtime_signal_tracing(
                signal_specs=((signal.SIGTERM, "SIGTERM"),),
                get_handler=lambda signum: signal.SIG_DFL,
                set_handler=lambda signum, handler: installed.append(signum),
            )

        self.assertEqual(installed, [])


if __name__ == "__main__":
    unittest.main()
