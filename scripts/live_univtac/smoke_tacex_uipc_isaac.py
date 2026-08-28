"""Import modified UIPC and TacEx after starting Isaac Sim headlessly."""

from __future__ import annotations

import importlib.metadata
import json

from isaaclab.app import AppLauncher  # type: ignore[import-not-found]


def main() -> int:
    """Launch Isaac, import the native tactile stack, and close cleanly."""
    launcher = AppLauncher(headless=True)
    simulation_app = launcher.app
    try:
        import tacex_uipc  # type: ignore[import-not-found]
        import uipc  # type: ignore[import-not-found]

        payload = {
            "isaac_app_running": simulation_app.is_running(),
            "status": "passed",
            "tacex_uipc": tacex_uipc.__file__,
            "tacex_uipc_version": importlib.metadata.version("tacex-uipc"),
            "uipc": uipc.__file__,
        }
        print(json.dumps(payload, sort_keys=True))
    finally:
        simulation_app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
