"""Import the pinned TacEx packages after starting Isaac Sim headlessly."""

from __future__ import annotations

import json

from isaaclab.app import AppLauncher  # type: ignore[import-not-found]


def main() -> int:
    """Launch Isaac, import TacEx extensions, and close the app cleanly."""
    launcher = AppLauncher(headless=True)
    simulation_app = launcher.app
    try:
        import tacex  # type: ignore[import-not-found]
        import tacex_assets  # type: ignore[import-not-found]
        import tacex_tasks  # type: ignore[import-not-found]
        import torch_scatter  # type: ignore[import-not-found]

        payload = {
            "isaac_app_running": simulation_app.is_running(),
            "status": "passed",
            "tacex": tacex.__file__,
            "tacex_assets": tacex_assets.__file__,
            "tacex_tasks": tacex_tasks.__file__,
            "torch_scatter": torch_scatter.__file__,
        }
        print(json.dumps(payload, sort_keys=True))
    finally:
        simulation_app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
