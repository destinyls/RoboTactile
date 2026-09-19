"""Canonical module entrypoint for Dream-Tac train759 materialization."""

from .materialize import (
    dry_run_plan,
    episode_destination,
    main,
    materialize_episode,
    materialize_train759,
    verify_materialization,
)

if __name__ == "__main__":
    raise SystemExit(main())

__all__ = [
    "dry_run_plan",
    "episode_destination",
    "main",
    "materialize_episode",
    "materialize_train759",
    "verify_materialization",
]
