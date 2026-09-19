#!/usr/bin/env python3
"""Single-task retrained N0 Clean entrypoint; run inside the Isaac runtime."""

from robotactile_benchmark.integrations.n0_twam.retrained_live import main

if __name__ == "__main__":
    main()
