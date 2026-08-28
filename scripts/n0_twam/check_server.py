#!/usr/bin/env python3
"""Fail-fast metadata probe for an already running official N0 server."""

from __future__ import annotations

import argparse
import json
import socket
from pathlib import Path

from robotactile_benchmark.transport.n0_official import load_official_n0_rpc


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=29601)
    parser.add_argument("--timeout", type=float, default=10.0)
    args = parser.parse_args()
    if args.timeout <= 0.0 or args.timeout > 60.0:
        raise SystemExit("--timeout must be in (0,60]")
    with socket.create_connection((args.host, args.port), timeout=args.timeout):
        pass
    rpc = load_official_n0_rpc(
        source_root=args.source_root,
        host=args.host,
        port=args.port,
    )
    try:
        metadata = dict(rpc.get_server_metadata())
    finally:
        rpc.close()
    print(
        json.dumps(
            {
                "host": args.host,
                "metadata": metadata,
                "official_protocol_reachable": True,
                "port": args.port,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
