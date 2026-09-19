#!/usr/bin/env python3
"""Receive one authorized binary asset into an isolated no-clobber directory."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allowed-root", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--expected-size", type=int, required=True)
    args = parser.parse_args()
    root = args.allowed_root.resolve(strict=True)
    target = args.target.absolute()
    if not target.is_relative_to(root) or target == root:
        raise ValueError("asset target is outside the isolated deployment")
    if target.exists() or target.is_symlink():
        raise FileExistsError("asset target already exists")
    if len(args.expected_sha256) != 64 or args.expected_size <= 0:
        raise ValueError("invalid expected asset identity")
    target.parent.mkdir(parents=True, exist_ok=False)
    partial = target.with_name(target.name + ".partial-v1")
    digest = hashlib.sha256()
    size = 0
    with partial.open("xb") as stream:
        for block in iter(lambda: sys.stdin.buffer.read(8 * 1024 * 1024), b""):
            digest.update(block)
            size += len(block)
            stream.write(block)
        stream.flush()
        os.fsync(stream.fileno())
    if size != args.expected_size or digest.hexdigest() != args.expected_sha256:
        raise ValueError("received asset does not match expected size and SHA256")
    os.link(partial, target)
    partial.unlink()
    receipt = {
        "target": str(target),
        "size_bytes": size,
        "sha256": digest.hexdigest(),
        "publication": "verified_no_clobber_hardlink",
    }
    with target.with_name(target.name + ".receipt.json").open("x") as stream:
        json.dump(receipt, stream, sort_keys=True, indent=2)
        stream.write("\n")
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
