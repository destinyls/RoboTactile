"""Contract tests for the Dream-Tac global 1D FSDP compatibility patch."""

from __future__ import annotations

import subprocess
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PATCH_PATH = (
    REPOSITORY_ROOT
    / "scripts"
    / "dream_tac"
    / "hcu_port"
    / "overlays"
    / "global_fsdp_1d.patch"
)
FSDP_HELPER_PATH = Path("cosmos_policy/_src/imaginaire/utils/fsdp_helper.py")
DTENSOR_HELPER_PATH = Path("cosmos_policy/_src/predict2/utils/dtensor_helper.py")

FSDP_V8_PREIMAGE = """def hsdp_device_mesh(
    replica_group_size=None, sharding_group_size=None, device=None
):
    world_size = distributed.get_world_size()
    if sharding_group_size is None:
        sharding_group_size = min(world_size, 8)
    sharding_group_size = min(sharding_group_size, world_size)
    if replica_group_size is None:
        replica_group_size = world_size // sharding_group_size

    device = device or "cuda"

    if world_size % sharding_group_size != 0:
        raise ValueError(
            f"World size {world_size} is not evenly divisible by sharding group size {sharding_group_size}."
        )

    if (world_size // sharding_group_size) % replica_group_size != 0:
        raise ValueError(
            f"The calculated number of replica groups is not evenly divisible by "
            f"replica_group_size {replica_group_size}."
        )

    device_mesh = init_device_mesh(
        device, (replica_group_size, sharding_group_size), mesh_dim_names=("replicate", "shard")
    )
    if device_mesh is None:
        raise RuntimeError("Failed to create a valid device mesh.")

    log.critical(
        f"Device mesh initialized with replica group size {replica_group_size} and sharding group size {sharding_group_size}"
    )

    return device_mesh
"""

DTENSOR_V8_PREIMAGE = '''class DTensorFastEmaModelUpdater:
    pass


def broadcast_dtensor_model_states(model: torch.nn.Module, mesh: DeviceMesh):
    """Broadcast model states from replicate mesh's rank 0."""
    replicate_group = mesh.get_group("replicate")
    all_ranks = dist.get_process_group_ranks(replicate_group)
    if len(all_ranks) == 1:
        return

    for _, tensor in itertools.chain(model.named_parameters(), model.named_buffers()):
        # Get src rank which is the first rank in each replication group
        src_rank = all_ranks[0]
        # Broadcast the local tensor
        local_tensor = get_local_tensor_if_DTensor(tensor)
        dist.broadcast(
            local_tensor,
            src=src_rank,
            group=replicate_group,
        )
'''


def _write_preimage(root: Path, relative_path: Path, source: str) -> None:
    target = root / relative_path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(source, encoding="utf-8")


def test_patch_is_scoped_to_global_1d_fsdp_compatibility() -> None:
    patch = PATCH_PATH.read_text(encoding="utf-8")

    assert patch.count("diff --git ") == 2
    assert "if replica_group_size == 1:" in patch
    assert 'mesh_dim_names=("shard",)' in patch
    assert 'mesh_dim_names=("replicate", "shard")' in patch
    assert (
        'if mesh.mesh_dim_names is None or "replicate" not in mesh.mesh_dim_names:'
    ) in patch


def test_patch_applies_to_exact_v8_function_preimages(tmp_path: Path) -> None:
    source_root = tmp_path / "dream-tac-v8"
    _write_preimage(source_root, FSDP_HELPER_PATH, FSDP_V8_PREIMAGE)
    _write_preimage(source_root, DTENSOR_HELPER_PATH, DTENSOR_V8_PREIMAGE)

    completed = subprocess.run(
        ["patch", "-f", "-N", "-p1", "-i", str(PATCH_PATH)],
        cwd=source_root,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    fsdp_source = (source_root / FSDP_HELPER_PATH).read_text(encoding="utf-8")
    dtensor_source = (source_root / DTENSOR_HELPER_PATH).read_text(encoding="utf-8")

    assert 'device, (sharding_group_size,), mesh_dim_names=("shard",)' in fsdp_source
    assert "else:\n        device_mesh = init_device_mesh(" in fsdp_source
    guard = 'if mesh.mesh_dim_names is None or "replicate" not in mesh.mesh_dim_names:'
    assert dtensor_source.index(guard) < dtensor_source.index(
        'mesh.get_group("replicate")'
    )
