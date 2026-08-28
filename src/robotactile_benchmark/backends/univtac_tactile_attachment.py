"""Source-bound GS Mini gel attachment repair for pinned UniVTAC assets."""

from __future__ import annotations

import hashlib
import importlib
import json
from collections.abc import Mapping, Sequence
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from robotactile_benchmark.backends.univtac_contracts import UniVTACContractError

_SUPPORTED_TASKS = frozenset(
    {
        "grasp_classify",
        "insert_HDMI",
        "insert_hole",
        "insert_tube",
        "lift_bottle",
        "lift_can",
        "pull_out_key",
        "put_bottle_in_shelf",
    }
)
_EXPECTED_TACTILES = {
    "left_tactile": "gelsight_mini_case_left",
    "right_tactile": "gelsight_mini_case_right",
}
_ATTACHMENT_VERTEX_INDICES = (
    0,
    1,
    3,
    4,
    5,
    8,
    9,
    13,
    14,
    21,
    24,
    40,
    41,
    43,
    44,
    45,
    51,
    52,
    53,
    54,
    58,
    59,
    60,
    61,
    67,
    68,
    69,
    70,
    71,
    72,
    73,
    74,
    77,
    78,
    79,
    80,
    82,
    84,
    85,
    86,
    87,
    96,
    99,
    100,
    101,
    103,
    104,
    106,
    107,
    108,
    110,
    111,
    112,
    121,
    122,
    123,
    124,
    125,
    126,
    127,
    128,
    129,
    130,
    131,
    132,
    133,
    134,
    146,
    147,
    148,
    149,
    150,
    151,
    152,
    153,
    154,
    155,
    156,
    157,
    161,
    162,
    163,
    165,
)
_ATTACHMENT_INDICES_SHA256 = (
    "579470561efd284f7d209ade762a73c24ee763de96151d98cdb9b659b8aaefd7"
)
_EXPECTED_POINT_SHAPE = (169, 3)
_MAX_RECONSTRUCTION_ERROR_M = 1e-6
_MAX_INIT_CURRENT_ERROR_M = 1e-5
_PINNED_ASSET_SHA256 = {
    "franka_gsmini_robot_usd": (
        "3047b1b7bfc982ad35222604b36069d30db431bd773b62fc91b11667b219a1d9"
    ),
    "gsmini_gelpad_high_res_usd": (
        "d055c15b020aab286fcc768778b6adc3833043e6f983471418d31022da13fa0e"
    ),
}
_PINNED_UNIVTAC_COMMIT = "05bcd3edb92237107efa40105292a24f1a9fd761"


def _attachment_prim_path(attachment: Any) -> str:
    cfg = getattr(attachment, "cfg", None)
    explicit = getattr(cfg, "isaac_rigid_prim_path", None)
    if isinstance(explicit, str) and explicit:
        return explicit
    rigid = getattr(attachment, "isaaclab_rigid_object", None)
    base = getattr(getattr(rigid, "cfg", None), "prim_path", None)
    body_name = getattr(cfg, "body_name", None)
    if not isinstance(base, str) or not base or not isinstance(body_name, str):
        raise UniVTACContractError("GS Mini attachment prim path is unavailable")
    return f"{base}/.*{body_name}"


def _populate_constructor_attachment(attachment: Any) -> bool:
    """Populate pinned indices before TacEx creates its animation callback."""

    cfg = getattr(attachment, "cfg", None)
    body_name = getattr(cfg, "body_name", None)
    if body_name not in set(_EXPECTED_TACTILES.values()):
        return False
    before_raw = getattr(attachment, "attachment_points_idx", None)
    if not isinstance(before_raw, Sequence):
        raise UniVTACContractError("GS Mini constructor indices are unavailable")
    before = tuple(int(item) for item in before_raw)
    if before:
        if before != _ATTACHMENT_VERTEX_INDICES:
            raise UniVTACContractError(
                "GS Mini constructor indices differ from pinned assets"
            )
        return False
    compute = getattr(attachment, "compute_attachment_data_from_given_ids", None)
    uipc_object = getattr(attachment, "uipc_object", None)
    meshes = getattr(uipc_object, "uipc_meshes", None)
    if not callable(compute) or not isinstance(meshes, Sequence) or len(meshes) != 1:
        raise UniVTACContractError("GS Mini constructor mesh contract is incomplete")
    mesh = meshes[0]
    positions = getattr(mesh, "positions", None)
    tetrahedra = getattr(mesh, "tetrahedra", None)
    if not callable(positions) or not callable(tetrahedra):
        raise UniVTACContractError("GS Mini constructor topology is unavailable")
    position_view = getattr(positions(), "view", None)
    topology = tetrahedra()
    topology_view = getattr(getattr(topology, "topo", lambda: None)(), "view", None)
    if not callable(position_view) or not callable(topology_view):
        raise UniVTACContractError("GS Mini constructor views are unavailable")
    tet_points = np.asarray(position_view())[:, :, 0]
    tet_indices = np.asarray(topology_view())[:, :, 0]
    offsets = np.asarray(
        compute(
            _attachment_prim_path(attachment),
            tet_points,
            tet_indices,
            _ATTACHMENT_VERTEX_INDICES,
        ),
        dtype=np.float32,
    )
    if (
        offsets.shape != (len(_ATTACHMENT_VERTEX_INDICES), 3)
        or not np.isfinite(offsets).all()
    ):
        raise UniVTACContractError("GS Mini constructor offsets are invalid")
    attachment.attachment_offsets = offsets
    attachment.attachment_points_idx = list(_ATTACHMENT_VERTEX_INDICES)
    attachment.num_attachment_points_per_obj = len(_ATTACHMENT_VERTEX_INDICES)
    attachment._robotactile_constructor_attachment = {
        "body_name": body_name,
        "indices_sha256": _ATTACHMENT_INDICES_SHA256,
        "offsets_sha256": _offsets_sha256(offsets),
        "repair_applied": True,
    }
    return True


def install_gsmini_attachment_constructor_compatibility(task_id: str) -> bool:
    """Install the pinned pre-initialization hook for supported GS Mini tasks."""

    if task_id not in _SUPPORTED_TASKS:
        return False
    module = importlib.import_module("tacex_uipc")
    attachment_type = getattr(module, "UipcIsaacAttachments", None)
    if not isinstance(attachment_type, type):
        raise UniVTACContractError("tacex_uipc attachment type is unavailable")
    if getattr(attachment_type, "_robotactile_constructor_hook", False) is True:
        return True
    upstream_init = vars(attachment_type).get("__init__")
    if not callable(upstream_init):
        raise UniVTACContractError("tacex_uipc attachment constructor is unavailable")

    def pinned_init(self: Any, *args: Any, **kwargs: Any) -> None:
        upstream_init(self, *args, **kwargs)
        _populate_constructor_attachment(self)

    attachment_class = cast(Any, attachment_type)
    attachment_class.__init__ = pinned_init
    attachment_class._robotactile_constructor_hook = True
    return True


def _host_array(value: Any, *, dtype: Any = np.float64) -> NDArray[Any]:
    detached = getattr(value, "detach", None)
    if callable(detached):
        value = detached()
    cpu = getattr(value, "cpu", None)
    if callable(cpu):
        value = cpu()
    return cast(NDArray[Any], np.asarray(value, dtype=dtype))


def _indices_sha256(indices: Sequence[int]) -> str:
    payload = json.dumps(list(indices), separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _offsets_sha256(offsets: NDArray[np.float32]) -> str:
    contiguous = np.ascontiguousarray(offsets, dtype=np.float32)
    return hashlib.sha256(contiguous.tobytes(order="C")).hexdigest()


def _repair_one_tactile(
    tactile: Any,
    *,
    tactile_name: str,
    expected_body_name: str,
) -> Mapping[str, object]:
    attachment = getattr(tactile, "attachment", None)
    gelpad = getattr(tactile, "gelpad", None)
    setup = getattr(tactile, "setup", None)
    get_attach_pose = getattr(tactile, "get_attach_pose", None)
    compute_aim = getattr(attachment, "_compute_aim_positions", None)
    body_name = getattr(getattr(attachment, "cfg", None), "body_name", None)
    initialized = getattr(attachment, "is_initialized", None)
    rigid_body_id = getattr(attachment, "rigid_body_id", None)
    gelpad_prim_path = str(getattr(getattr(gelpad, "cfg", None), "prim_path", ""))
    if (
        attachment is None
        or gelpad is None
        or not callable(setup)
        or not callable(get_attach_pose)
        or not callable(compute_aim)
        or body_name != expected_body_name
        or initialized is not True
        or rigid_body_id is None
        or not gelpad_prim_path.endswith(tactile_name.replace("_tactile", ""))
    ):
        raise UniVTACContractError(
            f"{tactile_name} GS Mini attachment contract is incomplete"
        )

    before_raw = getattr(attachment, "attachment_points_idx", None)
    if not isinstance(before_raw, Sequence):
        raise UniVTACContractError(f"{tactile_name} attachment indices are unavailable")
    before = tuple(int(item) for item in before_raw)
    if before and before != _ATTACHMENT_VERTEX_INDICES:
        raise UniVTACContractError(
            f"{tactile_name} attachment indices differ from pinned assets"
        )

    initial_points = _host_array(getattr(gelpad, "init_vertex_pos", None))
    data = getattr(gelpad, "data", None)
    if data is None:
        data = getattr(gelpad, "_data", None)
    current_points = _host_array(getattr(data, "nodal_pos_w", None))
    if (
        initial_points.shape != _EXPECTED_POINT_SHAPE
        or current_points.shape != _EXPECTED_POINT_SHAPE
        or not np.isfinite(initial_points).all()
        or not np.isfinite(current_points).all()
    ):
        raise UniVTACContractError(
            f"{tactile_name} gel vertices must be finite shape [169,3]"
        )
    init_current_max_abs_m = float(np.max(np.abs(current_points - initial_points)))
    if init_current_max_abs_m > _MAX_INIT_CURRENT_ERROR_M:
        raise UniVTACContractError(
            f"{tactile_name} gel moved before attachment compatibility hook"
        )

    pose = get_attach_pose()
    matrix_method = getattr(pose, "to_transformation_matrix", None)
    if not callable(matrix_method):
        raise UniVTACContractError(f"{tactile_name} attachment pose is unavailable")
    transform = _host_array(matrix_method())
    if transform.shape != (4, 4) or not np.isfinite(transform).all():
        raise UniVTACContractError(
            f"{tactile_name} attachment transform must be finite [4,4]"
        )

    selected = initial_points[list(_ATTACHMENT_VERTEX_INDICES)]
    offsets = np.asarray(
        (selected - transform[:3, 3]) @ transform[:3, :3], dtype=np.float32
    )
    reconstructed = offsets.astype(np.float64) @ transform[:3, :3].T + transform[:3, 3]
    reconstruction_error_m = float(np.max(np.abs(reconstructed - selected)))
    if reconstruction_error_m > _MAX_RECONSTRUCTION_ERROR_M:
        raise UniVTACContractError(
            f"{tactile_name} attachment offset reconstruction failed"
        )

    if before:
        offsets = _host_array(
            getattr(attachment, "attachment_offsets", None), dtype=np.float32
        )
        if offsets.shape != (len(_ATTACHMENT_VERTEX_INDICES), 3):
            raise UniVTACContractError(f"{tactile_name} attachment offsets are invalid")
        compute_aim(0.0)
    else:
        attachment.attachment_points_idx = list(_ATTACHMENT_VERTEX_INDICES)
        attachment.attachment_offsets = offsets
        attachment.num_attachment_points_per_obj = len(_ATTACHMENT_VERTEX_INDICES)
        compute_aim(0.0)
        setup()

    aim_positions = _host_array(getattr(attachment, "aim_positions", None))
    origin_points = _host_array(getattr(tactile, "origin_pts", None))
    attach_to_init = _host_array(getattr(tactile, "attach_to_init", None))
    aim_vs_initial_max_abs_m = float(np.max(np.abs(aim_positions - selected)))
    constructor_compatibility = getattr(
        attachment,
        "_robotactile_constructor_attachment",
        None,
    )
    constructor_native = isinstance(constructor_compatibility, Mapping)
    if (
        aim_positions.shape != (len(_ATTACHMENT_VERTEX_INDICES), 3)
        or origin_points.shape != (len(_ATTACHMENT_VERTEX_INDICES), 3)
        or attach_to_init.shape != (4, 4)
        or not np.isfinite(aim_positions).all()
        or not np.isfinite(origin_points).all()
        or not np.isfinite(attach_to_init).all()
    ):
        raise UniVTACContractError(
            f"{tactile_name} repaired attachment state is invalid"
        )
    if not constructor_native and aim_vs_initial_max_abs_m > _MAX_INIT_CURRENT_ERROR_M:
        raise UniVTACContractError(
            f"{tactile_name} repaired attachment aim is misregistered"
        )
    return {
        "after_count": len(_ATTACHMENT_VERTEX_INDICES),
        "aim_count": int(aim_positions.shape[0]),
        "attach_to_init_finite": True,
        "aim_vs_initial_max_abs_m": aim_vs_initial_max_abs_m,
        "before_count": len(before),
        "body_name": body_name,
        "aim_registered_to_initial": (
            aim_vs_initial_max_abs_m <= _MAX_INIT_CURRENT_ERROR_M
        ),
        "constructor_compatibility": constructor_compatibility,
        "gelpad_prim_path": gelpad_prim_path,
        "indices_sha256": _ATTACHMENT_INDICES_SHA256,
        "init_current_max_abs_m": init_current_max_abs_m,
        "offsets_sha256": _offsets_sha256(offsets),
        "origin_count": int(origin_points.shape[0]),
        "reconstruction_error_m": reconstruction_error_m,
        "repair_applied": not bool(before),
        "rigid_body_id": str(rigid_body_id),
    }


def repair_gsmini_tactile_attachments(task: Any, task_id: str) -> bool:
    """Repair empty runtime sweep results using pinned GS Mini attachment ids."""

    if task_id not in _SUPPORTED_TASKS:
        return False
    if getattr(getattr(task, "cfg", None), "tactile_sensor_type", None) != "gsmini":
        raise UniVTACContractError(f"{task_id} requires the pinned GS Mini sensor")
    if _indices_sha256(_ATTACHMENT_VERTEX_INDICES) != _ATTACHMENT_INDICES_SHA256:
        raise UniVTACContractError("pinned GS Mini attachment index hash mismatch")
    manager = getattr(task, "_tactile_manager", None)
    tactiles = getattr(manager, "tactiles", None)
    logger = getattr(task, "logger", None)
    log_info = getattr(logger, "info", None)
    if not isinstance(tactiles, Mapping) or not callable(log_info):
        raise UniVTACContractError("GS Mini tactile manager contract is incomplete")
    if set(tactiles) != set(_EXPECTED_TACTILES):
        raise UniVTACContractError("GS Mini tactile sensor registry mismatch")

    sides = {
        name: dict(
            _repair_one_tactile(
                tactiles[name],
                tactile_name=name,
                expected_body_name=body_name,
            )
        )
        for name, body_name in _EXPECTED_TACTILES.items()
    }
    witness = {
        "attachment_count_per_side": len(_ATTACHMENT_VERTEX_INDICES),
        "indices_sha256": _ATTACHMENT_INDICES_SHA256,
        "pinned_asset_sha256": dict(_PINNED_ASSET_SHA256),
        "pinned_univtac_commit": _PINNED_UNIVTAC_COMMIT,
        "sides": sides,
        "task_id": task_id,
    }
    task._robotactile_tactile_attachment = witness
    task._robotactile_tactile_attachment_witness = witness
    log_info(
        "ROBOTACTILE_TACTILE_ATTACHMENT "
        + json.dumps(witness, separators=(",", ":"), sort_keys=True)
    )
    return True


__all__ = [
    "install_gsmini_attachment_constructor_compatibility",
    "repair_gsmini_tactile_attachments",
]
