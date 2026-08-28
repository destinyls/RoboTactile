from __future__ import annotations

import json
import types

import numpy as np
import pytest

from robotactile_benchmark.backends import univtac_tactile_attachment
from robotactile_benchmark.backends.univtac_contracts import UniVTACContractError
from robotactile_benchmark.backends.univtac_tactile_attachment import (
    repair_gsmini_tactile_attachments,
)


class _Pose:
    def to_transformation_matrix(self) -> np.ndarray:
        matrix = np.eye(4, dtype=np.float64)
        matrix[:3, 3] = (0.3, -0.1, 0.2)
        return matrix


class _Attachment:
    def __init__(self, body_name: str, indices: list[int]) -> None:
        self.cfg = types.SimpleNamespace(body_name=body_name)
        self.is_initialized = True
        self.rigid_body_id = [3]
        self.attachment_points_idx = indices
        self.attachment_offsets = np.empty((0, 3), dtype=np.float32)
        self.num_attachment_points_per_obj = 0
        self.aim_positions = np.empty((0, 3), dtype=np.float32)

    def _compute_aim_positions(self, _dt: float = 0.0) -> None:
        self.aim_positions = np.asarray(
            self.attachment_offsets,
            dtype=np.float32,
        ) + np.asarray((0.3, -0.1, 0.2), dtype=np.float32)


class _Tactile:
    def __init__(self, body_name: str, indices: list[int]) -> None:
        points = np.arange(169 * 3, dtype=np.float64).reshape(169, 3) / 10000.0
        points += np.asarray((0.3, -0.1, 0.2), dtype=np.float64)
        self.gelpad = types.SimpleNamespace(
            cfg=types.SimpleNamespace(
                prim_path="/World/envs/env_.*/Robot/"
                + body_name.replace("gelsight_mini_case", "gelpad")
            ),
            data=types.SimpleNamespace(nodal_pos_w=points.copy()),
            init_vertex_pos=points.copy(),
        )
        self.attachment = _Attachment(body_name, indices)
        self.origin_pts = np.empty((0, 3), dtype=np.float64)
        self.attach_to_init = np.full((4, 4), np.nan, dtype=np.float64)
        self.setup_count = 0

    def get_attach_pose(self) -> _Pose:
        return _Pose()

    def setup(self) -> None:
        self.setup_count += 1
        selected = self.gelpad.data.nodal_pos_w[self.attachment.attachment_points_idx]
        self.origin_pts = selected - np.asarray((0.3, -0.1, 0.2))
        self.attach_to_init = np.eye(4, dtype=np.float64)


def _task(*, indices: list[int] | None = None) -> tuple[object, list[str]]:
    messages: list[str] = []
    values = [] if indices is None else indices
    tactiles = {
        "left_tactile": _Tactile("gelsight_mini_case_left", values.copy()),
        "right_tactile": _Tactile("gelsight_mini_case_right", values.copy()),
    }
    task = types.SimpleNamespace(
        _tactile_manager=types.SimpleNamespace(tactiles=tactiles),
        cfg=types.SimpleNamespace(tactile_sensor_type="gsmini"),
        logger=types.SimpleNamespace(info=messages.append),
    )
    return task, messages


@pytest.mark.parametrize(
    "task_id",
    [
        "grasp_classify",
        "insert_HDMI",
        "insert_hole",
        "insert_tube",
        "lift_bottle",
        "lift_can",
        "pull_out_key",
        "put_bottle_in_shelf",
    ],
)
def test_repair_restores_both_pinned_gsmini_attachments(task_id: str) -> None:
    task, messages = _task()

    assert repair_gsmini_tactile_attachments(task, task_id)

    witness = task._robotactile_tactile_attachment
    assert witness["attachment_count_per_side"] == 83
    assert witness["task_id"] == task_id
    assert len(messages) == 1
    logged = json.loads(messages[0].split(" ", 1)[1])
    assert logged["indices_sha256"] == (
        "579470561efd284f7d209ade762a73c24ee763de96151d98cdb9b659b8aaefd7"
    )
    for tactile in task._tactile_manager.tactiles.values():
        assert len(tactile.attachment.attachment_points_idx) == 83
        assert tactile.attachment.attachment_offsets.shape == (83, 3)
        assert tactile.attachment.aim_positions.shape == (83, 3)
        assert tactile.setup_count == 1
        assert np.isfinite(tactile.attach_to_init).all()


def test_repair_rejects_non_pinned_existing_indices() -> None:
    task, _messages = _task(indices=[0, 2])

    with pytest.raises(
        UniVTACContractError,
        match="attachment indices differ from pinned assets",
    ):
        repair_gsmini_tactile_attachments(task, "grasp_classify")


def test_constructor_native_attachment_records_post_start_aim_drift() -> None:
    indices = list(univtac_tactile_attachment._ATTACHMENT_VERTEX_INDICES)
    task, _messages = _task(indices=indices)
    for tactile in task._tactile_manager.tactiles.values():
        tactile.attachment.attachment_offsets = np.zeros((83, 3), dtype=np.float32)
        tactile.attachment._robotactile_constructor_attachment = {
            "repair_applied": True
        }
        tactile.setup()

    assert repair_gsmini_tactile_attachments(task, "lift_bottle")

    for side in task._robotactile_tactile_attachment["sides"].values():
        assert side["aim_registered_to_initial"] is False
        assert side["aim_vs_initial_max_abs_m"] > 1e-5


def test_constructor_compatibility_uses_upstream_precomputed_offset_path() -> None:
    class View:
        def __init__(self, value: np.ndarray) -> None:
            self._value = value

        def view(self) -> np.ndarray:
            return self._value

    class Mesh:
        def positions(self) -> View:
            return View(np.zeros((169, 3, 1), dtype=np.float32))

        def tetrahedra(self) -> object:
            return types.SimpleNamespace(
                topo=lambda: View(np.zeros((1, 4, 1), dtype=np.int32))
            )

    class ConstructorAttachment:
        def __init__(self) -> None:
            self.cfg = types.SimpleNamespace(
                body_name="gelsight_mini_case_left",
                isaac_rigid_prim_path=None,
            )
            self.isaaclab_rigid_object = types.SimpleNamespace(
                cfg=types.SimpleNamespace(prim_path="/World/envs/env_.*/Robot")
            )
            self.uipc_object = types.SimpleNamespace(uipc_meshes=[Mesh()])
            self.attachment_points_idx: list[int] = []

        def compute_attachment_data_from_given_ids(
            self,
            path: str,
            _points: np.ndarray,
            _topology: np.ndarray,
            indices: tuple[int, ...],
        ) -> np.ndarray:
            assert path.endswith("/.*gelsight_mini_case_left")
            assert len(indices) == 83
            return np.full((83, 3), 0.125, dtype=np.float32)

    attachment = ConstructorAttachment()

    assert univtac_tactile_attachment._populate_constructor_attachment(attachment)
    assert attachment.attachment_points_idx == list(
        univtac_tactile_attachment._ATTACHMENT_VERTEX_INDICES
    )
    assert np.all(attachment.attachment_offsets == np.float32(0.125))
    assert attachment._robotactile_constructor_attachment["repair_applied"] is True


def test_repair_is_task_scoped() -> None:
    assert not repair_gsmini_tactile_attachments(object(), "unknown_task")
