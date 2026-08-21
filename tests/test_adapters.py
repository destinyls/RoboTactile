import unittest
from dataclasses import replace

import numpy as np

from robotactile_benchmark.adapters.n0_twam import (
    AdapterStateError,
    N0Handshake,
    N0TwamQpos8Adapter,
    UnsupportedAvailabilityError,
)
from robotactile_benchmark.adapters.univtac import (
    ContactPhaseState,
    DepthPhaseTracker,
    UniVTACAliasManifest,
    UniVTACRecordBuilder,
)
from robotactile_benchmark.contracts import canonical_hash
from robotactile_benchmark.fixtures import make_synthetic_episode


def _raw_observation(depth_mm: float = 34.0, step_index: int = 0):
    return {
        "step": step_index,
        "observation": {
            "head": {"rgb": np.full((10, 12, 3), 20, dtype=np.uint8)},
            "wrist": {"rgb": np.full((10, 12, 3), 30, dtype=np.uint8)},
        },
        "tactile": {
            "left_tactile": {
                "rgb_marker": np.full((8, 9, 3), 60, dtype=np.uint8),
                "depth": np.full((8, 9), depth_mm, dtype=np.float32),
            },
            "right_tactile": {
                "rgb_marker": np.full((8, 9, 3), 70, dtype=np.uint8),
                "depth": np.full((8, 9), depth_mm, dtype=np.float32),
            },
        },
        "embodiment": {
            "joint": np.arange(9, dtype=np.float32),
            "ee": np.arange(7, dtype=np.float32),
        },
    }


def _aliases(**overrides):
    values = {
        "head_camera": "head",
        "wrist_camera": "wrist",
        "left_tactile": "left_tactile",
        "right_tactile": "right_tactile",
        "left_physical_source_id": "left-gsmini-serial-001",
        "right_physical_source_id": "right-gsmini-serial-002",
        "sensor_type": "gsmini",
        "far_plane_mm": 34.0,
        "calibration_id": "univtac-gsmini-demo-v1",
        "calibration_config_sha256": canonical_hash(
            "univtac-gsmini-demo-calibration-v1"
        ),
        "tactile_payload": "rgb_marker",
    }
    values.update(overrides)
    return UniVTACAliasManifest(**values)


def _builder(**alias_overrides):
    return UniVTACRecordBuilder(
        _aliases(**alias_overrides),
        phase_tracker=DepthPhaseTracker(on_threshold_mm=0.8, off_threshold_mm=0.3),
    )


def _valid_qpos8_action() -> np.ndarray:
    pose = np.asarray((0.0, -0.5, 0.0, -1.5, 0.0, 0.5, 0.0, 0.02), dtype=np.float32)
    return np.broadcast_to(pose[:, None, None], (8, 2, 4)).copy()


class UniVTACAdapterTests(unittest.TestCase):
    def test_depth_phase_tracker_exposes_explicit_hysteresis_state(self) -> None:
        tracker = DepthPhaseTracker(on_threshold_mm=0.8, off_threshold_mm=0.3)
        state = ContactPhaseState(in_contact=False)
        phases = []
        for indentation_mm in (0.0, 1.0, 1.1, 0.2, 0.0):
            phase, state = tracker.update(indentation_mm, state)
            phases.append(phase.value)

        self.assertEqual(
            phases,
            ["free", "contact_onset", "sustained_contact", "release", "free"],
        )

    def test_builder_keeps_phase_state_external_across_frames(self) -> None:
        builder = _builder()
        states = {
            "left": ContactPhaseState(False),
            "right": ContactPhaseState(False),
        }
        phases = []
        for step_index, depth_mm in enumerate((34.0, 32.8, 32.7, 33.9, 34.0)):
            record, states = builder.build_with_phase_states(
                _raw_observation(depth_mm, step_index),
                phase_states=states,
                episode_id="episode-1",
                task="insert_HDMI",
                seed=9,
                step_index=step_index,
            )
            phases.append(record.provenance_for("left").phase.value)

        self.assertEqual(
            phases,
            ["free", "contact_onset", "sustained_contact", "release", "free"],
        )

    def test_builder_uses_declared_aliases_and_canonicalizes_contract(self) -> None:
        builder = _builder()
        record = builder.build_stateless_single_frame(
            _raw_observation(step_index=4),
            episode_id="episode-1",
            task="insert_HDMI",
            seed=9,
            step_index=4,
        )

        self.assertEqual(record.observation.sensor("left").payload.dtype, np.uint8)
        self.assertEqual(record.provenance_for("left").source_index, 4)
        self.assertAlmostEqual(record.provenance_for("left").source_time_s, 4.0 / 120.0)
        self.assertEqual(record.observation.proprio.shape, (9,))

    def test_builder_fails_closed_instead_of_guessing_sensor_aliases(self) -> None:
        builder = _builder(left_tactile="left_gsmini", right_tactile="right_gsmini")

        with self.assertRaisesRegex(KeyError, "left_gsmini"):
            builder.build_stateless_single_frame(
                _raw_observation(),
                episode_id="episode-1",
                task="insert_HDMI",
                seed=9,
                step_index=0,
            )

    def test_builder_rejects_out_of_range_integer_rgb(self) -> None:
        builder = _builder()
        for invalid_value in (-1, 300):
            raw = _raw_observation()
            raw["tactile"]["left_tactile"]["rgb_marker"] = np.full(
                (8, 9, 3), invalid_value, dtype=np.int16
            )
            with (
                self.subTest(invalid_value=invalid_value),
                self.assertRaisesRegex(ValueError, r"\[0,255\]"),
            ):
                builder.build_stateless_single_frame(
                    raw,
                    episode_id="episode-1",
                    task="insert_HDMI",
                    seed=9,
                    step_index=0,
                )

    def test_builder_binds_source_provenance_to_raw_step_and_calibration(self) -> None:
        builder = _builder()
        with self.assertRaisesRegex(ValueError, "raw step"):
            builder.build_stateless_single_frame(
                _raw_observation(step_index=7),
                episode_id="episode-1",
                task="insert_HDMI",
                seed=9,
                step_index=6,
            )

        record = builder.build_stateless_single_frame(
            _raw_observation(step_index=7),
            episode_id="episode-1",
            task="insert_HDMI",
            seed=9,
            step_index=7,
        )
        self.assertEqual(record.provenance_for("left").source_index, 7)
        self.assertEqual(
            record.observation.sensor("left").calibration_id,
            "univtac-gsmini-demo-v1",
        )
        self.assertEqual(
            record.provenance_for("left").physical_source_id,
            "left-gsmini-serial-001",
        )

    def test_alias_manifest_rejects_duplicate_sources_and_bad_calibration(self) -> None:
        with self.assertRaisesRegex(ValueError, "tactile aliases"):
            _aliases(right_tactile="left_tactile")
        with self.assertRaisesRegex(ValueError, "physical tactile sources"):
            _aliases(right_physical_source_id="left-gsmini-serial-001")
        with self.assertRaisesRegex(ValueError, "calibration_config_sha256"):
            _aliases(calibration_config_sha256="not-a-sha")


class N0TwamAdapterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.handshake = N0Handshake(
            schema_version="robotactile-n0-v1",
            checkpoint_sha256="a" * 64,
            config_sha256="b" * 64,
            action_spec="qpos8_next_step",
            action_dim=8,
            camera_keys=("observation.images.top", "observation.images.wrist_l"),
            tactile_keys=(
                "observation.images.tactile_a",
                "observation.images.tactile_b",
            ),
            frame_chunk_size=2,
            action_per_frame=4,
            minimum_commit_keyframes=3,
            requires_commit=True,
            action_lower_bounds=(
                -2.8973,
                -1.7628,
                -2.8973,
                -3.0718,
                -2.8973,
                -0.0175,
                -2.8973,
                0.0,
            ),
            action_upper_bounds=(
                2.8973,
                1.7628,
                2.8973,
                -0.0698,
                2.8973,
                3.7525,
                2.8973,
                0.04,
            ),
        )

    def test_qpos8_transaction_is_reset_infer_commit_and_never_leaks_provenance(
        self,
    ) -> None:
        adapter = N0TwamQpos8Adapter(self.handshake)
        episode = make_synthetic_episode(length=10)
        record = episode[4]
        reset = adapter.reset(
            episode_id=record.observation.episode_id,
            prompt="insert HDMI",
            seed=record.observation.seed,
            task=record.observation.task,
        )
        adapter.ack_reset(adapter.success_response(reset))
        infer = adapter.build_infer(record)
        actions = adapter.ack_infer(
            adapter.success_response(infer, action=_valid_qpos8_action())
        )
        commit = adapter.build_commit(actions, (episode[5], episode[6], episode[7]))

        self.assertEqual(reset["reset"], True)
        self.assertEqual(actions.shape, (8, 2, 4))
        self.assertEqual(
            set(infer["obs"]),
            {"observation.images.top", "observation.images.wrist_l"},
        )
        self.assertEqual(
            set(infer["tactile"]),
            {
                "observation.images.tactile_a",
                "observation.images.tactile_b",
            },
        )
        self.assertNotIn("provenance", repr(infer))
        self.assertEqual(commit["compute_kv_cache"], True)
        np.testing.assert_array_equal(
            commit["current_state"], record.observation.proprio[:8]
        )
        with self.assertRaises(AdapterStateError):
            adapter.build_commit(actions, (episode[5], episode[6], episode[7]))
        with self.assertRaises(AdapterStateError):
            adapter.build_infer(record)

        adapter.ack_commit(adapter.success_response(commit))
        next_infer = adapter.build_infer(episode[7])
        self.assertEqual(next_infer["current_state"].shape, (8,))

    def test_failed_commit_requires_a_hard_episode_reset(self) -> None:
        adapter = N0TwamQpos8Adapter(self.handshake)
        episode = make_synthetic_episode(length=10)
        record = episode[4]
        reset = adapter.reset(
            episode_id=record.observation.episode_id,
            prompt="insert HDMI",
            seed=record.observation.seed,
            task=record.observation.task,
        )
        adapter.ack_reset(adapter.success_response(reset))
        infer = adapter.build_infer(record)
        actions = adapter.ack_infer(
            adapter.success_response(infer, action=_valid_qpos8_action())
        )
        adapter.build_commit(actions, (episode[5], episode[6], episode[7]))
        adapter.fail_commit()

        with self.assertRaises(AdapterStateError):
            adapter.ack_commit({})
        with self.assertRaises(AdapterStateError):
            adapter.build_infer(record)

        reset = adapter.reset(
            episode_id=record.observation.episode_id,
            prompt="insert HDMI",
            seed=record.observation.seed,
            task=record.observation.task,
        )
        adapter.ack_reset(adapter.success_response(reset))
        self.assertEqual(adapter.build_infer(record)["current_state"].shape, (8,))

    def test_failed_infer_requires_a_hard_episode_reset(self) -> None:
        adapter = N0TwamQpos8Adapter(self.handshake)
        record = make_synthetic_episode(length=10)[4]
        reset = adapter.reset(
            episode_id=record.observation.episode_id,
            prompt="insert HDMI",
            seed=record.observation.seed,
            task=record.observation.task,
        )
        adapter.ack_reset(adapter.success_response(reset))
        adapter.build_infer(record)
        adapter.fail_infer()

        with self.assertRaises(AdapterStateError):
            adapter.build_infer(record)

        reset = adapter.reset(
            episode_id=record.observation.episode_id,
            prompt="insert HDMI",
            seed=record.observation.seed,
            task=record.observation.task,
        )
        adapter.ack_reset(adapter.success_response(reset))
        self.assertEqual(adapter.build_infer(record)["current_state"].shape, (8,))

    def test_commit_requires_minimum_ordered_keyframe_chunk(self) -> None:
        episode = make_synthetic_episode(length=10)
        actions = _valid_qpos8_action()
        for keyframes, message in (
            ((episode[5], episode[6]), "at least 3"),
            ((episode[5], episode[7], episode[6]), "strictly increasing"),
            ((episode[3], episode[4], episode[5]), "after infer"),
        ):
            adapter = N0TwamQpos8Adapter(self.handshake)
            reset = adapter.reset(
                episode_id=episode[4].observation.episode_id,
                prompt="insert HDMI",
                seed=episode[4].observation.seed,
                task=episode[4].observation.task,
            )
            adapter.ack_reset(adapter.success_response(reset))
            infer = adapter.build_infer(episode[4])
            acknowledged = adapter.ack_infer(
                adapter.success_response(infer, action=actions)
            )
            with (
                self.subTest(message=message),
                self.assertRaisesRegex(ValueError, message),
            ):
                adapter.build_commit(acknowledged, keyframes)

    def test_frozen_tactile_required_model_rejects_missing_payload(self) -> None:
        adapter = N0TwamQpos8Adapter(self.handshake)
        record = make_synthetic_episode(length=10)[4]
        reset = adapter.reset(
            episode_id=record.observation.episode_id,
            prompt="insert HDMI",
            seed=record.observation.seed,
            task=record.observation.task,
        )
        adapter.ack_reset(adapter.success_response(reset))
        absent = record.observation.sensor("left").without_payload(declared=True)
        missing_record = record.with_observation(
            record.observation.replace_sensor(absent)
        )

        with self.assertRaises(UnsupportedAvailabilityError):
            adapter.build_infer(missing_record)

    def test_handshake_and_action_shape_fail_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "qpos8_next_step"):
            N0TwamQpos8Adapter(
                N0Handshake(
                    **{
                        **self.handshake.to_dict(),
                        "action_spec": "delta_ee_20d",
                    }
                )
            )
        adapter = N0TwamQpos8Adapter(self.handshake)
        with self.assertRaisesRegex(ValueError, r"\[8, F, 4\]"):
            adapter.decode_action(np.zeros((8, 12), dtype=np.float32))

    def test_transaction_ack_and_action_binding_fail_closed(self) -> None:
        adapter = N0TwamQpos8Adapter(self.handshake)
        episode = make_synthetic_episode(length=10)
        reset = adapter.reset(
            episode_id=episode[0].observation.episode_id,
            prompt="insert HDMI",
            seed=episode[0].observation.seed,
            task=episode[0].observation.task,
        )
        with self.assertRaisesRegex(AdapterStateError, "transaction ID"):
            adapter.ack_reset(
                {"status": "ok", "op": "reset", "transaction_id": "wrong"}
            )
        adapter.ack_reset(adapter.success_response(reset))
        infer = adapter.build_infer(episode[4])
        action = adapter.ack_infer(
            adapter.success_response(infer, action=_valid_qpos8_action())
        )
        with self.assertRaisesRegex(AdapterStateError, "does not match"):
            changed_action = action.copy()
            changed_action[0] += 0.01
            adapter.build_commit(changed_action, (episode[5], episode[6], episode[7]))
        with self.assertRaisesRegex(AdapterStateError, "cannot discard"):
            adapter.reset(
                episode_id=episode[0].observation.episode_id,
                prompt="insert HDMI",
                seed=episode[0].observation.seed,
                task=episode[0].observation.task,
            )

    def test_handshake_validates_version_hashes_and_required_commit(self) -> None:
        invalid_overrides = (
            {"schema_version": "legacy"},
            {"checkpoint_sha256": "not-a-sha"},
            {"frame_chunk_size": 0},
            {"frame_chunk_size": 3},
            {"camera_keys": ("top", "top")},
            {"requires_commit": False},
            {"requires_commit": 1},
            {"action_lower_bounds": (0.0,) * 8, "action_upper_bounds": (0.0,) * 8},
        )
        for override in invalid_overrides:
            with self.subTest(override=override), self.assertRaises(ValueError):
                N0TwamQpos8Adapter(
                    N0Handshake(**{**self.handshake.to_dict(), **override})
                )

    def test_adapter_binds_every_record_to_the_reset_episode(self) -> None:
        adapter = N0TwamQpos8Adapter(self.handshake)
        record = make_synthetic_episode(length=10)[4]
        reset = adapter.reset(
            episode_id="different-episode",
            prompt="insert HDMI",
            seed=record.observation.seed,
            task=record.observation.task,
        )
        adapter.ack_reset(adapter.success_response(reset))

        with self.assertRaisesRegex(AdapterStateError, "episode"):
            adapter.build_infer(record)

    def test_adapter_binds_records_to_reset_task_and_seed(self) -> None:
        record = make_synthetic_episode(length=10)[4]
        for task, seed, message in (
            ("other-task", record.observation.seed, "task"),
            (record.observation.task, record.observation.seed + 1, "seed"),
        ):
            adapter = N0TwamQpos8Adapter(self.handshake)
            reset = adapter.reset(
                episode_id=record.observation.episode_id,
                prompt="insert HDMI",
                seed=seed,
                task=task,
            )
            adapter.ack_reset(adapter.success_response(reset))
            with (
                self.subTest(message=message),
                self.assertRaisesRegex(AdapterStateError, message),
            ):
                adapter.build_infer(record)

    def test_ack_is_bound_to_episode_and_exact_request_digest(self) -> None:
        record = make_synthetic_episode(length=10)[4]
        for field, value, message in (
            ("episode_id", "wrong-episode", "episode ID"),
            ("request_sha256", "0" * 64, "request digest"),
        ):
            adapter = N0TwamQpos8Adapter(self.handshake)
            reset = adapter.reset(
                episode_id=record.observation.episode_id,
                prompt="insert HDMI",
                seed=record.observation.seed,
                task=record.observation.task,
            )
            response = adapter.success_response(reset)
            response[field] = value
            with (
                self.subTest(field=field),
                self.assertRaisesRegex(AdapterStateError, message),
            ):
                adapter.ack_reset(response)

        adapter = N0TwamQpos8Adapter(self.handshake)
        reset = adapter.reset(
            episode_id=record.observation.episode_id,
            prompt="insert HDMI",
            seed=record.observation.seed,
            task=record.observation.task,
        )
        reset["seed"] += 1
        with self.assertRaisesRegex(AdapterStateError, "content"):
            adapter.success_response(reset)

    def test_n0_rejects_fractional_dimensions_seed_and_noncanonical_rgb(self) -> None:
        for override in (
            {"action_dim": 8.0},
            {"frame_chunk_size": 2.0},
            {"action_per_frame": 4.0},
        ):
            with self.subTest(override=override), self.assertRaises(TypeError):
                N0Handshake(**{**self.handshake.to_dict(), **override})

        adapter = N0TwamQpos8Adapter(self.handshake)
        record = make_synthetic_episode(length=10)[4]
        with self.assertRaisesRegex(TypeError, "seed"):
            adapter.reset(
                episode_id=record.observation.episode_id,
                prompt="insert HDMI",
                seed=1.5,
                task=record.observation.task,
            )

        adapter = N0TwamQpos8Adapter(self.handshake)
        reset = adapter.reset(
            episode_id=record.observation.episode_id,
            prompt="insert HDMI",
            seed=record.observation.seed,
            task=record.observation.task,
        )
        adapter.ack_reset(adapter.success_response(reset))
        bad_vision = dict(record.observation.vision)
        bad_vision["top"] = bad_vision["top"].astype(np.float32)
        bad_record = record.with_observation(
            replace(record.observation, vision=bad_vision)
        )
        with self.assertRaisesRegex(ValueError, "canonical uint8"):
            adapter.build_infer(bad_record)

        with self.assertRaisesRegex(ValueError, "physical bounds"):
            adapter.decode_action(np.full((8, 2, 4), 1e6, dtype=np.float32))

    def test_live_handshake_must_match_expected_artifact_hashes(self) -> None:
        metadata = self.handshake.to_dict()
        rebuilt = N0Handshake.from_server_metadata(
            metadata,
            expected_checkpoint_sha256="a" * 64,
            expected_config_sha256="b" * 64,
        )
        self.assertEqual(rebuilt, self.handshake)
        with self.assertRaisesRegex(ValueError, "checkpoint"):
            N0Handshake.from_server_metadata(
                metadata,
                expected_checkpoint_sha256="c" * 64,
                expected_config_sha256="b" * 64,
            )
        with self.assertRaisesRegex(ValueError, "fields"):
            N0Handshake.from_server_metadata(
                {**metadata, "unexpected": True},
                expected_checkpoint_sha256="a" * 64,
                expected_config_sha256="b" * 64,
            )

    def test_next_infer_is_bound_to_the_last_acknowledged_commit(self) -> None:
        adapter = N0TwamQpos8Adapter(self.handshake)
        episode = make_synthetic_episode(length=10)
        reset = adapter.reset(
            episode_id=episode[4].observation.episode_id,
            prompt="insert HDMI",
            seed=episode[4].observation.seed,
            task=episode[4].observation.task,
        )
        adapter.ack_reset(adapter.success_response(reset))
        infer = adapter.build_infer(episode[4])
        action = adapter.ack_infer(
            adapter.success_response(infer, action=_valid_qpos8_action())
        )
        commit = adapter.build_commit(action, (episode[5], episode[6], episode[7]))
        adapter.ack_commit(adapter.success_response(commit))

        with self.assertRaisesRegex(AdapterStateError, "last committed"):
            adapter.build_infer(episode[8])
        self.assertEqual(adapter.build_infer(episode[7])["step_id"], 7)


if __name__ == "__main__":
    unittest.main()
