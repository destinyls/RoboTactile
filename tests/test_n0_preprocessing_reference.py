from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from robotactile_benchmark.integrations.n0_twam.preprocessing_reference import (
    CheckpointVideoContract,
    build_area_resize_command,
    build_h264_encode_command,
    build_training_decode_command,
    decode_checkpoint_jpeg,
    decode_live_numeric_proxy,
    reverse_rgb_channels,
)


def test_checkpoint_video_contract_rejects_invalid_parameters() -> None:
    with pytest.raises(ValueError, match="divisible"):
        CheckpointVideoContract(source_fps=30, target_fps=8)
    with pytest.raises(ValueError, match="CRF/GOP"):
        CheckpointVideoContract(crf=52)


def test_ffmpeg_commands_preserve_released_codec_and_area_contract() -> None:
    contract = CheckpointVideoContract()
    ffmpeg = Path("/opt/ffmpeg")
    encoded = build_h264_encode_command(
        ffmpeg,
        Path("clip.mp4"),
        width=480,
        height=270,
        contract=contract,
    )
    assert encoded[0] == str(ffmpeg)
    assert encoded[encoded.index("-crf") + 1] == "30"
    assert encoded[encoded.index("-g") + 1] == "2"
    pixel_format_positions = [
        index for index, value in enumerate(encoded) if value == "-pix_fmt"
    ]
    assert encoded[pixel_format_positions[-1] + 1] == "yuv420p"

    decoded = build_training_decode_command(
        ffmpeg,
        Path("clip.mp4"),
        width=256,
        height=256,
        contract=contract,
    )
    assert "fps=10,scale=256:256:flags=area" in decoded

    resized = build_area_resize_command(
        ffmpeg,
        input_width=480,
        input_height=270,
        output_width=256,
        output_height=256,
    )
    assert "scale=256:256:flags=area" in resized
    assert "fps=" not in " ".join(resized)


def test_reverse_rgb_channels_is_immutable_and_non_aliasing() -> None:
    source = np.asarray([[[1, 2, 3], [4, 5, 6]]], dtype=np.uint8)
    reversed_rgb = reverse_rgb_channels(source)
    np.testing.assert_array_equal(
        reversed_rgb,
        np.asarray([[[3, 2, 1], [6, 5, 4]]], dtype=np.uint8),
    )
    assert not reversed_rgb.flags.writeable
    source[0, 0, 0] = 99
    assert reversed_rgb[0, 0, 2] == 1


def test_legacy_cv2_writer_and_checkpoint_pil_decoder_swap_rb() -> None:
    cv2 = pytest.importorskip("cv2")
    simulator_rgb = np.zeros((32, 32, 3), dtype=np.uint8)
    simulator_rgb[..., 0] = 230
    simulator_rgb[..., 1] = 80
    simulator_rgb[..., 2] = 15
    ok, encoded = cv2.imencode(".jpg", simulator_rgb)
    assert ok

    live_proxy = decode_live_numeric_proxy(encoded)
    checkpoint_rgb = decode_checkpoint_jpeg(encoded)
    assert float(live_proxy[..., 0].mean()) > float(live_proxy[..., 2].mean())
    assert float(checkpoint_rgb[..., 0].mean()) < float(checkpoint_rgb[..., 2].mean())
    np.testing.assert_allclose(
        checkpoint_rgb.astype(np.int16),
        live_proxy[..., ::-1].astype(np.int16),
        atol=1,
    )
