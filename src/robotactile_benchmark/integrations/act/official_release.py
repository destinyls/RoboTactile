"""Public orchestration for the pinned official UniVTAC ACT release."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Optional

from robotactile_benchmark.closed_loop.artifact_io import canonical_json_bytes
from robotactile_benchmark.integrations.act.official_release_contracts import (
    OFFICIAL_ACT_PROFILES,
    OFFICIAL_ACT_REPOSITORY_ID,
    OFFICIAL_ACT_REVISION,
    OFFICIAL_ACT_TASKS,
    REFERENCE_EVIDENCE,
    RUNTIME_EVIDENCE,
    OfficialACTReleaseError,
    OfficialACTReleaseFile,
    OfficialACTReleaseLock,
    builtin_official_act_release_lock,
    load_official_act_release_lock,
    official_download_url,
)
from robotactile_benchmark.integrations.act.official_release_io import (
    OfficialACTInstallPlan,
    OfficialACTInstallResult,
    PlannedOfficialACTFile,
    install_official_act_release,
)

# task, profile, log size/hash, metadata size/hash. These immutable upstream
# objects are reference-only and never represent local simulator execution.
_REFERENCE_DATA = (
    (
        "grasp_classify",
        "univtac",
        14921,
        "ade305f2994f8df5a266341109761b51a7215508c8cf34570dd2333b56c0f03b",
        39131,
        "e333eba2e69e488c6f61b5481eadeb2399f74ad244f15a88606953109a08ad27",
    ),
    (
        "grasp_classify",
        "vision_only",
        14785,
        "7574d702106cb4d3b90267c196c864009a8b2fc1a585fdbfe76a474651bc5e24",
        39202,
        "c165d4e667febf56988fe44ddf53d7051a1215705be57abd7465c93e3d91bb57",
    ),
    (
        "insert_HDMI",
        "univtac",
        14718,
        "848fca8a8d61337c10c90e0fe6bf27f51b376aae9657b0a0c63feaba02a4a3e8",
        65519,
        "e9f29e18a5d3e26ac4527d1327d3d8e55749a994552aad600bfefc2bfbc687ad",
    ),
    (
        "insert_HDMI",
        "vision_only",
        14738,
        "96ec23e7ac02e37bbffd58463a711006e3cf7c0dab124dd6057b4e6277d0a464",
        65413,
        "441bb24af5e5bdfbc34491a044c6b2bb4f2ff086a704ea3b206b3fff12b31720",
    ),
    (
        "insert_hole",
        "univtac",
        14783,
        "69afef6fb2169be3803fd4a6d5b5feb25ed9faa9a73fb83af87b0e5aeb3f2579",
        48025,
        "3cc05b52f6b9047c8e5e4bfd5d612cb751c9bc870add1cc8daaebb15077a9f95",
    ),
    (
        "insert_hole",
        "vision_only",
        14759,
        "f31911b41da0e02701897dc11fe194014b4cd7cf3edabd8a1eb0414bb304e0d3",
        48615,
        "9d1e8d96c70b8f12b82ca004485798eafc2bc26ae32ecb0e0e320172780024ff",
    ),
    (
        "insert_tube",
        "univtac",
        14822,
        "e848033a46347d4e778c0968e293ee1cf37abd44da1e6134bf2ee8247207f0bc",
        63345,
        "c7556775d4f3b903fb46dc0decf00c6c065a2053214f631ef28743bfeaf3d826",
    ),
    (
        "insert_tube",
        "vision_only",
        14835,
        "5482ff85e006e1d48c67c7fd62ae1a2ecaa26929baf7bbfe427e6554ffed917a",
        63370,
        "6c46802d365ae000cab8ec5df608305f55cc8ba9afe47e99b67909a676ffecc5",
    ),
    (
        "lift_bottle",
        "univtac",
        14802,
        "b2dfb09f0256f00b14cbf4cedd0f97c9c7ba92f430f858a53abec277d0c626bb",
        38642,
        "affe218fa81f534b4d93f3978d27296bc7830ea2b6ca71c31450d03a102cb6ab",
    ),
    (
        "lift_bottle",
        "vision_only",
        14791,
        "06fe89931f06b605229754bd96d79832a570eaf395665577aca357e775476e63",
        38472,
        "15ee51dc669ccc3b23e9c66730607b14c476442dee3b554896b64ddb21739bea",
    ),
    (
        "lift_can",
        "univtac",
        14768,
        "853bd396623976be4895f0f65190727dae0d03a046d14239080ee9ff0fba528d",
        60530,
        "3373e15bf3527b1ae5225ec51ba7c787aa43221ee4db7236e08950555ae1ba1d",
    ),
    (
        "lift_can",
        "vision_only",
        14786,
        "e35e7a310e471690570e2b32067d878c7ee49175eff6c679799736d44ef67b30",
        60837,
        "7ff9ba85601557614fa947292f8d807006b1bcc85411cc18f5ab4b78bb341086",
    ),
    (
        "pull_out_key",
        "univtac",
        14810,
        "6e407ac00337740f3ebec013dd73680ca65b4dcf9135124d85f7c07544680f9c",
        52421,
        "6a2dc09ac97c77d729be1f28363780b2d9b071cba7ef796c09f8c8f7ed1412cf",
    ),
    (
        "pull_out_key",
        "vision_only",
        14802,
        "a018aab8b6f815ca66f4f2e0c7ea0fa61eeba8b9d66995206f679cb511a117c0",
        52697,
        "a7208ea14edb0d248accb02239f3e512edd10f8edb71ceafc4a02aec552df23e",
    ),
    (
        "put_bottle_in_shelf",
        "univtac",
        14633,
        "de5170961694fd4be373005d44cddfe870e1b45e549f0cfb2eaf000d36fbdb96",
        11764,
        "de04c87e7f9775f35055d26b759e3a08be737422492c6d0679bcd20062a8e5eb",
    ),
    (
        "put_bottle_in_shelf",
        "vision_only",
        14730,
        "af765e1eded89354b52c9817b20d2cee4973e007a51c2248fa3bcf1643803670",
        11780,
        "aec05e8ca8c01e5cb8a6e1183104eb13842041f30585ba8a81102c20f3b591c9",
    ),
)


def _selection(
    values: Optional[Iterable[str]],
    supported: Sequence[str],
    default: Sequence[str],
) -> tuple[str, ...]:
    raw = tuple(default if values is None else values)
    if (
        not raw
        or len(raw) != len(set(raw))
        or any(item not in supported for item in raw)
    ):
        raise OfficialACTReleaseError("ACT artifact selection is invalid")
    selected = set(raw)
    return tuple(item for item in supported if item in selected)


def _reference_files(
    tasks: tuple[str, ...], profiles: tuple[str, ...]
) -> list[PlannedOfficialACTFile]:
    result: list[PlannedOfficialACTFile] = []
    for (
        task,
        profile,
        log_size,
        log_sha,
        metadata_size,
        metadata_sha,
    ) in _REFERENCE_DATA:
        if task not in tasks or profile not in profiles:
            continue
        for name, size, sha256, kind in (
            ("log.log", log_size, log_sha, "reference_log"),
            ("metadata.json", metadata_size, metadata_sha, "reference_metadata"),
        ):
            remote = f"checkpoints/{task}/{profile}/{name}"
            destination = f"references/{OFFICIAL_ACT_REVISION}/{task}/{profile}/{name}"
            result.append(
                PlannedOfficialACTFile(
                    (destination,),
                    REFERENCE_EVIDENCE,
                    kind,
                    remote,
                    sha256,
                    size,
                    official_download_url(remote),
                )
            )
    return result


def build_official_act_install_plan(
    artifact_root: Path,
    *,
    lock: Optional[OfficialACTReleaseLock] = None,
    tasks: Optional[Iterable[str]] = None,
    profiles: Optional[Iterable[str]] = None,
    include_reference: bool = False,
) -> OfficialACTInstallPlan:
    """Build a network-free, canonical installation plan."""

    selected_lock = builtin_official_act_release_lock() if lock is None else lock
    if type(selected_lock) is not OfficialACTReleaseLock:
        raise TypeError("lock must be an exact OfficialACTReleaseLock")
    selected_tasks = _selection(tasks, OFFICIAL_ACT_TASKS, OFFICIAL_ACT_TASKS)
    selected_profiles = _selection(profiles, OFFICIAL_ACT_PROFILES, ("univtac",))
    planned: list[PlannedOfficialACTFile] = []
    for item in selected_lock.files:
        destinations: tuple[str, ...]
        if item.kind == "encoder":
            destinations = ("encoder.pth",)
        elif item.kind == "policy":
            if (
                item.task_id not in selected_tasks
                or item.profile not in selected_profiles
            ):
                continue
            destinations = (f"{item.task_id}/{item.profile}/policy_last.ckpt",)
        else:
            if item.task_id not in selected_tasks:
                continue
            destinations = tuple(
                f"{item.task_id}/{profile}/dataset_stats.pkl"
                for profile in selected_profiles
            )
        planned.append(
            PlannedOfficialACTFile(
                destinations,
                RUNTIME_EVIDENCE,
                item.kind,
                item.remote_path,
                item.sha256,
                item.size_bytes,
                official_download_url(item.remote_path),
            )
        )
    if include_reference:
        planned.extend(_reference_files(selected_tasks, selected_profiles))
    planned.sort(key=lambda item: item.remote_path)
    lock_sha256 = hashlib.sha256(
        canonical_json_bytes(selected_lock.to_dict())
    ).hexdigest()
    return OfficialACTInstallPlan(
        Path(artifact_root).expanduser().absolute(),
        tuple(planned),
        include_reference,
        lock_sha256,
        selected_profiles,
        selected_tasks,
    )


__all__ = [
    "OFFICIAL_ACT_PROFILES",
    "OFFICIAL_ACT_REPOSITORY_ID",
    "OFFICIAL_ACT_REVISION",
    "OFFICIAL_ACT_TASKS",
    "OfficialACTInstallPlan",
    "OfficialACTInstallResult",
    "OfficialACTReleaseError",
    "OfficialACTReleaseFile",
    "OfficialACTReleaseLock",
    "PlannedOfficialACTFile",
    "build_official_act_install_plan",
    "builtin_official_act_release_lock",
    "install_official_act_release",
    "load_official_act_release_lock",
]
