#!/usr/bin/env bash
# Run the eager-vs-lazy Franka parity probe in the pinned Dream-Tac container.
set -Eeuo pipefail

readonly IMAGE_REF="docker-registry-sh.tencentcloudcr.com/hg/hg_wm:v3.5"
readonly PYTHON_BIN="/opt/psi-policy-venvs/.venv_worldarena/bin/python"

die() { printf 'Dream-Tac parity error: %s\n' "$*" >&2; exit 2; }
[[ $# -eq 6 ]] || die "expected DATASET EAGER_SOURCE LAZY_SOURCE RUNTIME ROBOTACTILE HCU_ENV"

readonly dataset_root="$1" eager_root="$2" lazy_root="$3"
readonly runtime_root="$4" robotactile_root="$5" hcu_environment="$6"
for path in \
  "$dataset_root" "$eager_root" "$lazy_root" "$runtime_root" \
  "$robotactile_root" "$hcu_environment"; do
  [[ "$path" == /mnt/data/* ]] || die "all inputs must be below /mnt/data"
done
[[ -d "$dataset_root" && -d "$eager_root" && -d "$lazy_root" ]] || die "dataset/source missing"
[[ -d "$runtime_root" && -d "$robotactile_root" ]] || die "runtime/RoboTactile missing"
[[ -f "$hcu_environment" && ! -L "$hcu_environment" ]] || die "HCU environment missing"

readonly container_name="robotactile-dream-tac-lazy-parity-v1"
if docker container inspect "$container_name" >/dev/null 2>&1; then
  die "parity container name already exists"
fi

exec docker run --rm --name "$container_name" --entrypoint "" \
  --network host --ipc host \
  --mount type=bind,src=/opt/hyhal,dst=/opt/hyhal,readonly \
  --mount type=bind,src=/etc/hfm,dst=/etc/hfm,readonly \
  --mount type=bind,src=/mnt/data,dst=/mnt/data,readonly \
  --mount type=bind,src="$eager_root",dst=/workspace/Dream-Tac-HCU-eager,readonly \
  --mount type=bind,src="$lazy_root",dst=/workspace/Dream-Tac-HCU-port,readonly \
  --mount type=bind,src="$runtime_root",dst=/probe,readonly \
  --env PYTHONDONTWRITEBYTECODE=1 \
  --env PYTHONPATH=/workspace/Dream-Tac-HCU-port:/probe/site:/probe/site_cosmos:/probe/site_natten:/probe/site_xformers \
  "$IMAGE_REF" bash -lc \
  "source '$hcu_environment'; exec '$PYTHON_BIN' '$robotactile_root/scripts/dream_tac/hcu_port/probe_lazy_franka_parity.py' --dataset-root '$dataset_root/dataset' --eager-source /workspace/Dream-Tac-HCU-eager/cosmos_policy/datasets/franka_dataset.py --lazy-source /workspace/Dream-Tac-HCU-port/cosmos_policy/datasets/franka_dataset.py"
