# N0-VTLA integration example

N0-VTLA is a first-class RoboTactile `PolicyAdapter` backed by the official
ZMQ/msgpack server. The released UniVTAC checkpoint supports `insert_hole`, two
tactile views, qpos8 joint actions, and a 50-step action horizon.

The source, checkpoint, and runtime stay external to the Apache-2.0 wheel. The
all-zero hashes in `artifact_manifest.example.json` are schema examples and are
not runnable artifacts.

```bash
robotactile deployment init
bash integrations/install_univtac.sh
bash integrations/install_n0_vtla.sh

hf download NeoteAI/n0_VTLA_insert_hole \
  --revision 73a514c015c6745a14a3efdca92f25c6cfab5eb5 \
  --local-dir "$PWD/deployment/artifacts/models/n0_vtla/checkpoint"

robotactile integrations configure n0-vtla --task insert_hole
robotactile integrations doctor --model n0_vtla --task insert_hole
```

The adapter forwards the delivered (and therefore possibly faulted) left and
right tactile images to the model. It does not replace tactile tensors with
rest or black images. The official server must receive `reset` at episode start
and the simulator must execute the full 50-step chunk before the next predict.

Passing the static checks does not claim live inference, Isaac success, or the
paper's success rate. See `docs/model_integrations.md` for the exact runtime
command and evidence boundary.
