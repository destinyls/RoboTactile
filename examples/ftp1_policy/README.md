# FTP-1 policy integration example

FTP-1 is a first-class RoboTactile `PolicyAdapter` backed by an isolated
ZMQ/msgpack worker. The official release publishes six task-specific UniVTAC
checkpoints. Each worker performs the upstream temporal ensemble and returns
one absolute qpos8 action per fresh closed-loop observation.

The Apache-2.0 source remains pinned outside the main wheel. Model-weight
licensing is not declared by the upstream checkpoint repository; users must
review the upstream weight terms and the referenced Gemma terms before use.
The all-zero file hashes in `artifact_manifest.example.json` are schema
placeholders, not runnable evidence.

```bash
robotactile deployment init
bash integrations/install_univtac.sh
bash scripts/ftp1_policy/install_official_runtime.sh

hf download MJJJJ1064/ftp1_univtac_finetune \
  --revision 620ac69b4fffd2341300cfef1b1d224d56710ed3 \
  --include 'FTP1_UniVTAC_insert_hole_expert_gsmall_ftp1/19999/**' \
  --local-dir "$PWD/deployment/artifacts/models/ftp1_policy"
```

The adapter forwards canonical delivered RGB/tactile bytes unchanged to the
official worker (`upstream_passthrough_v1`). Therefore a faulted trial feeds
the actually delivered, fault-injected tactile frames rather than silently
restoring clean frames. Passing source/artifact checks alone does not claim
model inference, Isaac Sim success, or paper-level robustness results.
