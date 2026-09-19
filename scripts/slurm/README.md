# Westlake Slurm smoke entry

Run from a login shell so \`/soft/slurm/bin\` and the cluster module environment
are available:

\`\`\`bash
bash -l
cd /storage/yukaichengLab/zhangjiahuan/RoboTactile
bash scripts/slurm/submit_westlake.sh cpu
bash scripts/slurm/submit_westlake.sh gpu
\`\`\`

The CPU job runs one deterministic RoboTactile replay using the existing
repo-local source tree. The GPU job requests one GPU and records exactly one
Slurm-visible device through \`nvidia-smi\`; it does not load a model or allocate
training memory. Both write immutable timestamp/job-ID output directories
under \`deployment/outputs/slurm-smoke/\`, logs under
\`deployment/logs/slurm/\`, and a submission record under
\`deployment/runtime/slurm/jobs/\`.

These jobs establish \`SMOKE\` evidence only. They do not establish model
inference, simulator execution, \`CLOSED-LOOP\`, or \`OFFICIAL\` results.
