# RunPod Playbook — best practices, earned + official (2026-07-19)

Synthesis of RunPod docs + hard lessons from the unmuzzle M0–M2c GPU runs. The
governing principle: **the pod's compute is the only thing you should pay the pod
for.** Everything slow, flaky, or re-doable should happen off the critical path.

## 1. Persistence — the #1 win: NETWORK VOLUME for model weights
- Container disk is EPHEMERAL (wiped on stop/terminate). Network volumes are
  persistent NVMe (200–400 MB/s), survive pod death, and can attach to any pod.
- **Pre-load the base model onto a network volume ONCE.** We re-downloaded the
  62 GB R1-Distill base on *every* pod (M1, M2, M2b×2, M2c) — ~5 × 13 min wasted.
  With the model on a volume, new pods skip download entirely.
- Cost: $0.07/GB/mo (~$4.50/mo for 64 GB) — trivial vs the time saved.
- Attach with `--networkVolumeId <id>`; this PINS the pod to the volume's
  datacenter (trade-off: less GPU availability, no failover). Fine for 1×H100;
  NOT usable for 8×H100 in CA-MTL-1 (that DC has no network volumes).
- Also pre-stage scripts/datasets via the **S3-compatible API** (no compute needed).
- **REALITY CHECK (2026-07-19, our account/region):** the volume win is currently
  BLOCKED by a DC↔GPU mismatch. Our 1×H100 pods land in **AP-IN-1** (India — hence the
  slow ~170 KB/s link to a US Mac), and **AP-IN-1 does NOT support network volumes.**
  H100 stock in volume-capable DCs is Low/None (US-CA-2 Low, EU-NL-1 Low, US-TX-3/WA-1/
  IL-1/NC-1 None). So a volume would pin us to a DC with poor H100 availability. Mutation
  is `createNetworkVolume(input:{name,size,dataCenterId})` (also update/deleteNetworkVolume).
  **Revisit when H100 stock appears in a volume-capable DC (poll `gpuTypes(input:{id})
  {lowestPrice(input:{gpuCount:1,dataCenterId:DC}){stockStatus}}`), or opportunistically
  create the volume in US-CA-2 and try it first with fallback to AP-IN-1.** Until then the
  mitigation is: parallel-curl HF download (~80 MB/s, ~13 min) + keep transfers tiny.

## 2. Transfers — stop using raw scp on flaky links
- We hit **170 KB/s** inbound on some DCs; a 2.7 GB scp would've taken hours and
  truncated files silently.
- **Order of preference:** (a) network volume (no transfer at all); (b) `runpodctl
  send`/`receive` (croc relay, one-time code, pre-installed, no SSH — best for big
  files over bad links); (c) `rsync -avP` over SSH (incremental, resumable,
  compressed — restarts where it dropped); (d) raw scp only for tiny files.
- **The run should only need: fast HF download (outbound, ~80 MB/s) + a tiny script
  scp + a tiny result pull.** If a step needs a big INBOUND transfer, redesign it
  (e.g. retrain a 2.7 GB adapter on-pod from a 1 MB dataset instead of uploading).
- **Always checksum-verify a pushed file before launching against it.**
  scp-then-immediately-run RACES: the launch reads the half-written/old file.
  `md5 -q local` vs `md5sum remote` — only launch on match. (Cost us 2 failed DPO
  starts on stale files.)

## 3. Pod creation (the flags that actually matter)
- `--ports "22/tcp"` — FORCES a public-IP machine. Without it you can land on a
  proxy-only pod (ssh.runpod.io) where scp/scripted-exec don't work.
- `--startSSH --secureCloud` — secure cloud won't get preempted like community.
- `--containerDiskSize` big (≥160 GB) if you write a merged model locally;
  `--volumeSize` for /runpod. **A 32B bf16 merge is ~64 GB — put it on the
  CONTAINER disk, not a 100 GB /runpod volume that also holds the 62 GB base
  (base + merge > 100 GB = "No space left on device").**
- Get the endpoint via GraphQL `pod(input:{podId}){runtime{ports{ip publicPort
  privatePort isIpPublic}}}` (api key in ~/.runpod/config.toml). Editing console
  env vars RESTARTS the pod (new endpoint, wiped container disk; /runpod survives).

## 4. Environment gotchas
- Image python is PEP-668 externally-managed → `pip install --break-system-packages`.
- Library version drift is real (TRL 1.8 dropped `rpo_alpha` AND `max_prompt_length`
  from DPOConfig) → **introspect + filter kwargs** to the actual signature, don't
  hardcode. Pin versions across a multi-run campaign for comparability.
- Verify RAM/disk on the ACTUAL machine before betting a run on it (GiB≠GB; a CPU
  merge needs the RAM you think it has). RAM-OOM kills the whole pod+sshd (logs
  lost); GPU-OOM is recoverable.

## 5. Long jobs — never lose progress
- **Incremental save** on any long generate/eval loop. Our eval crashed at 93/98
  twice and lost everything because it only saved at the end. Save every N items.
- Bound each item (max_new_tokens) so one runaway generation can't hang/OOM the loop.
- Stream logs somewhere durable; a hard crash erases end-of-run stdout.

## 6. Orchestration & teardown
- **Poller: tear down ONLY on success; KEEP the pod on failure** for debug. A poller
  that nukes the pod on failure destroys the logs AND any trained artifact (cost us
  the M2b-run1 adapter — training was perfect, a save bug "failed" it, pod gone).
- Always arm a **safety idle-terminate** (e.g. 3 h) that kills the pod if no
  training/eval process is running — cost guard against a stuck/forgotten pod.
- Auto-teardown on success = GPU discipline (real $; agents are subscription).
- **Overlap everything off the critical path:** run CPU work (merge) concurrent
  with GPU work (rollouts/eval) and off-box work (agent judging); split a big eval
  across N concurrent GPU processes; mine more from data already in hand.
- Don't hot-swap a running pipeline (a mid-run kill/relaunch to "go faster" stalled
  a run once) — bake speed into the plan up front.

## 7. Cost model
- GPU is the only real-$ line ($2.99/hr for 1×H100). Tear down immediately.
- Network volume $0.07/GB/mo is worth it to kill repeated big downloads.
- Right-size: 1×H100 (80 GB) fits 32B 4-bit train + eval and even 3 concurrent
  jobs (~60 GB). Reach for multi-GPU only when the model won't fit.

Sources: [Network volumes](https://docs.runpod.io/storage/network-volumes),
[Transfer files](https://docs.runpod.io/pods/storage/transfer-files),
[GPU infra playbook](https://www.runpod.io/articles/guides/gpu-infrastructure-playbook-for-ai-startups).
