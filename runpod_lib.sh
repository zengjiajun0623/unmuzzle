#!/bin/bash
# runpod_lib.sh — hardened RunPod helpers codifying RUNPOD_PLAYBOOK.md.
# Source it:  source ~/llm-lab/unmuzzle-repo/runpod_lib.sh
# Requires: ~/runpodctl, ~/.ssh/runpod_ed25519, api key in ~/.runpod/config.toml

RP_AK() { python3 -c "import re;s=open('$HOME/.runpod/config.toml').read();m=re.search(r'apikey\s*=\s*(.+)',s);print(m.group(1).strip().strip('\"').strip(\"'\"))"; }

# rp_create <name> <gpuType> [--networkVolumeId ID] [extra flags...]
# Standard hardened flags: public-IP SSH, secure cloud, big container disk. Echoes pod id.
rp_create() {
  local name="$1" gpu="$2"; shift 2
  ~/runpodctl create pod --name "$name" --gpuType "$gpu" --gpuCount 1 \
    --imageName "runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404" \
    --containerDiskSize 180 --volumeSize 100 --volumePath /runpod \
    --ports "22/tcp" --startSSH --secureCloud --mem 64 --vcpu 16 --cost 3.5 "$@" 2>&1 \
    | grep -v deprecated | grep -oE '"[a-z0-9]{10,}"' | tr -d '"' | head -1
}

# rp_ssh <podId> <sshConfigOutPath> [hostAlias]  — wait for public TCP, write ssh config, wait for sshd
rp_ssh() {
  local pod="$1" out="$2" host="${3:-rppod}"; local ak; ak=$(RP_AK)
  local ep=""
  for i in $(seq 1 25); do
    ep=$(curl -s "https://api.runpod.io/graphql?api_key=$ak" -H 'Content-Type: application/json' \
      -d "{\"query\":\"query{pod(input:{podId:\\\"$pod\\\"}){runtime{ports{ip publicPort privatePort isIpPublic}}}}\"}" \
      | python3 -c "import json,sys;p=json.load(sys.stdin)['data']['pod'];r=(p or {}).get('runtime');ports=(r or {}).get('ports') or [];t=[x for x in ports if x['privatePort']==22 and x['isIpPublic']];print(t[0]['ip']+' '+str(t[0]['publicPort']) if t else '')" 2>/dev/null)
    [ -n "$ep" ] && break; sleep 15
  done
  [ -z "$ep" ] && { echo "no public TCP endpoint" >&2; return 1; }
  printf 'Host %s\n  HostName %s\n  Port %s\n  User root\n  IdentityFile ~/.ssh/runpod_ed25519\n  StrictHostKeyChecking no\n  UserKnownHostsFile /dev/null\n  ServerAliveInterval 20\n' \
    "$host" "$(echo $ep|cut -d' ' -f1)" "$(echo $ep|cut -d' ' -f2)" > "$out"
  for i in $(seq 1 10); do ssh -F "$out" -o ConnectTimeout=12 "$host" 'true' 2>/dev/null && { echo "$ep"; return 0; }; sleep 12; done
  echo "$ep"
}

# rp_push <sshCfg> <host> <destDir> <file...>  — scp + CHECKSUM-VERIFY each; retries; fails loud on mismatch
rp_push() {
  local cfg="$1" host="$2" dest="$3"; shift 3
  ssh -F "$cfg" -o ConnectTimeout=12 "$host" "mkdir -p $dest" 2>/dev/null
  local f b L R ok=0
  for f in "$@"; do
    b=$(basename "$f"); L=$(md5 -q "$f" 2>/dev/null || md5sum "$f"|cut -d' ' -f1)
    for t in 1 2 3; do
      scp -F "$cfg" "$f" "$host:$dest/$b" 2>/dev/null
      R=$(ssh -F "$cfg" -o ConnectTimeout=12 "$host" "md5sum $dest/$b 2>/dev/null|cut -d' ' -f1" 2>/dev/null|grep -vi warning|tail -1)
      [ "$L" = "$R" ] && { ok=$((ok+1)); break; }
      echo "  retry $b ($t)" >&2
    done
    [ "$L" != "$R" ] && { echo "PUSH FAILED (checksum) $b" >&2; return 1; }
  done
  echo "pushed+verified $ok file(s)"
}

# rp_push_big <sshCfg> <host> <destDir> <file>  — for LARGE files: runpodctl send/receive (croc relay), avoids slow scp
# (run receive on the pod; prints code to relay). Prefer a network volume over this entirely.
rp_push_big() {
  echo "For files >~500MB on a flaky link: DON'T scp. Options:" >&2
  echo "  1) network volume (pre-stage, no transfer)" >&2
  echo "  2) rsync -avP -e 'ssh -F $1' <file> $2:$3   (resumable)" >&2
  echo "  3) runpodctl send <file>  ->  code  ->  on pod: runpodctl receive <code>" >&2
}

# rp_pull <sshCfg> <host> <remoteFile> <localPath>  — scp with validate+retry (flaky link)
rp_pull() {
  local cfg="$1" host="$2" rem="$3" loc="$4"
  for t in 1 2 3 4; do
    scp -F "$cfg" "$host:$rem" "$loc" 2>/dev/null
    python3 -c "import json;json.load(open('$loc'))" 2>/dev/null && { echo "pulled $loc"; return 0; }
    [ -s "$loc" ] && { echo "pulled $loc (non-json)"; return 0; }
    sleep 15
  done
  echo "PULL FAILED $rem" >&2; return 1
}

# rp_safety <podId> <sshCfg> <host> <graceSec> <procRegex>  — idle-terminate guard (background it)
rp_safety() {
  local pod="$1" cfg="$2" host="$3" grace="$4" re="$5"
  sleep "$grace"
  local busy; busy=$(ssh -F "$cfg" -o ConnectTimeout=12 "$host" "ps -eo cmd|grep -E '$re'|grep -v grep|wc -l" 2>/dev/null)
  [ "${busy:-0}" = "0" ] && { ~/runpodctl remove pod "$pod" 2>&1|grep -v deprecated|tail -1; echo "SAFETY_TERMINATED (idle)"; } || echo "SAFETY_SKIP (busy)"
}

# Reminder printed on source
echo "runpod_lib loaded. Poller rule: teardown on SUCCESS only, KEEP pod on FAILURE. See RUNPOD_PLAYBOOK.md"
