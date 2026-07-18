#!/bin/bash
# Optimized Muon-vs-AdamW A/B driver: 3 arms TRAIN CONCURRENTLY on one H100 (each
# 4-bit QLoRA fits in ~12-15GB, 3x < 80GB), then BATCHED evals. ~2-3x faster than
# sequential+bs1, same matched science. Usage: run_ab_fast.sh <hf_model_id> <tag>
set -u
cd /workspace/m
export HF_HOME=/runpod/hf HF_HUB_DISABLE_XET=1 PYTHONUNBUFFERED=1
pip install -q --break-system-packages -U transformers peft bitsandbytes accelerate huggingface_hub >> ab.log 2>&1
MID="$1"; TAG="$2"
BASE=https://huggingface.co/$MID/resolve/main
MDIR=/runpod/$TAG; mkdir -p $MDIR
echo "=DL $(date -u +%H:%M)=" >> ab.log
python3 -c "from huggingface_hub import list_repo_files; open('/tmp/f.txt','w').write('\n'.join(list_repo_files('$MID')))" 2>>ab.log
# -f fails on HTTP errors (never saves an error body as a shard); --create-dirs for nested paths
cat /tmp/f.txt | xargs -P8 -I{} curl -fsSL --create-dirs --retry 8 --retry-delay 3 -o "$MDIR/{}" "$BASE/{}"
python3 -c "from transformers import AutoConfig; AutoConfig.from_pretrained('$MDIR')" >> ab.log 2>&1 || { echo "AB_FAILED: bad download"; exit 1; }
echo "=DL done $(date -u +%H:%M) size=$(du -sh $MDIR|cut -f1)=" >> ab.log

# --- 3 training arms CONCURRENTLY (background), each its own log ---
echo "=TRAIN x3 concurrent $(date -u +%H:%M)=" >> ab.log
python -u train_muon.py --model $MDIR --data train_v2.jsonl --out a_adamw --optimizer adamw --lr 1e-4 > tr_a.log 2>&1 &
PA=$!
python -u train_muon.py --model $MDIR --data train_v2.jsonl --out b_muon1 --optimizer muon  --lr 1e-4 > tr_b.log 2>&1 &
PB=$!
python -u train_muon.py --model $MDIR --data train_v2.jsonl --out c_muon2 --optimizer muon  --lr 2e-4 > tr_c.log 2>&1 &
PC=$!
wait $PA; RA=$?; wait $PB; RB=$?; wait $PC; RC=$?
if [ $RA -ne 0 ] || [ $RB -ne 0 ] || [ $RC -ne 0 ]; then
  echo "AB_FAILED train a=$RA b=$RB c=$RC"; tail -30 tr_a.log tr_b.log tr_c.log; exit 1; fi
echo "=TRAIN done $(date -u +%H:%M)=" >> ab.log

# --- batched evals (fp16, sequential to stay under VRAM for big models) ---
for pair in "base:" "a_adamw:a_adamw" "b_muon1:b_muon1" "c_muon2:c_muon2"; do
  name=${pair%%:*}; adp=${pair##*:}
  echo "=EVAL $name $(date -u +%H:%M)=" >> ab.log
  if [ -z "$adp" ]; then
    python -u eval_fast.py --model $MDIR --bench benchmark_china_v2.json --out eval_$name.json --bsz 32 >> ab.log 2>&1
  else
    python -u eval_fast.py --model $MDIR --adapter $adp --bench benchmark_china_v2.json --out eval_$name.json --bsz 32 >> ab.log 2>&1
  fi
done

tar czf ab_$TAG.tgz a_adamw b_muon1 c_muon2 2>/dev/null
NBASE=$(python -c "import json;print(len(json.load(open('eval_base.json'))['results']))" 2>/dev/null||echo 0)
NA=$(python -c "import json;print(len(json.load(open('eval_a_adamw.json'))['results']))" 2>/dev/null||echo 0)
NB=$(python -c "import json;print(len(json.load(open('eval_b_muon1.json'))['results']))" 2>/dev/null||echo 0)
NC=$(python -c "import json;print(len(json.load(open('eval_c_muon2.json'))['results']))" 2>/dev/null||echo 0)
if [ "$NBASE" -ge 260 ] && [ "$NA" -ge 260 ] && [ "$NB" -ge 260 ] && [ "$NC" -ge 260 ]; then echo "AB_OK base=$NBASE a=$NA b=$NB c=$NC"; else echo "AB_FAILED base=$NBASE a=$NA b=$NB c=$NC"; tail -30 ab.log tr_a.log tr_b.log tr_c.log; fi
