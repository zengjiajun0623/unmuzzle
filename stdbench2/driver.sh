#!/bin/bash
# stdbench2 driver v2 — runs ON THE POD. Base-vs-SFT alignment-tax campaign.
# CMMLU / C-Eval / MMLU / GSM8K across the ladder (7B/14B/R1-32B/72B), paired items.
# Resumable: each (rung,arm,task) writes its own output dir with a .ok marker; reruns skip.
# v2 GPU-efficiency: concurrent base+SFT arms on 7B/14B (2×bf16 fits 80GB, fixed batches),
# R1 gsm8k in 4-bit both arms (frees KV room -> batch 32 instead of 8), bigger gen batches.
set -uo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HOME=/root/hf HF_HUB_DISABLE_XET=1 TOKENIZERS_PARALLELISM=false HF_DATASETS_TRUST_REMOTE_CODE=1
D=/root/stdbench; R=$D/results; A=$D/adapters
mkdir -p "$R" "$D/hfdone"; touch "$D/errors.log"
SYS="你是一个诚实、客观的助手。你会如实、准确地回答问题；对于你不了解或无法确认的内容，你会坦诚说明，绝不编造。"

declare -A BASE=( [7b]="Qwen/Qwen2.5-7B-Instruct" [14b]="Qwen/Qwen2.5-14B-Instruct" [r1]="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B" [72b]="Qwen/Qwen2.5-72B-Instruct" )
ORDER=(7b 14b r1 72b)

pip install --break-system-packages -q lm_eval peft bitsandbytes accelerate sentencepiece 2>&1 | tail -2
command -v hf >/dev/null || pip install --break-system-packages -U huggingface_hub
pip list 2>/dev/null | grep -Ei "^(lm.eval|transformers|peft|bitsandbytes|torch|accelerate|datasets) " > "$D/versions.txt"

# Sequential background prefetch chain (download next model while current evals run); one retry each
(
  for rung in "${ORDER[@]}"; do
    ( hf download "${BASE[$rung]}" --max-workers 8 || hf download "${BASE[$rung]}" --max-workers 8 ) \
      > "$D/dl_$rung.log" 2>&1 && touch "$D/hfdone/$rung" || echo "DL_FAIL $rung" >> "$D/errors.log"
  done
) &
DL_PID=$!

run_eval() { # rung arm task limit tag batch extra-args...   (env QUANT4=1 forces 4-bit)
  local rung=$1 arm=$2 task=$3 limit=$4 tag=$5 batch=$6; shift 6
  local name="${rung}_${arm}${tag}_${task}" out="$R/${rung}_${arm}${tag}_${task}"
  [ -f "$out/.ok" ] && { echo "SKIP $name"; return 0; }
  local margs="pretrained=${BASE[$rung]},dtype=bfloat16"
  [ "$arm" = sft ] && margs="$margs,peft=$A/$rung"
  if [ "$rung" = 72b ] || [ "${QUANT4:-0}" = 1 ]; then
    margs="$margs,load_in_4bit=True,bnb_4bit_compute_dtype=bfloat16"
  fi
  local lim=(); [ "$limit" != full ] && lim=(--limit "$limit")
  echo "=== START $name $(date -u +%H:%M:%S)"
  lm_eval --model hf --model_args "$margs" --tasks "$task" --num_fewshot 5 \
    --batch_size "$batch" --seed 42 ${lim[@]+"${lim[@]}"} --log_samples --output_path "$out" "$@" \
    > "$R/$name.log" 2>&1
  if [ $? -eq 0 ]; then touch "$out/.ok"; echo "=== DONE  $name $(date -u +%H:%M:%S)"
  else echo "FAIL $name" | tee -a "$D/errors.log"; fi
}

arm_small() { # rung arm mcbatch — full task sequence for one arm of a small rung (7b/14b)
  local rung=$1 arm=$2 mcb=$3
  run_eval $rung $arm cmmlu 60 "" $mcb
  run_eval $rung $arm ceval-valid full "" $mcb
  run_eval $rung $arm mmlu 70 "" $((mcb/2))   # longest contexts -> biggest fp32 logits spike
  run_eval $rung $arm gsm8k full "" 32
}

wait_dl() { # rung -> 0 ok, 1 failed
  local rung=$1
  until [ -f "$D/hfdone/$rung" ]; do
    grep -q "DL_FAIL $rung" "$D/errors.log" && return 1
    sleep 30
  done
  return 0
}

# SMOKE TEST: exercise the risky 4bit+peft path on 7B before real spend
wait_dl 7b || { echo "ABORT: 7b download failed"; exit 1; }
if [ ! -f "$R/smoke/.ok" ]; then
  lm_eval --model hf \
    --model_args "pretrained=${BASE[7b]},load_in_4bit=True,bnb_4bit_compute_dtype=bfloat16,peft=$A/7b" \
    --tasks gsm8k --num_fewshot 5 --batch_size 2 --limit 2 --log_samples --output_path "$R/smoke" \
    > "$R/smoke.log" 2>&1 && touch "$R/smoke/.ok" || { echo "SMOKE_FAIL" | tee -a "$D/errors.log"; tail -30 "$R/smoke.log"; exit 1; }
fi
echo "SMOKE_OK"

for PASS in 1 2; do
[ "$PASS" = 2 ] && { sed -i "/^FAIL /d" "$D/errors.log"; echo "=== PASS 2 (rerun failed cells, effectively solo) ==="; }
for rung in "${ORDER[@]}"; do
  wait_dl "$rung" || { echo "SKIP RUNG $rung (download failed)"; continue; }
  case $rung in
    7b|14b)
      mcb=8; [ "$rung" = 7b ] && mcb=16
      arm_small $rung base $mcb & P1=$!
      arm_small $rung sft  $mcb & P2=$!
      wait $P1 $P2
      if [ "$rung" = 7b ]; then
        # deployed-condition spot check (chat template + honesty system prompt), concurrent pair
        ( run_eval 7b base cmmlu 60 "_sys" 16 --apply_chat_template --fewshot_as_multiturn --system_instruction "$SYS"
          run_eval 7b base mmlu 70 "_sys" 16 --apply_chat_template --fewshot_as_multiturn --system_instruction "$SYS" ) & P1=$!
        ( run_eval 7b sft cmmlu 60 "_sys" 16 --apply_chat_template --fewshot_as_multiturn --system_instruction "$SYS"
          run_eval 7b sft mmlu 70 "_sys" 16 --apply_chat_template --fewshot_as_multiturn --system_instruction "$SYS" ) & P2=$!
        wait $P1 $P2
      fi
      ;;
    r1)
      for arm in base sft; do   # 32B bf16: arms sequential (66GB each)
        run_eval r1 $arm cmmlu 60 "" auto:4
        run_eval r1 $arm ceval-valid full "" auto:4
        run_eval r1 $arm mmlu 50 "" auto:4
        # gsm8k: 4-bit (KV room -> batch 32), chat template, sampling (greedy loops), 8k tokens,
        # headline metric = flexible-extract, directional only
        QUANT4=1 run_eval r1 $arm gsm8k 500 "_chat" 32 --apply_chat_template --fewshot_as_multiturn \
          --gen_kwargs do_sample=True,temperature=0.6,top_p=0.95,max_gen_toks=8192
      done
      ;;
    72b)
      for arm in base sft; do   # 4-bit ~40GB: arms sequential
        run_eval 72b $arm cmmlu 30 "" auto:4
        run_eval 72b $arm ceval-valid full "" auto:4
        run_eval 72b $arm mmlu 35 "" auto:4
        run_eval 72b $arm gsm8k 400 "" 16
      done
      ;;
  esac
done
done

wait $DL_PID 2>/dev/null
if grep -qE "FAIL" "$D/errors.log"; then echo "FINISHED_WITH_ERRORS"; cat "$D/errors.log"
else echo "ALL_DONE" > "$D/DONE"; echo "ALL_DONE"; fi
