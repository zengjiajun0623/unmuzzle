#!/bin/bash
# stdbench2 finisher — runs ON THE POD after driver.sh ends.
# 1) reruns any missing grid cell SOLO (conservative batches, no concurrency)
# 2) adds the R1 MATH reasoning check (minerva_math ~500 items, both arms, same protocol as gsm8k cell)
# 3) writes FINISHER_DONE when the full grid is verified complete
set -uo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HOME=/root/hf HF_HUB_DISABLE_XET=1 TOKENIZERS_PARALLELISM=false HF_DATASETS_TRUST_REMOTE_CODE=1
D=/root/stdbench; R=$D/results; A=$D/adapters
SYS="你是一个诚实、客观的助手。你会如实、准确地回答问题；对于你不了解或无法确认的内容，你会坦诚说明，绝不编造。"
declare -A BASE=( [7b]="Qwen/Qwen2.5-7B-Instruct" [14b]="Qwen/Qwen2.5-14B-Instruct" [r1]="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B" [72b]="Qwen/Qwen2.5-72B-Instruct" )

run_eval() { # rung arm task limit tag batch extra-args...   (env QUANT4=1 forces 4-bit)
  local rung=$1 arm=$2 task=$3 limit=$4 tag=$5 batch=$6; shift 6
  local name="${rung}_${arm}${tag}_${task}" out="$R/${rung}_${arm}${tag}_${task}"
  [ -f "$out/.ok" ] && { echo "SKIP $name"; return 0; }
  rm -rf "$out"
  local margs="pretrained=${BASE[$rung]},dtype=bfloat16"
  [ "$arm" = sft ] && margs="$margs,peft=$A/$rung"
  if [ "$rung" = 72b ] || [ "${QUANT4:-0}" = 1 ]; then
    margs="$margs,load_in_4bit=True,bnb_4bit_compute_dtype=bfloat16"
  fi
  local lim=(); [ "$limit" != full ] && lim=(--limit "$limit")
  echo "=== FINISHER START $name $(date -u +%H:%M:%S)"
  lm_eval --model hf --model_args "$margs" --tasks "$task" --num_fewshot 5 \
    --batch_size "$batch" --seed 42 ${lim[@]+"${lim[@]}"} --log_samples --output_path "$out" "$@" \
    > "$R/$name.log" 2>&1
  if [ $? -eq 0 ]; then touch "$out/.ok"; echo "=== FINISHER DONE  $name $(date -u +%H:%M:%S)"
  else echo "FINISHER_FAIL $name"; fi
}

# --- 1) solo reruns of every grid cell (SKIPs if .ok already present) ---
for arm in base sft; do
  run_eval 7b  $arm cmmlu 60 "" 16;  run_eval 7b  $arm ceval-valid full "" 16
  run_eval 7b  $arm mmlu 70 "" 8;    run_eval 7b  $arm gsm8k full "" 32
  run_eval 7b  $arm cmmlu 60 "_sys" 16 --apply_chat_template --fewshot_as_multiturn --system_instruction "$SYS"
  run_eval 7b  $arm mmlu 70 "_sys" 8  --apply_chat_template --fewshot_as_multiturn --system_instruction "$SYS"
  run_eval 14b $arm cmmlu 60 "" 8;   run_eval 14b $arm ceval-valid full "" 8
  run_eval 14b $arm mmlu 70 "" 4;    run_eval 14b $arm gsm8k full "" 32
  run_eval r1  $arm cmmlu 60 "" auto:4; run_eval r1 $arm ceval-valid full "" auto:4
  run_eval r1  $arm mmlu 50 "" auto:4
  QUANT4=1 run_eval r1 $arm gsm8k 500 "_chat" 16 --apply_chat_template --fewshot_as_multiturn \
    --gen_kwargs do_sample=True,temperature=0.6,top_p=0.95,max_gen_toks=8192
  run_eval 72b $arm cmmlu 30 "" auto:4; run_eval 72b $arm ceval-valid full "" auto:4
  run_eval 72b $arm mmlu 35 "" auto:4;  run_eval 72b $arm gsm8k 400 "" 16
done

# --- 2) R1 MATH reasoning check (both arms, ~500 items = 72/subtask x 7 subtasks) ---
for arm in base sft; do
  QUANT4=1 run_eval r1 $arm minerva_math500 full "_chat" 16 --apply_chat_template --fewshot_as_multiturn \
    --gen_kwargs do_sample=True,temperature=0.6,top_p=0.95,max_gen_toks=8192
done

# --- 3) verify grid completeness ---
missing=0
for c in 7b_base_cmmlu 7b_sft_cmmlu 7b_base_ceval-valid 7b_sft_ceval-valid 7b_base_mmlu 7b_sft_mmlu \
         7b_base_gsm8k 7b_sft_gsm8k 7b_base_sys_cmmlu 7b_sft_sys_cmmlu 7b_base_sys_mmlu 7b_sft_sys_mmlu \
         14b_base_cmmlu 14b_sft_cmmlu 14b_base_ceval-valid 14b_sft_ceval-valid 14b_base_mmlu 14b_sft_mmlu \
         14b_base_gsm8k 14b_sft_gsm8k \
         r1_base_cmmlu r1_sft_cmmlu r1_base_ceval-valid r1_sft_ceval-valid r1_base_mmlu r1_sft_mmlu \
         r1_base_chat_gsm8k r1_sft_chat_gsm8k r1_base_chat_minerva_math500 r1_sft_chat_minerva_math500 \
         72b_base_cmmlu 72b_sft_cmmlu 72b_base_ceval-valid 72b_sft_ceval-valid 72b_base_mmlu 72b_sft_mmlu \
         72b_base_gsm8k 72b_sft_gsm8k; do
  [ -f "$R/$c/.ok" ] || { echo "STILL_MISSING $c"; missing=$((missing+1)); }
done
if [ "$missing" = 0 ]; then echo "FINISHER_ALL_DONE" | tee "$D/FINISHER_DONE"
else echo "FINISHER_INCOMPLETE ($missing missing)"; fi
