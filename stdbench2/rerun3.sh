#!/bin/bash
# rerun3.sh — post-reset: self-contained setup + the 3 remaining R1 cells.
# Idempotent: skips installed deps, downloaded weights, and .ok cells.
set -uo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HOME=/root/hf HF_HUB_DISABLE_XET=1 TOKENIZERS_PARALLELISM=false HF_DATASETS_TRUST_REMOTE_CODE=1
D=/root/stdbench; R=$D/results; A=$D/adapters
mkdir -p "$R"
M="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"

# deps (no-ops if the container disk survived)
python3 -c "import lm_eval" 2>/dev/null || pip install --break-system-packages -q lm_eval peft bitsandbytes accelerate sentencepiece
python3 -c "import datasets,sys; sys.exit(0 if datasets.__version__.startswith('3.') else 1)" 2>/dev/null \
  || pip install --break-system-packages -q "datasets==3.6.0"
# lm_eval 4bit patch for transformers 5 (idempotent)
python3 - <<'EOF'
path = '/usr/local/lib/python3.12/dist-packages/lm_eval/models/huggingface.py'
s = open(path).read()
if 'quantization_config = BitsAndBytesConfig(' in s:
    print('PATCH already present')
else:
    old = '''            if model_kwargs.get("load_in_4bit"):
                assert vparse(transformers.__version__) >= vparse("4.30.0"), (
                    "load_in_4bit requires transformers >= 4.30.0"
                )
                if compute_dtype := model_kwargs.get("bnb_4bit_compute_dtype"):
                    model_kwargs["bnb_4bit_compute_dtype"] = get_dtype(compute_dtype)'''
    new = '''            if model_kwargs.pop("load_in_4bit", None):
                from transformers import BitsAndBytesConfig
                _cd = model_kwargs.pop("bnb_4bit_compute_dtype", None)
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=get_dtype(_cd) if _cd else None,
                )'''
    assert old in s, 'PATTERN NOT FOUND'
    open(path, 'w').write(s.replace(old, new))
    print('PATCHED')
EOF

# weights (no-op if cache survived)
hf download "$M" --max-workers 8 > "$D/dl_r1_again.log" 2>&1 || hf download "$M" --max-workers 8 >> "$D/dl_r1_again.log" 2>&1

run_cell() { # name task limit extra...
  local name=$1 task=$2 limit=$3; shift 3
  local out="$R/$name"
  [ -f "$out/.ok" ] && { echo "SKIP $name"; return 0; }
  rm -rf "$out"
  echo "=== RERUN3 START $name $(date -u +%H:%M:%S)"
  lm_eval --model hf \
    --model_args "pretrained=$M,dtype=bfloat16,load_in_4bit=True,bnb_4bit_compute_dtype=bfloat16$EXTRA_MARGS" \
    --tasks "$task" --num_fewshot 5 --batch_size 16 --seed 42 --limit "$limit" --log_samples --output_path "$out" \
    --apply_chat_template --fewshot_as_multiturn \
    --gen_kwargs do_sample=True,temperature=0.6,top_p=0.95,max_gen_toks=8192 "$@" \
    > "$R/$name.log" 2>&1
  if [ $? -eq 0 ]; then touch "$out/.ok"; echo "=== RERUN3 DONE  $name $(date -u +%H:%M:%S)"
  else echo "RERUN3_FAIL $name"; tail -3 "$R/$name.log"; fi
}

EXTRA_MARGS=",peft=$A/r1" run_cell r1_sft_chat_gsm8k gsm8k 500
EXTRA_MARGS=""            run_cell r1_base_chat_minerva_math500 minerva_math500 500
EXTRA_MARGS=",peft=$A/r1" run_cell r1_sft_chat_minerva_math500 minerva_math500 500

ok=0
for c in r1_sft_chat_gsm8k r1_base_chat_minerva_math500 r1_sft_chat_minerva_math500; do
  [ -f "$R/$c/.ok" ] && ok=$((ok+1))
done
[ "$ok" = 3 ] && echo "RERUN3_ALL_DONE" | tee "$D/RERUN3_DONE" || echo "RERUN3_INCOMPLETE ($ok/3)"
