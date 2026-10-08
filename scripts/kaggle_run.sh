#!/usr/bin/env bash
# Run the core pipeline (NB1-NB4) on a Kaggle T4 notebook, unattended.
#
#   !curl -sL https://raw.githubusercontent.com/<user>/<repo>/main/scripts/kaggle_run.sh | bash
#
# Use "Save Version -> Save & Run All" so the run survives a closed browser tab.
# Whatever happens, the graded evidence (executed notebooks, screenshots,
# data/eval, data/pref, adapter json) is zipped to /kaggle/working/lab22_artifacts.zip.
set -o pipefail

REPO_URL=${REPO_URL:-https://github.com/t00-tuannguyen/K4-L3-Track3-Day22-DPO-ORPO-Alignment-NguyenTienTuan-2A202602595.git}
WORK=/tmp/lab22
OUT=/kaggle/working/out
export CUDA_VISIBLE_DEVICES=0  # Kaggle gives T4 x2; the lab uses one GPU

collect() {
  cd "$WORK" || return
  mkdir -p "$OUT/adapters/dpo" "$OUT/adapters/sft-mini" "$OUT/notebooks"
  cp -r submission data "$OUT/" 2>/dev/null
  rm -rf "$OUT/data/eval/lm_eval"
  cp notebooks/*.ipynb "$OUT/notebooks/" 2>/dev/null
  cp adapters/dpo/*.json "$OUT/adapters/dpo/" 2>/dev/null
  cp adapters/sft-mini/adapter_config.json "$OUT/adapters/sft-mini/" 2>/dev/null
  cd /kaggle/working && rm -f lab22_artifacts.zip && zip -qr lab22_artifacts.zip out pipeline.log
  ls -la /kaggle/working
}
trap collect EXIT

nvidia-smi --query-gpu=name,memory.total --format=csv
df -h /tmp /kaggle/working | tail -2

test -d "$WORK/.git" || git clone -q "$REPO_URL" "$WORK"
cd "$WORK" || exit 1

# Core stack only: llama-cpp-python (NB5) and lm-eval (NB6) are bonus and slow to build.
pip install -q "unsloth>=2026.10.1" "trl>=1.13,<1.14" "transformers>=5.2,<5.18" \
  "peft>=0.18,<1.0" "accelerate>=1.10,<2.0" "bitsandbytes>=0.48,<1.0" "datasets>=4.7,<5.0" \
  "matplotlib>=3.9,<4.0" "pandas>=2.2,<4.0" "pyarrow>=17" "jupytext>=1.16,<2.0" \
  nbconvert ipykernel "openai>=1.55,<4.0" "anthropic>=0.40,<2.0" 2>&1 | tail -3
python -c "import unsloth, trl, transformers; print('imports ok', trl.__version__, transformers.__version__)" || exit 1

make sft data dpo eval 2>&1 | tee /kaggle/working/pipeline.log
echo "PIPELINE_EXIT=${PIPESTATUS[0]}" | tee -a /kaggle/working/pipeline.log
