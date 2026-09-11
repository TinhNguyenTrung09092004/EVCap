#!/usr/bin/env bash
# Usage: bash kaggle/run_eval.sh [extra args forwarded to eval_evcap.py]
set -e
cd "$(dirname "$0")/.."

CKPT=${CKPT:-results/gpt2/last.pt}
OUT_PATH=${OUT_PATH:-results/gpt2_eval}
VAL_DIR=${VAL_DIR:-/kaggle/input/datasets/nadaibrahim/coco2014/val2014/val2014}

export TOKENIZERS_PARALLELISM=false
export TORCH_HOME=${TORCH_HOME:-/kaggle/temp/torch}

python -u eval_evcap.py \
  --device cuda:0 \
  --name_of_datasets coco \
  --ckpt "$CKPT" \
  --path_of_val_datasets data/coco/test_captions.json \
  --image_folder "$VAL_DIR/" \
  --out_path "$OUT_PATH" \
  --lm gpt2 \
  --lm_dtype fp32 \
  --vit_precision fp16 \
  --retrieval_backend torch \
  --beam_width 5 \
  --topn 9 \
  --num_query_token_txt 8 \
  "$@"

echo "========================== COCO EVAL =========================="
python -u kaggle/cocoeval.py --result_file_path "$OUT_PATH/coco_generated_captions.json"
