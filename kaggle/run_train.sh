#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/.."

NPROC=${NPROC:-$(python -c "import torch;print(torch.cuda.device_count())")}
OUT_DIR=${OUT_DIR:-results/gpt2}
BS=${BS:-12}

export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export OMP_NUM_THREADS=2
export TOKENIZERS_PARALLELISM=false
export TORCH_HOME=${TORCH_HOME:-/kaggle/temp/torch}
mkdir -p "$TORCH_HOME"

echo "GPUs=$NPROC  bs/gpu=$BS  effective batch=$((NPROC * BS))  out=$OUT_DIR"

torchrun --nproc_per_node "$NPROC" train_evcap.py \
  --out_dir "$OUT_DIR" \
  --dataset karpathy \
  --karpathy_splits train,restval \
  --lm gpt2 \
  --lm_dtype fp32 \
  --vit_precision fp16 \
  --retrieval_backend torch \
  --epochs 1 \
  --bs "$BS" \
  --num_workers 2 \
  --save_every 2000 \
  --max_txt_len 128 \
  --topn 9 \
  --num_query_token_txt 8 \
  "$@"
