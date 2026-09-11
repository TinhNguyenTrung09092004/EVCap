#!/usr/bin/env bash
set -e

pip install -q pycocoevalcap

# PTBTokenizer + METEOR in pycocoevalcap are java jars; without a JRE cocoeval.py
# silently drops METEOR and falls back to a regex tokenizer.
if ! command -v java >/dev/null 2>&1; then
  apt-get -qq update && apt-get -qq install -y default-jre >/dev/null
fi

# Only needed for --retrieval_backend faiss (the torch backend is the default and
# is numerically identical for an IndexFlatIP over L2-normalised vectors).
if [ "${WITH_FAISS:-0}" = "1" ]; then
  pip install -q faiss-cpu
fi

python - <<'PY'
import shutil, sys
print("java:", shutil.which("java") or "MISSING")
import pycocoevalcap; print("pycocoevalcap ok")
PY

echo "setup done"
