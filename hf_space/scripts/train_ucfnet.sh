#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
umask "${UCFNET_UMASK:-000}"

usage() {
  cat >&2 <<EOF
Usage: bash scripts/train_ucfnet.sh [-g GPUS] [-c CONFIG] [-p PORT] [-r RESUME] [-b BATCH] [-n] [-h]

  -g GPUS     comma-separated GPU list, e.g. 0,1,3,5      (default: 0)
  -c CONFIG   detector yaml: a bare name under
              DeepfakeBench/training/config/detector/ (e.g. ucfnet.yaml)
              or a path                                  (default: ucfnet.yaml)
  -p PORT     DDP master port                            (default: 29617)
  -r RESUME   resume from a checkpoint (latest.pth)
  -b BATCH    train batch size per GPU
  -n          disable DDP (single process)
  -h          show this help

Flags override the matching env vars (GPU_LIST / DETECTOR_TEMPLATE /
MASTER_PORT / RESUME / TRAIN_BATCH_SIZE / DDP), which still work too.
EOF
}

DETECTOR_DIR="$ROOT/DeepfakeBench/training/config/detector"
while getopts "g:c:p:r:b:nh" opt; do
  case "$opt" in
    g) GPU_LIST=$OPTARG ;;
    c) if [[ "$OPTARG" == */* ]]; then DETECTOR_TEMPLATE=$OPTARG; else DETECTOR_TEMPLATE="$DETECTOR_DIR/$OPTARG"; fi ;;
    p) MASTER_PORT=$OPTARG ;;
    r) RESUME=$OPTARG ;;
    b) TRAIN_BATCH_SIZE=$OPTARG ;;
    n) DDP=0 ;;
    h) usage; exit 0 ;;
    *) usage; exit 1 ;;
  esac
done
shift $((OPTIND - 1))

source "$ROOT/scripts/resolve_python.sh"
ENV_PYTHON=$(resolve_env_python "$ROOT")
DATA_ROOT=${DATA_ROOT:-$ROOT/UCF-Net-dataset}
PRETRAINED_ROOT=${PRETRAINED_ROOT:-$ROOT/pretrained}
DETECTOR_TEMPLATE=${DETECTOR_TEMPLATE:-$ROOT/DeepfakeBench/training/config/detector/ucfnet.yaml}
TRAIN_CONFIG=${TRAIN_CONFIG:-$ROOT/DeepfakeBench/training/config/train_config.yaml}
RUNTIME_CONFIG=${RUNTIME_CONFIG:-$ROOT/.runtime/configs/ucfnet.yaml}
LOG_DIR=${LOG_DIR:-$ROOT/runs/train/ucfnet}
LIBFIX_DIR=${LIBFIX_DIR:-$ROOT/.libfix}
GPU_LIST=${GPU_LIST:-0}
MASTER_PORT=${MASTER_PORT:-29617}
DDP=${DDP:-1}
RESUME=${RESUME:-}
RESUME_WEIGHTS_ONLY=${RESUME_WEIGHTS_ONLY:-0}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-}
TEST_BATCH_SIZE=${TEST_BATCH_SIZE:-}
TRAIN_WORKERS=${TRAIN_WORKERS:-}
MAX_TRAIN_ITERS=${MAX_TRAIN_ITERS:-}

resolve_file() {
  local path=$1
  if [[ "$path" = /* ]]; then
    printf '%s\n' "$path"
  else
    printf '%s\n' "$ROOT/$path"
  fi
}

resolve_dir() {
  local path=$1
  if [[ "$path" = /* ]]; then
    printf '%s\n' "$path"
  else
    printf '%s\n' "$ROOT/$path"
  fi
}

DATA_ROOT=$(resolve_dir "$DATA_ROOT")
PRETRAINED_ROOT=$(resolve_dir "$PRETRAINED_ROOT")
LOG_DIR=$(resolve_dir "$LOG_DIR")
LIBFIX_DIR=$(resolve_dir "$LIBFIX_DIR")
DETECTOR_TEMPLATE=$(resolve_file "$DETECTOR_TEMPLATE")
TRAIN_CONFIG=$(resolve_file "$TRAIN_CONFIG")
RUNTIME_CONFIG=$(resolve_file "$RUNTIME_CONFIG")
if [[ -n "$RESUME" ]]; then
  RESUME=$(resolve_file "$RESUME")
fi

require_env_python "$ENV_PYTHON"

if [[ ! -e "$LIBFIX_DIR/libcuda.so" ]]; then
  libcuda_so1=$(ldconfig -p 2>/dev/null | awk '/libcuda\.so\.1/{print $NF; exit}')
  if [[ -n "$libcuda_so1" ]]; then
    mkdir -p "$LIBFIX_DIR"
    ln -sf "$libcuda_so1" "$LIBFIX_DIR/libcuda.so"
  fi
fi
if [[ -d "$LIBFIX_DIR" ]]; then
  export LD_LIBRARY_PATH="$LIBFIX_DIR:/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

materialize_args=(
  "$ROOT/scripts/materialize_config.py"
  --repo-root "$ROOT" \
  --template "$DETECTOR_TEMPLATE" \
  --output "$RUNTIME_CONFIG" \
  --data-root "$DATA_ROOT" \
  --pretrained-root "$PRETRAINED_ROOT" \
  --log-dir "$LOG_DIR"
)
if [[ -n "$TRAIN_BATCH_SIZE" ]]; then
  materialize_args+=(--train-batch-size "$TRAIN_BATCH_SIZE")
fi
if [[ -n "$TEST_BATCH_SIZE" ]]; then
  materialize_args+=(--test-batch-size "$TEST_BATCH_SIZE")
fi
if [[ -n "$TRAIN_WORKERS" ]]; then
  materialize_args+=(--workers "$TRAIN_WORKERS")
fi
if [[ -n "$MAX_TRAIN_ITERS" ]]; then
  materialize_args+=(--max-train-iters "$MAX_TRAIN_ITERS")
fi
"$ENV_PYTHON" "${materialize_args[@]}" >/dev/null

IFS=',' read -r -a GPUS <<< "$GPU_LIST"
NPROC=${NPROC:-${#GPUS[@]}}

cmd=(
  "$ENV_PYTHON" -m torch.distributed.launch
  --master_port="$MASTER_PORT"
  --nproc_per_node="$NPROC"
  training/train.py
  --detector_path "$RUNTIME_CONFIG"
  --train_config_path "$TRAIN_CONFIG"
)

if [[ "$DDP" == "1" ]]; then
  cmd+=(--ddp)
fi
if [[ -n "$RESUME" ]]; then
  cmd+=(--resume "$RESUME")
fi
if [[ "$RESUME_WEIGHTS_ONLY" == "1" ]]; then
  cmd+=(--resume-weights-only)
fi

mkdir -p "$LOG_DIR"
chmod -R a+rwX "$LOG_DIR" 2>/dev/null || true
cd "$ROOT/DeepfakeBench"
echo "Runtime config: $RUNTIME_CONFIG"
echo "Logs/checkpoints: $LOG_DIR"
CUDA_VISIBLE_DEVICES="$GPU_LIST" "${cmd[@]}"
