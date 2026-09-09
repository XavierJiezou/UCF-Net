#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
umask "${UCFNET_UMASK:-000}"

usage() {
  cat >&2 <<EOF
Usage: bash scripts/test_ucfnet.sh -w WEIGHTS [-g GPUS] [-c CONFIG] [-m MODE] [-b BATCH] [-s N] [-h]

  -w WEIGHTS  trained model_best.pth                     (required)
  -g GPUS     comma-separated GPU list, e.g. 0,1,2,3     (default: 0)
  -c CONFIG   detector yaml: a bare name under
              DeepfakeBench/training/config/detector/ (e.g. ucfnet.yaml)
              or a path                                  (default: ucfnet.yaml)
  -m MODE     eval mode: cross-domain | in-domain | all  (default: cross-domain)
  -b BATCH    eval batch size                            (default: 96)
  -s N        smoke run on N samples only
  -h          show this help

Flags override the matching env vars (WEIGHTS / GPU_LIST / DETECTOR_TEMPLATE /
EVAL_MODE / BATCH_SIZE / SMOKE_SAMPLES), which still work too.
EOF
}

DETECTOR_DIR="$ROOT/DeepfakeBench/training/config/detector"
while getopts "w:g:c:m:b:s:h" opt; do
  case "$opt" in
    w) WEIGHTS=$OPTARG ;;
    g) GPU_LIST=$OPTARG ;;
    c) if [[ "$OPTARG" == */* ]]; then DETECTOR_TEMPLATE=$OPTARG; else DETECTOR_TEMPLATE="$DETECTOR_DIR/$OPTARG"; fi ;;
    m) EVAL_MODE=$OPTARG ;;
    b) BATCH_SIZE=$OPTARG ;;
    s) SMOKE_SAMPLES=$OPTARG ;;
    h) usage; exit 0 ;;
    *) usage; exit 1 ;;
  esac
done
shift $((OPTIND - 1))

source "$ROOT/scripts/resolve_python.sh"
ENV_PYTHON=$(resolve_env_python "$ROOT")
LIBFIX_DIR=${LIBFIX_DIR:-$ROOT/.libfix}
DATA_ROOT=${DATA_ROOT:-$ROOT/UCF-Net-dataset}
PRETRAINED_ROOT=${PRETRAINED_ROOT:-$ROOT/pretrained}
DETECTOR_TEMPLATE=${DETECTOR_TEMPLATE:-$ROOT/DeepfakeBench/training/config/detector/ucfnet.yaml}
TEST_CONFIG=${TEST_CONFIG:-$ROOT/DeepfakeBench/training/config/test_config.yaml}
RUNTIME_CONFIG=${RUNTIME_CONFIG:-$ROOT/.runtime/configs/ucfnet.yaml}
LOG_DIR=${LOG_DIR:-$ROOT/runs/train/ucfnet}
MODEL_NAME=${MODEL_NAME:-UCF-Net}
DETECTOR_TYPE=${DETECTOR_TYPE:-CLIP-DINO}
EVAL_MODE=${EVAL_MODE:-cross-domain}
GPU_LIST=${GPU_LIST:-0}
BATCH_SIZE=${BATCH_SIZE:-96}
WORKERS=${WORKERS:-4}
PIN_MEMORY=${PIN_MEMORY:-0}
SMOKE_SAMPLES=${SMOKE_SAMPLES:-}
WEIGHTS=${WEIGHTS:?Set WEIGHTS to a trained model_best.pth}

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
WEIGHTS=$(resolve_file "$WEIGHTS")

case "$EVAL_MODE" in
  cross-domain)
    TEST_JSON=${TEST_JSON:-$DATA_ROOT/splits/test_cross_domain.json}
    EXPECTED_SAMPLES=${EXPECTED_SAMPLES:-286448}
    ;;
  in-domain)
    TEST_JSON=${TEST_JSON:-$DATA_ROOT/splits/test_in_domain.json}
    EXPECTED_SAMPLES=${EXPECTED_SAMPLES:-1393675}
    ;;
  all)
    TEST_JSON=${TEST_JSON:-$DATA_ROOT/splits/test.json}
    EXPECTED_SAMPLES=${EXPECTED_SAMPLES:-1680123}
    ;;
  *)
    echo "Unsupported EVAL_MODE=$EVAL_MODE; use cross-domain, in-domain, or all." >&2
    exit 1
    ;;
esac

TEST_JSON=$(resolve_file "$TEST_JSON")

require_env_python "$ENV_PYTHON"
for path in "$WEIGHTS" "$TEST_JSON" "$TEST_CONFIG"; do
  if [[ ! -f "$path" ]]; then
    echo "Missing required file: $path" >&2
    exit 1
  fi
done

if [[ ! -d "$LIBFIX_DIR" ]]; then
  libcuda_so1=$(ldconfig -p 2>/dev/null | awk '/libcuda\.so\.1/{print $NF; exit}')
  if [[ -n "$libcuda_so1" ]]; then
    mkdir -p "$LIBFIX_DIR"
    ln -sf "$libcuda_so1" "$LIBFIX_DIR/libcuda.so"
  fi
fi
if [[ -d "$LIBFIX_DIR" ]]; then
  export LD_LIBRARY_PATH="$LIBFIX_DIR:/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

"$ENV_PYTHON" "$ROOT/scripts/materialize_config.py" \
  --repo-root "$ROOT" \
  --template "$DETECTOR_TEMPLATE" \
  --output "$RUNTIME_CONFIG" \
  --data-root "$DATA_ROOT" \
  --pretrained-root "$PRETRAINED_ROOT" \
  --log-dir "$LOG_DIR" >/dev/null

IFS=',' read -r -a GPUS <<< "$GPU_LIST"
SHARD_COUNT=${SHARD_COUNT:-${#GPUS[@]}}
RUN_ID=${RUN_ID:-$(basename "$(dirname "$WEIGHTS")")_${EVAL_MODE}_$(date '+%Y-%m-%d-%H-%M-%S')}
OUT_ROOT=${OUT_ROOT:-$ROOT/eval_logs/$RUN_ID}
RESULTS_DIR=${RESULTS_DIR:-$ROOT/results/$RUN_ID}

mkdir -p "$OUT_ROOT/raw" "$OUT_ROOT/logs" "$RESULTS_DIR"
chmod -R a+rwX "$OUT_ROOT" "$RESULTS_DIR" 2>/dev/null || true

export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
export NUMEXPR_NUM_THREADS=${NUMEXPR_NUM_THREADS:-1}
export TORCH_NUM_THREADS=${TORCH_NUM_THREADS:-1}
export PIN_MEMORY

{
  echo "started_at=$(date '+%F %T')"
  echo "root=$ROOT"
  echo "env_python=$ENV_PYTHON"
  echo "eval_mode=$EVAL_MODE"
  echo "test_json=$TEST_JSON"
  echo "dataset_root=$DATA_ROOT"
  echo "detector=$RUNTIME_CONFIG"
  echo "weights=$WEIGHTS"
  echo "gpu_list=$GPU_LIST"
  echo "shard_count=$SHARD_COUNT"
  echo "batch_size=$BATCH_SIZE"
  echo "workers=$WORKERS"
  echo "expected_samples=$EXPECTED_SAMPLES"
  echo "smoke_samples=$SMOKE_SAMPLES"
  echo "out_root=$OUT_ROOT"
  echo "results_dir=$RESULTS_DIR"
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu --format=csv,noheader,nounits
  fi
} | tee "$OUT_ROOT/logs/run_manifest.txt"

cat > "$OUT_ROOT/run_command.sh" <<EOF
#!/usr/bin/env bash
cd "$ROOT"
WEIGHTS="$WEIGHTS" EVAL_MODE="$EVAL_MODE" GPU_LIST="$GPU_LIST" BATCH_SIZE="$BATCH_SIZE" WORKERS="$WORKERS" SHARD_COUNT="$SHARD_COUNT" OUT_ROOT="$OUT_ROOT" RESULTS_DIR="$RESULTS_DIR" bash scripts/test_ucfnet.sh
EOF
chmod +x "$OUT_ROOT/run_command.sh"

pids=()
for shard in $(seq 0 $((SHARD_COUNT - 1))); do
  gpu=${GPUS[$((shard % ${#GPUS[@]}))]}
  extra_args=()
  if [[ -n "$SMOKE_SAMPLES" ]]; then
    extra_args+=(--max-samples "$SMOKE_SAMPLES")
  fi
  echo "[$(date '+%F %T')] starting $MODEL_NAME $EVAL_MODE shard $shard/$SHARD_COUNT on cuda:$gpu" | tee -a "$OUT_ROOT/logs/run.log"
  CUDA_VISIBLE_DEVICES="$gpu" "$ENV_PYTHON" "$ROOT/scripts/eval_test_json.py" \
    --repo-root "$ROOT" \
    --workdir "$ROOT/DeepfakeBench" \
    --detector-path "$RUNTIME_CONFIG" \
    --test-config-path "$TEST_CONFIG" \
    --weights-path "$WEIGHTS" \
    --test-json "$TEST_JSON" \
    --dataset-root "$DATA_ROOT" \
    --batch-size "$BATCH_SIZE" \
    --workers "$WORKERS" \
    --shard-index "$shard" \
    --shard-count "$SHARD_COUNT" \
    --raw-output "$OUT_ROOT/raw/shard_${shard}.npz" \
    --log-file "$OUT_ROOT/logs/shard_${shard}.log" \
    "${extra_args[@]}" \
    > "$OUT_ROOT/logs/shard_${shard}.stdout" 2>&1 &
  pids+=("$!")
  sleep 5
done

failed=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    failed=1
  fi
done

if [[ "$failed" != 0 ]]; then
  echo "[$(date '+%F %T')] $MODEL_NAME $EVAL_MODE evaluation failed; inspect $OUT_ROOT/logs" | tee -a "$OUT_ROOT/logs/run.log" >&2
  exit 1
fi

merge_args=(
  "$ROOT/scripts/merge_ucfnet_results.py"
  --model-raw "$OUT_ROOT"/raw/shard_*.npz
  --output-dir "$OUT_ROOT"
  --results-dir "$RESULTS_DIR"
  --model-name "$MODEL_NAME"
  --detector-type "$DETECTOR_TYPE"
  --eval-mode "$EVAL_MODE"
)
if [[ -n "$SMOKE_SAMPLES" ]]; then
  merge_args+=(--skip-sample-check --allow-partial-metrics)
else
  merge_args+=(--expected-samples "$EXPECTED_SAMPLES")
fi

"$ENV_PYTHON" "${merge_args[@]}" | tee -a "$OUT_ROOT/logs/run.log"
echo "[$(date '+%F %T')] wrote compact results to $RESULTS_DIR" | tee -a "$OUT_ROOT/logs/run.log"
