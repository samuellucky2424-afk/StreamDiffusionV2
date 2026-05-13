#!/bin/bash
set -eu

# Build the Svelte frontend, then launch the Python demo backend.
# Override HOST, PORT, GPU_IDS, STEP, MODEL_TYPE, CONDITIONING_SOURCE,
# and SEMANTIC_AVATAR_* via environment variables.
SCRIPT_DIR="$(CDPATH= cd -- "$(dirname "$0")" && pwd)"
FRONTEND_DIR="$SCRIPT_DIR/frontend"

PORT="${PORT:-7860}"
HOST="${HOST:-0.0.0.0}"
GPU_IDS="${GPU_IDS:-0}"
STEP="${STEP:-1}"
MODEL_TYPE="${MODEL_TYPE:-T2V-1.3B}"
USE_TAEHV="${USE_TAEHV:-0}"
USE_TENSORRT="${USE_TENSORRT:-0}"
FAST="${FAST:-0}"
CONDITIONING_SOURCE="${CONDITIONING_SOURCE:-rgb}"
SEMANTIC_AVATAR_TARGET_FPS="${SEMANTIC_AVATAR_TARGET_FPS:-8}"
SEMANTIC_AVATAR_JPEG_QUALITY="${SEMANTIC_AVATAR_JPEG_QUALITY:-68}"
SEMANTIC_AVATAR_MAX_INPUT_QUEUE_FRAMES="${SEMANTIC_AVATAR_MAX_INPUT_QUEUE_FRAMES:-16}"
SEMANTIC_AVATAR_IMAGE_FORMAT="${SEMANTIC_AVATAR_IMAGE_FORMAT:-JPEG}"
DEBUG_SEMANTIC_OVERLAY="${DEBUG_SEMANTIC_OVERLAY:-0}"
DEBUG_FACE_MASK="${DEBUG_FACE_MASK:-0}"

IFS=',' read -r -a GPU_ARRAY <<< "$GPU_IDS"
LOCAL_GPU_IDS="$(seq 0 $((${#GPU_ARRAY[@]} - 1)) | paste -sd, -)"

cd "$FRONTEND_DIR"
npm install
npm run build
echo "frontend build success"

cd "$SCRIPT_DIR"
TAEHV_FLAG=""
case "$(printf '%s' "$USE_TAEHV" | tr '[:upper:]' '[:lower:]')" in
  1|true|yes|on)
    TAEHV_FLAG="--use_taehv"
    ;;
esac

TENSORRT_FLAG=""
case "$(printf '%s' "$USE_TENSORRT" | tr '[:upper:]' '[:lower:]')" in
  1|true|yes|on)
    TENSORRT_FLAG="--use_tensorrt"
    ;;
esac

FAST_FLAG=""
case "$(printf '%s' "$FAST" | tr '[:upper:]' '[:lower:]')" in
  1|true|yes|on)
    FAST_FLAG="--fast"
    ;;
esac

DEBUG_SEMANTIC_OVERLAY_FLAG=""
case "$(printf '%s' "$DEBUG_SEMANTIC_OVERLAY" | tr '[:upper:]' '[:lower:]')" in
  1|true|yes|on)
    DEBUG_SEMANTIC_OVERLAY_FLAG="--debug-semantic-overlay"
    ;;
esac

DEBUG_FACE_MASK_FLAG=""
case "$(printf '%s' "$DEBUG_FACE_MASK" | tr '[:upper:]' '[:lower:]')" in
  1|true|yes|on)
    DEBUG_FACE_MASK_FLAG="--debug-face-mask"
    ;;
esac

CUDA_VISIBLE_DEVICES="$GPU_IDS" python main.py \
  --port "$PORT" \
  --host "$HOST" \
  --num_gpus "$(printf '%s' "$GPU_IDS" | awk -F',' '{print NF}')" \
  --gpu_ids "$LOCAL_GPU_IDS" \
  --step "$STEP" \
  --model_type "$MODEL_TYPE" \
  --conditioning_source "$CONDITIONING_SOURCE" \
  --semantic-avatar-target-fps "$SEMANTIC_AVATAR_TARGET_FPS" \
  --semantic-avatar-jpeg-quality "$SEMANTIC_AVATAR_JPEG_QUALITY" \
  --semantic-avatar-max-input-queue-frames "$SEMANTIC_AVATAR_MAX_INPUT_QUEUE_FRAMES" \
  --semantic-avatar-image-format "$SEMANTIC_AVATAR_IMAGE_FORMAT" \
  $DEBUG_SEMANTIC_OVERLAY_FLAG \
  $DEBUG_FACE_MASK_FLAG \
  $TAEHV_FLAG \
  $TENSORRT_FLAG \
  $FAST_FLAG
