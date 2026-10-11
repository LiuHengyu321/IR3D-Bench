#!/usr/bin/env bash
set -euo pipefail

# Check if the correct number of arguments is provided
if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <output_dir> <gt_dir> <gpu_id>"
    exit 1
fi

export CUDA_VISIBLE_DEVICES="$3"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
BLENDER_PATH="${BLENDER_PATH:-blender}"
: "${SAM_CHECKPOINT:?Set SAM_CHECKPOINT to the SAM ViT-H checkpoint path}"
: "${OPENAI_API_KEY:?Set OPENAI_API_KEY for LLM evaluation}"
if [[ ! -f "$SAM_CHECKPOINT" ]]; then
    echo "SAM_CHECKPOINT does not point to a file." >&2
    exit 1
fi


OUTPUT_DIR="$1"
GT_DIR="$2"


GT_JSON_MATCH_DIR="$GT_DIR/json_match"
GT_JSON_DIR="$GT_DIR/json"
GT_IMAGE_DIR="$GT_DIR/image"

PRED_JSON_DIR="$OUTPUT_DIR/json"
MAP_DIR="$OUTPUT_DIR/map"
RENDER_DIR="$OUTPUT_DIR/render"
SCORE_OUTPUT_DIR="$OUTPUT_DIR/score"

"$PYTHON_BIN" "$SCRIPT_DIR/pipeline/match_fast.py" \
    --map_dir "$MAP_DIR" \
    --pred_dir "$PRED_JSON_DIR" \
    --gt_dir "$GT_JSON_DIR"


"$BLENDER_PATH" --background \
    --python "$SCRIPT_DIR/pipeline/recon.py" -- "$PRED_JSON_DIR" "$RENDER_DIR"


"$PYTHON_BIN" "$SCRIPT_DIR/pipeline/metrics.py" \
    --map_dir "$MAP_DIR" \
    --render_dir "$RENDER_DIR" \
    --gt_json_dir "$GT_JSON_DIR" \
    --pred_json_dir "$PRED_JSON_DIR" \
    --gt_image_dir "$GT_IMAGE_DIR" \
    --output_dir "$SCORE_OUTPUT_DIR" \
    --llm_model_name "${LLM_MODEL_NAME:-gpt-4o}"
