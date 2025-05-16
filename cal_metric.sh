# Check if the correct number of arguments is provided
if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <output_dir> <gt_dir> <gpu_id>"
    exit 1
fi

export CUDA_VISIBLE_DEVICES="$3"

cd pipeline


OUTPUT_DIR="$1"
GT_DIR="$2"


GT_JSON_MATCH_DIR="$GT_DIR/json_match"
GT_JSON_DIR="$GT_DIR/json"
GT_IMAGE_DIR="$GT_DIR/image"

PRED_JSON_DIR="$OUTPUT_DIR/json"
MAP_DIR="$OUTPUT_DIR/map"
RENDER_DIR="$OUTPUT_DIR/render"
SCORE_OUTPUT_DIR="$OUTPUT_DIR/score"

python main_match_fast.py \
    --map_dir $MAP_DIR \
    --pred_dir $PRED_JSON_DIR \
    --gt_dir $GT_JSON_DIR


blender --background \
    --python main_recon.py -- $PRED_JSON_DIR $RENDER_DIR


python main_metrics_with_gpteval.py \
    --map_dir $MAP_DIR \
    --render_dir $RENDER_DIR \
    --gt_json_dir $GT_JSON_DIR \
    --pred_json_dir $PRED_JSON_DIR \
    --gt_image_dir $GT_IMAGE_DIR \
    --output_dir $SCORE_OUTPUT_DIR \
    --llm_model_name "gpt-4o" \
    --openai_api_key "Your-openai-key-here"