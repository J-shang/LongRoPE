# MODEL_PATH=/mnt/nishang/models/mistral_version/
# DATASET_PATH=/mnt/nishang/datasets/pg19-train-128k-search-phi31-tokenized-hf/
# RESULT_PATH=/mnt/nishang/longrope1-search/phi31-mscale/longrope-scale-72-init-1.19-init
mkdir -p $RESULT_PATH
# TARGET_LENGTH=131072

python evolution/search.py \
    --model $MODEL_PATH \
    --tokenized $DATASET_PATH \
    --algorithm dim_mono \
    --output-dir $RESULT_PATH \
    --target-length $TARGET_LENGTH \
    --dataset-min-tokens $TARGET_LENGTH \
    --samples 10 \
    --truncate \
    --attn-implementation flash_attention_2 \
    --attn-sliding-window $TARGET_LENGTH \
    --model-size-gb 10 \
    --init-factors /scratch/amlt_code/LongRoPE/longrope_multi_init.csv \
    --length-scale $LENGTH_SCALE \
    --num-proc 16 \
    --critical-dim $CRITICAL_DIM
