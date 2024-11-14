MODEL_PATH=/scratch/nishang/models/pgqa-tp1pp1-3000b-gbs8388608-mbs2-lr5e-4-HF
DATASET_PATH=/scratch/nishang/datasets/pg19-train-128k-search-phi4-omni-tokenized-hf

TARGET_LENGTH=131072
LENGTH_SCALE=64
CRITICAL_DIM=30

RESULT_PATH=/scratch/nishang/longrope-search/phi4-omni-mscale/critical_dim_$CRITICAL_DIM/

mkdir -p $RESULT_PATH

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
    --init-factors /scratch/nishang/LongRoPE/llama3-ntk-cd-init-128k.csv \
    --length-scale $LENGTH_SCALE \
    --num-proc 16 \
    --save-memory \
    --critical-dim $CRITICAL_DIM > log_phi4_omni_cd_$CRITICAL_DIM.txt 2>&1
