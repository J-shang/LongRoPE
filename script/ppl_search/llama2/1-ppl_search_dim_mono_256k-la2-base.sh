#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

path_dir=/your/path/to/store/model/or/dataset

source ./path_teamdrive.sh
path_dir=$path_team
model="${path_dir}/Llama-2-7b-hf/"

data_tokenized="${path_dir}/pg19_valid_mapped"
cache_dir="../cache_dir"

save_memory="\
--aggressive_mem_causal_lm \
--aggressive_mem_decoder \
--aggressive_mem_attn"


max_tokens=262144
serach_method="dim_mono"
scale=
python evolution/ppl_search_evolution.py \
    --model $model \
    --samples 5 \
    --s_pi_method $serach_method \
    --s_pi_init_para "./evolution/${serach_method}/init_alpha/${serach_method}_yarn_${scale}x.csv" \
    --factor $((max_tokens / 4096)) \
    --max_tokens $max_tokens \
    --min_tokens $max_tokens _
    --tokens_step 4000 \
    --tokenized $data_tokenized \
    --original_max_position_embeddings 4096 \
    --dataset_min_tokens $max_tokens \
    --sliding_window 65536 \
    --flash_attn \
    ${save_memory} \
    --cache_dir $cache_dir \
    --truncate 
