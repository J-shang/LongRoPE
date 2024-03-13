#!/bin/bash
export CUDA_VISIBLE_DEVICES=2

source ./path_teamdrive.sh
path_dir=$path_team

# model="${path_team}/Llama-2-7b-hf/"

model_128=${path_dir}/ft_out_model/cube_256k_from_128k/ck-600/
model_256="${path_dir}/ft_out_model/cube-128k-dim-piece-mono-500-#m0/ck-400/"

declare -A setting

# longrope 128k
setting["longrope_128k"]="-m ${model_128} --method longrope --finetuned --factor 32.0"

# longrope 256k
setting["longrope_256k"]="-m ${model_256} --method longrope --finetuned --factor 64.0"


mkdir -p evaluation/needle/logs evaluation/needle/img evaluation/needle/results

(
python -u evaluation/needle/needle_in_haystack.py \
    --s_len 0 --e_len 128000 \
    --context_lengths_min 1024 \
    --context_lengths_max 128000 \
    --context_lengths_num_intervals 40 \
    --model_provider LLaMA \
    --model_path ${model_128} \
    --result_path ./evaluation/needle/result/longrope_128k/ \
    ${setting["longrope_128k"]} \
    --flash_attn \
    --max_tokens 128000

) 2>&1  | tee -a evaluation/needle/logs/eval_longrope_128k.log

# python evaluation/needle/visualize.py 
