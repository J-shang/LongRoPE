#!/bin/bash
export CUDA_VISIBLE_DEVICES=7
# export HF_DATASETS_CACHE="/path/to/store/model"
export HF_DATASETS_CACHE="../cache"
# path_dir=/your/path/to/store/model/or/dataset
# your path to model

source ./path_teamdrive.sh
path_dir=$path_team


# model setting
declare -A setting

# base Mistral-7b
setting["base_mistral"]="--model ${path_dir}/Mistral-7B-v0.1/ --method pi --factor 1.0 --original-max-position-embeddings 4096"

# special: yarn Mistral -> origin postion 8k
# yarn Mistral 64k
setting["yarn_64k_mistral"]="--model ${path_team}/Yarn-Mistral-7b-64k --method yarn --finetuned --factor 16.0 --original-max-position-embeddings 8192"

# yarn Mistral 128k
setting["yarn_128k_mistral"]="--model ${path_team}/Yarn-Mistral-7b-128k --method yarn --finetuned --factor 16.0 --original-max-position-embeddings 8192"

# longrope Mistral 128k
setting["longrope_128k_mistral"]="--model ${path_dir}/ft_out_model/cube-128k-dim-piece-mono-500-#m0/ck-400 --method s_pi --s_pi_para ./evolution/test/result_alpha/ft_s_pi_131072_result.csv --finetuned --factor 32.0 --original-max-position-embeddings 4096"

# longrope Mistral 256k
setting["longrope_256k_mistral"]="--model ${path_dir}/ft_out_model/cube_256k_from_128k/ck-600 --method s_pi --s_pi_para ./evolution/test/result_alpha/ft_s_pi_262144_result_118.csv --finetuned --factor 64.0 --original-max-position-embeddings 4096"



# dataset setting


cache_dir="../cache_dir"
output_dir=./evaluation/result

save_memory="\
--aggressive-mem-causal_lm \
--aggressive-mem-decoder"
# save_memory="" # check

config_list=("base_mistral" "yarn_64k_mistral" "yarn_128k_mistral" "longrope_128k_mistral" "longrope_256k_mistral")
config_list=("yarn_64k_mistral" "yarn_128k_mistral" "longrope_128k_mistral" "longrope_256k_mistral") # check

echo "dataset PROOFPILE 10sample"
# max_tokens_list=(4096 8192 32768 65536 98304 131072)
max_tokens_list=(4096)

for config in "${config_list[@]}"; do
    for max_tokens in "${max_tokens_list[@]}"; do
        echo "####### $config, max-tokens=$max_tokens #############"
        python evaluation/perplexity.py \
            ${PROOFPILE_128k}\
            ${setting[$config]} \
            --max-tokens $max_tokens \
            --min-tokens $max_tokens \
            --tokens-step 2048 \
            --output-file "${output_dir}/t5_proofpile_${config}_${max_tokens}.csv" \
            --sliding_window_attention $max_tokens \
            --flash_attn \
            ${save_memory} \
            --cache_dir $cache_dir
    done
done

max_tokens_list=(262144)
for config in "${config_list[@]}"; do
    for max_tokens in "${max_tokens_list[@]}"; do
        echo "####### $config, max-tokens=$max_tokens #############"
        python evaluation/perplexity.py \
            ${PROOFPILE_128k}\
            ${setting[$config]} \
            --max-tokens $max_tokens \
            --min-tokens $max_tokens \
            --tokens-step 2048 \
            --output-file "${output_dir}/t5_proofpile_${config}_${max_tokens}.csv" \
            --sliding_window_attention $max_tokens \
            --flash_attn \
            ${save_memory} \
            --cache_dir $cache_dir
    done
done
