#! /bin/bash

l=1
lang=math
max_loc=50
h=30
num_steps=50
hidden_size=512
n_layers=5
num_head=8

agent=multi_output_graph_global_ppo
env_id=env_multi
weights_path=rlx_env_multi__multi_output_graph_global_ppo__math-5-full-ops__numpy_cost__20231021-140234
# agent=ppo
# env_id=env_single
# weights_path=rlx_env_single__ppo__math-5-full-ops__TinyASTSize__20231017-140341
dir=math-5-full-ops
#dir=math-5-100_150
gpu=0

set -e # stop on any error from now on

# fetch all exprs from the given dir,
# then run inference sequentially
files=(data/rlx/inputs/${dir}/*)
for fn in ${files[@]}; do
    # https://stackoverflow.com/questions/918886/how-do-i-split-a-string-on-a-delimiter-in-bash
    str_arr=(${fn//// })
    file_name=${str_arr[-1]}

    echo "inference: ${file_name}"
    python examples/run_rw_egg.py \
    -l $l \
    -t 0 \
    -h $h \
    --lang $lang \
    --max_loc $max_loc \
    --num_steps $num_steps \
    --hidden_size $hidden_size \
    --n_layers $n_layers \
    --num_head $num_head \
    --agent $agent \
    --env_id $env_id \
    --weights_path $weights_path  \
    --gpu $gpu \
    --fn $dir/$file_name
done
