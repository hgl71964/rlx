#! /bin/bash

l=1
lang=math
max_loc=50
num_steps=50
# agent=multi_output_ppo
agent=ppo
# env_id=env_multi
env_id=env_single
# weights_path=rlx_env_multi-v0__multi_output_ppo__None__20230922-022008
weights_path=rlx_env_single-v0__ppo__None__20230925-044705
dir=math-5-full-ops
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
    --lang $lang \
    --max_loc $max_loc \
    --num_steps $num_steps \
    --agent $agent \
    --env_id $env_id \
    --weights_path $weights_path  \
    --gpu $gpu \
    --fn $dir/$file_name
done