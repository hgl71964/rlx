#! /bin/bash

l=1
lang=math
max_loc=50
num_steps=50
agent=multi_output_ppo
env_id=env_multi
weights_path=rlx_env_multi-v0__multi_output_ppo__None__20230922-022008
dir=math-5-full-ops
gpu=0


set -e # stop on any error from now on

echo
echo "batch inference: dir=$dir; agent=$agent; gpu=$gpu"
echo

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
--dir $dir