#! /bin/bash

l=1
lang=math
max_loc=50
num_steps=50
hidden_size=128
n_layers=5
num_head=8

agent=multi_output_ppo
env_id=env_multi
weights_path=rlx_env_multi-v0__multi_output_ppo__math-5-0_100__TinyASTSize__20231016-195903
# agent=ppo
# env_id=env_single
# weights_path=rlx_env_single-v0__ppo__None__20230925-044705
#dir=math-5-full-ops
dir=math-5-100_150
gpu=1


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
--hidden_size $hidden_size \
--n_layers $n_layers \
--num_head $num_head \
--agent $agent \
--env_id $env_id \
--weights_path $weights_path  \
--gpu $gpu \
--dir $dir
