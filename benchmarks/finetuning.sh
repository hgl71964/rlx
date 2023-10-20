#! /bin/bash

l=1
lang=math
max_loc=50
h=30
num_steps=50
hidden_size=512
n_layers=5
num_head=8
num_mini_batch=16
total_timesteps=50000
annotation=fine_tuning

agent=multi_output_graph_global_ppo
env_id=env_multi
weights_path=rlx_env_multi__multi_output_graph_global_ppo__math-5-full-ops__20231018-224203
# agent=ppo
# env_id=env_single
# weights_path=rlx_env_single__ppo__math-5-full-ops__TinyASTSize__20231017-140341
dir=math-5-100_150
fn=math-5-128.pkl


set -e # stop on any error from now on

echo
echo "fine tuning: fn=$dir/$fn; agent=$agent;"
echo

python examples/run_rw_egg.py \
-l $l \
-t 1 \
-h $h \
--lang $lang \
--max_loc $max_loc \
--num_steps $num_steps \
--hidden_size $hidden_size \
--n_layers $n_layers \
--num_head $num_head \
--agent $agent \
--num_mini_batch $num_mini_batch \
--total_timesteps $total_timesteps \
--env_id $env_id \
--weights_path $weights_path  \
--gpu 1 \
--annotation $annotation \
--fn $dir/$fn
