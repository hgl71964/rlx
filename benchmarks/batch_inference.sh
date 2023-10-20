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
weights_path=rlx_env_multi__multi_output_graph_global_ppo__math-5-full-ops__20231018-224203
# agent=ppo
# env_id=env_single
# weights_path=rlx_env_single__ppo__math-5-full-ops__TinyASTSize__20231017-140341
dir=math-5-full-ops
#dir=math-5-100_150


set -e # stop on any error from now on

echo
echo "batch inference: dir=$dir; agent=$agent;"
echo

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
--gpu 0 \
--dir $dir

sleep 5

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
--gpu 1 \
--dir $dir
