#! /bin/bash

l=1
lang=math
dir=math-5-full-ops

set -e

# fetch all exprs from the given dir,
files=(data/rlx/inputs/${dir}/*)

# for fn in ${files[@]}; do
#     str_arr=(${fn//// })
#     file_name=${str_arr[-1]}
#
#     echo "egg: ${file_name}"
#     python examples/run_egg.py \
#     -l $l \
#     --lang $lang \
#     --annotation parallel_ast_size  \
#     --fn $dir/$file_name &
# done
#
# wait

parallel "(
	echo 'full name {};; file $dir/{/}'
	     python examples/run_egg.py \
	     -l $l \
	     --lang $lang \
	     --annotation parallel_ast_size  \
	     --fn $dir/{/}
	echo 'Done Building {/}')" ::: ${files[@]}
