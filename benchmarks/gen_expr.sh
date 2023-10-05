#! /bin/bash

for i in {1..100}; do 
    python examples/gen_expr.py -l 1 --save_len 10 --lang math --depth_lim 5 --seed $i; 
done