import os
import random
import datetime
import numpy as np
import pandas as pd

from rlx.extern.expr.expr_utils import (get_lang, new_egraph, add_df_meta,
                                        step, load_expr, cnt_op,
                                        solve_without_step, plot_expr)

from rust_lib import print_id

from absl import app
from absl import flags

FLAGS = flags.FLAGS
flags.DEFINE_integer("node_lim", 10000, "enode limit")
flags.DEFINE_integer("iter_lim", 100, "")
flags.DEFINE_integer("time_lim", 100, "")
flags.DEFINE_integer("backoff", 1, "")
flags.DEFINE_integer("seed", 0, "")
flags.DEFINE_integer("l", 0, "whether to log")

flags.DEFINE_integer("plot", 0, "whether to plot")

flags.DEFINE_string("lang", None, "")
flags.DEFINE_string("e", "greedy", "extractor; greedy or ilp")
flags.DEFINE_string("fn", None, "file name of the pre-generated expr")
flags.DEFINE_string("default_out_path", "data", "output dir")


def main(_):
    random.seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)

    # load
    lang = get_lang(FLAGS.lang)()
    print("=" * 40)
    if FLAGS.fn is None:
        expr = lang.gen_example_expr()
        print("Generated expression: ", expr)
    else:
        fn = f"{FLAGS.default_out_path}/rlx/inputs/"
        fn += FLAGS.fn
        expr = load_expr(lang, fn)
        print("Loaded expression: ", expr)
    print("=" * 40)

    #########################
    ####### solver ##########
    #########################
    print("=" * 40)
    print("[EGG] Solving expression", expr)
    print("=" * 40)
    # egg_df, best_expr = solve_expr_egg(lang, expr, FLAGS.node_lim)

    plot = bool(FLAGS.plot)
    if plot:
        plot_expr(
            expr,
            os.path.join(FLAGS.default_out_path, "viz", "initial_" + FLAGS.fn))

    # solve without step
    egraph = new_egraph(expr)
    base_cost, _ = egraph.extract(expr)
    print("[EGG] base cost:", base_cost)
    num_op = cnt_op(expr)
    print(f"[EGG] num of base ops: {num_op}")
    step_info, best_expr = solve_without_step(expr, lang, egraph, FLAGS)

    print("=" * 40)
    print(f"best cost {step_info.cost:.2f}")
    num_op = cnt_op(best_expr)
    print(f"[EGG] num of ops: {num_op}")
    # cost = cnt_op_cost(best_expr)
    # print(f"[EGG] cost: {cost}")
    print("=" * 40)

    if plot:
        plot_expr(
            best_expr,
            os.path.join(FLAGS.default_out_path, "viz", "final_" + FLAGS.fn))

    # save
    log = bool(FLAGS.l)
    if log:
        print("[LOG]:: ")
        source = f"{FLAGS.lang}_gen" if FLAGS.fn is None else FLAGS.fn.split(
            ".")[0]
        # t = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        run_name = f"egg_{FLAGS.e}_{FLAGS.node_lim}_{source}"
        save_path = f"{FLAGS.default_out_path}/rlx/runs/{run_name}"
        print("save path: ", save_path)

        if not os.path.exists(save_path):
            os.mkdir(save_path)
        # https://github.com/abseil/abseil-py/issues/57
        FLAGS.append_flags_into_file(save_path + "/hyperparams.txt")
        # egg_df.to_csv(f"{save_path}/egg.csv")


if __name__ == "__main__":
    app.run(main)
