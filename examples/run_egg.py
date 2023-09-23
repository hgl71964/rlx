import os
import random
import numpy as np
import pickle

from rlx.extern.expr.expr_utils import (get_lang, new_egraph, add_df_meta,
                                        load_expr, cnt_op, solve_without_step,
                                        plot_expr)

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
    if FLAGS.fn is None:
        expr = lang.gen_example_expr()
    else:
        fn = f"{FLAGS.default_out_path}/rlx/inputs/"
        fn += FLAGS.fn
        expr = load_expr(lang, fn)

    #########################
    ####### solver ##########
    #########################
    print("=" * 40)
    print("[EGG] Solving expression: ", expr)
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
    step_info, best_expr = solve_without_step(
        expr,
        lang,
        egraph,
        FLAGS.iter_lim,
        FLAGS.node_lim,
        FLAGS.time_lim,
        bool(FLAGS.backoff),
    )

    print("=" * 40)
    print(f"best cost {step_info.cost:.2f}")
    num_op = cnt_op(best_expr)
    print(f"[EGG] num of ops: {num_op}")
    # cost = cnt_op_cost(best_expr)
    # print(f"[EGG] cost: {cost}")
    print("[EGG] Opt expression: ", best_expr)
    print("=" * 40)

    # does use the same egraph to prove equivs make sense?
    # ok = egraph.equiv(expr, best_expr)
    # print(f"[EGG] verification: {ok}")

    if plot:
        plot_expr(
            best_expr,
            os.path.join(FLAGS.default_out_path, "viz", "final_" + FLAGS.fn))

    log = bool(FLAGS.l)
    if log:
        print("[LOG]:: ")
        source = f"{FLAGS.lang}_gen" if FLAGS.fn is None else FLAGS.fn.split(
            ".")[0]
        # t = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        run_name = f"{source}_{FLAGS.e}_{FLAGS.node_lim}"
        save_path = f"{FLAGS.default_out_path}/rlx/runs/egg/{run_name}"
        print("save path: ", save_path)

        if not os.path.exists(save_path):
            os.mkdir(save_path)
        # https://github.com/abseil/abseil-py/issues/57
        FLAGS.append_flags_into_file(save_path + "/hyperparams.txt")
        with open(save_path, "wb") as f:
            pickle.dump(step_info._asdict(), f)


if __name__ == "__main__":
    app.run(main)
