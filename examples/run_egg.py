import os
import time
import random
import numpy as np
import pickle

from rlx.extern.expr.expr_utils import (get_lang, new_egraph, load_expr,
                                        solve_without_step, plot_expr,
                                        load_all_exprs)

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
flags.DEFINE_string("default_out_path", "data", "output dir")

flags.DEFINE_string("fn", None, "file name of the pre-generated expr")
flags.DEFINE_string("dir", None, "directory pre-generated expr")


def main(_):
    random.seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)

    # load
    lang = get_lang(FLAGS.lang)()
    if FLAGS.fn is None and FLAGS.dir is None:
        raise RuntimeError(f"either fn or dir should be set")
    elif FLAGS.fn is not None and FLAGS.dir is not None:
        raise RuntimeError(f"only set fn or dir")
    elif FLAGS.dir is not None:
        fn = f"{FLAGS.default_out_path}/rlx/inputs/"
        fn = os.path.join(fn, FLAGS.dir)
        exprs, files = load_all_exprs(lang, fn)
    elif FLAGS.fn is not None:
        fn = f"{FLAGS.default_out_path}/rlx/inputs/"
        fn = os.path.join(fn, FLAGS.fn)
        expr = load_expr(lang, fn)
        exprs = [expr]

    #########################
    ####### solver ##########
    #########################
    if len(exprs) == 1:
        print("=" * 40)
        print("[EGG] Solving expression: ", exprs[0])
        print("=" * 40)
    # egg_df, best_expr = solve_expr_egg(lang, expr, FLAGS.node_lim)

    plot = bool(FLAGS.plot)
    if plot:
        print(f"[WARNING] only plotting the first expr")
        plot_expr(
            expr[0],
            os.path.join(FLAGS.default_out_path, "viz", "initial_" + FLAGS.fn))

    # solve without step
    old_costs = []
    opt_exprs = []
    opt_costs = []
    t1 = time.perf_counter()
    for expr in exprs:
        egraph = new_egraph(expr)
        base_cost, _ = egraph.extract(expr)
        step_info, best_expr = solve_without_step(
            expr,
            lang,
            egraph,
            FLAGS.iter_lim,
            FLAGS.node_lim,
            FLAGS.time_lim,
            bool(FLAGS.backoff),
        )
        old_costs.append(base_cost)
        opt_exprs.append(best_expr)
        opt_costs.append(step_info.cost)
        print(base_cost, step_info.cost)

    t2 = time.perf_counter()

    # result
    if len(opt_costs) == 1:
        logger.info("opt expression: %s", pformat(opt_exprs[0]))
        print(f"expr: {FLAGS.fn}; Costs: {old_costs[0]} -> {opt_costs[0]}")
    else:
        for i, (name, old, new) in enumerate(zip(files, old_costs, opt_costs)):
            name = name.split("/")[-1]
            print(f"expr{i}: {name}; Costs: {old} -> {new}")

    print(f"opt time {t2-t1:.4f}s")

    # TODO does use the same egraph to prove equivs make sense?
    # ok = egraph.equiv(expr, best_expr)
    # print(f"[EGG] verification: {ok}")

    if plot:
        print(f"[WARNING] only plotting the first opt expr")
        plot_expr(
            best_exprs[0],
            os.path.join(FLAGS.default_out_path, "viz", "final_" + FLAGS.fn))

    # log = bool(FLAGS.l)
    # if log:
    #     print("[LOG]:: ")
    #     source = f"{FLAGS.lang}_gen" if FLAGS.fn is None else FLAGS.fn.split(
    #         ".")[0]
    #     # t = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    #     run_name = f"{source}_{FLAGS.e}_{FLAGS.node_lim}"
    #     save_path = f"{FLAGS.default_out_path}/rlx/runs/egg/{run_name}"
    #     print("save path: ", save_path)

    #     if not os.path.exists(save_path):
    #         os.mkdir(save_path)
    #     # https://github.com/abseil/abseil-py/issues/57
    #     FLAGS.append_flags_into_file(save_path + "/hyperparams.txt")
    #     with open(save_path, "wb") as f:
    #         pickle.dump(step_info._asdict(), f)


if __name__ == "__main__":
    app.run(main)
