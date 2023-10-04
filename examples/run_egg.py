import os
import time
import random
import pickle
import numpy as np
from tqdm import tqdm
from pprint import pformat

from rlx.utils.common import get_logger
from rlx.extern.expr.expr_utils import (get_lang, new_egraph, load_expr,
                                        solve_without_step, plot_expr,
                                        load_all_exprs)

from rlx.extern.expr.math_def import expr_cost

from absl import app
from absl import flags

FLAGS = flags.FLAGS
flags.DEFINE_integer("node_lim", 10000, "enode limit")
flags.DEFINE_integer("iter_lim", 100, "")
flags.DEFINE_integer("time_lim", 100, "")
flags.DEFINE_integer("backoff", 1, "whether to use backoff scheduler")
flags.DEFINE_integer("seed", 0, "")

flags.DEFINE_integer("plot", 0, "whether to plot")

flags.DEFINE_string("lang", None, "")
flags.DEFINE_string("e", "greedy", "extractor; greedy or ilp")
flags.DEFINE_string("default_out_path", "data", "output dir")

flags.DEFINE_string("fn", None, "file name of the pre-generated expr")
flags.DEFINE_string("dir", None, "directory pre-generated expr")

logger = get_logger(__name__)


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
    if FLAGS.fn is not None:
        print("=" * 40)
        print("[EGG] Solving expression: ", expr)
        print("=" * 40)
    # egg_df, best_expr = solve_expr_egg(lang, expr, FLAGS.node_lim)

    plot = bool(FLAGS.plot)
    if plot:
        print(f"[WARNING] only plotting the first expr")
        plot_expr(
            exprs[0],
            os.path.join(FLAGS.default_out_path, "viz", "initial_" + FLAGS.fn))

    # solve without step
    old_costs = []
    opt_exprs = []
    step_infos = []
    t1 = time.perf_counter()
    for expr in tqdm(exprs):
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
            FLAGS.e,
        )
        old_costs.append(base_cost)
        opt_exprs.append(best_expr)
        step_infos.append(step_info)
    t2 = time.perf_counter()
    print(f"opt time {t2-t1:.2f}s")

    # result
    opt_costs = [expr_cost(opt_expr) for opt_expr in opt_exprs]
    if FLAGS.fn is not None:
        logger.info("opt expression: %s", pformat(opt_exprs[0]))
        print(f"expr: {FLAGS.fn}; Costs: {old_costs[0]} -> {opt_costs[0]}")

    elif FLAGS.dir is not None:
        results = {}
        for i, (
                name,
                info,
                old,
                new,
        ) in enumerate(zip(files, step_infos, old_costs, opt_costs)):
            name = name.split("/")[-1]
            print(f"expr{i}: {name}; Costs: {old} -> {new}")
            results[name] = (
                old,
                new,
                True,
                info.build_time,
                info.extract_time,
            )

        results["opt_time"] = t2 - t1
        run_name = f"{FLAGS.node_lim}_{FLAGS.e}_{FLAGS.dir}.pkl"
        result_path = f"{FLAGS.default_out_path}/runs/egg"

        if not os.path.exists(result_path):
            os.mkdir(result_path)
        result_path = os.path.join(result_path, run_name)
        with open(result_path, "wb") as f:
            pickle.dump(results, f)

    # TODO does use the same egraph to prove equivs make sense?
    # ok = egraph.equiv(expr, best_expr)
    # print(f"[EGG] verification: {ok}")

    if plot:
        print(f"[WARNING] only plotting the first opt expr")
        plot_expr(
            opt_exprs[0],
            os.path.join(FLAGS.default_out_path, "viz", "final_" + FLAGS.fn))


if __name__ == "__main__":
    app.run(main)
