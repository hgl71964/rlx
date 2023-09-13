import os
import random
import datetime
import numpy as np
import pandas as pd

from rlx.extern.expr.expr_utils import get_lang, new_egraph, add_df_meta, step, load_expr, cnt_op

from rust_lib import print_id

from absl import app
from absl import flags

FLAGS = flags.FLAGS
flags.DEFINE_integer("node_lim", 10000, "enode limit")
flags.DEFINE_integer("seed", 0, "")
flags.DEFINE_integer("l", 0, "whether to log")

flags.DEFINE_string("lang", "MATH", "")
flags.DEFINE_string("e", "greedy", "extractor; greedy or ilp")
flags.DEFINE_string("fn", None, "file name of the pre-generated expr")
flags.DEFINE_string("default_out_path", "data", "output dir")
flags.DEFINE_integer("ver", 0, "verbose")


def solve_expr_egg(lang, expr, node_lim):
    """
    Emulate egg's solver but WITHOUT an iteration limit.
    This will keep running until saturation,
        until a node limit, or time limit is reached.
    """
    egraph = new_egraph(expr)
    base_cost, _ = egraph.extract(expr)
    best_expr = expr
    print("[EGG] base cost:", base_cost)

    steps = []

    i = 0
    sat_counter = 0
    verbose = bool(FLAGS.ver)

    while True:
        action_to_apply = i % lang.num_rules
        if action_to_apply == 0:
            sat_counter = 0

        result, best_expr = step(action_to_apply, expr, lang, egraph, node_lim)
        steps.append(result)

        if verbose:
            print("=" * 40)
            print(result.stop_reason, result.num_applications,
                  result.num_enodes, result.num_eclasses)

        # normally it hits iter-limit and stop, thus apply rule one-step
        if result.stop_reason == 'NODE_LIMIT':
            print("***NODE limit***")
            break
        elif result.stop_reason == 'TIME_LIMIT':
            print("***TIME limit***")
            break  # egg stops optimizing
        elif result.stop_reason == 'SATURATED':
            sat_counter += 1

        if sat_counter == lang.num_rules:
            break  # egg has achieved saturation

        i += 1

    steps_df = pd.DataFrame(steps)
    steps_df = add_df_meta(steps_df, lang.name, "egg", base_cost, FLAGS.seed,
                           FLAGS.node_lim)

    print("=" * 40)
    print(f"[EGG] iter: {i}")
    # TODO add ILP extraction?
    print("greedy cost:", steps_df["cost"].iloc[-1])
    print("=" * 40)
    return steps_df, best_expr


def test_expr(lang, expr):
    for op in lang.all_operators():
        print(op)
        print_id(op)

    print("expr: ", type(expr))

    egraph = new_egraph(expr)
    base_cost, _ = egraph.extract(expr)


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

    # test_expr(lang, expr)
    # raise

    # egg solver
    print("=" * 40)
    print("[EGG] Solving expression", expr)
    print("=" * 40)
    egg_df, best_expr = solve_expr_egg(lang, expr, FLAGS.node_lim)

    print("=" * 40)
    num_op = cnt_op(best_expr)
    print(f"[EGG] num of ops: {num_op}")
    # cost = cnt_op_cost(best_expr)
    # print(f"[EGG] cost: {cost}")
    print("=" * 40)

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
        egg_df.to_csv(f"{save_path}/egg.csv")


if __name__ == "__main__":
    app.run(main)
