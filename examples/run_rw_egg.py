import os
import time
import random
import pickle
from pprint import pformat

import numpy as np
import torch

from rlx.rw_engine.parser import Parser
from rlx.rw_engine import RewriteEngine
from rlx.utils.common import get_logger

from rlx.extern.expr.expr_utils import expr_graph, get_lang, load_expr, load_all_exprs, verify_by_egraph

from rlx.extern.expr.math_def import define_rewrite_rules as math_rewrite_rules
from rlx.extern.expr.math_def import define_node_type as define_math_node_type
from rlx.extern.expr.math_def import reward_func as math_reward
from rlx.extern.expr.math_def import convert_rlxGraphs
from rlx.extern.expr.math_def import expr_cost

from rlx.extern.expr.prop_def import define_rewrite_rules as prop_rewrite_rules
from rlx.extern.expr.prop_def import define_node_type as define_prop_node_type
from rlx.extern.expr.prop_def import reward_func as prop_reward
from rlx.extern.expr.prop_def import rlxGraph2Prop

from absl import app
from absl import flags

FLAGS = flags.FLAGS
# yapf: disable
# extern
flags.DEFINE_integer("depth_lim", 2, "expression depth limit")
flags.DEFINE_string("lang", "math", "")
flags.DEFINE_string("fn", None, "file name of the pre-generated expr")
flags.DEFINE_string("dir", None, "directory pre-generated expr")
flags.DEFINE_string("default_out_path", "data", "output dir")
# flags.DEFINE_integer("load", 1, "whether to load from pre-generated expr?")
# common
flags.DEFINE_integer("t", 1, "whether to train?")
flags.DEFINE_integer("l", 1, "whether to log")
flags.DEFINE_integer("viz", 0, "whether to visualize the ast?")
flags.DEFINE_integer("seed", 3407, "")
flags.DEFINE_string("plot", None, "path to plot the initial graph")
# env
flags.DEFINE_string("env_id", "env_multi-v0", "")
flags.DEFINE_integer("a", 0, "whether to AsyncEnv?")
flags.DEFINE_integer("total_timesteps", int(1e6), "1e6 = 1 million")
flags.DEFINE_integer("num_envs", 4, "")
flags.DEFINE_integer("num_mini_batch", 8, "")
flags.DEFINE_integer("h", 256, "hard horizon")
flags.DEFINE_integer("num_steps", 512, "num of steps to roll out")
flags.DEFINE_integer("max_loc", 128, "maximum location consider")
flags.DEFINE_integer("normalize_reward", 0, "whether to normalize the reward?")
# agent
flags.DEFINE_string("agent", "multi_output_ppo", "which RL agent to train")
flags.DEFINE_string("weights_path", None, "path to pre-trained weights")
flags.DEFINE_string("agent_id", None, "agent id")
flags.DEFINE_integer("anneal_lr", 1, "")
flags.DEFINE_integer("gae", 1, "")
flags.DEFINE_integer("norm_adv", 1, "")
flags.DEFINE_integer("clip_vloss", 1, "")
flags.DEFINE_integer("update_epochs", 4, "")
flags.DEFINE_float("lr", 2.5e-4, "")
flags.DEFINE_float("gamma", 0.99, "")
flags.DEFINE_float("gae_lambda", 0.95, "")
flags.DEFINE_float("clip_coef", 0.2, "")
flags.DEFINE_float("ent_coef", 0.01, "")
flags.DEFINE_float("vf_coef", 0.5, "")
flags.DEFINE_float("max_grad_norm", 0.5, "")
flags.DEFINE_float("target_kl", None, "")
flags.DEFINE_integer("gpu", 1, "whether to use gpu")
# GNN
flags.DEFINE_integer("num_head", 8, "num of heads in GAT")
flags.DEFINE_integer("n_layers", 3, "num of GAT layers")
flags.DEFINE_integer("hidden_size", 128, "hidden_size of GNN")
flags.DEFINE_integer("use_dropout", 0, "")
flags.DEFINE_integer("vgat", 2, "version of gat")

# logger
logger = get_logger(__name__)
# yapf: enable


def main(_):
    # ===== seed =====
    random.seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)
    torch.manual_seed(FLAGS.seed)
    torch.backends.cudnn.deterministic = True

    # ===== load =====
    lang = get_lang(FLAGS.lang)()
    logger.info("=" * 40)
    if FLAGS.fn is None and FLAGS.dir is None:
        raise RuntimeError(f"either fn or dir should be set")
    elif FLAGS.fn is not None and FLAGS.dir is not None:
        raise RuntimeError(f"only set fn or dir")
    elif FLAGS.dir is not None:
        fn = f"{FLAGS.default_out_path}/rlx/inputs/"
        fn = os.path.join(fn, FLAGS.dir)
        exprs, files = load_all_exprs(lang, fn)
        logger.info(f"Loaded exprs from files: {files}")
    elif FLAGS.fn is not None:
        fn = f"{FLAGS.default_out_path}/rlx/inputs/"
        fn = os.path.join(fn, FLAGS.fn)
        expr = load_expr(lang, fn)
        exprs = [expr]
        logger.info("Loaded expr: %s", pformat(expr))
    logger.info("=" * 40)

    # ===== register domain =====
    if FLAGS.lang == "math":
        define_types = define_math_node_type
        define_rewrite_rules = math_rewrite_rules
        conversion = convert_rlxGraphs
        callback_reward_function = math_reward
    elif FLAGS.lang == "prop":
        # define_types = define_prop_node_type
        # define_rewrite_rules = prop_rewrite_rules
        # conversion = rlxGraph2Prop
        # callback_reward_function = prop_reward
        raise NotImplementedError(f"Need to redo")
    else:
        raise NotImplementedError(f"Unsupported lang: {FLAGS.lang}")

    node_types, _, _ = define_types()
    rewrite_rules = define_rewrite_rules(node_types)
    expr_graphs = [expr_graph(expr, node_types) for expr in exprs]

    # for plot initial expr
    if FLAGS.plot is not None:
        logger.warning("only plotting the first graph")
        parser = Parser(expr_graphs[0])  # ONLY plot the first graph
        parser.viz(parser.edges,
                   os.path.join(FLAGS.default_out_path, "viz", FLAGS.plot),
                   check=False)

    # ===== rewrite engine =====
    rw_eng = RewriteEngine(
        expr_graphs,
        rewrite_rules,
        callback_reward_function,
        FLAGS,
    )

    # ===== train/inference =====
    t = bool(FLAGS.t)
    if t:
        rw_eng.train()
    else:
        print("=" * 40)
        opt_time = rw_eng.run()
        print(f"opt time {opt_time:.4f}s")

        # result
        opt_exprs = conversion(lang.all_operators(), rw_eng.envs)
        old_costs = [expr_cost(expr) for expr in exprs]
        opt_costs = [expr_cost(expr) for expr in opt_exprs]

        t1 = time.perf_counter()
        oks = verify_by_egraph(lang, exprs, opt_exprs)
        t2 = time.perf_counter()
        print(f"verification: {oks} ")
        print(f"verify time: {t2-t1:.2f}s")
        print(f"all_verified?: {all(oks)}")
        print("=" * 40)

        results = {}
        if FLAGS.fn is not None:
            logger.info("opt expression: %s", pformat(opt_exprs[0]))
            print(f"expr: {FLAGS.fn}; Costs: {old_costs[0]} -> {opt_costs[0]}")
            results[FLAGS.fn] = (old_costs[0], opt_costs[0], oks[0])
            results["opt_time"] = opt_time
            fn = FLAGS.fn.split("/")[-1]
            result_path = f"results_{fn}"

        elif FLAGS.dir is not None:
            for i, (
                    name,
                    old,
                    new,
                    ok,
            ) in enumerate(zip(files, old_costs, opt_costs, oks)):
                name = name.split("/")[-1]
                print(f"expr{i}: {name}; Costs: {old} -> {new}")
                results[name] = (old, new, ok)
            results["opt_time"] = opt_time
            result_path = f"{FLAGS.dir}.pkl"

        l = bool(FLAGS.l)
        if l:
            result_dir_path = os.path.join(
                f"{FLAGS.default_out_path}/runs/",
                FLAGS.weights_path,
                "results",
            )
            if not os.path.exists(result_dir_path):
                os.makedirs(result_dir_path)
            full_path = os.path.join(
                result_dir_path,
                result_path,
            )
            with open(full_path, "wb") as f:
                pickle.dump(results, f)

        # v1 = verify(opt_expr)
        # v2 = verify(expr)
        # assert np.isclose(v1, v2), f"verify failed: {v1}, {v2}"


if __name__ == "__main__":
    app.run(main)
