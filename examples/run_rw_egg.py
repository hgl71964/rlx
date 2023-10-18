import os
import time
import random
import pickle
from pprint import pformat
from copy import deepcopy

import numpy as np
import torch

from rlx.rw_engine.parser import Parser
from rlx.rw_engine import RewriteEngine
from rlx.utils.common import get_logger

from rlx.extern.expr.expr_utils import expr_graph, get_lang, load_expr, load_all_exprs, verify_by_egraph, plot_expr_graph
from rlx.extern.expr.math_def import define_rewrite_rules as math_rewrite_rules
from rlx.extern.expr.math_def import define_types as define_math_node_type
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
# common
flags.DEFINE_integer("t", 1, "whether to train?")
flags.DEFINE_integer("l", 1, "whether to log")
flags.DEFINE_integer("viz", 0, "whether to visualize the ast?")
flags.DEFINE_integer("seed", 3407, "")
flags.DEFINE_integer("plot", 0, "whether to plot")
flags.DEFINE_string("annotation", None, "append to save name")
# env
flags.DEFINE_string("env_id", "env_multi-v0", "")
flags.DEFINE_integer("a", 0, "whether to AsyncEnv; not support for now")
flags.DEFINE_integer("total_timesteps", int(1e6), "1e6 = 1 million")
flags.DEFINE_integer("num_envs", 4, "")
flags.DEFINE_integer("num_mini_batch", 8, "")
flags.DEFINE_integer("h", 256, "hard horizon")
flags.DEFINE_integer("num_steps", 512, "num of steps to roll out")
flags.DEFINE_integer("max_loc", 50, "maximum location consider")
flags.DEFINE_integer("normalize_reward", 0, "whether to normalize the reward?")
# agent
flags.DEFINE_string("agent", "multi_output_ppo", "which RL agent to train")
flags.DEFINE_string("weights_path", None, "path to pre-trained weights")
flags.DEFINE_string("agent_id", None, "agent ckpt id for inference; None -> latest")
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
        raise NotImplementedError(f"Unsupported lang: {FLAGS.lang}")
    else:
        raise NotImplementedError(f"Unsupported lang: {FLAGS.lang}")

    types, _, _ = define_types()
    rewrite_rules = define_rewrite_rules(types)
    expr_graphs = [expr_graph(expr, types) for expr in exprs]

    # if plot to debug
    plot = bool(FLAGS.plot)
    if plot:
        file_name = FLAGS.fn.split(
            "/")[-1] if FLAGS.fn is not None else FLAGS.dir
        logger.warning("only plotting the first graph")
        parser = Parser([deepcopy(expr_graphs[0])])
        parser.viz(parser.all_edges[0],
                   os.path.join(
                       FLAGS.default_out_path,
                       "viz",
                       "rlx_initial_" + file_name,
                   ),
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
        info = rw_eng.run()
        opt_time = info["opt_time"]
        inf_time = info["inf_time"]
        cnt = info["iter"]
        print(f"opt time {opt_time:.4f}s")

        # result
        opt_exprs, opt_costs = conversion(lang.all_operators(), rw_eng.envs)
        old_costs = [expr_cost(expr) for expr in exprs]

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
            results["inf_time"] = inf_time
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
            results["inf_time"] = inf_time
            device = "gpu" if torch.cuda.is_available() and bool(
                FLAGS.gpu) else "cpu"
            result_path = f"{device}_{FLAGS.dir}_dir.pkl"

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

        if plot:
            file_name = FLAGS.fn.split(
                "/")[-1] if FLAGS.fn is not None else FLAGS.dir
            plot_expr_graph(
                parser, rw_eng.envs,
                os.path.join(
                    FLAGS.default_out_path,
                    "viz",
                    "rlx_final_" + file_name,
                ))

        # v1 = verify(opt_expr)
        # v2 = verify(expr)
        # assert np.isclose(v1, v2), f"verify failed: {v1}, {v2}"


if __name__ == "__main__":
    app.run(main)
