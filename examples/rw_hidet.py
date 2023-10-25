import time
import random
from pprint import pformat
from copy import deepcopy

from functools import partial

import numpy as np
import torch

from rlx.frontend.registry import get_types
from rlx.rw_engine.parser import Parser
from rlx.rw_engine import RewriteEngine
from rlx.utils.common import get_logger
from rlx.extern.hidet.hidet_utils import *
from rlx.extern.hidet.hidet_rules import *

import hidet

from absl import app
from absl import flags

FLAGS = flags.FLAGS

# yapf: disable
'''
set PYTHONPATH to HIDET_HOME/python
'''
# extern
flags.DEFINE_string("fn", None, "name of the model; e.g. resnet50")
flags.DEFINE_string("default_out_path", "data", "output dir")
flags.DEFINE_integer("c", 1, "whether to cache operators")
flags.DEFINE_integer("s", 1, "search level: [0, 1, 2]")
# common
flags.DEFINE_integer("t", 1, "whether to train?")
flags.DEFINE_integer("l", 1, "whether to log")
flags.DEFINE_integer("viz", 0, "whether to visualize the ast?")
flags.DEFINE_integer("seed", 3407, "")
flags.DEFINE_integer("plot", 0, "whether to plot")
flags.DEFINE_integer("verbose", 1, "whether print in reward function")
flags.DEFINE_string("annotation", None, "append to save name")
# env
flags.DEFINE_string("env_id", "env_multi-v0", "")
flags.DEFINE_integer("a", 0, "whether to AsyncEnv?")
flags.DEFINE_integer("total_timesteps", int(1e5), "1e6 = 1 million")
flags.DEFINE_integer("num_envs", 1, "")
flags.DEFINE_integer("num_mini_batch", 2, "")
flags.DEFINE_integer("h", 30, "hard horizon")
flags.DEFINE_integer("num_steps", 128, "num of steps to roll out")
flags.DEFINE_integer("max_loc", 64, "maximum location consider")
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
flags.DEFINE_integer("n_layers", 5, "num of GAT layers")
flags.DEFINE_integer("hidden_size", 512, "hidden_size of GNN")
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

    # ===== hidet config =====
    # change the cache directory
    c = bool(FLAGS.c)
    hidet.option.cache_operator(c)
    hidet.option.cache_dir('./data/cache')
    # hidet.option.cache_operator(False)  # force not to cache, so compile from sratch

    # save the tensor program level ir in operator cache
    # hidet.option.save_lower_ir()
    hidet.option.search_space(FLAGS.s)

    # ===== load =====
    # hidet_graph = hidet_model_from_onnx_path(FLAGS.fn)
    hidet_graph = get_hidet_model(FLAGS.fn)
    # for i, node in enumerate(hidet_graph.nodes):
    #     for inp in node.inputs:
    #         if inp.storage is not None:
    #             print(i)
    # print(hidet_graph)

    ###### RUN a round trip #####
    # dfg = convert_to_dataflow_graph(hidet_graph)
    # converted_hidet_graph = convert_to_hidet_graph(dfg.get_edges())
    # verify_graph(hidet_graph, converted_hidet_graph)
    ###### RUN a round trip #####

    ###### start to rw ######
    dfg = convert_to_dataflow_graph(hidet_graph)

    # for plot
    plot = bool(FLAGS.plot)
    if plot:
        logger.warning("only plotting the first graph")
        parser = Parser([deepcopy(dfg)])
        parser.viz(parser.all_edges[0],
                   os.path.join(
                       FLAGS.default_out_path,
                       "viz",
                       "rlx_initial_" + FLAGS.fn,
                   ),
                   check=False)

    node_types, _, _ = get_types()
    rewrite_rules = define_rewrite_rules(node_types)
    rw_eng = RewriteEngine(
        [dfg],
        rewrite_rules,
        partial(reward_func, bool(FLAGS.verbose)),
        FLAGS,
    )

    t = bool(FLAGS.t)
    if t:
        rw_eng.train()
    else:
        opt_graph = rw_eng.run()
        opt_hidet_graph = convert_to_hidet_graph(opt_graph.get_edges())
        verify_graph(hidet_graph, opt_hidet_graph)
        print("----- Benchmark -----")
        bench_hidet_graph(opt_hidet_graph)
        print("----- Benchmark -----")
        print()

        if plot:
            logger.warning("only plotting the first graph")
            parser = Parser([deepcopy(dfg)])
            parser.viz(parser.all_edges[0],
                       os.path.join(
                           FLAGS.default_out_path,
                           "viz",
                           "rlx_final_" + FLAGS.fn,
                       ),
                       check=False)


if __name__ == "__main__":
    app.run(main)
