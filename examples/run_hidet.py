import random

import numpy as np
import torch

from rlx.utils.common import get_logger

from rlx.extern.hidet.hidet_utils import *
from rlx.extern.hidet.hidet_def import *

import hidet
from hidet.graph.transforms import (
    conv_channel_last_pass,
    subgraph_rewrite_pass,
    automatic_mix_precision_pass,
    selective_quantize_pass,
    resolve_variant_pass,
    fuse_operator_pass,
    eliminate_barrier_pass,
)

from absl import app
from absl import flags

FLAGS = flags.FLAGS
'''
this not only benchmarks hidet, but also builds operator cache
'''
# yapf: disable
# extern
flags.DEFINE_string("fn", None, "name of the model; e.g. resnet50")
flags.DEFINE_string("default_out_path", "data", "output dir")
flags.DEFINE_integer("c", 1, "whether to cache operators")
flags.DEFINE_integer("s", 1, "search level: [0, 1, 2]")
flags.DEFINE_integer("verify", 1, "whether to verify")
flags.DEFINE_integer("l", 1, "no use")
flags.DEFINE_string("verbose", None, "no use")
# common
flags.DEFINE_integer("viz", 0, "whether to visualize the ast?")
flags.DEFINE_integer("seed", 3407, "")

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

    # NOTE: cause error if False, cannot rm (tmp?) file, device not empty
    hidet.option.debug_cache_tuning(True)

    # ===== load =====
    # hidet_graph = hidet_model_from_onnx_path(FLAGS.fn)
    hidet_graph = get_hidet_model(FLAGS.fn)
    # print(hidet_graph)

    # benchmark before run
    print("----- Init Benchmark -----")
    bench_hidet_graph(hidet_graph)
    latency = hidet_graph.latency()
    print(f"graph latency: {latency:.2f}ms")
    print("----- Init Benchmark -----")
    print()

    # round trip
    v = bool(FLAGS.verify)
    if v:
        dfg = convert_to_dataflow_graph(hidet_graph)
        converted_hidet_graph = convert_to_hidet_graph(dfg.get_edges())
        verify_graph(hidet_graph, converted_hidet_graph)
    # raise

    # optimization
    my_passes = [
        # fold_const_pass(),
        conv_channel_last_pass(),
        # subgraph_rewrite_pass(),
        automatic_mix_precision_pass(),
        selective_quantize_pass(),
        resolve_variant_pass(),
        fuse_operator_pass(),
        eliminate_barrier_pass(),
    ]
    with hidet.graph.PassContext() as ctx:
        # INSTRUMENTS
        # save the computation graph level ir
        # ctx.save_graph_instrument(out_dir='./outs/graphs')
        # if use_fp16:
        #     ctx.set_precision('float16')
        #     ctx.set_mma('mma')
        # run customized passes; and not running instruments
        opt_graph = hidet_graph
        for opt_pass in my_passes:
            opt_graph = opt_pass(opt_graph)

        # could use a graph instruments
        # print("----")
        # print(graph)
        # print("----")

    if v:
        dfg = convert_to_dataflow_graph(opt_graph)
        converted_hidet_graph = convert_to_hidet_graph(dfg.get_edges())
        verify_graph(opt_graph, converted_hidet_graph)

    print()
    print("----- Opt graph runtime -----")
    bench_hidet_graph(opt_graph)
    latency = opt_graph.latency()
    print(f"opt graph latency: {latency:.2f}ms")
    print()

    with hidet.graph.PassContext() as ctx:
        # run the default passes
        opt_graph = hidet.graph.optimize(hidet_graph)
    if v:
        dfg = convert_to_dataflow_graph(opt_graph)
        converted_hidet_graph = convert_to_hidet_graph(dfg.get_edges())
        verify_graph(opt_graph, converted_hidet_graph)

    print()
    print("----- Full Opt graph runtime -----")
    bench_hidet_graph(opt_graph)
    latency = opt_graph.latency()
    print(f"Full Opt graph graph latency: {latency:.2f}ms")
    print()


if __name__ == "__main__":
    app.run(main)
