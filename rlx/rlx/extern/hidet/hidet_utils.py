import os
import numpy as np
from collections import namedtuple

# import ctypes

import torch

import onnx  # type: ignore

from rlx.utils.common import get_logger
from rlx.extern.hidet.hidet_def import *
from rlx.extern.hidet.hidet_conversion import *

# extern
import hidet
from hidet.utils import benchmark_func

# storage
# from hidet.graph.impl.dlpack import DLPackStorage, DLTensorDeleter
# from hidet.runtime.storage import Storage

# logger
logger = get_logger(__name__)


def reward_func(
    graph: Graph,
    init: bool,
    terminated: bool,
    stats: dict,
) -> float:
    hidet_graph = convert_to_hidet_graph(graph.get_edges())
    my_passes = [
        fold_const_pass(),
        subgraph_rewrite_pass(),
        automatic_mix_precision_pass(),
        subgraph_rewrite_pass(),
        resolve_variant_pass(),
        fuse_operator_pass(),
        eliminate_barrier_pass(),
    ]
    with hidet.graph.PassContext() as ctx:
        for p in my_passes:
            hidet_graph = p(hidet_graph)

    latency = bench_hidet_graph(hidet_graph, False)
    if init:
        assert (latency != 0), f"initial reward cannot be zero {latency}"
        stats["init_latency"] = latency
        stats["last_latency"] = latency
        return latency

    reward = (stats["last_latency"] - latency) / stats["init_latency"]
    stats["last_latency"] = latency
    return reward


def verify_graph(g1, g2):
    assert isinstance(g1, hidet.FlowGraph), f"g1 is {type(g1)}"
    assert isinstance(g2, hidet.FlowGraph), f"g2 is {type(g2)}"

    data = g1.dummy_inputs()
    cuda_graph = g1.cuda_graph()
    (out1, ) = cuda_graph.run(data)
    del cuda_graph

    cuda_graph = g2.cuda_graph()
    (out2, ) = cuda_graph.run(data)
    del cuda_graph

    np.testing.assert_allclose(
        actual=out1.cpu().numpy(),
        desired=out2.cpu().numpy(),
        rtol=1e-5,
        atol=1e-5,
    )
    logger.info(f"verify graphs OK!!")


def bench_hidet_graph(graph: hidet.FlowGraph, v=True) -> float:
    data = graph.dummy_inputs()
    cuda_graph = graph.cuda_graph()
    (output, ) = cuda_graph.run(data)
    latency = benchmark_func(lambda: cuda_graph.run())
    if v:
        logger.info('Hidet: {:.3f} ms'.format(latency))
    return latency


def get_hidet_model(model: str):
    if model[:4] == "bert" or model == "gpt2":
        return _hidet_model_from_pytorch(model)

    return _hidet_model_from_onnx_path(model)


def _hidet_model_from_onnx_path(model: str):
    # load from onnx
    if model[-1] == "/":
        model = model[:-1]
    onnx_path = "data/models/"
    full_path = os.path.join(onnx_path, model, model + ".onnx")
    logger.info(f"load from {full_path}")
    logger.info('{}: {:.1f} MiB'.format(full_path,
                                        os.path.getsize(full_path) / (2**20)))

    hidet_onnx_module = hidet.graph.frontend.from_onnx(full_path)

    # get input shape
    print("====================")
    print("INPUTS")
    model = onnx.load(full_path)
    inputs = {}
    inputs_shape = []
    for inp in model.graph.input:
        shape = str(inp.type.tensor_type.shape.dim)
        shape = [int(s) for s in shape.split() if s.isdigit()]
        inputs[inp.name] = shape
        inputs_shape.append(shape)
    logger.info(inputs)

    print("====================")
    print("TRACE")
    assert len(inputs) == 1, f"len(inputs) == {len(inputs)}"

    # XXX: a hack to convert [3, 244, 244] -> [1, 3, 244, 244]
    if len(inputs_shape[0]) == 3:
        inputs_shape[0] = [1] + inputs_shape[0]
    # data = hidet.randn([1, 3, 224, 224], device='cuda')
    data = hidet.randn(inputs_shape[0], device='cuda')

    symbol_data = hidet.symbol_like(data)
    symbol_output = hidet_onnx_module(symbol_data)
    graph = hidet.trace_from(symbol_output)
    return graph


def _hidet_model_from_pytorch(model: str, batch_size=1, seq_length=1):
    """
    the following export to onnx and back have problems
        "bert-base-uncased",
        "bert-large-uncased",
        "bert-base-cased",
        "bert-large-cased",
        "gpt2",
    """
    from transformers.utils.fx import symbolic_trace  # type: ignore
    repo_name = 'huggingface/pytorch-transformers'
    torch_model = torch.hub.load(repo_name, 'model', model)
    torch_model = torch_model.cuda().eval()

    input_ids = torch.zeros((batch_size, seq_length),
                            dtype=torch.long,
                            device='cuda')
    args = {
        'input_ids': input_ids,
        "inputs_embeds": None,
    }

    # fx from transfomers
    # traced_model = symbolic_trace(model, input_names=["input_ids", "attention_mask", "token_type_ids"])

    # NOTE: hidet has not supported Ops
    traced_model = symbolic_trace(torch_model, input_names=["input_ids"])
    hidet_module = hidet.graph.frontend.from_torch(traced_model, None)

    data = hidet.randn([batch_size, seq_length], device='cuda')
    symbol_data = hidet.symbol_like(data)
    symbol_output = hidet_module(symbol_data)
    graph = hidet.trace_from(symbol_output)
    return graph
