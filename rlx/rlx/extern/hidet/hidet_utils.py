import os
import numpy as np
from collections import namedtuple

# import ctypes
import torch
import onnx

from rlx.utils.common import get_logger
from rlx.extern.hidet.hidet_def import *
from rlx.extern.hidet.hidet_conversion import *

# utils
from hidet.utils import benchmark_func

logger = get_logger(__name__)


def reward_func(
    verbose: bool,
    graph: Graph,
    init: bool,
    terminated: bool,
    stats: dict,
) -> float:
    hidet_graph = convert_to_hidet_graph(graph.get_edges())
    my_passes = [
        conv_channel_last_pass(),
        # subgraph_rewrite_pass(),
        automatic_mix_precision_pass(),
        selective_quantize_pass(),
        resolve_variant_pass(),
        fuse_operator_pass(),
        eliminate_barrier_pass(),
    ]
    with hidet.graph.PassContext() as ctx:
        for p in my_passes:
            hidet_graph = p(hidet_graph)

    # TODO search with cudaGraph will OOM; how to free?
    # latency = bench_hidet_graph(hidet_graph, verbose)
    latency = hidet_graph.latency()
    if init:
        assert (latency != 0), f"initial reward cannot be zero {latency}"
        stats["init_latency"] = latency
        stats["last_latency"] = latency
        return latency

    if verbose:
        print(
            f"init latency: {stats['init_latency']:.2f}; last latency: {stats['last_latency']:.2f}; latency: {latency:.2f}"
        )
    reward = (stats["last_latency"] - latency) / stats["init_latency"]
    stats["last_latency"] = latency
    return reward


def verify_graph(g1, g2):
    assert isinstance(g1, hidet.FlowGraph), f"g1 is {type(g1)}"
    assert isinstance(g2, hidet.FlowGraph), f"g2 is {type(g2)}"

    data = g1.dummy_inputs()
    cuda_graph = g1.cuda_graph()
    out1 = cuda_graph.run(data)
    del cuda_graph

    cuda_graph = g2.cuda_graph()
    out2 = cuda_graph.run(data)
    del cuda_graph

    if not isinstance(out1, list):
        out1 = [out1]
    if not isinstance(out2, list):
        out2 = [out2]

    for o1, o2 in zip(out1, out2):
        np.testing.assert_allclose(
            actual=o1.cpu().numpy(),
            desired=o2.cpu().numpy(),
            rtol=1e-5,
            atol=1e-5,
        )
    logger.info(f"verify graphs OK!!")


def bench_hidet_graph(graph: hidet.FlowGraph, v=True) -> float:
    data = graph.dummy_inputs()
    cuda_graph = graph.cuda_graph()
    _ = cuda_graph.run(data)
    latency = benchmark_func(lambda: cuda_graph.run())
    if v:
        logger.info('Hidet: {:.3f} ms'.format(latency))
    return latency


def get_hidet_model(model: str):
    # if model[:4] == "bert" or model == "gpt2":
    #     return _hidet_model_from_pytorch(model)

    if model == "gpt-2":
        return _hidet_model_from_hidet(model)

    return _hidet_model_from_onnx_path(model)


def _hidet_model_from_onnx_path(model: str):
    # load from onnx
    if model[-1] == "/":
        model = model[:-1]
    model_name = model
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
    # assert len(inputs) == 1, f"len(inputs) == {len(inputs)}"

    # XXX: a hack to convert [3, 244, 244] -> [1, 3, 244, 244]
    if model_name == "resnet50" and len(inputs_shape[0]) == 3:
        print('ad')
        inputs_shape[0] = [1] + inputs_shape[0]
    # data = hidet.randn([1, 3, 224, 224], device='cuda')
    data = hidet.randn(inputs_shape[0], device='cuda')

    symbol_data = hidet.symbol_like(data)
    symbol_output = hidet_onnx_module(symbol_data)
    graph = hidet.trace_from(symbol_output)
    return graph


def _hidet_model_from_hidet(model):
    onnx_path = "data/models/"
    model_name = "model_124M_seq40_fp32.hf"
    hf_path = os.path.join(onnx_path, model, model_name)
    return hidet.load_graph(hf_path)


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

    # see: transformer.bert.modelling_bert.BertModel.forward (to satisfy all args)
    input_ids = torch.zeros((batch_size, seq_length),
                            dtype=torch.long,
                            device='cuda')
    input_embeds = torch.randn((batch_size, seq_length, 768),
                               dtype=torch.float,
                               device='cuda')
    attention_mask = torch.ones((batch_size, seq_length, 768),
                                dtype=torch.long,
                                device='cuda')
    head_mask = torch.ones((seq_length, 768), dtype=torch.long, device='cuda')
    use_cache = False
    output_hidden_states = False
    output_attentions = False

    args = {
        # 'input_ids': input_ids,
        'input_ids': None,
        "inputs_embeds": input_embeds,
        'attention_mask': attention_mask,
        'head_mask': head_mask,
        'use_cache': use_cache,
        'output_hidden_states': output_hidden_states,
        'output_attentions': output_attentions,
    }

    # NOTE: hidet has not supported Ops; fx from transfomers
    traced_model = symbolic_trace(torch_model, input_names=["input_ids"])
    hidet_module = hidet.graph.frontend.from_torch(traced_model, None)
    # hidet_module = hidet.graph.frontend.from_torch(torch_model, args)

    data = hidet.randn([batch_size, seq_length], device='cuda')
    symbol_data = hidet.symbol_like(data)
    symbol_output = hidet_module(symbol_data)
    graph = hidet.trace_from(symbol_output)
    return graph
