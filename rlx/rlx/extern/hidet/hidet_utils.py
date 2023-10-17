import os
import numpy as np
from collections import namedtuple

# import ctypes

import torch
from transformers.utils.fx import symbolic_trace

import onnx  # type: ignore

from rlx.utils.common import get_logger
from rlx.frontend.registry import get_node_type
from rlx.frontend import Graph, Edge, Node
from rlx.extern.hidet.hidet_def import *

# extern
import hidet
from hidet.utils import benchmark_func
from hidet.graph import ops  # see frontend/onnx for how to build op

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
    logger.info(f"verify graphs")


def hidet_lookup(hidet_op: hidet.Operator):
    all_node_type, _, _ = get_node_type()
    op_class = type(hidet_op)
    if op_class not in OP_MAP:
        raise RuntimeError(f"Unsupport hidet op type {hidet_op} | {op_class}")
    return all_node_type[OP_MAP[op_class]]


def hidet_reverse_loopup(dfg_op, inputs):
    node_type = dfg_op.get_type().name
    # arithmeticOp
    if node_type == "add":
        assert len(inputs) == 2, f"len(inputs) == {len(inputs)}"
        # this returns a Tensor
        return ops.add(inputs[0], inputs[1])
    elif node_type == "sub":
        assert len(inputs) == 2, f"len(inputs) == {len(inputs)}"
        return ops.subtract(inputs[0], inputs[1])
    elif node_type == "mul":
        assert len(inputs) == 2, f"len(inputs) == {len(inputs)}"
        return ops.multiply(inputs[0], inputs[1])
    elif node_type == "div":
        assert len(inputs) == 2, f"len(inputs) == {len(inputs)}"
        return ops.divide(inputs[0], inputs[1])
    elif node_type == "sin":
        assert len(inputs) == 1, f"len(inputs) == {len(inputs)}"
        return ops.sin(inputs[0])
    elif node_type == "cos":
        assert len(inputs) == 1, f"len(inputs) == {len(inputs)}"
        return ops.cos(inputs[0])
    elif node_type == "tan":
        assert len(inputs) == 1, f"len(inputs) == {len(inputs)}"
        return ops.tan(inputs[0])
    elif node_type == "tanh":
        assert len(inputs) == 1, f"len(inputs) == {len(inputs)}"
        return ops.tanh(inputs[0])
    elif node_type == "multiply_scalar":
        assert len(inputs) == 1, f"len(inputs) == {len(inputs)}"
        return ops.multiply(inputs[0], dfg_op.attr[1]["scalar"])
    elif node_type == "add_scalar":
        assert len(inputs) == 1, f"len(inputs) == {len(inputs)}"
        return ops.add(inputs[0], dfg_op.attr[1]["scalar"])
    elif node_type == "divide_scalar":
        assert len(inputs) == 1, f"len(inputs) == {len(inputs)}"
        return ops.divide(inputs[0], dfg_op.attr[1]["scalar"])
    elif node_type == "pow":
        assert len(inputs) == 2, f"len(inputs) == {len(inputs)}"
        return ops.pow(inputs[0], inputs[1])
    elif node_type == "sqrt":
        assert len(inputs) == 1, f"len(inputs) == {len(inputs)}"
        return ops.sqrt(inputs[0])
    elif node_type == "erf":
        assert len(inputs) == 1, f"len(inputs) == {len(inputs)}"
        return ops.erf(inputs[0])

    # transformOp
    elif node_type == "reshape":
        assert len(inputs) == 1, f"len(inputs) == {len(inputs)}"
        return ops.reshape(inputs[0], dfg_op.attr[1]["shape"])
    elif node_type == "rearrange":
        assert len(inputs) == 1, f"len(inputs) == {len(inputs)}"
        return ops.rearrange(inputs[0], dfg_op.attr[1]["plan"])
    elif node_type == "squeeze":
        assert len(inputs) == 1, f"len(inputs) == {len(inputs)}"
        return ops.squeeze(inputs[0], dfg_op.attr[1]["dims"])
    elif node_type == "unsqueeze":
        assert len(inputs) == 1, f"len(inputs) == {len(inputs)}"
        return ops.unsqueeze(inputs[0], dfg_op.attr[1]["dims"])
    elif node_type == "flatten":
        assert len(inputs) == 1, f"len(inputs) == {len(inputs)}"
        return ops.flatten(inputs[0], dfg_op.attr[1]["start_dim"],
                           dfg_op.attr[1]["end_dim"])
    elif node_type == "permute_dims":
        assert len(inputs) == 1, f"len(inputs) == {len(inputs)}"
        return ops.permute_dims(inputs[0], dfg_op.attr[1]["axes"])
    elif node_type == "cast":
        assert len(inputs) == 1, f"len(inputs) == {len(inputs)}"
        return ops.cast(inputs[0], dfg_op.attr[1]["dtype"])
    elif node_type == "concat":
        return ops.concat(inputs, dfg_op.attr[1]["axis"])
    elif node_type == "take":
        assert len(inputs) == 2, f"len(inputs) == {len(inputs)}"
        return ops.take(inputs[0], inputs[1], dfg_op.attr[1]["axis"])
    elif node_type == "gather":
        assert len(inputs) == 2, f"len(inputs) == {len(inputs)}"
        return ops.gather(inputs[0], inputs[1], dfg_op.attr[1]["axis"])
    elif node_type == "strided_slice":
        # assert len(inputs) == 2, f"len(inputs) == {len(inputs)}"
        # return ops.strided_slice(inputs[0], inputs[1])
        raise RuntimeError(f"Unsupported node type: {node_type}")
    elif node_type == "broadcast":
        assert len(inputs) == 1, f"len(inputs) == {len(inputs)}"
        return ops.broadcast(inputs[0], dfg_op.attr[1]["shape"])
    elif node_type == "pad":
        assert len(inputs) == 1, f"len(inputs) == {len(inputs)}"
        return ops.pad(inputs[0], dfg_op.attr[1]["pads"],
                       dfg_op.attr[1]["mode"], dfg_op.attr[1]["value"])

    # activation
    elif node_type == "relu":
        assert len(inputs) == 1, f"len(inputs) == {len(inputs)}"
        return ops.relu(inputs[0])
    elif node_type == "leaky_relu":
        assert len(inputs) == 1, f"len(inputs) == {len(inputs)}"
        return ops.leaky_relu(inputs[0], dfg_op.attr[1]["alpha"])
    elif node_type == "sigmoid":
        assert len(inputs) == 1, f"len(inputs) == {len(inputs)}"
        return ops.sigmoid(inputs[0])
    elif node_type == "clip":
        assert len(inputs) == 1, f"len(inputs) == {len(inputs)}"
        return ops.clip(inputs[0], dfg_op.attr[1]["min_val"],
                        dfg_op.attr[1]["max_val"])
    elif node_type == "gelu":
        assert len(inputs) == 1, f"len(inputs) == {len(inputs)}"
        return ops.gelu(inputs[0])
    elif node_type == "softmax":
        assert len(inputs) == 1, f"len(inputs) == {len(inputs)}"
        return ops.softmax(inputs[0], dfg_op.attr[1]["axis"])

    # pool
    elif node_type == "max_pool2d":
        assert len(inputs) == 1, f"len(inputs) == {len(inputs)}"
        return ops.max_pool2d(inputs[0], dfg_op.attr[1]["kernel"],
                              dfg_op.attr[1]["stride"],
                              dfg_op.attr[1]["padding"])
    elif node_type == "avg_pool2d":
        assert len(inputs) == 1, f"len(inputs) == {len(inputs)}"
        return ops.avg_pool2d(inputs[0], dfg_op.attr[1]["kernel"],
                              dfg_op.attr[1]["stride"],
                              dfg_op.attr[1]["padding"])

    # tensor compute
    elif node_type == "conv2d":
        assert len(inputs) == 2, f"len(inputs) == {len(inputs)}"
        return ops.conv2d(inputs[0], inputs[1], dfg_op.attr[1]["stride"],
                          dfg_op.attr[1]["dilations"],
                          dfg_op.attr[1]["groups"])
    elif node_type == "matmul":
        assert len(inputs) == 2, f"len(inputs) == {len(inputs)}"
        return ops.matmul(inputs[0], inputs[1])
    elif node_type == "batch_matmul":
        assert len(inputs) == 2, f"len(inputs) == {len(inputs)}"
        return ops.batch_matmul(inputs[0], inputs[1], dfg_op.attr[1]["mma"])
    elif node_type == "attention":
        mask = None if len(inputs) == 3 else inputs[3]
        return ops.attention(inputs[0], inputs[1], inputs[2], mask)

    # reduce
    elif node_type == "reduce_sum":
        assert len(inputs) == 1, f"len(inputs) == {len(inputs)}"
        return ops.sum(inputs[0],
                       dims=dfg_op.attr[1]["dims"],
                       keep_dim=dfg_op.attr[1]["keepdims"])
    elif node_type == "reduce_mean":
        assert len(inputs) == 1, f"len(inputs) == {len(inputs)}"
        return ops.mean(inputs[0],
                        dims=dfg_op.attr[1]["dims"],
                        keep_dim=dfg_op.attr[1]["keepdims"])
    else:
        raise RuntimeError(f"Unsupported node type: {node_type}")


def convert_to_dataflow_graph(graph: hidet.FlowGraph):
    cnt = 0
    built = {}  # 1-to-1 mapping
    _, const_type, var_type = get_node_type()

    def dfs(obj):
        nonlocal cnt
        if obj in built:
            return built[obj]

        if isinstance(obj, hidet.Operator):
            inputs = []
            for inp in obj.inputs:
                out = dfs(inp)
                inputs.append(out)
                assert isinstance(out,
                                  DFG_Edge), f"expect Tensor, got {type(out)}"

            node_type = hidet_lookup(obj)
            node = DFG_Op(
                cnt,
                attr=(
                    obj.name,
                    obj.attrs,
                    # task is not necessary, build func from ops.* will build task automatically
                    # obj.task,
                ),
                node_type=node_type,
                inputs=inputs)
            cnt += 1
            built[obj] = node
            return node

        if isinstance(obj, hidet.Tensor):
            trace = None
            if obj.trace is not None:
                trace = dfs(obj.trace[0])
                assert isinstance(trace,
                                  DFG_Op), f"expect DFG_Op, got {type(trace)}"

            # NOTE: doesn't work for deepcopy
            # tensor = DFG_Edge(cnt,
            #                   attr=(obj.shape,
            #                         obj.dtype,
            #                         obj.device,
            #                         obj.layout,

            #                         # NOTE: storage cannot deepcopy
            #                         obj.storage.device if obj.storage is not None else None,
            #                         obj.storage.addr if obj.storage is not None else None,
            #                         obj.storage.num_bytes if obj.storage is not None else None,
            #                         obj.storage.free_handler if obj.storage is not None else None,
            #                         ),
            #                   edge_type=None,
            #                   trace=trace)

            edge_type = var_type if obj.storage is None else const_type
            if obj.storage is None:
                tensor = DFG_Edge(cnt,
                                  attr=(
                                      obj.shape,
                                      obj.dtype,
                                      obj.device,
                                      obj.layout,
                                      None,
                                  ),
                                  edge_type=edge_type,
                                  trace=trace)
            else:
                storage_id = add_storage(obj.storage)
                tensor = DFG_Edge(cnt,
                                  attr=(
                                      obj.shape,
                                      obj.dtype,
                                      obj.device,
                                      obj.layout,
                                      storage_id,
                                  ),
                                  edge_type=edge_type,
                                  trace=trace)

            if trace is not None:
                trace.outputs.append(tensor)

            cnt += 1
            built[obj] = tensor
            return tensor

        raise RuntimeError(f"Unsupport type {type(obj)}")

    for out in graph.outputs:
        dfs(out)

    nodes = [v for _, v in built.items() if isinstance(v, DFG_Op)]
    edges = [v for _, v in built.items() if isinstance(v, DFG_Edge)]
    dfg = DataflowGraph(nodes, edges)
    return dfg


def convert_to_hidet_graph(edges: list[Edge]):
    # find outputs
    outputs = []
    for e in edges:
        if len(e.uses) == 0:
            outputs.append(e)

    built = {}

    def dfs(obj):
        if obj in built:
            return built[obj]

        if isinstance(obj, Node):
            inputs = []
            for inp in obj.inputs:
                out = dfs(inp)
                inputs.append(out)
                assert isinstance(
                    out, hidet.Tensor), f"expect Tensor, got {type(out)}"

            # FIXME could be multiple outputs
            out = hidet_reverse_loopup(obj, inputs)
            built[obj] = out
            return out

        if isinstance(obj, Edge):
            if obj.trace is None:
                tensor = hidet.Tensor(
                    shape=obj.attr[0],
                    dtype=obj.attr[1],
                    device=obj.attr[2],
                    layout=obj.attr[3],
                    trace=None,
                    storage=None
                    # if obj.attr[4] is None else STORAGE_CACHE[obj.attr[4]],
                    if obj.attr[4] is None else get_storage(obj.attr[4]),
                )
                built[obj] = tensor
                return tensor
            else:
                out = dfs(obj.trace)
                built[obj] = out
                return out

        raise RuntimeError(f"Unsupport type {type(obj)}")

    outs = []
    for out in outputs:
        o = dfs(out)
        outs.append(o)

    hidet_graph = hidet.FlowGraph(outs)
    return hidet_graph


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
