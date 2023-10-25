from rlx.utils.common import get_logger
from rlx.frontend.registry import get_types
from rlx.frontend import Edge, Node
from rlx.extern.hidet.hidet_def import *

# extern
from hidet.graph import ops  # see frontend/onnx for how to build op

# storage
# from hidet.graph.impl.dlpack import DLPackStorage, DLTensorDeleter
# from hidet.runtime.storage import Storage

logger = get_logger(__name__)

# NOTE: storage cannot be deepcopy neither its attributes
STORAGE_CACHE = {}
_storage_id = 0


def get_storage(idx):
    assert idx in STORAGE_CACHE, f"{idx} not in storage cache"
    return STORAGE_CACHE[idx]


def add_storage(storage):
    global _storage_id
    local = _storage_id
    STORAGE_CACHE[local] = storage
    _storage_id += 1
    return local


#########################
##### look up table #####
#########################
def hidet_lookup(hidet_op: hidet.Operator, types):
    op_class = type(hidet_op)
    if op_class not in OP_MAP:
        raise RuntimeError(f"Unsupport hidet op type {hidet_op} | {op_class}")
    return types[OP_MAP[op_class]]


def hidet_reverse_loopup(dfg_op: DFG_Op, inputs: list[hidet.Tensor]):
    assert isinstance(dfg_op, DFG_Op), f"expected DFG_Op, got {type(dfg_op)}"
    for inp in inputs:
        assert isinstance(
            inp, hidet.Tensor), f"expected hidet.Tensor, got {type(inp)}"
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
        return ops.multiply(inputs[0], dfg_op.attr.attrs["scalar"])
    elif node_type == "add_scalar":
        assert len(inputs) == 1, f"len(inputs) == {len(inputs)}"
        return ops.add(inputs[0], dfg_op.attr.attrs["scalar"])
    elif node_type == "divide_scalar":
        assert len(inputs) == 1, f"len(inputs) == {len(inputs)}"
        return ops.divide(inputs[0], dfg_op.attr.attrs["scalar"])
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
        return ops.reshape(inputs[0], dfg_op.attr.attrs["shape"])
    elif node_type == "rearrange":
        assert len(inputs) == 1, f"len(inputs) == {len(inputs)}"
        return ops.rearrange(inputs[0], dfg_op.attr.attrs["plan"])
    elif node_type == "squeeze":
        assert len(inputs) == 1, f"len(inputs) == {len(inputs)}"
        return ops.squeeze(inputs[0], dfg_op.attr.attrs["dims"])
    elif node_type == "unsqueeze":
        assert len(inputs) == 1, f"len(inputs) == {len(inputs)}"
        return ops.unsqueeze(inputs[0], dfg_op.attr.attrs["dims"])
    elif node_type == "flatten":
        assert len(inputs) == 1, f"len(inputs) == {len(inputs)}"
        return ops.flatten(inputs[0], dfg_op.attr.attrs["start_dim"],
                           dfg_op.attr.attrs["end_dim"])
    elif node_type == "permute_dims":
        assert len(inputs) == 1, f"len(inputs) == {len(inputs)}"
        return ops.permute_dims(inputs[0], dfg_op.attr.attrs["axes"])
    elif node_type == "cast":
        assert len(inputs) == 1, f"len(inputs) == {len(inputs)}"
        return ops.cast(inputs[0], dfg_op.attr.attrs["dtype"])
    elif node_type == "concat":
        return ops.concat(inputs, dfg_op.attr.attrs["axis"])
    elif node_type == "take":
        assert len(inputs) == 2, f"len(inputs) == {len(inputs)}"
        return ops.take(inputs[0], inputs[1], dfg_op.attr.attrs["axis"])
    elif node_type == "gather":
        assert len(inputs) == 2, f"len(inputs) == {len(inputs)}"
        return ops.gather(inputs[0], inputs[1], dfg_op.attr.attrs["axis"])
    elif node_type == "strided_slice":
        assert len(inputs) == 1, f"len(inputs) == {len(inputs)}"
        # TODO one output?
        return ops.strided_slice(
            inputs[0],
            dfg_op.attr.attrs["starts"],
            dfg_op.attr.attrs["ends"],
            dfg_op.attr.attrs["axes"],
            dfg_op.attr.attrs["strides"],
        )
    elif node_type == "broadcast":
        assert len(inputs) == 1, f"len(inputs) == {len(inputs)}"
        return ops.broadcast(inputs[0], dfg_op.attr.attrs["shape"])
    elif node_type == "pad":
        assert len(inputs) == 1, f"len(inputs) == {len(inputs)}"
        return ops.pad(
            inputs[0],
            dfg_op.attr.attrs["pads"],
            dfg_op.attr.attrs["mode"],
            dfg_op.attr.attrs["value"],
        )

    # activation
    elif node_type == "relu":
        assert len(inputs) == 1, f"len(inputs) == {len(inputs)}"
        return ops.relu(inputs[0])
    elif node_type == "leaky_relu":
        assert len(inputs) == 1, f"len(inputs) == {len(inputs)}"
        return ops.leaky_relu(inputs[0], dfg_op.attr.attrs["alpha"])
    elif node_type == "sigmoid":
        assert len(inputs) == 1, f"len(inputs) == {len(inputs)}"
        return ops.sigmoid(inputs[0])
    elif node_type == "clip":
        assert len(inputs) == 1, f"len(inputs) == {len(inputs)}"
        return ops.clip(inputs[0], dfg_op.attr.attrs["min_val"],
                        dfg_op.attr.attrs["max_val"])
    elif node_type == "gelu":
        assert len(inputs) == 1, f"len(inputs) == {len(inputs)}"
        return ops.gelu(inputs[0])
    elif node_type == "softmax":
        assert len(inputs) == 1, f"len(inputs) == {len(inputs)}"
        return ops.softmax(inputs[0], dfg_op.attr.attrs["axis"])

    # pool
    elif node_type == "max_pool2d":
        assert len(inputs) == 1, f"len(inputs) == {len(inputs)}"
        return ops.max_pool2d(
            inputs[0],
            dfg_op.attr.attrs["kernel"],
            dfg_op.attr.attrs["stride"],
            dfg_op.attr.attrs["padding"],
        )
    elif node_type == "avg_pool2d":
        assert len(inputs) == 1, f"len(inputs) == {len(inputs)}"
        return ops.avg_pool2d(
            inputs[0],
            dfg_op.attr.attrs["kernel"],
            dfg_op.attr.attrs["stride"],
            dfg_op.attr.attrs["padding"],
        )

    # tensor compute
    elif node_type == "conv2d":
        assert len(inputs) == 2, f"len(inputs) == {len(inputs)}"
        return ops.conv2d(
            inputs[0],
            inputs[1],
            dfg_op.attr.attrs["stride"],
            dfg_op.attr.attrs["dilations"],
            dfg_op.attr.attrs["groups"],
        )
    elif node_type == "matmul":
        assert len(inputs) == 2, f"len(inputs) == {len(inputs)}"
        return ops.matmul(inputs[0], inputs[1])
    elif node_type == "batch_matmul":
        assert len(inputs) == 2, f"len(inputs) == {len(inputs)}"
        return ops.batch_matmul(inputs[0], inputs[1], dfg_op.attr.attrs["mma"])
    elif node_type == "attention":
        mask = None if len(inputs) == 3 else inputs[3]
        return ops.attention(inputs[0], inputs[1], inputs[2], mask)

    # reduce
    elif node_type == "reduce_sum":
        assert len(inputs) == 1, f"len(inputs) == {len(inputs)}"
        return ops.sum(inputs[0],
                       dims=dfg_op.attr.attrs["dims"],
                       keep_dim=dfg_op.attr.attrs["keepdims"])
    elif node_type == "reduce_mean":
        assert len(inputs) == 1, f"len(inputs) == {len(inputs)}"
        return ops.mean(inputs[0],
                        dims=dfg_op.attr.attrs["dims"],
                        keep_dim=dfg_op.attr.attrs["keepdims"])
    else:
        raise RuntimeError(f"Unsupported node type: {node_type}")


#########################
####### Converter #######
#########################
def convert_to_dataflow_graph(graph: hidet.FlowGraph):
    cnt = 0
    built = {}  # 1-to-1 mapping
    types, const_type, var_type = get_types()

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

            node_type = hidet_lookup(obj, types)
            node = DFG_Op(
                cnt,
                attr=NodeAttribute(
                    name=obj.name,
                    attrs=obj.attrs,
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

            edge_type = var_type if obj.storage is None else const_type

            if obj.storage is None:
                storage_id = None
            else:
                storage_id = add_storage(obj.storage)
            tensor = DFG_Edge(cnt,
                              attr=EdgeAttribute(
                                  shape=obj.shape,
                                  dtype=obj.dtype,
                                  device=obj.device,
                                  layout=obj.layout,
                                  storage_id=storage_id,
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

            out = hidet_reverse_loopup(obj, inputs)
            if isinstance(out, list):
                # FIXME could be multiple outputs
                raise RuntimeError(
                    f"doesnt support multiple outputs for now {out}")
            built[obj] = out
            return out

        if isinstance(obj, Edge):
            if obj.trace is None:
                tensor = hidet.Tensor(
                    shape=obj.attr.shape,
                    dtype=obj.attr.dtype,
                    device=obj.attr.device,
                    layout=obj.attr.layout,
                    trace=None,
                    # if obj.attr[4] is None else STORAGE_CACHE[obj.attr[4]],
                    storage=None if obj.attr.storage_id is None else
                    get_storage(obj.attr.storage_id),
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


def edge2tensor(edge: Edge) -> hidet.Tensor:
    assert isinstance(edge, DFG_Edge), f"expect DFG_Edge, got {type(edge)}"
    return hidet.Tensor(
        shape=edge.attr.shape,
        dtype=edge.attr.dtype,
        device=edge.attr.device,
        layout=edge.attr.layout,
        trace=None,
        # if edge.attr[4] is None else STORAGE_CACHE[edge.attr[4]],
        storage=None
        if edge.attr.storage_id is None else get_storage(edge.attr.storage_id),
    )
