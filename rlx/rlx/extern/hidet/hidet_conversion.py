from rlx.utils.common import get_logger
from rlx.frontend.registry import get_types
from rlx.frontend import Edge, Node
from rlx.extern.hidet.hidet_def import *

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
