from collections import namedtuple

from rlx.utils.common import get_logger
from rlx.frontend.registry import register_types
from rlx.frontend import Graph, Node, Edge

# extern
import hidet
from hidet.graph import ops  # see frontend/onnx for how to build op

from hidet.graph.ops.arithmetic import AddScalarOp, SubScalarOp, RSubScalarOp, MultiplyScalarOp, DivideScalarOp, RDivideScalarOp, SqrtOp, ErfOp, ExpOp, Expm1Op, LogOp, Log2Op, Log10Op, Log1pOp, RsqrtOp, PowOp, NegativeOp, ReciprocalOp, AddOp, SubtractOp, MultiplyOp, DivideOp, SinOp, CosOp, TanOp, SinhOp, CoshOp, TanhOp, AcosOp, AsinOp, AtanOp, Atan2Op, AcoshOp, AsinhOp, AtanhOp, SquareOp, CubeOp, AbsOp, FloorOp, RoundOp, TruncOp, CeilOp, IsFiniteOp, IsInfOp, IsNanOp, SignOp, RightShiftOp, LeftShiftOp, BitwiseAndOp, BitwiseNotOp, BitwiseOrOp, BitwiseXorOp, ModOp, WhereOp, MaxOp, MinOp

from hidet.graph.ops.compare import EqualOp, NotEqualOp, LessOp, GreaterOp, LessEqualOp, GreaterEqualOp, LogicalNotOp, LogicalAndOp, LogicalOrOp, LogicalXorOp

from hidet.graph.ops.transform import ReshapeOp, RearrangeOp, SqueezeOp, UnsqueezeOp, FlattenOp, PermuteDimsOp, CastOp, ConcatOp, TakeOp, GatherOp, StridedSliceOp, BroadcastOp, PadOp, TileOp

from hidet.graph.ops.pool import MaxPool2dOp, MaxPool3dOp, AvgPool2dOp, AvgPool3dOp, AdaptivePoolOp, AdaptiveAvgPool1dOp, AdaptiveAvgPool2dOp, AdaptiveAvgPool3dOp, AdaptiveMaxPool1dOp, AdaptiveMaxPool2dOp, AdaptiveMaxPool3dOp

from hidet.graph.ops.activation import ReluOp, LeakyReluOp, SigmoidOp, HardSigmoidOp, ClipOp, GeluOp, SiluOp, PReluOp, HardSwishOp, ThresholdOp, HardTanhOp, EluOp, SeluOp, CeluOp, LogSigmoidOp, HardShrinkOp, TanhShrinkOp, SoftSignOp, SoftPlusOp, SoftShrinkOp, SoftmaxOp

from hidet.graph.ops.image import Resize2dOp
from hidet.graph.ops.cumulative import CumulativeSumOp
from hidet.graph.ops.special import BarrierOp

from hidet.graph.ops.conv1d import Conv1dOp
from hidet.graph.ops.conv1d_transpose.conv1d_transpose import Conv1dTransposeOp
from hidet.graph.ops.attention.attention import AttnOp

from hidet.graph.ops.conv2d.conv2d import Conv2dOp
from hidet.graph.ops.conv3d.conv3d import Conv3dOp

from hidet.graph.ops.matmul.matmul import MatmulOp
from hidet.graph.ops.matmul.batch_matmul import BatchMatmulOp

from hidet.graph.ops.reduce.reduce import ReduceSumOp, ReduceMeanOp

# pass
# from hidet.graph.transforms.fold_const import fold_const_pass
# from hidet.graph.transforms.subgraph_rewrite import subgraph_rewrite_pass
# from hidet.graph.transforms.automatic_mix_precision import automatic_mix_precision_pass
# from hidet.graph.transforms.resolve_variant import resolve_variant_pass
# from hidet.graph.transforms.fuse_operator import fuse_operator_pass
# from hidet.graph.transforms.eliminate_barrier import eliminate_barrier_pass

from hidet.graph.transforms import (
    conv_channel_last_pass,
    subgraph_rewrite_pass,
    automatic_mix_precision_pass,
    selective_quantize_pass,
    resolve_variant_pass,
    fuse_operator_pass,
    eliminate_barrier_pass,
)

# utils
from hidet.utils import benchmark_func
from hidet.utils import same_list, initialize
from hidet.graph.ops.utils.tensor_utils import normalize_dim

logger = get_logger(__name__)

NodeAttribute = namedtuple("NodeAttribute", ["name", "attrs"])
EdgeAttribute = namedtuple("EdgeAttribute", [
    "shape",
    "dtype",
    "device",
    "layout",
    "storage_id",
])


########################################
## override user-API for hidet  ########
########################################
class DFG_Edge(Edge):

    def __init__(self, idx, attr, edge_type, trace):
        self.idx = idx
        self.attr = attr
        self.edge_type = edge_type
        self.uses = []
        self.trace = trace

    def get_idx(self):
        return self.idx

    def get_type(self):
        return self.edge_type

    def set_type(self, edge_type):
        self.edge_type = edge_type

    def get_attr(self) -> EdgeAttribute:
        return self.attr

    def set_attr(self, attr):
        self.attr = attr

    def get_uses(self):
        return self.uses

    def set_uses(self, uses):
        self.uses = uses

    def get_trace(self):
        return self.trace

    @staticmethod
    def get_nums_embedding():
        return 0

    def get_embedding(self):
        return []


class DFG_Op(Node):

    def __init__(self, idx, attr, node_type, inputs):
        self.idx = idx
        self.attr = attr
        self.node_type = node_type
        self.inputs = inputs
        self.outputs = []

        for inp in inputs:
            inp.uses.append(self)

    def get_idx(self):
        return self.idx

    def get_type(self):
        return self.node_type

    def set_type(self, node_type):
        self.node_type = node_type

    def get_attr(self) -> NodeAttribute:
        return self.attr

    def set_attr(self, attr):
        self.attr = attr

    def get_inputs(self):
        return self.inputs

    def get_outputs(self):
        return self.outputs

    @staticmethod
    def get_nums_embedding():
        return 0

    def get_embedding(self):
        return []

    def out(self, idx=-1, attr=None, edge_type=None):
        o = DFG_Edge(idx, attr=attr, edge_type=edge_type, trace=self)
        self.outputs.append(o)
        return o


class DataflowGraph(Graph):

    def __init__(self, nodes, edges):
        self.nodes = nodes
        self.edges = edges

    def get_nodes(self):
        return self.nodes

    def get_edges(self):
        return self.edges


#########################
##### look up table #####
#########################
OP_MAP = {
    # arithmeticOp
    AddOp: "add",
    SubtractOp: "sub",
    MultiplyOp: "mul",
    DivideOp: "div",
    SinOp: "sin",
    CosOp: "cos",
    TanOp: "tan",
    TanhOp: "tanh",
    MultiplyScalarOp: "multiply_scalar",
    AddScalarOp: "add_scalar",
    DivideScalarOp: "divide_scalar",
    PowOp: "pow",
    SqrtOp: "sqrt",
    ErfOp: "erf",
    NegativeOp: "neg",

    # transformOp
    ReshapeOp: "reshape",
    RearrangeOp: "rearrange",
    SqueezeOp: "squeeze",
    UnsqueezeOp: "unsqueeze",
    FlattenOp: "flatten",
    PermuteDimsOp: "permute_dims",
    CastOp: "cast",
    ConcatOp: "concat",
    TakeOp: "take",
    GatherOp: "gather",
    StridedSliceOp: "strided_slice",
    BroadcastOp: "broadcast",
    PadOp: "pad",
    # TileOp: "tile",

    # activation
    ReluOp: "relu",
    LeakyReluOp: "leaky_relu",
    SigmoidOp: "sigmoid",
    # HardSigmoidOp: "hard_sigmoid",
    ClipOp: "clip",
    GeluOp: "gelu",
    # SiluOp: "silu",
    # PReluOp: "prelu",
    # HardSwishOp: "hardswish",
    # ThresholdOp: "threshold",
    # HardTanhOp: "hardtanh",
    # EluOp: "elu",
    # SeluOp: "selu",
    # CeluOp: "celu",
    # LogSigmoidOp: "log_sigmoid",
    # HardShrinkOp: "hard_shrink",
    # TanhShrinkOp: "tanh_shrink",
    # SoftSignOp: "soft_sign",
    # SoftPlusOp: "softplus",
    # SoftShrinkOp: "soft_shrink",
    SoftmaxOp: "softmax",

    # pool
    MaxPool2dOp: "max_pool2d",
    AvgPool2dOp: "avg_pool2d",

    # tensor compute
    Conv2dOp: "conv2d",
    MatmulOp: "matmul",
    BatchMatmulOp: "batch_matmul",
    AttnOp: "attention",

    # reduce
    ReduceSumOp: "reduce_sum",
    ReduceMeanOp: "reduce_mean",
}

NODE_TYPE = ["Var", "Const"] + [v for _, v in OP_MAP.items()]
register_types(NODE_TYPE)


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
            dfg_op.attr.attrs["padding"],
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
