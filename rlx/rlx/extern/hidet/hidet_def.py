from collections import namedtuple

from rlx.utils.common import get_logger
from rlx.frontend.registry import register_types
from rlx.frontend import Graph, Node, Edge

# extern
import hidet

from hidet.graph.ops.definitions.arithmetic import AddScalarOp, SubScalarOp, RSubScalarOp, MultiplyScalarOp, DivideScalarOp, RDivideScalarOp, SqrtOp, ErfOp, ExpOp, Expm1Op, LogOp, Log2Op, Log10Op, Log1pOp, RsqrtOp, PowOp, NegativeOp, ReciprocalOp, AddOp, SubtractOp, MultiplyOp, DivideOp, SinOp, CosOp, TanOp, SinhOp, CoshOp, TanhOp, AcosOp, AsinOp, AtanOp, Atan2Op, AcoshOp, AsinhOp, AtanhOp, SquareOp, CubeOp, AbsOp, FloorOp, RoundOp, TruncOp, CeilOp, IsFiniteOp, IsInfOp, IsNanOp, SignOp, RightShiftOp, LeftShiftOp, BitwiseAndOp, BitwiseNotOp, BitwiseOrOp, BitwiseXorOp, ModOp, WhereOp, MaxOp, MinOp

from hidet.graph.ops.definitions.compare import EqualOp, NotEqualOp, LessOp, GreaterOp, LessEqualOp, GreaterEqualOp, LogicalNotOp, LogicalAndOp, LogicalOrOp, LogicalXorOp

from hidet.graph.ops.definitions.transform import ReshapeOp, RearrangeOp, SqueezeOp, UnsqueezeOp, FlattenOp, PermuteDimsOp, CastOp, ConcatOp, TakeOp, GatherOp, StridedSliceOp, BroadcastOp, PadOp, TileOp

from hidet.graph.ops.definitions.pool import MaxPool2dOp, MaxPool3dOp, AvgPool2dOp, AvgPool3dOp, AdaptivePoolOp, AdaptiveAvgPool1dOp, AdaptiveAvgPool2dOp, AdaptiveAvgPool3dOp, AdaptiveMaxPool1dOp, AdaptiveMaxPool2dOp, AdaptiveMaxPool3dOp

from hidet.graph.ops.definitions.activation import ReluOp, LeakyReluOp, SigmoidOp, HardSigmoidOp, ClipOp, GeluOp, SiluOp, PReluOp, HardSwishOp, ThresholdOp, HardTanhOp, EluOp, SeluOp, CeluOp, LogSigmoidOp, HardShrinkOp, TanhShrinkOp, SoftSignOp, SoftPlusOp, SoftShrinkOp, SoftmaxOp

from hidet.graph.ops.definitions.image import Resize2dOp
from hidet.graph.ops.definitions.cumulative import CumulativeSumOp
from hidet.graph.ops.definitions.special import BarrierOp

from hidet.graph.ops.definitions.conv1d import Conv1dOp
from hidet.graph.ops.definitions.conv1d_transpose.conv1d_transpose import Conv1dTransposeOp
from hidet.graph.ops.definitions.attention.attention import AttnOp

from hidet.graph.ops.definitions.conv2d.conv2d import Conv2dOp
from hidet.graph.ops.definitions.conv3d.conv3d import Conv3dOp

from hidet.graph.ops.definitions.matmul.matmul import MatmulOp
from hidet.graph.ops.definitions.matmul.batch_matmul import BatchMatmulOp

from hidet.graph.ops.definitions.reduce.reduce import ReduceSumOp, ReduceMeanOp

# pass
from hidet.graph.transforms.fold_const import fold_const_pass
from hidet.graph.transforms.subgraph_rewrite import subgraph_rewrite_pass
from hidet.graph.transforms.automatic_mix_precision import automatic_mix_precision_pass
from hidet.graph.transforms.resolve_variant import resolve_variant_pass
from hidet.graph.transforms.fuse_operator import fuse_operator_pass
from hidet.graph.transforms.eliminate_barrier import eliminate_barrier_pass

# utils
from hidet.utils import benchmark_func
from hidet.utils import same_list, initialize
from hidet.graph.ops.definitions.utils.tensor_utils import normalize_dim

logger = get_logger(__name__)

NodeAttribute = namedtuple("NodeAttribute", ["name", "attrs"])
EdgeAttribute = namedtuple("EdgeAttribute", [
    "shape",
    "dtype",
    "device",
    "layout",
    "storage_id",
])

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

########################################
## override user-API for hidet  ########
########################################
register_types(NODE_TYPE)


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
