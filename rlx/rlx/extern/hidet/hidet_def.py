import os
import math
import time
import pickle
import pandas as pd
from collections import namedtuple

import onnx  # type: ignore

from rlx.utils.common import get_logger
from rlx.frontend.registry import register_node_type
from rlx.frontend import RewriteRule, Graph, Node, Edge, node_pattern, const_pattern, symbol_pattern

# extern
import hidet  # type: ignore

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

# logger
logger = get_logger(__name__)

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
register_node_type(NODE_TYPE)


class DFG_Edge(Edge):
    def __init__(self, idx, attr, edge_type, trace):
        self.idx = idx
        self.attr = attr
        self.edge_type = edge_type
        self.uses = []
        self.trace = trace

    def get_type(self):
        return self.edge_type

    def get_attr(self):
        return self.attr

    def get_uses(self):
        return self.uses

    def get_trace(self):
        return self.trace


class DFG_Op(Node):
    def __init__(self, idx, attr, node_type, inputs):
        self.idx = idx
        self.attr = attr
        self.node_type = node_type
        self.inputs = inputs
        self.outputs = []

        for inp in inputs:
            inp.uses.append(self)

    def get_type(self):
        return self.node_type

    def get_attr(self):
        return self.attr

    def get_inputs(self):
        return self.inputs

    def get_outputs(self):
        return self.outputs


class DataflowGraph(Graph):
    def __init__(self, nodes, edges):
        self.nodes = nodes
        self.edges = edges

    def get_nodes(self):
        return self.nodes

    def get_edges(self):
        return self.edges


#########################################
########### rewrite rules! ##############
#########################################


#### transform rules ####
class tr1(RewriteRule):
    def __init__(self, node_type):
        self.node_type = node_type
        self.name = "reshape(x) * scale => "
        self.x, self.tx = symbol_pattern()
        self.scale, self.tscale = const_pattern()

    def source_pattern(self):
        reshape = node_pattern(self.node_types["reshape"], [self.x], 1)
        mul = node_pattern(self.node_types["mul"], [reshape, self.scale], 1)
        return [mul]

    def target_pattern(self):
        unsqueeze = node_pattern(self.node_types["unsqueeze"], [self.tscale],
                                 1)
        return [out]


class tr2(RewriteRule):
    def __init__(self):
        self.name = "reshape(x) + bias => "
        self.a, self.ta = const_pattern()
        self.b, self.tb = const_pattern()

    def source_pattern(self):
        return []

    def target_pattern(self):
        return []


##### arithmetic rules ####
class ar1(RewriteRule):
    def __init__(self, node_types):
        self.name = "a + x => x + a"
        self.a, self.ta = const_pattern()
        self.x, self.tx = symbol_pattern()
        self.node_types = node_types

    def source_pattern(self):
        add = node_pattern(self.node_types["add"], [self.a, self.x], 1)
        return [add]

    def target_pattern(self):
        out = node_pattern(self.node_types["add"], [self.tx, self.ta], 1)
        return [out]


class ar2(RewriteRule):
    def __init__(self, node_types):
        self.name = "x - a => x + (-a)"
        self.a, self.ta = const_pattern()
        self.x, self.tx = symbol_pattern()
        self.node_types = node_types

    def source_pattern(self):
        sub = node_pattern(self.node_types["sub"], [self.x, self.a], 1)
        return [sub]

    def target_pattern(self):
        neg = node_pattern(self.node_types["neg"], [self.ta], 1)
        add = node_pattern(self.node_types["add"], [self.tx, neg], 1)
        return [add]


class ar3(RewriteRule):
    def __init__(self, node_types):
        self.name = "(x + a) + b => x + (a + b)"
        self.a, self.ta = const_pattern()
        self.b, self.tb = const_pattern()
        self.x, self.tx = symbol_pattern()
        self.node_types = node_types

    def source_pattern(self):
        add = node_pattern(self.node_types["add"], [self.a, self.x], 1)
        add2 = node_pattern(self.node_types["add"], [self.b, add], 1)
        return [add2]

    def target_pattern(self):
        add = node_pattern(self.node_types["add"], [self.tb, self.ta], 1)
        out = node_pattern(self.node_types["add"], [self.tx, add], 1)
        return [out]


class ar4(RewriteRule):
    def __init__(self, node_types):
        self.name = "(x + a) * b => x * b + a * b"
        self.a, self.ta = const_pattern()
        self.b, self.tb = const_pattern()
        self.x, self.tx = symbol_pattern()
        self.node_types = node_types

    def source_pattern(self):
        add = node_pattern(self.node_types["add"], [self.a, self.x], 1)
        mul = node_pattern(self.node_types["mul"], [self.b, add], 1)
        return [mul]

    def target_pattern(self):
        mul1 = node_pattern(self.node_types["mul"], [self.tx, self.tb], 1)
        mul2 = node_pattern(self.node_types["mul"], [self.ta, self.tb], 1)
        add = node_pattern(self.node_types["add"], [mul1, mul2], 1)
        return [add]


class ar5(RewriteRule):
    def __init__(self, node_types):
        self.name = "(x + a) + (y + b) => (x + y) + (a + b)"
        self.a, self.ta = const_pattern()
        self.b, self.tb = const_pattern()
        self.x, self.tx = symbol_pattern()
        self.y, self.ty = symbol_pattern()
        self.node_types = node_types

    def source_pattern(self):
        add1 = node_pattern(self.node_types["add"], [self.a, self.x], 1)
        add2 = node_pattern(self.node_types["add"], [self.y, self.b], 1)
        add3 = node_pattern(self.node_types["add"], [add1, add2], 1)
        return [add3]

    def target_pattern(self):
        add1 = node_pattern(self.node_types["add"], [self.tx, self.ty], 1)
        add2 = node_pattern(self.node_types["add"], [self.ta, self.tb], 1)
        add3 = node_pattern(self.node_types["add"], [add1, add2], 1)
        return [add3]


##### matmul rules ####
class mr1(RewriteRule):
    def __init__(self):
        self.name = "matmul(x, c1)|matmul(x, c2) ==> matmul(x, concat(c1, c2)) followed by split"
        self.x, self.tx = symbol_pattern()
        self.c1, self.tc1 = const_pattern()
        self.c2, self.tc2 = const_pattern()

    def source_pattern(self):
        matmul1 = node_pattern(node_types["matmul"], [self.x, self.c1], 1)
        matmul2 = node_pattern(node_types["matmul"], [self.x, self.c2], 1)
        return [matmul1, matmul2]

    def target_pattern(self):
        concat = node_pattern(node_types["concat"], [self.tc1, self.tc2], 1)
        matmul = node_pattern(node_types["matmul"], [self.tx, concat], 1)
        # TODO add attr
        split = node_pattern(node_types["split"], [matmul], 2)
        return split


class mr2(RewriteRule):
    def __init__(self):
        self.name = "matmul(x, c1)|matmul(x, c2)|matmul(x, c3) => matmul(x, concat(c1, c2, c3)) followed by split"
        self.x, self.tx = symbol_pattern()
        self.c1, self.tc1 = const_pattern()
        self.c2, self.tc2 = const_pattern()
        self.c3, self.tc3 = const_pattern()

    def source_pattern(self):
        matmul1 = node_pattern(node_types["matmul"], [self.x, self.c1], 1)
        matmul2 = node_pattern(node_types["matmul"], [self.x, self.c2], 1)
        matmul3 = node_pattern(node_types["matmul"], [self.x, self.c3], 1)
        return [matmul1, matmul2, matmul3]

    def target_pattern(self):
        concat = node_pattern(node_types["concat"],
                              [self.tc1, self.tc2, self.tc3], 1)
        matmul = node_pattern(node_types["matmul"], [self.tx, concat], 1)
        # TODO add attr
        split = node_pattern(node_types["split"], [matmul], 3)
        return split


class mr3(RewriteRule):
    def __init__(self):
        self.name = "3 branches of matmul(x, branch c) + branch b ==> matmul(x, c) + b followed by split"
        self.x, self.tx = symbol_pattern()
        self.c1, self.tc1 = const_pattern()
        self.c2, self.tc2 = const_pattern()
        self.c3, self.tc3 = const_pattern()
        self.b1, self.tb1 = const_pattern()
        self.b2, self.tb2 = const_pattern()
        self.b3, self.tb3 = const_pattern()

    def source_pattern(self):
        matmul1 = node_pattern(node_types["matmul"], [self.x, self.c1], 1)
        matmul2 = node_pattern(node_types["matmul"], [self.x, self.c2], 1)
        matmul3 = node_pattern(node_types["matmul"], [self.x, self.c3], 1)
        add1 = node_pattern(node_types["+"], [matmul1, self.b1], 1)
        add2 = node_pattern(node_types["+"], [matmul2, self.b2], 1)
        add3 = node_pattern(node_types["+"], [matmul3, self.b3], 1)
        return [add1, add2, add3]

    def target_pattern(self):
        # assume Tensor shape is attr
        concat1 = node_pattern(node_types["concat"],
                               [self.tc1, self.tc2, self.tc3], 1)
        concat2 = node_pattern(node_types["concat"],
                               [self.tb1, self.tb2, self.tb3], 1)
        matmul = node_pattern(node_types["matmul"], [self.tx, concat1], 1)
        add = node_pattern(node_types["+"], [matmul, concat2], 1)
        split = node_pattern(node_types["split"], [add], 3)
        return split

    def deps(self):
        pass


##### attn rules ####
class attnr1(RewriteRule):
    def __init__(self):
        # TODO q, k is symbol or const?
        self.name = "matmul(q,k) * c1 => matmul(c1 * q, k)"
        self.q, self.tq = const_pattern()
        self.k, self.tk = const_pattern()
        self.c1, self.tc1 = const_pattern()

    def source_pattern(self):
        matmul1 = node_pattern(node_types["matmul"], [self.q, self.k], 1)
        mul = node_pattern(node_types["*"], [matmul1, self.c1], 1)
        return [mul]

    def target_pattern(self):
        mul = node_pattern(node_types["*"], [self.tc1, self.tq], 1)
        matmul = node_pattern(node_types["matmul"], [mul, self.tk], 1)
        return [matmul]


class attnr2(RewriteRule):
    def __init__(self):
        # TODO q, k is symbol or const?
        self.name = "matmul(q,k) / c1 => matmul(q/c1, k)"
        self.q, self.tq = const_pattern()
        self.k, self.tk = const_pattern()
        self.c1, self.tc1 = const_pattern()

    def source_pattern(self):
        matmul1 = node_pattern(node_types["matmul"], [self.q, self.k], 1)
        mul = node_pattern(node_types["/"], [matmul1, self.c1], 1)
        return [mul]

    def target_pattern(self):
        div = node_pattern(node_types["/"], [self.tc1, self.tq], 1)
        matmul = node_pattern(node_types["matmul"], [div, self.tk], 1)
        return [matmul]


class attnr3(RewriteRule):
    def __init__(self):
        self.name = "matmul(Softmax(matmul(q, k)), v) => attn(q, k, v)"
        # TODO how to design attr
        self.q, self.tq = const_pattern()
        self.k, self.tk = const_pattern()
        self.v, self.tv = const_pattern()

    def source_pattern(self):
        qk = node_pattern(node_types["matmul"], [self.q, self.k], 1)
        softmax = node_pattern(node_types["Softmax"], [qk], 1)
        qkv = node_pattern(node_types["matmul"], [softmax, self.v], 1)
        return [qkv]

    def target_pattern(self):
        attn = node_pattern(node_types["attn"], [self.tq, self.tk, self.tv], 1)
        return [attn]

    def register_deps(self):
        # TODO how to design attr
        pass


class attnr4(RewriteRule):
    def __init__(self):
        self.name = "matmul(Softmax(matmul(q, k) + mask), v) => attn(q, k, v, mask)"
        # TODO how to design attr
        self.q, self.tq = const_pattern()
        self.k, self.tk = const_pattern()
        self.v, self.tv = const_pattern()
        self.mask, self.tm = const_pattern()

    def source_pattern(self):
        qk = node_pattern(node_types["matmul"], [self.q, self.k], 1)
        mask = node_pattern(node_types["+"], [qk, self.mask], 1)
        softmax = node_pattern(node_types["Softmax"], [mask], 1)
        qkv = node_pattern(node_types["matmul"], [softmax, self.v], 1)
        return [qkv]

    def target_pattern(self):
        attn = node_pattern(node_types["masked_attn"],
                            [self.tq, self.tk, self.tv, self.tm], 1)
        return [attn]

    def register_deps(self):
        # TODO how to design attr
        pass


##### conv rules ####
class cr1(RewriteRule):
    def __init__(self):
        self.name = "conv2d(x, w) * scale => conv2d(x, w * scale)"
        self.x, self.tx = symbol_pattern()
        self.w, self.tw = const_pattern()
        self.scale, self.tscale = const_pattern()

    def source_pattern(self):
        conv = node_pattern(node_types["conv2d"], [self.x, self.w], 1)
        mul = node_pattern(node_types["*"], [conv, self.scale], 1)
        return [mul]

    def target_pattern(self):
        mul = node_pattern(node_types["*"], [self.tw, self.tscale], 1)
        conv = node_pattern(node_types["conv2d"], [self.tx, mul], 1)
        return [conv]

    def register_deps(self):
        pass


class cr2(RewriteRule):
    def __init__(self):
        self.name = "conv2d(x, w1)|conv2d(x, w2) => conv2d(x, w1 + w2)"
        self.x, self.tx = symbol_pattern()
        self.w1, self.tw1 = const_pattern()
        self.w2, self.tw2 = const_pattern()

    def source_pattern(self):
        conv1 = node_pattern(node_types["conv2d"], [self.x, self.w1], 1)
        conv2 = node_pattern(node_types["conv2d"], [self.x, self.w2], 1)
        return [conv1, conv2]

    def target_pattern(self):
        concat = node_pattern(node_types["concat"], [self.tw1, self.tw2], 1)
        conv = node_pattern(node_types["conv2d"], [self.tx, concat], 1)
        split = node_pattern(node_types["split"], [conv], 2)
        return split

    def register_deps(self):
        pass


class cr3(RewriteRule):
    def __init__(self):
        self.name = "conv2d(x, w1)|conv2d(x, w2)|conv2d(x, w3) => conv2d(x, w1 + w2 + w3)"
        self.x, self.tx = symbol_pattern()
        self.w1, self.tw1 = const_pattern()
        self.w2, self.tw2 = const_pattern()
        self.w3, self.tw3 = const_pattern()

    def source_pattern(self):
        conv1 = node_pattern(node_types["conv2d"], [self.x, self.w1], 1)
        conv2 = node_pattern(node_types["conv2d"], [self.x, self.w2], 1)
        conv3 = node_pattern(node_types["conv2d"], [self.x, self.w3], 1)
        return [conv1, conv2, conv3]

    def target_pattern(self):
        concat = node_pattern(node_types["concat"],
                              [self.tw1, self.tw2, self.tw3], 1)
        conv = node_pattern(node_types["conv2d"], [self.tx, concat], 1)
        split = node_pattern(node_types["split"], [conv], 3)
        return split

    def register_deps(self):
        pass


def define_rewrite_rules(node_types):

    return [
        ar1(node_types),
        ar2(node_types),
        ar3(node_types),
        ar4(node_types),
        ar5(node_types),
    ]
