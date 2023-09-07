from enum import Enum

NODE_TYPE: Enum = None
ConstType = None
VarType = None
REWRITE_RULES: list = []


def register_node_type(str_types: list[str]):
    global NODE_TYPE, ConstType, VarType

    # check
    assert NODE_TYPE is None, "NODE_TYPE has been registered!"
    assert "Var" in str_types, "node type must have 'Var'"
    assert "Const" in str_types, "node type must have 'Const'"

    NODE_TYPE = Enum("node_type", str_types)
    ConstType = NODE_TYPE["Const"]
    VarType = NODE_TYPE["Var"]


def get_node_type():
    global NODE_TYPE, ConstType, VarType
    assert NODE_TYPE is not None, "NODE_TYPE is not registered!"
    assert ConstType is not None, "ConstType is not registered!"
    assert VarType is not None, "VarType is not registered!"
    return NODE_TYPE, ConstType, VarType


def clear_node_type():
    global NODE_TYPE, ConstType, VarType
    NODE_TYPE = None
    ConstType = None
    VarType = None
