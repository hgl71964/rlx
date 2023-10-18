from enum import Enum

TYPES: Enum = None
ConstType = None
VarType = None
REWRITE_RULES: list = []


def register_types(str_types: list[str]):
    global TYPES, ConstType, VarType

    # check
    assert TYPES is None, "TYPES has been registered!"
    assert "Var" in str_types, "node type must have 'Var'"
    assert "Const" in str_types, "node type must have 'Const'"

    TYPES = Enum("typeS", str_types)
    ConstType = TYPES["Const"]
    VarType = TYPES["Var"]


def get_types():
    global TYPES, ConstType, VarType
    assert TYPES is not None, "TYPES is not registered!"
    assert ConstType is not None, "ConstType is not registered!"
    assert VarType is not None, "VarType is not registered!"
    return TYPES, ConstType, VarType


def clear_types():
    global TYPES, ConstType, VarType
    TYPES = None
    ConstType = None
    VarType = None
