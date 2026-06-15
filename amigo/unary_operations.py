from .expressions import Expr, VarNode, UnaryNode, BinaryNode, ConstNode, PassiveNode


def abs(expr: Expr):
    return Expr(UnaryNode("fabs", expr))


def fabs(expr: Expr):
    return Expr(UnaryNode("fabs", expr))


def sqrt(expr: Expr):
    return Expr(UnaryNode("sqrt", expr))


def sin(expr: Expr):
    return Expr(UnaryNode("sin", expr))


def sqrt(expr: Expr):
    return Expr(UnaryNode("sqrt", expr))


def asin(expr: Expr):
    return Expr(UnaryNode("asin", expr))


def cos(expr: Expr):
    return Expr(UnaryNode("cos", expr))


def acos(expr: Expr):
    return Expr(UnaryNode("acos", expr))


def tan(expr: Expr):
    return Expr(UnaryNode("tan", expr))


def atan(expr: Expr):
    return Expr(UnaryNode("atan", expr))


def sinh(expr: Expr):
    return Expr(UnaryNode("sinh", expr))


def asinh(expr: Expr):
    return Expr(UnaryNode("asinh", expr))


def cosh(expr: Expr):
    return Expr(UnaryNode("cosh", expr))


def acosh(expr: Expr):
    return Expr(UnaryNode("acosh", expr))


def tanh(expr: Expr):
    return Expr(UnaryNode("tanh", expr))


def atanh(expr: Expr):
    return Expr(UnaryNode("atanh", expr))


def exp(expr: Expr):
    return Expr(UnaryNode("exp", expr))


def log(expr: Expr):
    return Expr(UnaryNode("log", expr))


def atan2(a: Expr, b: Expr):
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        raise ValueError("Neither argument is active")
    elif isinstance(a, (int, float)):
        return Expr(BinaryNode("atan2", ConstNode(value=a), b))
    elif isinstance(b, (int, float)):
        return Expr(BinaryNode("atan2", a, ConstNode(value=b)))
    elif isinstance(a, Expr) and isinstance(b, Expr):
        return Expr(BinaryNode("atan2", a, b))
    else:
        raise TypeError("Types not recognized for atan2")


def min2(a: Expr, b: Expr):
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        raise ValueError("Neither argument is active")
    elif isinstance(a, (int, float)):
        return Expr(BinaryNode("min2", ConstNode(value=a), b))
    elif isinstance(b, (int, float)):
        return Expr(BinaryNode("min2", a, ConstNode(value=b)))
    elif isinstance(a, Expr) and isinstance(b, Expr):
        return Expr(BinaryNode("min2", a, b))
    else:
        raise TypeError("Types not recognized for min2")


def max2(a: Expr, b: Expr):
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        raise ValueError("Neither argument is active")
    elif isinstance(a, (int, float)):
        return Expr(BinaryNode("max2", ConstNode(value=a), b))
    elif isinstance(b, (int, float)):
        return Expr(BinaryNode("max2", a, ConstNode(value=b)))
    elif isinstance(a, Expr) and isinstance(b, Expr):
        return Expr(BinaryNode("max2", a, b))
    else:
        raise TypeError("Types not recognized for max2")


def passive(expr: Expr):
    """Force an expression to be passive"""
    if not isinstance(expr.node, VarNode):
        raise TypeError("Passive type must be a variable")
    return Expr(PassiveNode(expr))


# binary = ["MatMatMult", "MatSum"]

# VecScale
# VecSum
# VecCross
# VecSymOuterProduct

# VecNorm
# VecOuter


# unary = [""]
# MatScale()
# MatInv
# MatDet
# MatGreenStrain
# SymMatMultTrace
# unary = ["SymMatSum", ]


# def norm(v : Expr):
