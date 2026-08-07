"""
Neural Network Builder — activations.py

Activation math. Every function here takes all its inputs as arguments and
returns a value -- no DOM access, no mutation of network topology state.
The one exception is reading state.CUSTOM_ACT_NAMES, a fixed constant table
(not app state), for the custom-expression eval namespace.
"""
import math
import re
import traceback
import numpy as np

import state

FUNC_CALL_RE = re.compile(r'([A-Za-z_][A-Za-z_0-9]*)\(')
DIGIT_BEFORE_RE = re.compile(r'(?<=[0-9])(?=[a-zA-Z(])')
LETTER_BEFORE_DIGIT_RE = re.compile(r'(?<=[a-zA-Z)])(?=[0-9])')
PAREN_ADJACENT_RE = re.compile(r'(?<=\))(?=\()')

def sigmoid_numpy(x):
    return 1.0 / (1.0 + np.exp(-x))

def apply_activation(x: float, fn: str, custom_activation: dict | None = None) -> float:
    if fn == "relu":
        return max(0.0, x)
    elif fn == "sigmoid":
        return sigmoid_numpy(x)
    elif fn == "tanh":
        return math.tanh(x)
    elif fn == "softplus":
        return math.log(1.0 + math.exp(x))
    elif fn == "custom":
        return apply_custom_activation(x, custom_activation or {"expr": "x", "pieces": []})
    else:
        return x

def _paren_repl(m):
    name = m.group(1)
    return name + "(" if name in state.CUSTOM_ACT_NAMES else name + "*("

def normalize_expr(expr: str) -> str:
    """Insert implicit-multiplication `*` (e.g. `3x`, `2(x+1)`) while leaving
    calls to known functions like `abs(x)` or `sqrt(x)` untouched."""
    cleaned = expr.replace("^", "**")
    cleaned = FUNC_CALL_RE.sub(_paren_repl, cleaned)
    cleaned = DIGIT_BEFORE_RE.sub("*", cleaned)
    cleaned = LETTER_BEFORE_DIGIT_RE.sub("*", cleaned)
    cleaned = PAREN_ADJACENT_RE.sub("*", cleaned)
    return cleaned

def safe_eval_expr(expr: str, x: float) -> float:
    if not expr or not expr.strip():
        return x
    cleaned = normalize_expr(expr)
    ns = dict(state.CUSTOM_ACT_NAMES)
    ns["x"] = x
    try:
        return float(eval(cleaned, {"__builtins__": {}}, ns))
    except Exception:
        print("Custom activation eval error:\n" + traceback.format_exc())
        return x

def parse_bound(raw: str):
    if raw is None:
        return None
    s = raw.strip().lower().replace("∞", "inf")
    if s == "":
        return None
    try:
        return float(s)
    except ValueError:
        return None

def apply_custom_activation(x: float, custom_activation: dict) -> float:
    pieces = custom_activation["pieces"]
    if not pieces:
        return safe_eval_expr(custom_activation["expr"], x)
    for p in pieces:
        lo, hi = parse_bound(p["lo"]), parse_bound(p["hi"])

        if lo is None:
            lo_ok = True
        elif p["lo_op"] == "<":
            lo_ok = lo < x
        elif p["lo_op"] == "<=":
            lo_ok = lo <= x
        elif p["lo_op"] == ">":
            lo_ok = lo > x
        elif p["lo_op"] == ">=":
            lo_ok = lo >= x
        else:
            lo_ok = True

        if hi is None:
            hi_ok = True
        elif p["hi_op"] == "<":
            hi_ok = x < hi
        elif p["hi_op"] == "<=":
            hi_ok = x <= hi
        elif p["hi_op"] == ">":
            hi_ok = x > hi
        elif p["hi_op"] == ">=":
            hi_ok = x >= hi
        else:
            hi_ok = True

        if lo_ok and hi_ok:
            return safe_eval_expr(p["expr"], x)
    return 0.0
