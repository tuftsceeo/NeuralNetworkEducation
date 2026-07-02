"""
Neural Network Builder — main.py
PyScript 2026.3.1
"""
import math
import asyncio
from pyscript import document, window, when
from pyscript.ffi import create_proxy
from Device import Element
import legoeducation as le

import traceback
import threading
import concurrent.futures

from pyscript.js_modules import Plotly
import plot

# ── State ──────────────────────────────────────────────────────────────────────

rows: list[dict] = [] # row is a dict
row_counter      = 0
devices: list[Element] = []
is_running       = False
all_plots: dict[str, object] = {}

debug_mode      = False
input_values: dict[int, float]  = {}   # rid -> raw input reading
eq_values: dict[int, float]     = {}   # rid -> pre-activation (weighted sum + bias)
output_values: dict[int, float] = {}   # rid -> post-activation value

ACTIVATION_OPTIONS = [
    ("None",     ""),
    ("ReLU",     "relu"),
    ("Sigmoid",  "sigmoid"),
    ("Tanh",     "tanh"),
    ("Softplus", "softplus"),
    ("Custom",   "custom"),
]

ARROW_COLOR = "#1e40af"

custom_activation = {"expr": "x", "pieces": []}
piece_counter = 0

CUSTOM_ACT_NAMES = {
    "abs": abs, "max": max, "min": min, "round": round,
    "sqrt": math.sqrt, "exp": math.exp, "log": math.log,
    "log2": math.log2, "log10": math.log10,
    "sin": math.sin, "cos": math.cos, "tan": math.tan,
    "pi": math.pi, "e": math.e, "inf": math.inf,
}

import re

IMPLICIT_MULT_RE = re.compile(r'(?<=[0-9])(?=[a-zA-Z(])|(?<=[a-zA-Z)])(?=[0-9(])')

def normalize_expr(expr: str) -> str:
    cleaned = expr.replace("^", "**")
    # insert * between number/letter/paren boundaries like "3x" -> "3*x", "2(x" -> "2*(x"
    cleaned = IMPLICIT_MULT_RE.sub("*", cleaned)
    return cleaned

def safe_eval_expr(expr: str, x: float) -> float:
    if not expr or not expr.strip():
        return x
    cleaned = normalize_expr(expr)
    ns = dict(CUSTOM_ACT_NAMES)
    ns["x"] = x
    try:
        return float(eval(cleaned, {"__builtins__": {}}, ns))
    except Exception:
        print("Custom activation eval error:\n" + traceback.format_exc())
        return x

def parse_bound(raw: str):
    """Blank -> unbounded (None). Accepts 'inf', '-inf', '∞', '-∞', or a number."""
    if raw is None:
        return None
    s = raw.strip().lower().replace("∞", "inf")
    if s == "":
        return None
    try:
        return float(s)
    except ValueError:
        return None

def apply_custom_activation(x: float) -> float:
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

# ── Helpers ────────────────────────────────────────────────────────────────────
def apply_activation(x: float, fn: str) -> float:
    if fn == "relu":
        return max(0.0, x)
    elif fn == "sigmoid":
        return 1.0 / (1.0 + math.exp(-x))
    elif fn == "tanh":
        return math.tanh(x)
    elif fn == "softplus":
        return math.log(1.0 + math.exp(x))
    elif fn == "custom":
        return apply_custom_activation(x)
    else:
        return x
        
def device_by_name(name: str) -> Element | None:
    for d in devices:
        if d.name == name:
            return d
    return None

def get_id(id_: str):
    return document.getElementById(id_)

def row_by_id(rid: int) -> dict | None:
    for r in rows:
        if r["id"] == rid:
            return r
    return None

def get_device_options_html() -> str:
    opts = '<option class = "dev-dropdown" value="">— device —</option>'
    for device in devices:
        opts += f'<option value="{device.name}">{device.name}</option>'
    return opts

def get_in_channels_html(device: Element | None = None) -> str:
    if device is None:
        return '<option value="">— value —</option>'
    try:
        state = device.state  # dict parsed from JSON
        return "".join(
            f'<option value="{key}">{key}</option>' for key in state.keys()
        )
    except Exception:
        return '<option value="">— value —</option>'

def get_out_channels_html(device: Element | None = None) -> str:
    if device is None:
        return '<option value="">— value —</option>'
    try:
        opts = device.get_out_list()  # dict parsed from JSON
        return "".join(
            f'<option value="{opt}">{opt}</option>' for opt in opts
        )
    except Exception:
        return '<option value="">— value —</option>'

def populate_act_select():
    sel = get_id("act-select")
    if not sel:
        return
    html = ""
    for label, val in ACTIVATION_OPTIONS:
        html += f'<option value="{val}">{label}</option>'
    sel.innerHTML = html

# ── SVG arrow primitives ───────────────────────────────────────────────────────

def svg_path(d: str, stroke_w: float = 2.0):
    p = document.createElementNS("http://www.w3.org/2000/svg", "path")
    p.setAttribute("d", d)
    p.setAttribute("fill", "none")
    p.setAttribute("stroke", ARROW_COLOR)
    p.setAttribute("stroke-width", str(stroke_w))
    p.setAttribute("stroke-linecap", "round")
    p.setAttribute("stroke-linejoin", "round")
    return p

def arrowhead(svg_el, x2: float, y2: float, dx: float, dy: float,
              stroke_w: float = 2.0, size: float = 9.0):
    """V-shaped arrowhead at (x2,y2) pointing in direction (dx,dy)."""
    mag = math.sqrt(dx * dx + dy * dy)
    if mag < 1e-9:
        return
    ux, uy = dx / mag, dy / mag
    spread = math.radians(26)
    cs, sn = math.cos(spread), math.sin(spread)
    wx1, wy1 =  ux * cs - uy * sn,  ux * sn + uy * cs
    wx2, wy2 =  ux * cs + uy * sn, -ux * sn + uy * cs
    ax1, ay1 = x2 - size * wx1, y2 - size * wy1
    ax2, ay2 = x2 - size * wx2, y2 - size * wy2
    d = f"M{x2:.2f},{y2:.2f} L{ax1:.2f},{ay1:.2f} M{x2:.2f},{y2:.2f} L{ax2:.2f},{ay2:.2f}"
    svg_el.appendChild(svg_path(d, stroke_w))

def debug_label(svg_el, x: float, y: float, value: float):
    text = f"{value:.2f}"
    char_w, pad_x, h = 6.2, 6, 18
    w = len(text) * char_w + pad_x * 2

    rect = document.createElementNS("http://www.w3.org/2000/svg", "rect")
    rect.setAttribute("x", str(x - w / 2))
    rect.setAttribute("y", str(y - h / 2))
    rect.setAttribute("width", str(w))
    rect.setAttribute("height", str(h))
    rect.setAttribute("rx", "5")
    rect.setAttribute("fill", "#1a1d2e")
    rect.setAttribute("stroke", "#ffffff")
    rect.setAttribute("stroke-width", "1")

    txt = document.createElementNS("http://www.w3.org/2000/svg", "text")
    txt.setAttribute("x", str(x))
    txt.setAttribute("y", str(y + 3.5))
    txt.setAttribute("text-anchor", "middle")
    txt.setAttribute("font-family", "JetBrains Mono, monospace")
    txt.setAttribute("font-size", "10")
    txt.setAttribute("font-weight", "700")
    txt.setAttribute("fill", "#ffffff")
    txt.textContent = text

    svg_el.appendChild(rect)
    svg_el.appendChild(txt)


def straight_arrow(svg_el, x1, y1, x2, y2, stroke_w: float = 2.0, debug_value=None):
    """Straight line with arrowhead, optionally labeled with a live value."""
    d = f"M{x1:.2f},{y1:.2f} L{x2:.2f},{y2:.2f}"
    svg_el.appendChild(svg_path(d, stroke_w))
    arrowhead(svg_el, x2, y2, x2 - x1, y2 - y1, stroke_w)

    if debug_mode and debug_value is not None:
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        dx, dy = x2 - x1, y2 - y1
        length = math.sqrt(dx * dx + dy * dy) or 1.0
        # perpendicular offset so the label sits beside/above the line, not on top of it
        px, py = -dy / length, dx / length
        debug_label(svg_el, mx + px * 11, my + py * 11, debug_value)

# ── Main redraw ────────────────────────────────────────────────────────────────

def redraw_arrows():
    svg_el  = get_id("arrow-svg")
    wrap_el = get_id("network-wrap")
    if not svg_el or not wrap_el or not rows:
        if svg_el:
            svg_el.innerHTML = ""
        return

    svg_el.innerHTML = ""
    wr = wrap_el.getBoundingClientRect()

    def rel(el):
        r = el.getBoundingClientRect()
        return {
            "x": r.left - wr.left,
            "y": r.top  - wr.top,
            "w": r.width,
            "h": r.height,
        }

    in_pts   = []
    eq_pts   = []
    eq_r_pts = []
    out_pts  = []

    for row in rows:
        rid    = row["id"]
        in_el  = get_id(f"cell-in-{rid}")
        eq_el  = get_id(f"cell-eq-{rid}")
        out_el = get_id(f"cell-out-{rid}")
        if in_el and eq_el and out_el:
            ir  = rel(in_el)
            er  = rel(eq_el)
            or_ = rel(out_el)
            in_pts.append(  (ir["x"] + ir["w"],  ir["y"] + ir["h"] / 2, rid) )
            eq_pts.append(  (er["x"],             er["y"] + er["h"] / 2, rid) )
            eq_r_pts.append((er["x"] + er["w"],   er["y"] + er["h"] / 2, rid) )
            out_pts.append( (or_["x"],             or_["y"] + or_["h"] / 2, rid) )

    if not in_pts:
        return

    act_el = get_id("act-box")

    # ── 1. Input → Equation: straight lines, spread at both ends ─────────────
    SPREAD_FRAC = 0.55
    n_rows = len(in_pts)

    in_heights = {}
    eq_heights = {}
    for row in rows:
        rid   = row["id"]
        in_el = get_id(f"cell-in-{rid}")
        eq_el = get_id(f"cell-eq-{rid}")
        if in_el:
            in_heights[rid] = rel(in_el)["h"]
        if eq_el:
            eq_heights[rid] = rel(eq_el)["h"]

    for src_idx, (ix, iy, i_rid) in enumerate(in_pts):
        for tgt_idx, (ex, ey, e_rid) in enumerate(eq_pts):
            if n_rows > 1:
                src_h    = in_heights.get(i_rid, 120)
                spread_h = min(src_h * SPREAD_FRAC, (n_rows - 1) * 14)
                tail_y   = iy - spread_h / 2 + (tgt_idx / (n_rows - 1)) * spread_h

                eq_h     = eq_heights.get(e_rid, 80)
                spread_h = min(eq_h * SPREAD_FRAC, (n_rows - 1) * 14)
                head_y   = ey - spread_h / 2 + (src_idx / (n_rows - 1)) * spread_h
            else:
                tail_y = iy
                head_y = ey

            same = i_rid == e_rid
            straight_arrow(svg_el, ix, tail_y, ex, head_y,
                           stroke_w=2.2 if same else 1.8,
                           debug_value=input_values.get(i_rid))

    # ── 2. Equation → Activation: horizontal straight arrows ─────────────────
    if act_el and eq_r_pts:
        act_x = rel(act_el)["x"]
        for (ex, ey, e_rid) in eq_r_pts:
            straight_arrow(svg_el, ex, ey, act_x, ey, stroke_w=2.0,
                           debug_value=eq_values.get(e_rid))

    # ── 3. Activation → Output: horizontal straight arrows ───────────────────
    if act_el and out_pts:
        act_rx = rel(act_el)["x"] + rel(act_el)["w"]
        for (ox, oy, o_rid) in out_pts:
            straight_arrow(svg_el, act_rx, oy, ox, oy, stroke_w=2.0,
                           debug_value=output_values.get(o_rid))

# ── Row HTML ───────────────────────────────────────────────────────────────────

def make_left_row_html(row: dict) -> str:
    rid  = row["id"]
    name = row["name"]

    eq_parts = ""
    for i, r in enumerate(rows):
        coeff_val = row["weights"][i] if i < len(row["weights"]) else 1.0
        if i > 0:
            eq_parts += '<span class="eq-op">+</span>'
        eq_parts += (
            f'<input type="number" step="any" value="{coeff_val:.2f}"'
            f' class="eq-num-input" id="coeff-{rid}-{i}"'
            f' data-row="{rid}" data-idx="{i}" />'
            f'<span class="eq-var" id="var-label-{rid}-{i}">{r["name"]}</span>'
        )
    eq_parts += (
        f'<span class="eq-op">+</span>'
        f'<input type="number" step="any" value="{row["bias"]:.2f}"'
        f' class="eq-bias-input" id="bias-{rid}" data-row="{rid}" />'
    )

    dev_opts = get_device_options_html()

    return f"""
<div class="neuron-row neuron-row-left" id="row-left-{rid}" data-row="{rid}">

    <div class="cell-label">
        <input type="text" class="name-input" id="name-{rid}"
               value="{name}" data-row="{rid}" maxlength="12" />
    </div>

    <div class="cell-node" id="cell-in-{rid}">
        <div class="node-card input-node">
            <div class="node-header">
                <select class="node-device-select" id="dev-in-{rid}">{dev_opts}</select>
                <span class="node-reading" id="reading-in-{rid}">—</span>
            </div>
            <div class="node-plot">
                <div class="plot-canvas" id="plot-in-{rid}" width="170" height="70"></div>
            </div>
            <div class="plot-footer">
                <select class="channel-select" id="chan-in-{rid}">— value —</select>
            </div>
        </div>
    </div>

    <div class="cell-arrow-gap-in"></div>

    <div class="cell-node" id="cell-eq-{rid}">
        <div class="eq-node" id="eq-node-{rid}">
            <div class="eq-inline" id="eq-inline-{rid}">
                {eq_parts}
            </div>
        </div>
    </div>

</div>
"""

def make_right_row_html(row: dict) -> str:
    rid      = row["id"]
    dev_opts = get_device_options_html()

    return f"""
<div class="neuron-row neuron-row-right" id="row-right-{rid}" data-row="{rid}">

    <div class="cell-node" id="cell-out-{rid}">
        <div class="node-card output-node">
            <div class="node-header">
                <select class="node-device-select" id="dev-out-{rid}">{dev_opts}</select>
                <span class="node-reading" id="reading-out-{rid}">—</span>
            </div>
            <div class="node-plot">
                <div class="plot-canvas" id="plot-out-{rid}" width="170" height="70"></div>
            </div>
            <div class="plot-footer">
                <select class="channel-select" id="chan-out-{rid}">— value —</select>
            </div>
        </div>
    </div>

    <div class="cell-delete">
        <button class="btn-delete-row" id="del-{rid}" data-row="{rid}" title="Remove neuron">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none"
                 stroke="currentColor" stroke-width="2.5">
                <path d="M3 6h18M8 6V4h8v2M19 6l-1 14H6L5 6"/>
            </svg>
        </button>
    </div>

</div>
"""

# ── Equation field sync ────────────────────────────────────────────────────────

def make_piece_html(p: dict) -> str:
    pid = p["id"]
    def op_opts(current):
        out = ""
        for val, sym in (("<", "&lt;"), ("<=", "&le;"), (">", "&gt;"), (">=", "&ge;")):
            sel = "selected" if val == current else ""
            out += f'<option value="{val}" {sel}>{sym}</option>'
        return out
    return f"""
<div class="custom-piece-row" data-piece="{pid}">
    <input type="text" class="custom-eq-input small" id="piece-expr-{pid}" value="{p['expr']}" />
    <span class="piece-if">if</span>
    <input type="text" class="piece-num-input" id="piece-lo-{pid}" placeholder="-∞" value="{p['lo']}" />
    <select class="piece-op-select" id="piece-lo-op-{pid}">{op_opts(p['lo_op'])}</select>
    <span class="piece-x">x</span>
    <select class="piece-op-select" id="piece-hi-op-{pid}">{op_opts(p['hi_op'])}</select>
    <input type="text" class="piece-num-input" id="piece-hi-{pid}" placeholder="∞" value="{p['hi']}" />
    <button class="btn-remove-piece" id="remove-piece-{pid}">×</button>
</div>
"""

def render_custom_pieces():
    list_el    = get_id("custom-pieces-list")
    wrap_el    = get_id("custom-pieces-wrap")
    default_el = get_id("custom-default-expr")
    if not list_el:
        return
    pieces = custom_activation["pieces"]
    if pieces:
        wrap_el.classList.remove("hidden")
        default_el.classList.add("hidden")
        list_el.innerHTML = "".join(make_piece_html(p) for p in pieces)
        for p in pieces:
            bind_piece_events(p["id"])
    else:
        wrap_el.classList.add("hidden")
        default_el.classList.remove("hidden")
    window.setTimeout(create_proxy(lambda: redraw_arrows()), 60)

def find_piece(pid: int):
    return next((p for p in custom_activation["pieces"] if p["id"] == pid), None)

def bind_piece_events(pid: int):
    def bind(elid, evname, key, is_select=False):
        el = get_id(elid)
        if not el:
            return
        def h(evt):
            p = find_piece(pid)
            if p:
                p[key] = evt.target.value
        el.addEventListener(evname, create_proxy(h))

    bind(f"piece-expr-{pid}",   "input",  "expr")
    bind(f"piece-lo-{pid}",     "input",  "lo")
    bind(f"piece-hi-{pid}",     "input",  "hi")
    bind(f"piece-lo-op-{pid}",  "change", "lo_op")
    bind(f"piece-hi-op-{pid}",  "change", "hi_op")

    rm = get_id(f"remove-piece-{pid}")
    if rm:
        def h(evt):
            remove_piece(pid)
        rm.addEventListener("click", create_proxy(h))

def add_piece(evt=None):
    global piece_counter
    piece_counter += 1
    seed_expr = "x"
    if not custom_activation["pieces"]:
        default_el = get_id("custom-default-expr")
        seed_expr = default_el.value if default_el else custom_activation["expr"]
    custom_activation["pieces"].append({
        "id": piece_counter, "expr": seed_expr,
        "lo": "", "lo_op": "<", "hi": "", "hi_op": "<",
    })
    render_custom_pieces()

def remove_piece(pid: int):
    custom_activation["pieces"] = [p for p in custom_activation["pieces"] if p["id"] != pid]
    render_custom_pieces()

def bind_custom_box_events():
    default_el = get_id("custom-default-expr")
    if default_el:
        def h(evt):
            custom_activation["expr"] = evt.target.value
        default_el.addEventListener("input", create_proxy(h))

    add_btn = get_id("add-piece-btn")
    if add_btn:
        add_btn.addEventListener("click", create_proxy(add_piece))

def on_act_select_change(evt):
    val = evt.target.value
    box = get_id("custom-act-box")
    if not box:
        return
    is_custom = (val == "custom")
    box.classList.toggle("hidden", not is_custom)
    get_id("act-box").classList.toggle("has-custom", is_custom)
    get_id("act-col").classList.toggle("act-col-custom", is_custom)
    window.setTimeout(create_proxy(lambda: redraw_arrows()), 60)

def sync_all_eq_fields():
    for row in rows:
        rid       = row["id"]
        container = get_id(f"eq-inline-{rid}")
        if not container:
            continue
        while len(row["weights"]) < len(rows):
            row["weights"].append(1.0)
        html = ""
        for i, r in enumerate(rows):
            coeff_val = row["weights"][i]
            if i > 0:
                html += '<span class="eq-op">+</span>'
            html += (
                f'<input type="number" step="any" value="{coeff_val:.2f}"'
                f' class="eq-num-input" id="coeff-{rid}-{i}"'
                f' data-row="{rid}" data-idx="{i}" />'
                f'<span class="eq-var" id="var-label-{rid}-{i}">{r["name"]}</span>'
            )
        html += (
            f'<span class="eq-op">+</span>'
            f'<input type="number" step="any" value="{row["bias"]:.2f}"'
            f' class="eq-bias-input" id="bias-{rid}" data-row="{rid}" />'
        )
        container.innerHTML = html
        bind_eq_inputs(rid)

def sync_var_labels():
    for i, r in enumerate(rows):
        label = r["name"]
        for row in rows:
            el = get_id(f"var-label-{row['id']}-{i}")
            if el:
                el.textContent = label

# ── Event binding ──────────────────────────────────────────────────────────────

def bind_eq_inputs(rid: int):
    row = row_by_id(rid)
    if not row:
        return
    for i in range(len(rows)):
        inp = get_id(f"coeff-{rid}-{i}")
        if inp:
            def make_ch(r, idx):
                def h(evt):
                    try:
                        r["weights"][idx] = float(evt.target.value)
                    except (ValueError, TypeError):
                        pass
                return create_proxy(h)
            inp.addEventListener("input", make_ch(row, i))
    bias_inp = get_id(f"bias-{rid}")
    if bias_inp:
        def make_bh(r):
            def h(evt):
                try:
                    r["bias"] = float(evt.target.value)
                except (ValueError, TypeError):
                    pass
            return create_proxy(h)
        bias_inp.addEventListener("input", make_bh(row))

def on_channel_dropdown_change(evt):
    sel = evt.target
    sel_id = sel.id
    channel = sel.value

    if sel_id.startswith("chan-in-"):
        rid = int(sel_id[len("chan-in-"):])
        dev_sel = get_id(f"dev-in-{rid}")
        plot_id = f"plot-in-{rid}"
    elif sel_id.startswith("chan-out-"):
        rid = int(sel_id[len("chan-out-"):])
        dev_sel = get_id(f"dev-out-{rid}")
        plot_id = f"plot-out-{rid}"
    else:
        return

    if not dev_sel:
        return

    dev_name = dev_sel.value
    row = row_by_id(rid)
    if not row:
        return

    plot_obj = all_plots.get(plot_id)

    # Determine which prev keys to use based on side
    prev_dev_key = "prev_in_device" if plot_id.startswith("plot-in-") else "prev_out_device"
    prev_chan_key = "prev_in_channel" if plot_id.startswith("plot-in-") else "prev_out_channel"

    # Remove from previous device
    prev_dev_name = row.get(prev_dev_key)
    if prev_dev_name:
        prev_dev = device_by_name(prev_dev_name)
        if prev_dev and plot_obj and plot_obj in prev_dev.plots:
            idx = prev_dev.plots.index(plot_obj)
            prev_dev.plots.pop(idx)
            if idx < len(prev_dev.plot_vars):
                prev_dev.plot_vars.pop(idx)

    # Add to new device
    new_dev = device_by_name(dev_name)
    if new_dev and channel and plot_obj:
        new_dev.plots.append(plot_obj)
        new_dev.plot_vars.append(channel)

    row[prev_dev_key] = dev_name
    row[prev_chan_key] = channel

def bind_row_events(rid: int):
    row = row_by_id(rid)
    if not row:
        return

    name_el = get_id(f"name-{rid}")
    if name_el:
        def make_nh(r):
            def h(evt):
                new = evt.target.value.strip() or r["name"]
                r["name"] = new
                sync_var_labels()
            return create_proxy(h)
        name_el.addEventListener("input", make_nh(row))

    del_btn = get_id(f"del-{rid}")
    if del_btn:
        def make_dh(r):
            def h(evt):
                delete_row(r["id"])
            return create_proxy(h)
        del_btn.addEventListener("click", make_dh(row))

    bind_eq_inputs(rid)
    
    handler = create_proxy(on_device_dropdown_change)
    for pfx in ("dev-in-", "dev-out-"):
        sel = get_id(f"{pfx}{rid}")
        if sel:
            sel.addEventListener("change", handler)

    chan_handler = create_proxy(on_channel_dropdown_change)
    for chan_id in (f"chan-in-{rid}", f"chan-out-{rid}"):
        chan_el = get_id(chan_id)
        if chan_el:
            chan_el.addEventListener("change", chan_handler)

def on_device_dropdown_change(evt):
    sel = evt.target
    sel_id = sel.id
    dev_name = sel.value

    if sel_id.startswith("dev-in-"):
        rid = int(sel_id[len("dev-in-"):])
        chan_id = f"chan-in-{rid}"
        plot_id = f"plot-in-{rid}"
        prev_dev_key = "prev_in_device"
        prev_chan_key = "prev_in_channel"
    elif sel_id.startswith("dev-out-"):
        rid = int(sel_id[len("dev-out-"):])
        chan_id = f"chan-out-{rid}"
        plot_id = f"plot-out-{rid}"
        prev_dev_key = "prev_out_device"
        prev_chan_key = "prev_out_channel"
    else:
        return

    chan_sel = get_id(chan_id)
    if not chan_sel:
        return

    matched = next((d for d in devices if d.name == dev_name), None)
    chan_sel.innerHTML = get_in_channels_html(matched) if chan_id.startswith("chan-in") else get_out_channels_html(matched)

    row = row_by_id(rid)
    if not row:
        return

    plot_obj = all_plots.get(plot_id)

    # Remove from previous device
    prev_dev_name = row.get(prev_dev_key)
    if prev_dev_name:
        prev_dev = device_by_name(prev_dev_name)
        if prev_dev and plot_obj and plot_obj in prev_dev.plots:
            idx = prev_dev.plots.index(plot_obj)
            prev_dev.plots.pop(idx)
            if idx < len(prev_dev.plot_vars):
                prev_dev.plot_vars.pop(idx)

    # Add to new device using first channel as default
    if matched and plot_obj:
        first_channel = chan_sel.options.item(0).value if chan_sel.options.length > 0 else None
        if first_channel:
            matched.plots.append(plot_obj)
            matched.plot_vars.append(first_channel)

    row[prev_dev_key] = dev_name
    row[prev_chan_key] = chan_sel.options.item(0).value if chan_sel.options.length > 0 else None
# ── Row CRUD ───────────────────────────────────────────────────────────────────

def add_row(evt=None):
    global row_counter
    row_counter += 1
    rid = row_counter
    n   = len(rows) + 1
    name = chr((ord('x') - ord('a') + (n - 1)) % 26 + ord('a'))
    row = {
        "id":          rid,
        "name":        name,
        "weights":     [1.0] * n ,
        "bias":        0.0,
        "input_data":  [],
        "output_data": [],
    }
    rows.append(row)

    left_container = get_id("rows-container")
    lw = document.createElement("div")
    lw.innerHTML = make_left_row_html(row)
    left_container.appendChild(lw.firstElementChild)

    right_container = get_id("rows-out-container")
    rw = document.createElement("div")
    rw.innerHTML = make_right_row_html(row)
    right_container.appendChild(rw.firstElementChild)

    bind_row_events(rid)

    def make_plots(rid=rid):
        all_plots[f"plot-in-{rid}"]  = plot.plot(f"plot-in-{rid}")
        all_plots[f"plot-out-{rid}"] = plot.plot(f"plot-out-{rid}")
    
    window.setTimeout(create_proxy(make_plots), 60)
    sync_all_eq_fields()
    window.setTimeout(create_proxy(lambda: redraw_arrows()), 60)


def delete_row(rid: int):
    global rows
    remove_row_plots(rid)

    for suffix in ("left", "right"):
        el = get_id(f"row-{suffix}-{rid}")
        if el:
            el.remove()

    idx = next((i for i, r in enumerate(rows) if r["id"] == rid), None)
    if idx is None:
        return

    rows = [r for r in rows if r["id"] != rid]

    for r in rows:
        if idx < len(r["weights"]):
            r["weights"].pop(idx)

    sync_all_eq_fields()
    window.setTimeout(create_proxy(lambda: redraw_arrows()), 60)

def remove_row_plots(rid: int):
    """Detach this row's plots from their devices' plots/plot_vars lists,
    and drop them from all_plots since the row no longer exists."""
    row = row_by_id(rid)
    if not row:
        return

    for side in ("in", "out"):
        plot_id = f"plot-{side}-{rid}"
        plot_obj = all_plots.get(plot_id)
        prev_dev_name = row.get(f"prev_{side}_device")

        if prev_dev_name and plot_obj:
            dev = device_by_name(prev_dev_name)
            if dev and plot_obj in dev.plots:
                idx = dev.plots.index(plot_obj)
                dev.plots.pop(idx)
                if idx < len(dev.plot_vars):
                    dev.plot_vars.pop(idx)

        all_plots.pop(plot_id, None)

# ── Device management ──────────────────────────────────────────────────────────

async def add_device_chip(dev: Element):
    """Insert a chip for `name` above the connect button."""
    dl  = get_id("device-list")
    btn = get_id("add-device-btn")
    
    name = dev.myble.device.name
    chip = document.createElement("div")
    chip.className = "device-row"
    chip.id = f"chip-{name}"
    chip.innerHTML = (
        f'<div class="device-indicator"></div>'
        f'<span class="device-name">{name}</span>'
        f'<button class="btn-disconnect" title="Disconnect">'
        f'    <svg width="11" height="11" viewBox="0 0 24 24" fill="none"'
        f'         stroke="currentColor" stroke-width="2.5">'
        f'        <path d="M18 6L6 18M6 6l12 12"/>'
        f'    </svg>'
        f'</button>'
    )

    disc = chip.querySelector(".btn-disconnect")

    async def make_disc(dev):
        async def handler(evt):
            chip_el = get_id(f"chip-{dev.myble.device.name}")
            if chip_el:
                chip_el.remove()
            await dev.disconnect()
            devices.remove(dev)
            refresh_device_dropdowns()
        return create_proxy(handler)

    disc.addEventListener("click", await make_disc(dev))

    # Insert before the button so chips stack above it
    dl.insertBefore(chip, btn)
    refresh_device_dropdowns()


def refresh_device_dropdowns():
    dev_opts = get_device_options_html()
    for row in rows:
        rid = row["id"]
        for pfx in ("dev-in-", "dev-out-"):
            sel = get_id(f"{pfx}{rid}")
            if sel:
                cur = sel.value
                sel.innerHTML = dev_opts
                sel.value = cur

async def create_new_device(evt=None):
    new_dev = Element()
    await new_dev.connect()
    print("out of connect")
    if not new_dev.hub or not new_dev.hub.connected:
        return
    devices.append(new_dev)
    await add_device_chip(new_dev)
    refresh_device_dropdowns()
    

# ── Play / Stop ────────────────────────────────────────────────────────────────

def play_network(evt=None):
    global is_running
    is_running = True
    get_id("play-btn").setAttribute("disabled", "")
    get_id("stop-btn").removeAttribute("disabled")
    document.body.classList.add("running")
    asyncio.ensure_future(loop_network())

def stop_network(evt=None):
    global is_running
    is_running = False
    get_id("stop-btn").setAttribute("disabled", "")
    get_id("play-btn").removeAttribute("disabled")
    document.body.classList.remove("running")
    for device in devices:
        try:
            device.stop()
        except Exception:
            print("Caught: " + e)

async def loop_network():
    while is_running:
        forward()
        if debug_mode:
            redraw_arrows()
        await asyncio.sleep(0.05)

def forward():
    # 1. Collect input values for every row
    input_vals = []
    for row in rows:
        rid = row["id"]
        dev_name = get_id(f"dev-in-{rid}").value
        channel  = get_id(f"chan-in-{rid}").value
        dev = device_by_name(dev_name)
        try:
            val = float(dev.state[channel]) if dev and channel else 0.0
        except (KeyError, TypeError, ValueError):
            val = 0.0
        input_vals.append(val)
        input_values[rid] = val          # NEW

    # 2. Compute each row's output
    act_fn = get_id("act-select").value
    for row in rows:
        rid = row["id"]
        weighted = sum(row["weights"][i] * input_vals[i] for i in range(len(input_vals)))
        pre_act  = weighted + row["bias"]
        eq_values[rid] = pre_act          # NEW

        post_act = apply_activation(pre_act, act_fn)
        output_values[rid] = post_act     # NEW

        result = int(post_act)
        print("result is: " + str(result))
        run_output(get_id(f"chan-out-{rid}").value, get_id(f"dev-out-{rid}").value, result)

def run_output(variable, dev_name, value):
    device = device_by_name(dev_name)
    if value > 100:
        value = 100
    elif value < -100:
        value = -100
    if variable == "Speed":
        device.set_speed(value)
    elif variable == "LeftSpeed":
        device.set_speedL(value)
    elif variable == "RightSpeed":
        device.set_speedR(value)
    elif variable == "BothSpeed":
        device.set_speed(value)
    else:
        print("Cannot set " + variable)
    
# ── Activation help popover ────────────────────────────────────────────────────

def open_act_help(evt=None):
    get_id("act-help-popover").classList.remove("hidden")

def close_act_help(evt=None):
    get_id("act-help-popover").classList.add("hidden")

# ── Resize observer ────────────────────────────────────────────────────────────

def setup_resize_observer():
    wrap = get_id("network-wrap")
    if not wrap:
        return
    def on_resize(entries, observer):
        redraw_arrows()
    observer = window.ResizeObserver.new(create_proxy(on_resize))
    observer.observe(wrap)

# ── Static wiring ──────────────────────────────────────────────────────────────

@when("click", "#add-device-btn")
async def _on_add_device(evt):
    await create_new_device()

@when("click", "#add-row-btn")
def _on_add_row(evt):
    add_row()

@when("click", "#play-btn")
def _on_play(evt):
    play_network()

@when("click", "#stop-btn")
def _on_stop(evt):
    stop_network()

@when("click", "#act-help-btn")
def _on_act_help(evt):
    open_act_help()

@when("click", "#close-act-help-btn")
def _on_close_act_help(evt):
    close_act_help()

@when("change", "#debug-toggle")
def _on_debug_toggle(evt):
    global debug_mode
    debug_mode = evt.target.checked
    redraw_arrows()

# ── Boot ───────────────────────────────────────────────────────────────────────

def boot():
    get_id("loading-splash").style.display = "none"
    get_id("page-wrap").style.display = "flex"

    populate_act_select()
    get_id("act-select").addEventListener("change", create_proxy(on_act_select_change))
    bind_custom_box_events()

    add_row()
    setup_resize_observer()
    window.setTimeout(create_proxy(lambda: redraw_arrows()), 120)

boot()