"""Builds and updates the top network-diagram panel: the neuron chain, the
weight badges on each connecting arrow, and the chain-rule labels that
appear underneath a weight the moment its gradient is revealed."""
import asyncio

from pyscript import document
from pyscript.ffi import create_proxy

import state
import network_model
from activations import ACTIVATION_SYMBOL

layers_track_el = state.get_id("layers-track")
output_readout_el = state.get_id("output-readout")

# A small curved arrow icon, always drawn the same way: it starts near the
# top right (where the neuron "ahead" of this weight sits, since gradients
# flow backward from output to input) and curves down-left into the
# chain-rule text below it -- a stylized stand-in for "this number came
# from the layer ahead of you."
CURVED_ARROW_SVG = (
    '<svg class="curved-grad-arrow" width="30" height="22" viewBox="0 0 30 22" '
    'fill="none" xmlns="http://www.w3.org/2000/svg">'
    '<path d="M27 3 C 27 13, 18 17, 4 17" stroke="#7c3aed" stroke-width="2" '
    'stroke-linecap="round" fill="none"/>'
    '<path d="M4 17 L9 14.5 M4 17 L8 20" stroke="#7c3aed" stroke-width="2" '
    'stroke-linecap="round"/>'
    '</svg>'
)


def make_el(tag, class_name=None, text=None, id_=None):
    e = document.createElement(tag)
    if class_name:
        e.className = class_name
    if text is not None:
        e.textContent = text
    if id_:
        e.id = id_
    return e


def fmt(x, n=3):
    try:
        return f"{x:.{n}f}"
    except (ValueError, TypeError):
        return str(x)


# ── Skeleton build (full rebuild whenever the topology changes) ──────────

def build_diagram():
    layers_track_el.innerHTML = ""
    for idx, layer in enumerate(state.layers):
        lid = layer["id"]
        pos = idx + 1

        layers_track_el.appendChild(make_el("div", "flow-arrow", text="→"))

        conn_cell = make_el("div", "conn-cell")
        weight_badge = make_el("div", "weight-badge", id_=f"weight-badge-{lid}")
        conn_cell.appendChild(weight_badge)
        conn_cell.appendChild(make_el("div", "chain-rule-slot", id_=f"chain-rule-{lid}"))
        layers_track_el.appendChild(conn_cell)

        neuron_box = make_el("div", "neuron-box", id_=f"neuron-box-{lid}")
        neuron_box.appendChild(make_el("div", "neuron-title", text=f"n{pos}"))

        select = document.createElement("select")
        select.className = "act-select"
        select.id = f"act-select-{lid}"
        for label, value in state.ACTIVATION_OPTIONS:
            opt = document.createElement("option")
            opt.value = value
            opt.textContent = label
            if value == layer["act"]:
                opt.selected = True
            select.appendChild(opt)
        neuron_box.appendChild(select)

        remove_btn = make_el("button", "btn-remove-neuron", text="×", id_=f"remove-layer-{lid}")
        remove_btn.title = "Remove this layer"
        neuron_box.appendChild(remove_btn)

        layers_track_el.appendChild(neuron_box)

        select.addEventListener("change", create_proxy(
            lambda evt, lid=lid: on_activation_change(lid, evt.target.value)))
        remove_btn.addEventListener("click", create_proxy(lambda evt, lid=lid: on_remove_layer(lid)))

    render_weight_badges()
    clear_chain_rule_slots()
    render_output_readout()


def on_activation_change(lid, value):
    network_model.set_layer_activation(lid, value)
    import main
    main.on_topology_changed()


def on_remove_layer(lid):
    network_model.remove_layer(lid)
    import main
    main.on_topology_changed()


# ── Live value rendering ──────────────────────────────────────────────────

def _weight_badge_html(layer, pos):
    html = f"w<sub>{pos}</sub> = <span class='w-color'>{fmt(layer['w'])}</span>"
    if state.biases_enabled:
        html += f"<br>b<sub>{pos}</sub> = <span class='b-color'>{fmt(layer['b'])}</span>"
    return html


def render_weight_badges():
    for idx, layer in enumerate(state.layers):
        badge = state.get_id(f"weight-badge-{layer['id']}")
        if badge:
            badge.innerHTML = _weight_badge_html(layer, idx + 1)


def render_output_readout():
    if output_readout_el is None:
        return
    if state.forward_cache is not None:
        output_readout_el.textContent = f"L = {fmt(state.forward_cache['mean_loss'])}"
    else:
        output_readout_el.textContent = "L = –"


def clear_chain_rule_slots():
    for layer in state.layers:
        slot = state.get_id(f"chain-rule-{layer['id']}")
        if slot:
            slot.innerHTML = ""


def render_chain_rule(entry):
    """Fills in the chain-rule label under the weight this plan entry
    updated, and kicks off the grow-shrink pulse on its weight badge."""
    pos = entry["layer_pos"]
    slot = state.get_id(f"chain-rule-{entry['layer_id']}")
    if slot:
        act_name = ACTIVATION_SYMBOL.get(entry["act"], "")
        upstream = "L" if entry["is_last"] else f"a_{{{pos + 1}}}"
        formula = f"dL/dw<sub>{pos}</sub> = dL/da<sub>{pos}</sub> · da<sub>{pos}</sub>/dw<sub>{pos}</sub>"
        nums = (f"= {fmt(entry['avg_grad_out'])} · {fmt(entry['avg_local'])} "
                f"≈ {fmt(entry['grad_w'])}")
        bias_line = ""
        if state.biases_enabled:
            bias_line = (
                f"<div class='chain-rule-formula'>dL/db<sub>{pos}</sub> = dL/da<sub>{pos}</sub> "
                f"· da<sub>{pos}</sub>/db<sub>{pos}</sub></div>"
                f"<div class='chain-rule-nums'>≈ {fmt(entry['grad_b'])}</div>"
            )
        slot.innerHTML = (
            f"<div class='chain-rule-inner'>{CURVED_ARROW_SVG}"
            f"<div class='chain-rule-body'>"
            f"<div class='chain-rule-formula'>{formula}</div>"
            f"<div class='chain-rule-nums'>{nums}</div>{bias_line}"
            f"</div></div>"
        )
    render_weight_badges()
    asyncio.ensure_future(pulse_weight(entry["layer_id"]))


PULSE_DURATION = 0.45


async def pulse_weight(lid):
    badge = state.get_id(f"weight-badge-{lid}")
    if badge is None:
        return
    badge.classList.add("weight-pulse")
    await asyncio.sleep(PULSE_DURATION)
    badge.classList.remove("weight-pulse")


def highlight_layer(lid, active: bool):
    box = state.get_id(f"neuron-box-{lid}")
    if box:
        if active:
            box.classList.add("neuron-active")
        else:
            box.classList.remove("neuron-active")
