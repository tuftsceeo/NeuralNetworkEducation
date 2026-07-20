"""The step machine: forward pass -> one backward-reveal step per layer
(right to left) -> epoch complete. Mirrors gradient-descent-visualization's
do_step()/run_epoch()/play_loop() pattern, generalized to N layers."""
import asyncio

import state
import network_model
import diagram_render
import plots

step_btn = state.get_id("step-btn")
epoch_btn = state.get_id("epoch-btn")
back_step_btn = state.get_id("back-step-btn")
back_epoch_btn = state.get_id("back-epoch-btn")
play_pause_btn = state.get_id("play-pause-btn")


def ensure_initialized() -> bool:
    return bool(state.layers) and bool(state.dataset)


def update_back_button_states():
    can_go_back = len(state.history) > 1
    back_step_btn.disabled = not can_go_back
    back_epoch_btn.disabled = not can_go_back


def _clear_all_highlights():
    for layer in state.layers:
        diagram_render.highlight_layer(layer["id"], False)


def do_step():
    if not ensure_initialized():
        return

    if state.step_index == 0:
        state.forward_cache = network_model.compute_forward_cache()
        state.plan = network_model.build_backward_plan()
        diagram_render.clear_chain_rule_slots()
        diagram_render.render_output_readout()
        _clear_all_highlights()
        state.step_index = 1
    else:
        entry = network_model.apply_plan_step(state.step_index)
        diagram_render.render_chain_rule(entry)
        _clear_all_highlights()
        diagram_render.highlight_layer(entry["layer_id"], True)
        plots.update_fit_curve()

        if state.step_index >= len(state.plan):
            state.loss_history.append((state.epoch, state.forward_cache["mean_loss"]))
            plots.update_loss_plot()
            state.epoch += 1
            state.step_index = 0
        else:
            state.step_index += 1

    network_model.take_snapshot()
    update_back_button_states()


def run_epoch():
    if not ensure_initialized():
        return
    do_step()
    while state.step_index != 0:
        do_step()


def run_epoch_turbo():
    """Used while Play is running: does the whole epoch's math and applies
    every layer's update in one shot, rendering the result once instead of
    once per layer -- faster and less visually busy than stepping."""
    if not ensure_initialized():
        return

    state.forward_cache = network_model.compute_forward_cache()
    state.plan = network_model.build_backward_plan()
    for i in range(1, len(state.plan) + 1):
        network_model.apply_plan_step(i)

    diagram_render.clear_chain_rule_slots()
    diagram_render.render_output_readout()
    diagram_render.render_weight_badges()
    plots.update_fit_curve()

    state.loss_history.append((state.epoch, state.forward_cache["mean_loss"]))
    plots.update_loss_plot()
    state.epoch += 1
    state.step_index = 0

    network_model.take_snapshot()
    update_back_button_states()


def do_backward_step():
    if len(state.history) <= 1:
        return
    state.history.pop()
    snap = state.history[-1]
    network_model.restore_snapshot(snap)
    _clear_all_highlights()
    diagram_render.render_weight_badges()
    diagram_render.render_output_readout()
    diagram_render.clear_chain_rule_slots()
    if state.plan and 1 <= state.step_index <= len(state.plan):
        for i in range(1, state.step_index + 1):
            entry = state.plan[i - 1]
            diagram_render.render_chain_rule(entry)
        diagram_render.highlight_layer(state.plan[state.step_index - 1]["layer_id"], True)
    plots.update_fit_curve()
    plots.update_loss_plot()
    update_back_button_states()


def backward_epoch():
    if len(state.history) <= 1:
        return
    do_backward_step()
    while state.step_index != 0 and len(state.history) > 1:
        do_backward_step()


# ── Play / Pause ────────────────────────────────────────────────────────

PLAY_DELAY = 0.05
_play_task = None


async def play_loop():
    while state.playing:
        run_epoch_turbo()
        await asyncio.sleep(PLAY_DELAY)


def enable_training_controls(enabled: bool):
    step_btn.disabled = not enabled
    epoch_btn.disabled = not enabled


def start_playing():
    global _play_task
    if not ensure_initialized():
        return
    state.playing = True
    play_pause_btn.textContent = "❚❚"
    play_pause_btn.classList.add("is-playing")
    play_pause_btn.title = "Pause"
    _play_task = asyncio.ensure_future(play_loop())


def stop_playing():
    state.playing = False
    play_pause_btn.textContent = "▶"
    play_pause_btn.classList.remove("is-playing")
    play_pause_btn.title = "Play"


def on_play_pause_click(evt=None):
    if state.playing:
        stop_playing()
    else:
        start_playing()


def on_step_click(evt=None):
    if state.playing:
        stop_playing()
    do_step()


def on_epoch_click(evt=None):
    if state.playing:
        stop_playing()
    run_epoch()


def on_back_step_click(evt=None):
    if state.playing:
        stop_playing()
    do_backward_step()


def on_back_epoch_click(evt=None):
    if state.playing:
        stop_playing()
    backward_epoch()
