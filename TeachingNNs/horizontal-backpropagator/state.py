"""Module-level state shared across the horizontal-backpropagator UI.

Every other module reaches into this one via `import state` and reads/writes
state.x (never `from state import x`) so mutations are visible everywhere --
PyScript's cross-module rebinding only works through attribute access.
"""
from pyscript import document

# Each layer is a single neuron: {id, w, b, act}. The chain is strictly
# linear -- layer i's only input is layer (i-1)'s activation (or the raw
# input x for layer 0).
layers: list[dict] = []
layer_counter = 0

biases_enabled = False
lr = 0.1

# Dataset: [{"id": int, "x": float, "y": float}, ...]
dataset: list[dict] = []
data_counter = 0

ACTIVATION_OPTIONS = [
    ("None", "none"),
    ("ReLU", "relu"),
    ("Sigmoid", "sigmoid"),
    ("Tanh", "tanh"),
    ("Softplus", "softplus"),
]

# ── Training state machine ───────────────────────────────────────────────
# step_index 0 == ready to run a forward pass. step_index k (1..len(layers))
# means the backward reveal has been applied through plan[k-1] (plan is
# ordered from the LAST layer to the FIRST, since backprop runs right to
# left). step_index wraps back to 0 once every layer's update is revealed.
epoch = 0
step_index = 0
plan: list[dict] = []          # this epoch's full backward plan, computed once at step 0
forward_cache: dict | None = None
loss_history: list[tuple[int, float]] = []
initialized = False
playing = False

# Snapshot stack for backward step/epoch (see network_model.take_snapshot).
history: list[dict] = []

INIT_WEIGHT_RANGE = (-1.0, 1.0)
RANDOM_POINT_X_RANGE = (-3.0, 3.0)
RANDOM_POINT_Y_RANGE = (-3.0, 3.0)


def get_id(id_: str):
    return document.getElementById(id_)


def next_layer_id() -> int:
    global layer_counter
    layer_counter += 1
    return layer_counter


def next_data_id() -> int:
    global data_counter
    data_counter += 1
    return data_counter
