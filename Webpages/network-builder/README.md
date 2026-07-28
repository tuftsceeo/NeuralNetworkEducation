# Neural Network Builder

A hands-on webpage for visually building a neural network. You can add inputs, hidden layers, neurons, activation functions, and outputs, then connect it to LEGO CS + AI hardware over Bluetooth, so sensor readings become network inputs and computed outputs drive motors, lights, and sounds. It is designed to teach users how neural networks are structured in an approachable way

## Features

- **Devices card** — "Connect device" triggers a Web Bluetooth scan/pairing flow; connected LEGO devices (motors, color sensor, controller, etc.) appear as removable chips.
- **Live network diagram** — columns for Inputs, per-layer Neurons + Activation blocks, and Outputs, connected by dynamically drawn SVG arrows.
- **Editable topology** — add an Input, add a hidden Layer, add a Neuron to a layer, or add an Output; each item is deletable (deleting a layer's first neuron deletes the whole layer; delete buttons hide when only one item remains).
- **Editable neuron equations** — each neuron shows a live-editable weighted-sum equation (weight + bias inputs).
- **Activation functions per layer** — None, ReLU, Sigmoid, Tanh, Softplus, or Custom (via a piecewise expression editor with range-bound pieces), plus a "?" help popover explaining each function.
- **Device/channel binding** — each input/output node has a dropdown to bind it to a connected device's channel, with a live Plotly sparkline of its current value.
- **Zoom controls** to rescale the diagram, with automatic arrow realignment on resize.
- **Play/Stop** — runs or stops the network's live forward-pass loop; a **Debug** toggle overlays live numeric values on the diagram's arrows.

## Files

| Path | Purpose |
|---|---|
| `index.html` | Page skeleton, styling/font loading, activation-help popover markup; boots `main.py`. |
| `main.py` | Wires all click/change event handlers and boots the initial 1-input/1-layer/1-output network. |
| `Device.py` | Hardware abstraction layer for LEGO devices — BLE connect/disconnect, per-device-type state parsing, output routing, and WASM/Pyodide compatibility patches for the `legoeducation` library. |
| `plot.py` | Thin Plotly.js wrapper for rolling-buffer input/output sparklines. |
| `glue/state.py` | Single source of truth for topology, counters, zoom/debug flags, activation options. |
| `glue/bindings.py` | Attaches DOM event listeners (weights, biases, device/channel selects, delete buttons) to model items. |
| `glue/sync.py` | Keeps the DOM in sync with state after add/remove operations. |
| `network/network_model.py` | Topology lookups and the core `forward()` pass (reads devices → propagates layers → routes outputs). |
| `network/network_actions.py` | CRUD for inputs/neurons/layers/outputs, plus the Play/Stop async loop. |
| `network/activations.py` | Activation math (ReLU/Sigmoid/Tanh/Softplus) and a safe custom-expression evaluator. |
| `network/activation_editor.py` | UI logic for the activation dropdown and custom piecewise-function editor. |
| `misc_ui/templates.py` | HTML string builders for input/neuron/output/layer DOM blocks. |
| `misc_ui/arrows.py` | SVG arrow-drawing engine (including fan-out arrows between layers and debug value labels) and full diagram redraw. |
| `misc_ui/zoom.py` | Zoom in/out and `ResizeObserver`-based arrow realignment. |
| `JS/ble.js` | Web Bluetooth wrapper class (`BLEDevice`) for scan/connect/notify/send/disconnect. |
| `JS/plotly.js` | PyScript JS-module loader/re-export for Plotly. |
| `styles.css` | All visual styling — cards, diagram, nodes, arrows, activation editor, popovers, zoom. |