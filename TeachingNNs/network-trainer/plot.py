from pyscript import document, window, when, Event
from pyscript.js_modules import Plotly
from collections import deque
import js
import json

class plot():
    def __init__(self, id):
        self.size = 100
        self.num = 12
        self.id = id
        self.buffer = []
        self.traces = js.JSON.parse(json.dumps([
            {'y': 0, 'mode': 'lines', 'name': 'Element Data', 'line': {'color': '#82ca9d'}}
        ]))
        for n in range(self.num):
            self.buffer.append(deque([], self.size))
        layout = js.JSON.parse(json.dumps({
            "margin": {"l": 20, "r": 10, "t": 10, "b": 20},
            "xaxis": {
                "title": {"font": {"size": 10}},
                "tickfont": {"size": 9},
                "showticklabels": False
            },
            "yaxis": {
                "tickfont": {"size": 9},
                "autorange": True
            },
            "showlegend": False
        }))
        Plotly.newPlot(self.id, self.traces, layout)

    def addPoints(self, num, data):
        lines = []
        num = (num - 1) * 2
        for i, d in enumerate(data):
            self.buffer[num + i].append(d)
            lines.append(list(self.buffer[num + i]))
        return js.JSON.parse(json.dumps({'y': lines}))

    def updatePlot(self, update):
        Plotly.update(self.id, update)


class scatter_plot():
    """Persistent (x, y) marker plot -- unlike `plot` above (a scrolling
    line over an implicit time axis), points here accumulate and stay put
    so a set of test inputs/targets can be compared against the network's
    fitted curve."""

    def __init__(self, id, max_points=300):
        self.id = id
        self.max_points = max_points
        self.xs = deque([], max_points)
        self.ys = deque([], max_points)
        self.traces = js.JSON.parse(json.dumps([
            {'x': [], 'y': [], 'mode': 'markers', 'type': 'scatter',
             'name': 'Test points', 'marker': {'color': '#82ca9d', 'size': 8}}
        ]))
        layout = js.JSON.parse(json.dumps({
            "margin": {"l": 28, "r": 10, "t": 10, "b": 22},
            "xaxis": {"tickfont": {"size": 9}, "zeroline": True},
            "yaxis": {"tickfont": {"size": 9}, "autorange": True, "zeroline": True},
            "showlegend": False
        }))
        Plotly.newPlot(self.id, self.traces, layout)

    def addPoint(self, x, y):
        self.xs.append(x)
        self.ys.append(y)
        return js.JSON.parse(json.dumps({'x': [list(self.xs)], 'y': [list(self.ys)]}))

    def addPoints(self, num, data):
        # Compatibility no-op: an output's device/channel hookup still routes
        # live hardware notifications here (see Device.py's notif_callback),
        # but this plot is meant to only move on explicit addPoint() calls
        # from Test(), so continuous notifications are ignored.
        return js.JSON.parse(json.dumps({}))

    def updatePlot(self, update):
        Plotly.update(self.id, update)

    def clear(self):
        self.xs.clear()
        self.ys.clear()
        self.updatePlot(js.JSON.parse(json.dumps({'x': [[]], 'y': [[]]})))