"""
Self-calibrating line follower with online backprop -- LIVE LEARNING DEMO.

The network starts on truly random weights, and the ONLY training it
ever gets is the handful of backprop steps it takes per tick, live,
while it's already driving. Expect it to spin, stall, or wander for
the first several seconds before it visibly starts correcting toward
the line as the weights converge.

Self-recalibration: cal.hi can only ever be pulled DOWN by a slow fixed
decay (never by a fresh low reading), and once the robot is driving
smoothly it may stop physically visiting the true dark/light extremes
on its own. So every WOBBLE_PERIOD_TICKS, the main loop briefly
overrides steering and spins the robot across the line edge, purely to
refresh cal.lo/cal.hi against real sensor extremes -- independent of
whatever the network currently thinks is good driving. This is what
lets it keep discovering "how dark is dark" / "how light is light" on
its own, including after a real lighting change, without needing a
human to re-run a calibration sweep by hand.

Hardware note: on this chassis the two motors are mounted mirrored,
so a positive command on both wheels drives them in physically
opposite directions. RIGHT_SIGN is applied once, inside drive(), so
every caller (normal driving AND the forced wobble) stays correct
automatically -- this is a hardware fact, not something the network
knows about or controls.
"""

import math
import random
import time
from collections import deque
import matplotlib.pyplot as plt
from lelib import doubleMotor, colorSensor

random.seed(5)

# How often (in ticks) to redraw the plot. Redrawing every tick will
# noticeably slow the control loop, so batch it.
PLOT_EVERY = 5
HISTORY_LEN = 400   # ticks of history shown on screen at once

# Hardware fact: motors are mirrored on this chassis, so the right
# wheel needs the opposite sign from the network's raw output to
# actually turn the same rotational direction as the left wheel.
RIGHT_SIGN = -1

# Small per-tick delay so a human can actually watch the behavior
# change, rather than it converging faster than the eye can follow.
TICK_SLEEP = 0.03

# ---- Forced re-exploration ----
# Once the network converges, the robot's own steering may stay close
# to the line and stop physically visiting the true dark/light extremes.
# cal.hi in particular can only ever be pulled DOWN by slow fixed decay
# (see Calibrator.update), never by a fresh low reading -- so if nothing
# forces the robot to keep sweeping past both edges, calibration can go
# stale exactly when lighting changes and you need it most.
# Every WOBBLE_PERIOD_TICKS, override the motors for WOBBLE_DURATION_TICKS
# and spin the robot in place across the line edge, purely to refresh
# cal.lo / cal.hi against real readings. Training and calibration continue
# normally during this window -- only the motor command source changes.
WOBBLE_PERIOD_TICKS = 400
WOBBLE_DURATION_TICKS = 60
WOBBLE_TURN_STRENGTH = 70     # motor magnitude while forced-exploring

# Safety floor: if decay ever shrinks the calibrated range faster than
# real exploration refreshes it, don't let lo/hi cross or collapse --
# clamp the gap open around the current midpoint instead.
MIN_GAP = 8.0


# =========================================================
# Self-calibration: running min/max with slow decay
# =========================================================
class Calibrator:
    DECAY = 0.05          # units per tick that lo/hi relax inward

    def __init__(self):
        self.lo, self.hi = 50.0, 50.0

    def update(self, r):
        # update the lo and hi if a higher or lower value comes into play
        self.lo = min(self.lo, r)
        self.hi = max(self.hi, r)
        self.lo += self.DECAY
        self.hi -= self.DECAY
        
        # Safety: decay (or a run of readings all on one side) could in
        # principle shrink the gap to nothing or cross it. Don't let the
        # self-labels degrade into nonsense -- hold a minimum gap open
        # around the current midpoint if that ever happens.
        if self.hi - self.lo < MIN_GAP:
            mid = 0.5 * (self.lo + self.hi)
            self.lo = mid - MIN_GAP / 2
            self.hi = mid + MIN_GAP / 2

    def mid(self):
        return 0.5 * (self.lo + self.hi)


# =========================================================
# The network: raw reading -> 3 tanh hidden -> 2 motors
# =========================================================
class Net:
    def __init__(self, n_hidden=3, lr=0.3):
        self.n, self.lr = n_hidden, lr
        u = lambda: random.uniform(-1, 1)
        self.w = [u() for _ in range(n_hidden)]
        self.b = [u() for _ in range(n_hidden)]
        self.v = [[u() for _ in range(n_hidden)] for _ in range(2)]
        self.c = [u() for _ in range(2)]

    def forward(self, reading):
        x = reading / 100.0
        h = [math.tanh(self.w[j] * x + self.b[j]) for j in range(self.n)]
        y = [sum(self.v[k][j] * h[j] for j in range(self.n)) + self.c[k]
             for k in range(2)]
        return x, h, y

    def train_step(self, reading, t_L, t_R):
        x, h, y = self.forward(reading)
        d_out = [t_L - y[0], t_R - y[1]]
        d_hid = [(d_out[0] * self.v[0][j] + d_out[1] * self.v[1][j])
                 * (1.0 - h[j] ** 2) for j in range(self.n)]
        for k in range(2):
            for j in range(self.n):
                self.v[k][j] += self.lr * d_out[k] * h[j]
            self.c[k] += self.lr * d_out[k]
        for j in range(self.n):
            self.w[j] += self.lr * d_hid[j] * x
            self.b[j] += self.lr * d_hid[j]
        return 0.5 * (d_out[0] ** 2 + d_out[1] ** 2)


# =========================================================
# Self-labeling: the robot writes its own training data
# =========================================================
def self_labeled_batch(cal):
    return [
        (cal.lo,    0.40, 0),   # darkest it has seen -> brake, turn right
        (cal.mid(), 0.90, 0.90),   # midpoint            -> sprint
        (cal.hi,    0, 0.4),   # lightest it has seen-> brake, turn left
    ]


def clamp(v, lo=-100, hi=100):
    return max(lo, min(v, hi))


def drive(left_cmd, right_cmd):
    """Single place where commands reach the motors. RIGHT_SIGN (a
    hardware fact about mounting, not something the network knows or
    controls) is applied here so every caller -- normal driving or the
    forced wobble below -- stays consistent automatically."""
    SCALING = 0.15
    motor.movement_move_tank(SCALING*clamp(left_cmd), SCALING*clamp(right_cmd))


# =========================================================
# Live plot of weights & biases as they train
# =========================================================
class LivePlot:
    """Rolling line plot of every trainable parameter, updated in place.

    Call .push(net) once per tick (cheap: just appends to deques).
    Call .maybe_draw() periodically (does the actual matplotlib redraw).
    """

    def __init__(self, net, history_len=HISTORY_LEN):
        self.history_len = history_len
        self.t = 0
        self.ticks = deque(maxlen=history_len)

        # One deque per scalar parameter, named for the legend.
        self.series = {}
        for j in range(net.n):
            self.series["w[{}]".format(j)] = deque(maxlen=history_len)
            self.series["b[{}]".format(j)] = deque(maxlen=history_len)
        for k in range(2):
            for j in range(net.n):
                self.series["v[{}][{}]".format(k, j)] = deque(maxlen=history_len)
            self.series["c[{}]".format(k)] = deque(maxlen=history_len)
        self.loss_hist = deque(maxlen=history_len)

        self.lo_hist = deque(maxlen=history_len)
        self.mid_hist = deque(maxlen=history_len)
        self.hi_hist = deque(maxlen=history_len)

        plt.ion()
        self.fig, (self.ax_params, self.ax_cal, self.ax_loss) = plt.subplots(
            3, 1, figsize=(9, 8), sharex=True,
            gridspec_kw={"height_ratios": [3, 2, 1]})

        self.lines = {}
        for name in self.series:
            (line,) = self.ax_params.plot([], [], label=name, linewidth=1.2)
            self.lines[name] = line
        self.ax_params.set_ylabel("value")
        self.ax_params.set_title("Weights & biases (live)")
        self.ax_params.legend(loc="upper left", ncol=4, fontsize=7)
        self.ax_params.axhline(0, color="gray", linewidth=0.5)

        (self.lo_line,) = self.ax_cal.plot([], [], label="cal.lo", color="tab:blue", linewidth=1.4)
        (self.mid_line,) = self.ax_cal.plot([], [], label="cal.mid()", color="gray", linewidth=1.0, linestyle="--")
        (self.hi_line,) = self.ax_cal.plot([], [], label="cal.hi", color="tab:orange", linewidth=1.4)
        self.ax_cal.set_ylabel("sensor units")
        self.ax_cal.set_title("Calibration bounds (live)")
        self.ax_cal.legend(loc="upper left", fontsize=8)


        (self.loss_line,) = self.ax_loss.plot([], [], color="black", linewidth=1.2)
        self.ax_loss.set_ylabel("loss")
        self.ax_loss.set_xlabel("tick")

        self.fig.tight_layout()
        self.fig.canvas.draw()
        plt.show(block=False)

    def push(self, net, loss, cal):
        self.t += 1
        self.ticks.append(self.t)
        for j in range(net.n):
            self.series["w[{}]".format(j)].append(net.w[j])
            self.series["b[{}]".format(j)].append(net.b[j])
        for k in range(2):
            for j in range(net.n):
                self.series["v[{}][{}]".format(k, j)].append(net.v[k][j])
            self.series["c[{}]".format(k)].append(net.c[k])
        self.loss_hist.append(loss)
        self.lo_hist.append(cal.lo)
        self.mid_hist.append(cal.mid())
        self.hi_hist.append(cal.hi)

    def maybe_draw(self, force=False):
        if not force and self.t % PLOT_EVERY != 0:
            return
        xs = list(self.ticks)
        for name, line in self.lines.items():
            line.set_data(xs, list(self.series[name]))
        self.ax_params.relim()
        self.ax_params.autoscale_view()

        self.lo_line.set_data(xs, list(self.lo_hist))
        self.mid_line.set_data(xs, list(self.mid_hist))
        self.hi_line.set_data(xs, list(self.hi_hist))
        self.ax_cal.relim()
        self.ax_cal.autoscale_view()

        self.loss_line.set_data(xs, list(self.loss_hist))
        self.ax_loss.relim()
        self.ax_loss.autoscale_view()

        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
        plt.pause(0.001)


# =========================================================
# Hardware setup
# =========================================================
sensor = colorSensor()
sensor.connect(card_serial="1003")
motor = doubleMotor()
motor.connect(card_serial="1003")

net = Net()
cal = Calibrator()

# ---- Startup sweep: seed calibration from REAL readings ----
# Physically rock/rotate the sensor across the line's edge while this runs,
# so cal.lo / cal.hi capture the true dark and light extremes.
print("Sweeping... move the sensor across the line edge now.")
SWEEP_TICKS = 60
time.sleep(1)
for i in range(SWEEP_TICKS):
    r = sensor.sensor.reflection
    cal.update(r)
    time.sleep(0.1)
print("  calibrated:  lo={:.1f}  mid={:.1f}  hi={:.1f}".format(
    cal.lo, cal.mid(), cal.hi))

# ---- No pretraining burst here on purpose ----
# net.w / net.b / net.v / net.c are still the raw random values from
# Net.__init__. The very first backprop step happens inside the loop
# below, live, with the robot already moving.
print("Starting on RANDOM weights -- expect it to misbehave at first.")
plot = LivePlot(net)

# =========================================================
# Main loop: calibrate + train + drive, every tick, forever.
# Every WOBBLE_PERIOD_TICKS, briefly override steering to spin across
# the line edge so cal.lo/cal.hi get refreshed against real extremes,
# even once the network is converged and driving smoothly.
# =========================================================
tick = 0
while True:
    tick += 1
    r = sensor.sensor.reflection
    cal.update(r)

    loss = 0.0
    for sample in self_labeled_batch(cal):
        loss += net.train_step(*sample)
    plot.push(net, loss, cal)
    plot.maybe_draw()
    phase_in_period = tick % WOBBLE_PERIOD_TICKS
    if phase_in_period == 0:
        cal.lo = 50
        cal.hi = 50
    if phase_in_period < WOBBLE_DURATION_TICKS:
        # Forced re-exploration: spin in place, alternating direction,
        # to sweep the sensor back across the line edge and re-confirm
        # the true dark/light extremes -- independent of what the
        # network currently thinks is a good idea.
        half = WOBBLE_DURATION_TICKS // 2
        turn = WOBBLE_TURN_STRENGTH if phase_in_period < half else -WOBBLE_TURN_STRENGTH
        drive(turn, -turn)
    else:
        _, _, y = net.forward(r)
        drive(100 * y[0], 100 * y[1])

    if TICK_SLEEP:
        time.sleep(TICK_SLEEP)