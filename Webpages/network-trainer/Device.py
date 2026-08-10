import asyncio
import threading
import concurrent.futures
import legoeducation as le
import legoeducation.background_worker as bw
import legoeducation.basic_device as bd
from pyscript import document
from pyscript.js_modules import ble
from pyscript.ffi import create_proxy
import json

import state

def _nowait_event_wait(self, timeout=None):
    return self._flag
threading.Event.wait = _nowait_event_wait

def _nowait_future_result(self, timeout=None):
    if self._state == 'FINISHED':
        return self._Future__get_result()
    return None
concurrent.futures.Future.result = _nowait_future_result

_worker_started = False

def _wasm_start_thread(self):
    global _worker_started
    if _worker_started:
        return
    _worker_started = True
    self.loop = asyncio.get_event_loop()
    self.loop_ready.set()
    asyncio.ensure_future(_wasm_worker_loop(self))

def _wasm_put_request(self, request):
    asyncio.ensure_future(self.async_put_request(request))

bw.Worker.start_thread = _wasm_start_thread
bw.Worker.put_request  = _wasm_put_request

async def _wasm_worker_loop(worker):
    worker._js_ble_registry = {}
    while True:
        try:
            req = await worker.request_queue.get()
            if req is None:
                break
            topic = req.get('topic')
            if topic == 'send':
                device  = req.get('msg')
                message = req.get('msg2')
                js_ble  = worker._js_ble_registry.get(id(device))
                if js_ble is not None and message is not None:
                    try:
                        await js_ble.send(list(message))
                    except Exception as e:
                        print(f"BLE send error: {e}")
            elif topic == 'connect':
                cb = req.get('msg3')
                if cb:
                    cb(True)
            elif topic == 'disconnect':
                device = req.get('msg')
                js_ble = worker._js_ble_registry.pop(id(device), None)
                if js_ble is not None:
                    js_ble.disconnect()
            elif topic == 'scan':
                cb = req.get('msg2')
                if cb:
                    cb([])
        except Exception as e:
            print(f"Worker loop error: {e}")

bd.my_worker.start_thread()

def flush_pending_commands():
    """Drop any not-yet-sent BLE requests.

    The run loop enqueues a 'send' request every tick (main.py's 50ms
    loop_network) but _wasm_worker_loop sends them strictly one at a
    time, awaiting each BLE write. If BLE round-trips run slower than
    the tick rate, motor-speed updates pile up in request_queue faster
    than they drain. Stop's motor_stop() request would otherwise land
    at the tail of that backlog, so the robot keeps executing stale
    speed commands for a while after Stop is pressed. Call this right
    before issuing the stop command so it isn't stuck behind them.
    """
    q = bd.my_worker.request_queue
    while not q.empty():
        try:
            q.get_nowait()
        except Exception:
            break

_DEVICE_MAP = {
    "Double Motor": le.DoubleMotor,
    "Single Motor": le.SingleMotor,
    "Color Sensor": le.ColorSensor,
    "Controller":   le.Controller,
}

def _dedupe_name(base_name: str, existing_names: list[str]) -> str:
    """Append ' (1)', ' (2)', etc. if base_name is already in use."""
    if base_name not in existing_names:
        return base_name
    i = 1
    while f"{base_name} ({i})" in existing_names:
        i += 1
    return f"{base_name} ({i})"

NOTIFY_UUID = '0000fd02-0002-1000-8000-00805f9b34fb'

class Element:
    def __init__(self):
        self.myble = ble.BLEDevice.new()
        self.hub = None
        self.name = None
        self.plots = []
        self.plot_vars = []
        self.in_list = []
        self.out_list = []
        self.list_done = False
        self.state = None
        self.hardware_state = {"LightColor": le.LEGO_COLOR_WHITE, 
                               "LightPattern": le.LIGHT_PATTERN_BREATHE, 
                               "LightIntensity": 100, 
                               "BeepPattern": le.SOUND_PATTERN_BEEP_SINGLE, 
                               "BeepFrequency": 1,}

    async def connect(self, existing_names: list[str] | None = None):
        print("Scanning...")
        try:
            await self.myble.scan()
        except Exception as e:
            print(f"Scan cancelled or failed: {e}")
            return

        raw_name = self.myble.name
        
        if not raw_name:
            print("No device selected.")
            return

        base_name = str(raw_name)
        self.name = _dedupe_name(base_name, existing_names)

        hub_cls = None
        for key, cls in _DEVICE_MAP.items():
            if key in self.name:
                hub_cls = cls
                self.build_state()
                break

        if hub_cls is None:
            print(f"Unknown device type: {self.name}")
            return

        try:
            self.hub = hub_cls()

            _hub = self.hub
            self._notification_proxy = create_proxy(
                lambda data: asyncio.ensure_future(
                    _hub._device_callback(NOTIFY_UUID, bytes(data.to_py()))
                )
            )
            self.myble.callback = self._notification_proxy

            SERVICE_UUID = '0000fd02-0000-1000-8000-00805f9b34fb'
            WRITE_UUID   = '0000fd02-0001-1000-8000-00805f9b34fb'
            success = await self.myble.connect(SERVICE_UUID, WRITE_UUID, NOTIFY_UUID)
            if not success:
                print("BLE connect() returned false")
                self.hub = None
                return

            print("BLE GATT connected, setting up hub...")
            
            self.hub.device = self.hub  # <-- THIS is the fix

            registry = getattr(bd.my_worker, '_js_ble_registry', None)
            if registry is not None:
                registry[id(self.hub)] = self.myble 
                print(f"Registered hub id={id(self.hub)}")
            else:
                print("WARNING: worker registry not found")

            self.hub.connected = True
            
            await asyncio.sleep(0)
            
            print("Sending device_notification_request...")
            self.hub.device_notification_request(100, blocking=False)
            
            # Wait for the first real notification to arrive (up to 2 seconds)
            max_wait = 2.0
            interval = 0.05
            waited = 0.0
            while not self.hub._first_notification_ready.is_set() and waited < max_wait:
                await asyncio.sleep(interval)
                waited += interval
        
            
            if self.hub._first_notification_ready.is_set():
                print("First notification received, sensor values ready.")
            else:
                print("Warning: timed out waiting for first notification, sensor values may still be nan.")
            
            print(f"Hub ready: {self.name}")

            self.hub.set_notification_callback(self.notif_callback)

        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"Hub setup failed: {e}")
            self.hub = None
            
    async def disconnect(self):
        if self.hub and self.hub.connected:
            self.hub.disconnect()
            self.hub.connected = False
        await asyncio.sleep(0)
        self.hub = None
        self.myble = None   

    def build_state(self):
        if "Double Motor" in self.name:
            self.state = {
                "LeftPosition": None, "RightPosition": None, "LeftAngle": None, "RightAngle": None,
                "LeftSpeed": None, "RightSpeed": None, "RightPower": None, "LeftPower": None,
                "Yaw": None, "Pitch": None, "Roll": None,
                "AccelerationX": None, "AccelerationY": None, "AccelerationZ": None,
                "GyroX": None, "GyroY": None, "GyroZ": None,
            }
        elif "Single Motor" in self.name:
            self.state = {
                "Position": None, "Speed": None, "Angle": None,
            }
        elif "Controller" in self.name:
            self.state = {
                "LeftPercent": None, "RightPercent": None, 
                "LeftAngle": None, "RightAngle": None,
            }
        else:
            self.state = {
                "Reflection": None, "Color": None, 
                "Hue": None, "Saturation": None, "Value": None,
                "Red": None, "Green": None, "Blue": None,
            }

    def get_out_list(self):
        if "Double Motor" in self.name:
            return ["LeftSpeed", "RightSpeed", "BothSpeed", "LightIntensity", "BeepFrequency"]
        elif "Single Motor" in self.name:
            return ["Speed", "LightIntensity", "BeepFrequency"]
        elif "Color Sensor" in self.name:
            return ["LightIntensity", "BeepFrequency"] #["LightColor", "LightPattern", "LightIntensity", "BeepPattern", "BeepFrequency"]
        elif "Controller" in self.name:
            return ["LightIntensity", "BeepFrequency"] #["LightColor", "LightPattern", "LightIntensity", "BeepPattern", "BeepFrequency"]

    def notif_callback(self, data):                    
        parsed_items = le.device_notification_parser(data)
        for parsed_item in parsed_items:
    
            if isinstance(parsed_item, le.MotorNotification):
                if "Double Motor" in self.name:
                    side = "Left" if parsed_item.motorBitMask == le.MOTOR_BITS_LEFT else "Right"
                    self.state[side + "Position"] = parsed_item.position
                    self.state[side + "Angle"] = parsed_item.absolutePosition
                    self.state[side + "Speed"] = parsed_item.speed
                    self.state[side + "Power"] = parsed_item.power
                else:
                    self.state["Position"] = parsed_item.position
                    self.state["Angle"] = parsed_item.absolutePosition
                    self.state["Speed"] = parsed_item.speed
                    self.state["Power"] = parsed_item.power
    
            elif isinstance(parsed_item, le.ImuDeviceNotification):
                self.state["Yaw"] = parsed_item.yaw
                self.state["Pitch"] = parsed_item.pitch
                self.state["Roll"] = parsed_item.roll
                self.state["AccelerationX"] = parsed_item.accelerometerX
                self.state["AccelerationY"] = parsed_item.accelerometerY
                self.state["AccelerationZ"] = parsed_item.accelerometerZ
                self.state["GyroX"] = parsed_item.gyroscopeX
                self.state["GyroY"] = parsed_item.gyroscopeX
                self.state["GyroZ"] = parsed_item.gyroscopeX

            elif isinstance(parsed_item, le.ColorSensorNotification):
                self.state["Reflection"] = parsed_item.reflection
                self.state["Color"] = parsed_item.color
                self.state["Hue"] = parsed_item.hue
                self.state["Saturation"] = parsed_item.saturation
                self.state["Value"] = parsed_item.value
                self.state["Red"] = parsed_item.rawRed
                self.state["Green"] = parsed_item.rawGreen
                self.state["Blue"] = parsed_item.rawBlue

            elif isinstance(parsed_item, le.ControllerNotification):
                self.state["LeftPercent"] = parsed_item.leftPercent
                self.state["RightPercent"] = parsed_item.rightPercent
                self.state["LeftAngle"] = parsed_item.leftAngle
                self.state["RightAngle"] = parsed_item.rightAngle

        # LIVE PLOTTING STUFF
        for graph, var in zip(self.plots, self.plot_vars):
            val = 0
            if var == "BothSpeed":
                val = (self.state["LeftSpeed"] - self.state["RightSpeed"])/2
            elif var in self.hardware_state:
                try:
                    val = self.hardware_state[var]
                except Exception as e:
                    print("Cannot plot. Error: " + e)
            else: 
                val = self.state[var]
            update = graph.addPoints(1, [val])
            graph.updatePlot(update)

    def set_speed(self, speed):
        if "Single Motor" in self.name:
            #self.hub.motor_set_speed(speed, blocking=False)
            self.hub.motor_run(speed=speed, blocking=False)
        elif "Double Motor" in self.name:
            #self.hub.movement_set_speed(speed, blocking=False)
            self.hub.movement_move(speed=speed, blocking=False)
        else:
            print("cant set speed for " + self.name)

    def set_speedL(self, speed):
        if "Double Motor" in self.name:
            #self.hub.motor_set_speed(speed, motor=le.MOTOR_LEFT, blocking=False)
            self.hub.motor_run(speed=speed, motor=le.MOTOR_LEFT, blocking=False)
        else:
            print("cant set speedL for " + self.name)

    def set_speedR(self, speed):
        if "Double Motor" in self.name:
            #self.hub.motor_set_speed(speed, motor=le.MOTOR_RIGHT, blocking=False)
            self.hub.motor_run(speed=speed, motor=le.MOTOR_RIGHT, blocking=False)
        else:
            print("cant set speedL for " + self.name)
        
    def stop(self):
        if "Single Motor" in self.name:
            self.hub.motor_stop()
        elif "Double Motor" in self.name:
            self.hub.motor_stop(motor=le.MOTOR_BOTH)
        else:
            print("Cant stop " + self.name)

    def set_light(self, variable, value):
        self.hardware_state[variable] = value
        self.hub.light_color(self.hardware_state["LightColor"], intensity=self.hardware_state["LightIntensity"], blocking=False) #removed pattern=self.hardware_state["LightPattern"],
        
    def set_beep(self, variable, value):
        self.hardware_state[variable] = value
        freq = self.hardware_state["BeepFrequency"] if self.hardware_state["BeepFrequency"] != 0 else 1 # dont want it to beep at 0
        self.hub.beep(frequency=freq, blocking=False)


# ── Lookup / options-list helpers ────────────────────────────────────────────

def device_by_name(name: str) -> "Element | None":
    for d in state.devices:
        if d.name == name:
            return d
    return None

def get_device_options_html() -> str:
    opts = '<option class="dev-dropdown" value="">— select device —</option>'
    for device in state.devices:
        opts += f'<option value="{device.name}">{device.name}</option>'
    return opts

def _channel_options_html(source_fn) -> str:
    """Shared fallback-on-exception shape for get_in_channels_html/
    get_out_channels_html: try to pull the option keys from `source_fn`,
    fall back to a single placeholder option on any error (e.g. the device
    doesn't expose that list yet)."""
    try:
        keys = source_fn()
        return "".join(f'<option value="{key}">{key}</option>' for key in keys)
    except Exception as e:
        print("got error: " + e)
        return '<option value="">— value —</option>'

def get_in_channels_html(device: "Element | None" = None) -> str:
    if device is None:
        return '<option value="">— value —</option>'
    return _channel_options_html(lambda: device.state.keys())

def get_out_channels_html(device: "Element | None" = None) -> str:
    if device is None:
        return '<option value="">— value —</option>'
    return _channel_options_html(device.get_out_list)

# ── Device management (connect/disconnect UI) ────────────────────────────────

async def add_device_chip(dev: "Element"):
    dl = document.getElementById("device-list")

    name = dev.name
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
            import sync
            chip_el = document.getElementById(f"chip-{dev.myble.device.name}")
            if chip_el:
                chip_el.remove()
            await dev.disconnect()
            state.devices.remove(dev)
            sync.refresh_device_dropdowns()
        return create_proxy(handler)

    disc.addEventListener("click", await make_disc(dev))
    dl.appendChild(chip)
    import sync
    sync.refresh_device_dropdowns()

async def create_new_device(evt=None):
    import sync
    new_dev = Element()
    existing_names = [d.name for d in state.devices]
    await new_dev.connect(existing_names=existing_names)
    print("out of connect")
    if not new_dev.hub or not new_dev.hub.connected:
        return
    state.devices.append(new_dev)
    await add_device_chip(new_dev)
    sync.refresh_device_dropdowns()

# ── Virtual controller (no-Bluetooth fallback) ───────────────────────────────
# Stands in for a real Controller+Motor pair on browsers without Web
# Bluetooth (Safari/Firefox, see main.py's check_browser_support()). Exposes
# LeftPercent/RightPercent input channels -- same names a real LEGO
# Controller reports -- driven by two <input type=range> sliders, and a
# "Speed" output channel -- same as a real Motor -- that spins a CSS
# animation instead of a physical motor. Because run_output() dispatches on
# fixed channel names (Device.py:~475), reusing "Speed" here means it needs
# no changes at all: it just calls this class's set_speed() like it would
# any real device's.
class VirtualDevice:
    def __init__(self, name="Virtual Controller"):
        self.name = name
        self.state = {"LeftPercent": 0.0, "RightPercent": 0.0}
        self.plots = []
        self.plot_vars = []

    def get_out_list(self):
        return ["Speed"]

    def set_speed(self, value):
        _set_virtual_motor_speed(value)

    def stop(self):
        _set_virtual_motor_speed(0)

def _set_virtual_motor_speed(value):
    # value arrives already clamped to [-100, 100] by run_output()'s "Speed"
    # branch (or is exactly 0, from stop()).
    label = document.getElementById("virtual-motor-value")
    if label:
        label.textContent = str(value)

    blade = document.getElementById("virtual-motor-blade")
    if not blade:
        return
    if value == 0:
        blade.style.animationPlayState = "paused"
        return
    # Faster spin at higher magnitude: 2s/rev at |value|=0 down to ~0.15s/rev at 100.
    duration = max(0.15, 2.0 - abs(value) / 100 * 1.85)
    blade.style.animationPlayState = "running"
    blade.style.animationDuration = f"{duration}s"
    blade.style.animationDirection = "reverse" if value < 0 else "normal"

def _push_virtual_plot_points(dev: "VirtualDevice"):
    """Mirrors the "LIVE PLOTTING STUFF" loop in Element.notif_callback --
    a real device's BLE notification pushes a fresh point into every
    live-plot attached to it (dev.plots/dev.plot_vars, wired up by
    bindings.py whenever an Input node picks this device+channel); a slider
    move is this device's equivalent of "new data just arrived"."""
    for graph, var in zip(dev.plots, dev.plot_vars):
        if var not in dev.state:
            continue
        update = graph.addPoints(1, [dev.state[var]])
        graph.updatePlot(update)

_VIRTUAL_SLIDERS = (
    ("virtual-slider-left",  "virtual-slider-left-value",  "LeftPercent"),
    ("virtual-slider-right", "virtual-slider-right-value", "RightPercent"),
)

# elem_id -> in-flight spring-back Task, so grabbing a slider again mid-animation
# cancels the old return instead of fighting it.
_virtual_slider_anim_tasks: dict[str, "asyncio.Task"] = {}

def _cancel_virtual_slider_anim(elem_id: str):
    task = _virtual_slider_anim_tasks.pop(elem_id, None)
    if task and not task.done():
        task.cancel()

def _apply_virtual_slider_value(elem_id, value_label_id, dev, channel, val):
    slider = document.getElementById(elem_id)
    if slider:
        slider.value = str(val)
    dev.state[channel] = val
    label = document.getElementById(value_label_id)
    if label:
        label.textContent = str(int(round(val)))
    _push_virtual_plot_points(dev)

async def _spring_virtual_slider_to_zero(elem_id, value_label_id, dev, channel, start_value):
    """Physical joystick sliders are spring-loaded and snap back to center
    the instant you let go; this eases the virtual slider back to 0 the same
    way over a quick ~250ms instead of leaving it wherever it was dropped."""
    try:
        steps, duration = 15, 0.25
        for i in range(1, steps + 1):
            eased = 1 - (1 - i / steps) ** 3   # ease-out cubic
            _apply_virtual_slider_value(elem_id, value_label_id, dev, channel, start_value * (1 - eased))
            await asyncio.sleep(duration / steps)
    except asyncio.CancelledError:
        pass
    finally:
        # Only clear our own registration -- if we got cancelled and a new
        # spring-back task already replaced us in the dict, leave it alone.
        if _virtual_slider_anim_tasks.get(elem_id) is asyncio.current_task():
            _virtual_slider_anim_tasks.pop(elem_id, None)

def _bind_virtual_slider(elem_id: str, value_label_id: str, dev: "VirtualDevice", channel: str):
    slider = document.getElementById(elem_id)
    if not slider:
        return

    def on_input(evt):
        _cancel_virtual_slider_anim(elem_id)
        _apply_virtual_slider_value(elem_id, value_label_id, dev, channel, float(evt.target.value))
    slider.addEventListener("input", create_proxy(on_input))

    def on_release(evt):
        current = float(slider.value)
        if current == 0:
            return
        _virtual_slider_anim_tasks[elem_id] = asyncio.ensure_future(
            _spring_virtual_slider_to_zero(elem_id, value_label_id, dev, channel, current))
    slider.addEventListener("change", create_proxy(on_release))

def _zero_virtual_sliders(dev: "VirtualDevice"):
    for elem_id, value_label_id, channel in _VIRTUAL_SLIDERS:
        _cancel_virtual_slider_anim(elem_id)
        _apply_virtual_slider_value(elem_id, value_label_id, dev, channel, 0)

def enable_virtual_controller():
    """Called once at boot when navigator.bluetooth is missing: hides the
    (non-functional) Connect device button, shows the slider/motor panel in
    its place, and registers the virtual device so it appears in every
    Input/Output node's device dropdown immediately -- no "connect" step,
    since there's nothing to connect to."""
    device_list = document.getElementById("device-list")
    if device_list:
        device_list.classList.add("hidden")
    panel = document.getElementById("virtual-controller-panel")
    if panel:
        panel.classList.remove("hidden")

    dev = VirtualDevice()
    state.devices.append(dev)
    for elem_id, value_label_id, channel in _VIRTUAL_SLIDERS:
        _bind_virtual_slider(elem_id, value_label_id, dev, channel)

    zero_btn = document.getElementById("zero-virtual-sliders-btn")
    if zero_btn:
        zero_btn.addEventListener("click", create_proxy(lambda evt: _zero_virtual_sliders(dev)))

    import sync
    sync.refresh_device_dropdowns()

# ── Output routing ───────────────────────────────────────────────────────────

def run_output(variable, dev_name, value):
    device = device_by_name(dev_name)
    if not device:
        return
    if "Speed" in variable:
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
    elif variable == "LightColor":
        device.set_light(variable, value)
    elif variable == "LightPattern":
        device.set_light(variable, value)
    elif variable == "LightIntensity":
        if value > 100:
            value = 100
        elif value < 0:
            value = 0
        device.set_light(variable, value)
    elif variable == "BeepPattern":
        device.set_beep(variable, value)
    elif variable == "BeepFrequency":
        if value < 0:
            value = 0
        elif value > 2700:
            value = 2700
        device.set_beep(variable, value)
    else:
        print("Cannot set " + str(variable))
