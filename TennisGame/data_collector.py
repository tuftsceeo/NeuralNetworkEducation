"""
collect_data.py — PC side of the data collection pipeline.

Listens on USB serial for IMU recordings from the SPIKE hub and saves
them to a CSV file. Run this BEFORE starting the hub program.

Usage:
    python collect_data.py                  # auto-detects port
    python collect_data.py COM3             # specify port explicitly
    python collect_data.py COM3 my_data.csv # specify port and output file

The CSV has a header row followed by one row per recording:
    label, g0_ax, g0_ay, g0_az, g0_gx, g0_gy, g0_gz, g1_ax, ...
"""

import sys
import csv
import os
import serial
import serial.tools.list_ports

# ── Configuration ─────────────────────────────────────────────────────────────

BAUD_RATE   = 115200
OUTPUT_FILE = "gesture_data.csv"
NUM_SAMPLES = 30
NUM_AXES    = 6   # ax, ay, az, gx, gy, gz
GESTURES    = ["Rock", "Paper", "Scissors"]

# ── CSV header ────────────────────────────────────────────────────────────────

def make_header():
    """Build column names: label, t0_ax, t0_ay, ..., t29_gz"""
    cols = ["label"]
    axis_names = ["ax", "ay", "az", "gx", "gy", "gz"]
    for t in range(NUM_SAMPLES):
        for axis in axis_names:
            cols.append(f"t{t}_{axis}")
    return cols


# ── Port detection ────────────────────────────────────────────────────────────

def find_spike_port():
    """Try to auto-detect the SPIKE Prime USB serial port."""
    ports = serial.tools.list_ports.comports()
    for p in ports:
        # SPIKE Prime shows up as a USB CDC device
        if "USB" in (p.description or "") or "LEGO" in (p.description or ""):
            return p.device
    # If we can't detect it, list what's available and ask
    print("Could not auto-detect SPIKE port. Available ports:")
    for p in ports:
        print(f"  {p.device}  —  {p.description}")
    return input("Enter port manually (e.g. COM3 or /dev/ttyUSB0): ").strip()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    # Parse command line args
    port        = sys.argv[1] if len(sys.argv) > 1 else None
    output_file = sys.argv[2] if len(sys.argv) > 2 else OUTPUT_FILE

    if port is None:
        port = find_spike_port()

    print(f"Connecting to {port} at {BAUD_RATE} baud...")
    ser = serial.Serial(port, BAUD_RATE, timeout=2)
    print(f"Connected. Saving recordings to '{output_file}'")
    print("─" * 50)

    # Open CSV — append mode so you can run this multiple sessions
    file_exists = os.path.exists(output_file)
    csv_file    = open(output_file, "a", newline="")
    writer      = csv.writer(csv_file)

    if not file_exists:
        writer.writerow(make_header())
        csv_file.flush()

    # Wait for hub to signal it's ready
    print("Waiting for hub to start... (start the program on the hub now)")
    while True:
        line = ser.readline().decode("utf-8", errors="replace").strip()
        if line == "READY":
            print("Hub is ready!")
            break

    # Count existing recordings so we can show running totals
    counts = {i: 0 for i in range(len(GESTURES))}

    current_gesture_idx  = None
    current_gesture_name = None

    print("\nPress the hub's centre button to record each gesture.")
    print("The hub cycles through Rock → Paper → Scissors automatically.\n")

    while True:
        try:
            raw = ser.readline().decode("utf-8", errors="replace").strip()
        except KeyboardInterrupt:
            break

        if not raw:
            continue

        # ── GESTURE line: tells us what the next recording will be ────────────
        if raw.startswith("GESTURE,"):
            parts = raw.split(",")
            current_gesture_idx  = int(parts[1])
            current_gesture_name = parts[2]
            total = counts[current_gesture_idx]
            print(f"Next gesture: {current_gesture_name}  "
                  f"(recorded so far: {total})  — press hub button when ready")

        # ── SAMPLE line: the actual IMU data ─────────────────────────────────
        elif raw.startswith("SAMPLE,"):
            parts  = raw.split(",")
            label  = int(parts[1])
            values = [float(v) for v in parts[2:]]

            expected = NUM_SAMPLES * NUM_AXES
            if len(values) != expected:
                print(f"  WARNING: expected {expected} values, got {len(values)} — skipping")
                continue

            writer.writerow([label] + values)
            csv_file.flush()

            counts[label] += 1
            print(f"  ✓ Saved {GESTURES[label]} recording "
                  f"#{counts[label]}  (total: {sum(counts.values())})")

        # ── anything else is a debug print from the hub ───────────────────────
        else:
            print(f"  HUB: {raw}")

    # Wrap up
    csv_file.close()
    ser.close()

    print("\n─" * 50)
    print("Collection complete. Recordings saved:")
    for i, name in enumerate(GESTURES):
        print(f"  {name}: {counts[i]}")
    print(f"  Total: {sum(counts.values())} rows in '{output_file}'")


if __name__ == "__main__":
    main()