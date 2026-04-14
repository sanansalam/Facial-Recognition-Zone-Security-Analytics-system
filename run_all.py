#!/usr/bin/env python3
"""
run_all.py — Launch all Edge AI Security services in one terminal.

Usage:
  cd /home/sana/Bank project/video_ingestion_standalone
  export PYTHONPATH=$PYTHONPATH:.
  python3 run_all.py

Press Ctrl+C to stop everything.
"""
import subprocess
import sys
import signal
import os
import time
import threading

BASE = os.path.dirname(os.path.abspath(__file__))
ENV = {
    **os.environ, 
    "PYTHONPATH": f"{BASE}:{os.environ.get('PYTHONPATH', '')}",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1"
}

SERVICES = [
    ("video_ingestion ",  [sys.executable, "video_ingestion/app/main.py"]),
    ("motion_detect   ",  [sys.executable, "motion_detection/app/main.py"]),
    ("ai_inference    ",  [sys.executable, "ai_inference/app/main.py"]),
    ("sop_state_mach  ",  [sys.executable, "sop_state_machine/app/main.py"]),
    ("event_logger    ",  [sys.executable, "event_logger/app/main.py"]),
]

procs = []


def stream_output(name, proc):
    """Stream stdout+stderr from a subprocess with a label prefix."""
    import select
    while True:
        rlist = [proc.stdout, proc.stderr]
        readable, _, _ = select.select(rlist, [], [], 0.1)
        for stream in readable:
            line = stream.readline()
            if line:
                print(f"[{name}] {line}", end="", flush=True)
        if proc.poll() is not None:
            # Drain remaining output
            for line in proc.stdout.readlines():
                print(f"[{name}] {line}", end="", flush=True)
            for line in proc.stderr.readlines():
                print(f"[{name}] {line}", end="", flush=True)
            print(f"[{name}] ⚠️  Process exited with code {proc.returncode}")
            break


def shutdown(sig, frame):
    print("\n\n🛑 Stopping all services...")
    for name, proc in procs:
        proc.terminate()
    time.sleep(1)
    for name, proc in procs:
        if proc.poll() is None:
            proc.kill()
    print("✅ All services stopped.")
    sys.exit(0)


signal.signal(signal.SIGINT, shutdown)
signal.signal(signal.SIGTERM, shutdown)


def main():
    print("🚀 Starting Edge AI Security Pipeline (V5 — 5 services)")
    print("=" * 55)

    # Start services with a small delay between each
    delays = [0, 2, 4, 6, 8, 10]  # Explicitly allow all 6 services
    for i, (name, cmd) in enumerate(SERVICES):
        delay = delays[i] if i < len(delays) else 2
        if delay:
            print(f"  ⏳ Waiting {delay}s before starting {name.strip()}...")
            time.sleep(delay)

        print(f"  ▶  Starting [{name.strip()}]...")
        proc = subprocess.Popen(
            cmd,
            cwd=BASE,
            env=ENV,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        procs.append((name, proc))
        t = threading.Thread(target=stream_output, args=(name, proc), daemon=True)
        t.start()

    print("\n✅ All services launched! Logs streaming below.")
    print("   Press Ctrl+C to stop everything.\n")
    print("=" * 55)

    # Keep main thread alive
    while True:
        # Check if any critical service died or finished
        for name, proc in procs:
            if proc.poll() is not None:
                if proc.returncode != 0:
                    print(f"\n⚠️  [{name.strip()}] died unexpectedly! Stopping all...")
                    shutdown(None, None)
                elif "video_ingestion" in name:
                    print("\n🎬 All videos finished playing! Generating final security report...")
                    shutdown(None, None)
        time.sleep(2)


if __name__ == "__main__":
    main()
