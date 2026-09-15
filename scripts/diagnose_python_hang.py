"""Run an existing Python command with periodic, non-terminating stack dumps.

Example: python scripts/diagnose_python_hang.py -- -m package.module arguments
No model or evaluation settings are changed. Ctrl+C also records all stacks.
"""
import argparse
import threading
import traceback
from datetime import datetime
from pathlib import Path
import runpy
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interval", type=float, default=60,
                        help="Seconds between stack snapshots (default: 60)")
    parser.add_argument("--log", type=Path,
                        default=Path("outputs/checks/hang_stacks.log"))
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command
    if command and command[0] == "--":
        command = command[1:]
    if not command or (command[0] == "-m" and len(command) < 2):
        parser.error("provide -- -m module [arguments] or -- script.py [arguments]")
    if args.interval <= 0:
        parser.error("--interval must be positive")
    args.log.parent.mkdir(parents=True, exist_ok=True)
    old_argv = sys.argv
    old_path = list(sys.path)
    with args.log.open("a", encoding="utf-8") as log:
        log.write(f"\n=== Started {datetime.now().astimezone().isoformat()} ===\n")
        log.flush()
        print(f"Stack snapshots every {args.interval:g}s: {args.log.resolve()}", flush=True)
        stopped = threading.Event()
        lock = threading.Lock()

        def snapshot():
            with lock:
                log.write(f"\nTimeout snapshot {datetime.now().astimezone().isoformat()}\n")
                for ident, frame in sys._current_frames().items():
                    log.write(f"\nThread {ident}:\n")
                    traceback.print_stack(frame, file=log)
                log.flush()

        def watch():
            while not stopped.wait(args.interval):
                snapshot()

        # This runtime crashed inside native _Py_DumpTracebackThreads. Use
        # GIL-protected Python frames instead; this cannot diagnose a held GIL.
        watcher = threading.Thread(target=watch, name="stack-diagnostic", daemon=True)
        watcher.start()
        try:
            if command[0] == "-m":
                sys.path.insert(0, str(Path.cwd()))
                sys.argv = command[1:]
                runpy.run_module(command[1], run_name="__main__", alter_sys=True)
            else:
                sys.path.insert(0, str(Path(command[0]).resolve().parent))
                sys.argv = command
                runpy.run_path(command[0], run_name="__main__")
        except KeyboardInterrupt:
            log.write("\n=== KeyboardInterrupt ===\n")
            log.flush()
            snapshot()
            raise
        finally:
            stopped.set()
            watcher.join()
            sys.argv = old_argv
            sys.path[:] = old_path
            log.write(f"\n=== Finished {datetime.now().astimezone().isoformat()} ===\n")


if __name__ == "__main__":
    main()
