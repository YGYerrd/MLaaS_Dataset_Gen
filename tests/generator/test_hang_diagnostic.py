from pathlib import Path
import subprocess
import sys


LAUNCHER = Path(__file__).resolve().parents[2] / "scripts" / "diagnose_python_hang.py"


def test_watchdog_captures_stack_without_terminating_work(tmp_path):
    target = tmp_path / "slow.py"
    target.write_text("import time, sys\ntime.sleep(0.4)\nprint(sys.argv[1])\n")
    log = tmp_path / "stacks.log"
    result = subprocess.run([sys.executable, str(LAUNCHER), "--interval", "0.05",
                             "--log", str(log), "--", str(target), "finished"],
                            capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stderr
    assert "finished" in result.stdout
    stacks = log.read_text()
    assert "Timeout" in stacks and 'slow.py", line 2' in stacks
    assert "=== Finished" in stacks


def test_module_launch_preserves_arguments_and_exit_code(tmp_path):
    (tmp_path / "probe.py").write_text("import sys\nprint(sys.argv[1])\nsys.exit(7)\n")
    result = subprocess.run([sys.executable, str(LAUNCHER), "--log", str(tmp_path / "stacks.log"),
                             "--", "-m", "probe", "hello"], cwd=tmp_path,
                            capture_output=True, text=True, timeout=15)
    assert result.returncode == 7
    assert "hello" in result.stdout
