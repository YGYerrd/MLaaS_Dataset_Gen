import os
import pickle
import subprocess
import sys
from types import SimpleNamespace

import pytest

from mlaas_data_generator.cli import miopen_worker as worker
from mlaas_data_generator.cli import run_manifest as runner
from mlaas_data_generator.cli.main import build_parser


@pytest.mark.parametrize("family,device,hip,platform,architecture,expected", [
    ("detr", "auto", "7.2", "win32", "gfx1201", True),
    ("conditional-detr", "cuda:0", "7.2", "win32", "gfx1201:sramecc-", True),
    ("deformable-detr", "cuda", "7.2", "win32", "gfx1201", True),
    ("yolos", "cuda", "7.2", "win32", "gfx1201", False),
    ("segformer", "cuda", "7.2", "win32", "gfx1201", False),
    ("upernet", "cuda", "7.2", "win32", "gfx1201", False),
    ("dpt", "cuda", "7.2", "win32", "gfx1201", False),
    ("detr", "cpu", "7.2", "win32", "gfx1201", False),
    ("detr", "cuda", None, "win32", "gfx1201", False),
    ("detr", "cuda", "7.2", "linux", "gfx1201", False),
    ("detr", "cuda", "7.2", "win32", "gfx1151", False),
    ("detr", "cuda", "7.2", "win32", "", False),
])
def test_scope(monkeypatch, family, device, hip, platform, architecture, expected):
    monkeypatch.setattr(worker.sys, "platform", platform)
    cuda = SimpleNamespace(is_available=lambda: True, current_device=lambda: 0,
                           get_device_properties=lambda index: SimpleNamespace(gcnArchName=architecture))
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(version=SimpleNamespace(hip=hip), cuda=cuda))
    cfg = dict(model_family=family, device=device)
    assert worker.should_isolate(cfg) is expected
    assert not worker.should_isolate({**cfg, "_miopen_fast_isolated": True})


def test_selected_device_controls_policy(monkeypatch):
    monkeypatch.setattr(worker.sys, "platform", "win32")
    cuda = SimpleNamespace(is_available=lambda: True, current_device=lambda: 1,
        get_device_properties=lambda index: SimpleNamespace(gcnArchName={0: "gfx1151", 1: "gfx1201"}[index]))
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(version=SimpleNamespace(hip="7.2"), cuda=cuda))
    assert worker.should_isolate({"model_family": "detr", "device": "auto"})
    assert not worker.should_isolate({"model_family": "detr", "device": "cuda:0"})
    assert worker.should_isolate({"model_family": "detr", "device": "cuda:1"})
    cuda.is_available = lambda: False
    assert not worker.should_isolate({"model_family": "detr"})


@pytest.mark.parametrize("model,family", [
    ("facebook/detr-resnet-50", "detr"),
    ("facebook/detr-resnet-50-dc5", "detr"),
    ("microsoft/conditional-detr-resnet-50", "conditional-detr"),
    ("SenseTime/deformable-detr", "deformable-detr"),
    ("unknown/detr-looking-name", ""),
])
def test_registry_resolution_without_model_name_guessing(model, family):
    assert worker.registered_family({"hf_model_id": model}) == family


def test_manifest_family_metadata_resolution():
    assert worker.registered_family({"hf_service_meta_json": '{"model_family":"conditional-detr"}'}) == "conditional-detr"
    assert worker.registered_family({"hf_service_meta_json": {"model_family": "deformable_detr"}}) == "deformable-detr"
    assert worker.registered_family({"hf_service_meta_json": "invalid", "hf_model_id": "facebook/detr-resnet-50"}) == "detr"


def test_child_environment_isolated_from_parent_and_next_process(monkeypatch):
    monkeypatch.delenv("MIOPEN_FIND_MODE", raising=False)
    real_popen = subprocess.Popen
    def launch(command, **kwargs):
        # Exercise the real process boundary without importing any ML backend.
        request, response = command[-2:]
        code = "import os,pickle,sys; pickle.dump([os.environ.get('MIOPEN_FIND_MODE'),os.environ.get('CUDA_VISIBLE_DEVICES')],open(sys.argv[1],'wb'))"
        return real_popen([sys.executable, "-c", code, response], **kwargs)
    monkeypatch.setattr(worker.subprocess, "Popen", launch)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2")
    assert worker.run_isolated([], grouped=True) == ["FAST", "2"]
    assert "MIOPEN_FIND_MODE" not in os.environ
    with real_popen([sys.executable, "-c", "import os; print(os.environ.get('MIOPEN_FIND_MODE'))"], stdout=subprocess.PIPE, text=True) as child:
        assert child.communicate()[0].strip() == "None"


@pytest.mark.parametrize("grouped", [True, False])
def test_child_marks_config_and_uses_requested_executor(monkeypatch, tmp_path, grouped):
    request, response = tmp_path / "in.pkl", tmp_path / "out.pkl"
    payload = [{"idx": 0, "ordinal": 1, "resolved": {"service_id": "a", "model_family": "detr"}}]
    request.write_bytes(pickle.dumps({"payload": payload, "grouped": grouped}))
    monkeypatch.setattr(sys, "argv", ["worker", str(request), str(response)])
    monkeypatch.setattr(runner, "_execute_hf_group_worker", lambda items: [items[0]["resolved"]])
    monkeypatch.setattr(runner, "_execute_entry_worker", lambda item: item["resolved"])
    worker.main()
    result = pickle.loads(response.read_bytes())[0]
    assert result["_miopen_fast_isolated"] and result["miopen_process_isolated"]
    assert result["miopen_find_mode"] == "FAST"


def test_cli_flag_removed():
    import inspect
    assert "miopen_fast_vision" not in inspect.signature(runner.run_manifest).parameters
    for parser, args in [(build_parser(), ["run-manifest", "--file", "x.csv"]),
                         (runner.build_parser(), ["--file", "x.csv"])]:
        assert not hasattr(parser.parse_args(args), "miopen_fast_vision")
        with pytest.raises(SystemExit):
            parser.parse_args(args + ["--miopen-fast-vision"])


def test_worker_failure_preserves_already_committed_services(monkeypatch, tmp_path):
    import sqlite3
    path = tmp_path / "services.db"
    with sqlite3.connect(path) as conn:
        conn.execute("CREATE TABLE services(service_id TEXT,status TEXT)")
        conn.executemany("INSERT INTO services VALUES (?,?)", [("done", "completed"), ("pending", "running")])
    entries = [runner.ManifestEntry(idx=i, ordinal=i+1, validation=runner.RowValidation(True),
                resolved={"service_id": sid, "db_path": str(path)}) for i,sid in enumerate(("done", "pending"))]
    def fail(*args, **kwargs):
        raise RuntimeError("child exited")
    monkeypatch.setattr(worker, "run_isolated", fail)
    recorded = []
    def record(entry, **kwargs):
        recorded.append(entry.resolved["service_id"])
        return {"service_id": entry.resolved["service_id"], "status": "failed"}
    monkeypatch.setattr(runner, "_record_entry_failure", record)
    monkeypatch.setattr(runner, "_record_manifest_progress_result", lambda result: None)
    results = runner._execute_isolated_miopen(entries, grouped=True)
    assert [r["status"] for r in results] == ["success", "failed"]
    assert recorded == ["pending"]


def test_interruption_terminates_child_and_preserves_parent_environment(monkeypatch):
    monkeypatch.delenv("MIOPEN_FIND_MODE", raising=False)
    class Child:
        waits = 0
        terminated = False
        def wait(self):
            self.waits += 1
            if self.waits == 1:
                raise KeyboardInterrupt()
            return -1
        def terminate(self):
            self.terminated = True
    child = Child()
    monkeypatch.setattr(worker.subprocess, "Popen", lambda *args, **kwargs: child)
    with pytest.raises(KeyboardInterrupt):
        worker.run_isolated([], grouped=True)
    assert child.terminated and child.waits == 2
    assert "MIOPEN_FIND_MODE" not in os.environ
