from pathlib import Path


def test_active_pipeline_docs_and_code_do_not_expose_federated_core_terms():
    root = Path(__file__).resolve().parents[2]
    checked_paths = [
        root / "README.md",
        root / "mlaas_data_generator" / "config.py",
        root / "mlaas_data_generator" / "cli",
        root / "mlaas_data_generator" / "services",
        root / "mlaas_data_generator" / "storage",
    ]
    banned = [
        "FedAvg",
        "global_model",
        "num_clients",
        "num_rounds",
        "client",
        "client_participation_rate",
        "aggregation_weight",
    ]
    offenders = []
    for path in checked_paths:
        files = [path] if path.is_file() else list(path.rglob("*.py")) + list(path.rglob("*.sql"))
        for file_path in files:
            text = file_path.read_text(encoding="utf-8")
            for term in banned:
                if term in text:
                    offenders.append(f"{file_path.relative_to(root)}: {term}")
    assert offenders == []
