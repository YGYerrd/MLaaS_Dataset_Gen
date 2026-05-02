from mlaas_data_generator.test.test_cases import COMPOSITION_BENCHMARKS


def test_composition_benchmarks_have_consistent_reporting_schema():
    assert COMPOSITION_BENCHMARKS
    for bench in COMPOSITION_BENCHMARKS:
        assert "name" in bench
        assert isinstance(bench.get("stages"), list) and bench["stages"]
        report = bench.get("report")
        assert isinstance(report, dict)
        assert "primary_metric" in report
        assert "secondary_metric" in report
        assert "stage_metrics" in report
        for stage in bench["stages"]:
            assert stage in report["stage_metrics"]
