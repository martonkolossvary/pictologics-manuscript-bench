"""Keep smoke requests compatible with the real adapter workload protocol."""

import argparse
import json

from bench.adapters.protocol import (
    add_common_arguments,
    requested_benchmark_workload,
    requested_families,
)
from scripts import smoke_adapter_environments


def test_pyradiomics_smoke_preserves_canonical_texture_request(tmp_path, monkeypatch):
    # No native dependency is needed: exercise the real smoke caller and the
    # adapter's actual argument/contract validation at the process boundary.
    requests = []

    def validate_request(adapter, **options):
        parser = argparse.ArgumentParser()
        add_common_arguments(parser)
        args = parser.parse_args([
            "--image", options["image"],
            "--mask", options["mask"],
            "--families", ",".join(options["families"]),
            "--benchmark-workload", options["benchmark_workload"],
        ])
        supported, unsupported = requested_families(adapter, args)
        workload = requested_benchmark_workload(adapter, args, supported)
        requests.append((workload, options["families"], unsupported))
        assert options["timed"] is False
        return {"values": {"all": {"finite_fixture_feature": 1.0}}}, {}

    monkeypatch.setattr(smoke_adapter_environments.env, "verify_profile", lambda *a, **k: None)
    monkeypatch.setattr(smoke_adapter_environments, "run_adapter_process", validate_request)
    results = smoke_adapter_environments.smoke(["pyradiomics"], tmp_path, 300.0)

    assert [request[0] for request in requests] == ["morphology", "intensity", "texture"]
    assert requests[-1][1] == ["histogram", "glcm", "glrlm", "glszm", "gldzm", "ngtdm", "ngldm"]
    assert requests[-1][2] == ["gldzm"]
    assert all(result["status"] == "passed" for result in results)
    assert json.loads((tmp_path / "smoke-results.json").read_text()) == results
