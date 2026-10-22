import ast
import json
from pathlib import Path

from scripts.build_micro_calibration_notebook import (
    MICRO_CALIBRATION_WEIGHTS,
    build_micro_calibration_variant,
)


def test_micro_calibration_builder(tmp_path: Path):
    source_nb = Path("notebooks/submission/golden_fusion/rsna-knee-golden-fusion.ipynb")
    source_meta = Path("notebooks/submission/golden_fusion/kernel-metadata.json")

    assert source_nb.exists(), "Source notebook missing"
    assert source_meta.exists(), "Source metadata missing"

    # Track original file states
    orig_nb_stat = source_nb.stat()
    orig_meta_stat = source_meta.stat()

    out_dir = tmp_path / "micro_calibration"
    out_dir.mkdir()

    # Run builder
    out_nb, out_meta = build_micro_calibration_variant(
        source_notebook=source_nb, source_metadata=source_meta, output_dir=out_dir
    )

    # Assert source unmodified
    assert source_nb.stat().st_mtime == orig_nb_stat.st_mtime
    assert source_meta.stat().st_mtime == orig_meta_stat.st_mtime

    # Assert outputs created
    assert out_nb.exists(), "Output notebook was not created"
    assert out_meta.exists(), "Output metadata sidecar was not created"

    # Validate Sidecar JSON (metadata)
    with open(out_meta, "r") as f:
        meta_json = json.load(f)

    assert meta_json["id"] == "aleixlopez/rsna-knee-micro-calibration-20261021"
    assert meta_json["is_private"] is True
    assert meta_json["enable_gpu"] is True
    assert meta_json["enable_internet"] is False

    with open(source_meta, "r") as f:
        orig_meta_json = json.load(f)

    assert meta_json["dataset_sources"] == orig_meta_json["dataset_sources"]
    assert meta_json["competition_sources"] == orig_meta_json["competition_sources"]

    # Validate Notebook JSON
    with open(out_nb, "r") as f:
        nb_json = json.load(f)

    # Check internal metadata
    nb_meta = nb_json.get("metadata", {})
    assert "kaggle" in nb_meta, "Internal notebook Kaggle metadata missing"
    kaggle_meta = nb_meta["kaggle"]
    assert kaggle_meta.get("isGpuEnabled") is True, (
        "Internal notebook Kaggle metadata must stay GPU-enabled"
    )
    assert kaggle_meta.get("isInternetEnabled") is False, (
        "Internal notebook Kaggle metadata must stay Internet-disabled"
    )

    # Check that all targets are present in MICRO_CALIBRATION_WEIGHTS
    expected_weights = {
        "Fracture": (0.975, 0.025),
        "Contusion": (0.980, 0.020),
        "MCL": (0.985, 0.015),
        "Medial Meniscus": (0.990, 0.010),
        "Lateral Meniscus": (0.990, 0.010),
        "Lateral OA": (0.995, 0.005),
        "Medial OA": (0.995, 0.005),
        "PF OA": (0.995, 0.005),
        "ACL": (0.995, 0.005),
        "Effusion": (0.995, 0.005),
        "Synovitis": (0.995, 0.005),
        "Baker's": (0.995, 0.005),
    }

    assert MICRO_CALIBRATION_WEIGHTS == expected_weights, (
        "MICRO_CALIBRATION_WEIGHTS dictionary does not match expected playbook map"
    )

    # Check that these exact weights are assigned to FINAL_SUPER_WEIGHTS in the notebook source
    found_weights = False
    found_warning = False
    for cell in nb_json.get("cells", []):
        cell_source = "".join(cell.get("source", []))

        if (
            cell.get("cell_type") == "code"
            and "FINAL_SUPER_WEIGHTS" in cell_source
            and "=" in cell_source
        ):
            # Find the assignment and parse it
            for line in cell_source.splitlines():
                if line.strip().startswith("FINAL_SUPER_WEIGHTS ="):
                    # Extract the dictionary part assuming it's multi-line or single-line
                    # The simplest way to parse it robustly without a full AST parser for the whole cell
                    # is to parse the whole cell text and find the Assignment node.
                    try:
                        parsed_ast = ast.parse(cell_source)
                        for node in ast.walk(parsed_ast):
                            if isinstance(node, ast.Assign):
                                for target in node.targets:
                                    if (
                                        isinstance(target, ast.Name)
                                        and target.id == "FINAL_SUPER_WEIGHTS"
                                    ):
                                        parsed_dict = ast.literal_eval(node.value)
                                        assert parsed_dict == expected_weights, (
                                            "FINAL_SUPER_WEIGHTS in code cell does not match MICRO_CALIBRATION_WEIGHTS"
                                        )
                                        found_weights = True
                    except SyntaxError:
                        pass

        if (
            "unvalidated public-LB-only experiment" in cell_source
            or "no CV claim" in cell_source
            or "no CV validation" in cell_source
        ):
            found_warning = True

    assert found_weights, (
        "Did not find FINAL_SUPER_WEIGHTS definition with exact playbook weight map in notebook"
    )
    assert found_warning, (
        "Did not find 'unvalidated public-LB-only experiment' or 'no CV validation' warning in notebook"
    )
