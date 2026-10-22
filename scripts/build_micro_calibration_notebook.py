from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

MICRO_CALIBRATION_WEIGHTS: dict[str, tuple[float, float]] = {
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


def build_micro_calibration_variant(
    source_notebook: str | Path, source_metadata: str | Path, output_dir: str | Path
) -> tuple[Path, Path]:
    """
    Builds a micro-calibration notebook variant.

    Args:
        source_notebook: Path to the golden_fusion notebook.
        source_metadata: Path to the golden_fusion kernel-metadata.json.
        output_dir: Directory to write the new files.

    Returns:
        tuple[Path, Path]: Paths to the new notebook and metadata files.
    """
    source_nb_path = Path(source_notebook)
    source_meta_path = Path(source_metadata)
    out_dir_path = Path(output_dir)
    out_dir_path.mkdir(parents=True, exist_ok=True)

    with open(source_nb_path, "r", encoding="utf-8") as f:
        nb_json: dict[str, Any] = json.load(f)

    with open(source_meta_path, "r", encoding="utf-8") as f:
        meta_json: dict[str, Any] = json.load(f)

    # 1. Update Notebook JSON
    # Locate the cell with FINAL_SUPER_WEIGHTS
    found_cells: list[int] = []
    for i, cell in enumerate(nb_json.get("cells", [])):
        if cell.get("cell_type") != "code":
            continue

        cell_source = "".join(cell.get("source", []))
        if "FINAL_SUPER_WEIGHTS" in cell_source and "=" in cell_source:
            found_cells.append(i)

    if len(found_cells) != 1:
        raise ValueError(
            f"Expected exactly 1 code cell defining FINAL_SUPER_WEIGHTS, found {len(found_cells)}."
        )

    target_cell_idx = found_cells[0]
    target_cell = nb_json["cells"][target_cell_idx]
    original_source = "".join(target_cell["source"])

    # Regex to replace the FINAL_SUPER_WEIGHTS block. We will replace the entire definition.
    # We find the assignment, assuming it ends with a closing brace `}`.
    pattern = re.compile(r"FINAL_SUPER_WEIGHTS\s*=\s*\{.*?\}(\s*)", re.DOTALL)

    if not pattern.search(original_source):
        raise ValueError("Could not regex match FINAL_SUPER_WEIGHTS block for replacement.")

    # Format the new weights block
    new_weights_str = "FINAL_SUPER_WEIGHTS = {\n"
    for k, v in MICRO_CALIBRATION_WEIGHTS.items():
        new_weights_str += f'    "{k}": {v},\n'
    new_weights_str += "}\n"

    new_source = pattern.sub(new_weights_str, original_source, count=1)

    # Split back into lines to preserve notebook format
    new_source_lines = [
        line + "\n" if not line.endswith("\n") else line for line in new_source.splitlines()
    ]
    if original_source.endswith("\n") and not new_source_lines[-1].endswith("\n"):
        new_source_lines[-1] += "\n"
    elif not original_source.endswith("\n") and new_source_lines[-1].endswith("\n"):
        new_source_lines[-1] = new_source_lines[-1][:-1]

    nb_json["cells"][target_cell_idx]["source"] = new_source_lines

    # Add the Markdown warning cell at the top
    warning_cell = {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "**Exp-22 micro-calibration; unvalidated public-LB-only experiment; no CV claim.**"
        ],
    }
    nb_json["cells"].insert(0, warning_cell)

    # Set internal kaggle metadata
    if "metadata" not in nb_json:
        nb_json["metadata"] = {}
    if "kaggle" not in nb_json["metadata"]:
        nb_json["metadata"]["kaggle"] = {}

    nb_json["metadata"]["kaggle"]["isGpuEnabled"] = True
    nb_json["metadata"]["kaggle"]["isInternetEnabled"] = False

    # 2. Update Kernel Metadata JSON
    out_meta_json = {
        "id": "aleixlopez/rsna-knee-micro-calibration-20261021",
        "title": "RSNA Knee Micro-Calibration Exp-22",
        "code_file": "rsna-knee-micro-calibration-20261021.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": False,
        "dataset_sources": meta_json.get("dataset_sources", []),
        "competition_sources": meta_json.get("competition_sources", []),
        "kernel_sources": meta_json.get("kernel_sources", []),
        "model_sources": meta_json.get("model_sources", []),
    }

    # 3. Write outputs
    out_nb_path = out_dir_path / "rsna-knee-micro-calibration-20261021.ipynb"
    out_meta_path = out_dir_path / "kernel-metadata.json"

    with open(out_nb_path, "w", encoding="utf-8") as f:
        json.dump(nb_json, f, indent=2, ensure_ascii=False)
        f.write("\n")

    with open(out_meta_path, "w", encoding="utf-8") as f:
        json.dump(out_meta_json, f, indent=2, ensure_ascii=False)
        f.write("\n")

    return out_nb_path, out_meta_path
