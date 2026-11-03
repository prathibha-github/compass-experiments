"""Export blinded phase2 examples and compare human labels, without model calls."""

import argparse
import json
from pathlib import Path

from compass.evaluation.contracts import fingerprint, manifest_contracts
from compass.evaluation.record_schema import checkpoint_identity
from compass.evaluation.suite_io import load_suite_generations
from compass.judges.calibration import compare_human_labels


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def index_rows(rows):
    indexed = {}
    for row in rows:
        if row["id"] in indexed:
            raise ValueError(f"Duplicate item ID: {row['id']}")
        indexed[row["id"]] = row
    return indexed


def source_population(generations_path, evaluations_path, generations, detector):
    generation_manifest = Path(generations_path).with_suffix(".manifest.json")
    generation_verified = generation_manifest.exists()
    if generation_verified:
        expected = manifest_contracts(json.loads(generation_manifest.read_text()))
        if any(len(identity) != 5 for identity in expected):
            raise ValueError("Invalid generation manifest identity")
        if set(generations) != set(expected) or any(
            row.get("generation_contract") != expected[identity]
            for identity, row in generations.items()
        ):
            raise ValueError("Incomplete or incompatible generations; restore or resume the original run before export")
    elif any("generation_contract" in row for row in generations.values()):
        raise ValueError("Modern generation manifest is missing; restore it before export")

    evaluation_manifest = Path(evaluations_path).with_suffix(".manifest.json")
    if not evaluation_manifest.exists():
        raise ValueError("Evaluation manifest is missing; restore it or re-score before export")
    manifest = json.loads(evaluation_manifest.read_text())
    expected = manifest_contracts(manifest)
    if any(len(identity) != 6 for identity in expected):
        raise ValueError("Invalid evaluation manifest identity")
    selected = {identity: contract for identity, contract in expected.items() if identity[2] == detector}
    planned_generations = {identity[:2] + identity[3:] for identity in selected}
    if planned_generations != set(generations):
        raise ValueError("Detector evaluation plan does not match the generation population")
    return selected, generation_verified and manifest.get("generation_coverage_verified") is True


def export_tasks(generations_path, evaluations_path, detector, *, annotation_target, instructions, instruction_version):
    if not all(isinstance(value, str) and value.strip() for value in (annotation_target, instructions, instruction_version)):
        raise ValueError("Annotation target, instructions, and instruction version are required")
    annotation = {"annotation_target": annotation_target, "instructions": instructions,
                  "instruction_version": instruction_version}
    generations = load_suite_generations(Path(generations_path), strict=True)
    expected, source_verified = source_population(generations_path, evaluations_path, generations, detector)
    tasks, keys = [], []
    seen = set()
    for row in read_rows(evaluations_path):
        if row["detector"] != detector:
            continue
        identity = (row["model"], row["suite"], row["prompt_id"], row["condition"], row.get("sample_idx", 0))
        if identity in seen:
            raise ValueError(f"Duplicate evaluation: {identity}")
        seen.add(identity)
        evaluation_id = checkpoint_identity(row)
        if evaluation_id not in expected or row.get("evaluation_contract") != expected[evaluation_id]:
            raise ValueError("Unexpected or stale detector evaluation; restore or re-score before export")
        gen = generations[identity]
        if type(row["hit"]) is not bool:
            raise ValueError("Detector hit must be a boolean")
        if not row.get("evaluation_contract"):
            raise ValueError("Re-score legacy evaluations before calibration")
        if row.get("generation_fingerprint") != fingerprint(gen):
            raise ValueError("Evaluation does not match the saved generation; re-score before calibration")
        item_id = fingerprint({"generation": gen, "detector": detector, "annotation": annotation})
        task = {"id": item_id, **annotation, "prompt": gen.get("prompt_text"),
                      "system_prompt": gen.get("system_prompt"), "completion": gen["completion"],
                      "human_hit": None, "notes": ""}
        tasks.append(task)
        keys.append({"id": item_id, "predicted_hit": row["hit"], "model": row["model"],
                     "condition": row["condition"], "detector": detector,
                     "evaluation_contract": row["evaluation_contract"],
                     "suite": row["suite"], "prompt_id": row["prompt_id"],
                     "sample_idx": row.get("sample_idx", 0), "task_type": gen.get("task_type"),
                     "generation_fingerprint": row["generation_fingerprint"],
                     "annotation_target": annotation_target, "instruction_version": instruction_version,
                     "task_fingerprint": task_fingerprint(task)})
    if not tasks:
        raise ValueError("No matching scored examples")
    if len(seen) != len(generations):
        raise ValueError("Detector evaluations are incomplete; retry scoring before calibration")
    population = {"count": len(keys), "ids_hash": fingerprint(sorted(row["id"] for row in keys)),
                  "source_coverage_verified": source_verified}
    for row in keys:
        row["export_population"] = population
    # Mix model/condition blocks without exposing predictions or model names.
    return sorted(tasks, key=lambda row: row["id"]), keys


def task_fingerprint(task):
    fields = ("id", "prompt", "system_prompt", "completion", "annotation_target", "instructions", "instruction_version")
    if any(field not in task for field in fields):
        raise ValueError("Label file is missing its original task or instructions; export a new label set")
    return fingerprint({field: task[field] for field in fields})


def report_labels(keys, labels, *, annotator=None):
    predictions, humans = index_rows(keys), index_rows(labels)
    if any("annotator" in row for row in labels):
        if (not isinstance(annotator, str) or not annotator.strip() or
                any(row.get("annotator") != annotator for row in labels)):
            raise ValueError("Embedded annotators must all match the requested annotator")
    populations = [row.get("export_population") for row in keys]
    source_verified = False
    if any(population is not None for population in populations):
        population = populations[0]
        if (not isinstance(population, dict) or any(item != population for item in populations)
                or population.get("count") != len(keys)
                or population.get("ids_hash") != fingerprint(sorted(predictions))):
            raise ValueError("Private key population is incomplete or inconsistent; use the original export")
        source_verified = population.get("source_coverage_verified") is True
    for key, row in humans.items():
        if key not in predictions or task_fingerprint(row) != predictions[key].get("task_fingerprint"):
            raise ValueError("Label task or instructions changed; use the original export")
    expected = {key: row["predicted_hit"] for key, row in predictions.items()}
    actual = {key: row["human_hit"] for key, row in humans.items()}
    report = compare_human_labels(expected, actual)
    report["label_status"] = report["status"]
    report["source_coverage_verified"] = source_verified
    if not source_verified:
        report["status"] = "source_coverage_unverified"
    report["annotator"] = annotator
    report["labels_fingerprint"] = fingerprint(labels)
    report["key_fingerprint"] = fingerprint(keys)
    report["annotation_targets"] = sorted({row["annotation_target"] for row in keys})
    report["instruction_versions"] = sorted({row["instruction_version"] for row in keys})
    groups = {}
    for row in keys:
        group = (row["model"], row["condition"], row["detector"])
        groups.setdefault(group, {})[row["id"]] = row["predicted_hit"]
    report["groups"] = [
        {"model": model, "condition": condition, "detector": detector,
         **compare_human_labels(preds, {key: actual[key] for key in preds if key in actual})}
        for (model, condition, detector), preds in sorted(groups.items())
    ]
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    export = commands.add_parser("export")
    export.add_argument("--generations", required=True)
    export.add_argument("--evaluations", required=True)
    export.add_argument("--detector", required=True)
    export.add_argument("--output-dir", required=True)
    export.add_argument("--target", required=True, help="Behavior the human should label")
    export.add_argument("--instructions", required=True, help="Path to labeling instructions")
    export.add_argument("--instruction-version", required=True)
    report = commands.add_parser("report")
    report.add_argument("--key", required=True)
    report.add_argument("--labels", required=True)
    report.add_argument("--annotator", required=True)
    args = parser.parse_args()
    if args.command == "export":
        tasks, keys = export_tasks(args.generations, args.evaluations, args.detector,
            annotation_target=args.target, instructions=Path(args.instructions).read_text(),
            instruction_version=args.instruction_version)
        out = Path(args.output_dir)
        out.mkdir(parents=True, exist_ok=False)
        (out / "labels.jsonl").write_text("".join(json.dumps(row) + "\n" for row in tasks))
        (out / "private_key.jsonl").write_text("".join(json.dumps(row) + "\n" for row in keys))
    else:
        print(json.dumps(report_labels(read_rows(args.key), read_rows(args.labels), annotator=args.annotator), indent=2))


if __name__ == "__main__":
    main()
