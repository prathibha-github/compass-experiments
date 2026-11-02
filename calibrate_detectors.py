"""Export blinded phase2 examples and compare human labels, without model calls."""

import argparse
import json
from pathlib import Path

from compass.evaluation.contracts import fingerprint
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


def export_tasks(generations_path, evaluations_path, detector):
    generations = load_suite_generations(Path(generations_path), strict=True)
    tasks, keys = [], []
    seen = set()
    for row in read_rows(evaluations_path):
        if row["detector"] != detector:
            continue
        identity = (row["model"], row["suite"], row["prompt_id"], row["condition"], row.get("sample_idx", 0))
        if identity in seen:
            raise ValueError(f"Duplicate evaluation: {identity}")
        seen.add(identity)
        gen = generations[identity]
        if type(row["hit"]) is not bool:
            raise ValueError("Detector hit must be a boolean")
        if not row.get("evaluation_contract"):
            raise ValueError("Re-score legacy evaluations before calibration")
        if row.get("generation_fingerprint") != fingerprint(gen):
            raise ValueError("Evaluation does not match the saved generation; re-score before calibration")
        item_id = fingerprint({"generation": gen, "detector": detector})
        tasks.append({"id": item_id, "prompt": gen.get("prompt_text"),
                      "system_prompt": gen.get("system_prompt"), "completion": gen["completion"],
                      "human_hit": None, "notes": ""})
        keys.append({"id": item_id, "predicted_hit": row["hit"], "model": row["model"],
                     "condition": row["condition"], "detector": detector,
                     "evaluation_contract": row["evaluation_contract"]})
    if not tasks:
        raise ValueError("No matching scored examples")
    if len(seen) != len(generations):
        raise ValueError("Detector evaluations are incomplete; retry scoring before calibration")
    # Mix model/condition blocks without exposing predictions or model names.
    return sorted(tasks, key=lambda row: row["id"]), keys


def report_labels(keys, labels):
    predictions, humans = index_rows(keys), index_rows(labels)
    expected = {key: row["predicted_hit"] for key, row in predictions.items()}
    actual = {key: row["human_hit"] for key, row in humans.items()}
    report = compare_human_labels(expected, actual)
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
    report = commands.add_parser("report")
    report.add_argument("--key", required=True)
    report.add_argument("--labels", required=True)
    args = parser.parse_args()
    if args.command == "export":
        tasks, keys = export_tasks(args.generations, args.evaluations, args.detector)
        out = Path(args.output_dir)
        out.mkdir(parents=True, exist_ok=False)
        (out / "labels.jsonl").write_text("".join(json.dumps(row) + "\n" for row in tasks))
        (out / "private_key.jsonl").write_text("".join(json.dumps(row) + "\n" for row in keys))
    else:
        print(json.dumps(report_labels(read_rows(args.key), read_rows(args.labels)), indent=2))


if __name__ == "__main__":
    main()
