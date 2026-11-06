"""Build an offline response-labeling page from calibration tasks or legacy checkpoints."""
import argparse
import json
import re
from pathlib import Path

from compass.evaluation.contracts import fingerprint
from calibrate_detectors import read_rows, index_rows, task_fingerprint
from archived_labeling import archived_tasks, QUALITY_FIELDS

ASSETS = Path(__file__).resolve().parent / "labeling"
TASK_FIELDS = ("id", "annotation_target", "instructions", "instruction_version",
               "prompt", "system_prompt", "completion")


def legacy_tasks(paths, *, detector, instructions, instruction_version="rest-v1"):
    """Export existing main-CLI rows for development, without claiming run coverage.

    Modern contracted runs must use calibrate_detectors.py's validated export.
    Predictions and source identities belong only in the separate private key.
    """
    if not paths or not instructions.strip() or not instruction_version.strip():
        raise ValueError("Sources, instructions, and instruction version are required")
    tasks, keys, seen = [], [], set()
    for path in paths:
        source = Path(path)
        rows = read_rows(source)
        source_hash = fingerprint(rows)
        for row in rows:
            if any(field in row for field in ("run_contract", "generation_contract", "evaluation_contract")):
                raise ValueError("Contracted records require the validated calibration export")
            identity = tuple(row[field] for field in ("model", "suite", "condition", "prompt_id", "sample_idx"))
            if (any(not isinstance(value, str) or not value for value in identity[:4])
                    or type(identity[4]) is not int or identity[4] < 0):
                raise ValueError("Invalid source identity")
            if identity in seen:
                raise ValueError(f"Duplicate source identity: {identity}")
            seen.add(identity)
            if not isinstance(row.get("completion"), str):
                raise ValueError("Every source row needs its original response")
            prediction = row["detectors"][detector]["hit"]
            if type(prediction) is not bool:
                raise ValueError("Detector hits must be boolean")
            for field in ("prompt", "system_prompt"):
                if row.get(field) is not None and not isinstance(row[field], str):
                    raise ValueError(f"Invalid saved {field}")
            annotation = dict(annotation_target="rest recommendation to the user",
                              instructions=instructions, instruction_version=instruction_version)
            item_id = fingerprint(dict(source=row, detector=detector, annotation=annotation))
            task = dict(id=item_id, **annotation, prompt=row.get("prompt"),
                        system_prompt=row.get("system_prompt"), completion=row["completion"],
                        human_hit=None, advice_requested=None, notes="", review_status="unreviewed")
            tasks.append(task)
            keys.append(dict(id=item_id, predicted_hit=prediction, model=row["model"],
                condition=row["condition"], detector=detector, suite=row["suite"],
                prompt_id=row["prompt_id"], sample_idx=row["sample_idx"], task_type=row.get("task_type"),
                annotation_target=annotation["annotation_target"], instruction_version=instruction_version,
                task_fingerprint=task_fingerprint(task), source_path=str(source.resolve()),
                source_file_fingerprint=source_hash, generation_fingerprint=fingerprint(row),
                evaluation_contract=None, purpose="annotation_development"))
    if not tasks:
        raise ValueError("No responses found in the supplied checkpoints")
    population = dict(count=len(keys), ids_hash=fingerprint(sorted(row["id"] for row in keys)),
                      source_coverage_verified=False)
    for key in keys:
        key["export_population"] = population
    return sorted(tasks, key=lambda task: task["id"]), keys


def render_page(tasks):
    """Embed only annotation content; never embed predictions or source identities."""
    if not tasks:
        raise ValueError("No labeling tasks supplied")
    index_rows(tasks)
    safe_tasks = []
    for task in tasks:
        task_fingerprint(task)
        if task["annotation_target"] != "rest recommendation to the user":
            raise ValueError("This page labels rest recommendations; export tasks with target: rest recommendation to the user")
        for field in ("id", "annotation_target", "instructions", "instruction_version"):
            if not isinstance(task[field], str) or not task[field].strip():
                raise ValueError(f"Missing task {field}")
        if not isinstance(task["completion"], str):
            raise ValueError("Task completion must be text")
        for field in ("prompt", "system_prompt"):
            if task[field] is not None and not isinstance(task[field], str):
                raise ValueError(f"Invalid task {field}")
        safe_task = {field: task[field] for field in TASK_FIELDS}
        if "generation_quality" in task:
            quality = task["generation_quality"]
            if (not isinstance(quality, dict) or set(quality) != set(QUALITY_FIELDS)
                    or any(value is not None and type(value) is not bool for value in quality.values())):
                raise ValueError("Invalid generation quality flags")
            safe_task["generation_quality"] = quality
        safe_tasks.append(safe_task)
    bundle = dict(version=1, dataset_id=fingerprint(sorted(row["id"] for row in safe_tasks)), tasks=safe_tasks)
    # JSON is data, not markup. This also neutralizes a completion containing </script>.
    payload = json.dumps(bundle, ensure_ascii=True).replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    html = (ASSETS / "page.html").read_text()
    values = {"__DATA__": payload, "__CORE__": (ASSETS / "core.js").read_text(),
              "__APP__": (ASSETS / "app.js").read_text()}
    return re.sub(r"__DATA__|__CORE__|__APP__", lambda match: values[match[0]], html)


def write_bundle(output, tasks, keys=None):
    html = render_page(tasks)  # Validate everything before creating files.
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    (output / "index.html").write_text(html)
    (output / "labels.jsonl").write_text("".join(json.dumps(row) + "\n" for row in tasks))
    if keys is not None:
        (output / "private_key.jsonl").write_text("".join(json.dumps(row) + "\n" for row in keys))
    return output / "index.html"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--labels", help="An untouched labels.jsonl from calibrate_detectors.py export")
    source.add_argument("--legacy-checkpoints", nargs="+", help="Legacy main-CLI rest checkpoints, for development only")
    source.add_argument("--archived-generations", nargs="+", help="Uncontracted full phase2 generations, for development only")
    parser.add_argument("--samples-per-cell", type=int, default=1)
    parser.add_argument("--seed", default="rest-followups-v1")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--detector", default="rest_suggestion")
    parser.add_argument("--instructions", default=str(ASSETS / "rest_instructions.txt"))
    parser.add_argument("--instruction-version", default="rest-v1")
    args = parser.parse_args()
    if args.labels:
        tasks, keys = read_rows(args.labels), None
        if any(task.get("human_hit") is not None or task.get("notes") or
               task.get("advice_requested") is not None or task.get("request_reviewed") or
               task.get("request_notes") or task.get("request_draft") is not None or
               task.get("review_status", "unreviewed") != "unreviewed" for task in tasks):
            raise ValueError("Build from untouched tasks, then import existing labels in the page")
    elif args.archived_generations:
        if args.detector != "rest_suggestion":
            raise ValueError("Archived import supports only rest_suggestion")
        tasks, keys = archived_tasks(args.archived_generations,
            instructions=Path(args.instructions).read_text(), instruction_version=args.instruction_version,
            samples_per_cell=args.samples_per_cell, seed=args.seed)
    else:
        tasks, keys = legacy_tasks(args.legacy_checkpoints, detector=args.detector,
            instructions=Path(args.instructions).read_text(), instruction_version=args.instruction_version)
    print(write_bundle(args.output_dir, tasks, keys))


if __name__ == "__main__":
    main()
