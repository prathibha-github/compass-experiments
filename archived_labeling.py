"""Sample saved, uncontracted follow-up responses for annotation development."""
from collections import defaultdict
from pathlib import Path

from compass.detectors.heuristic import PhraseSetDetector
from compass.detectors.suites import SUITE_REGISTRY
from compass.evaluation.contracts import detector_definition, fingerprint
from calibrate_detectors import read_rows, task_fingerprint

QUALITY_FIELDS = ('hit_token_cap', 'is_fragment', 'quality_flagged')


def archived_tasks(paths, *, instructions, instruction_version='rest-v1',
                   samples_per_cell=1, seed='rest-followups-v1'):
    """Use full saved responses and re-run only the current phrase detector.

    Sampling is deterministic within source/model/suite/prompt/condition cells.
    Missing manifests and system context remain missing; this is not a validated
    reconstruction of a historical experiment or a held-out calibration set.
    """
    if (not paths or not instructions.strip() or not instruction_version.strip()
            or type(samples_per_cell) is not int or samples_per_cell < 1
            or not isinstance(seed, str) or not seed):
        raise ValueError('Sources, instructions, version, positive sample count, and seed are required')
    annotation = dict(annotation_target='rest recommendation to the user',
                      instructions=instructions, instruction_version=instruction_version)
    cells, seen, prompts, sources = defaultdict(list), set(), {}, []
    definitions, detectors = {}, {}
    # Validate every row before selecting any examples, including unused samples.
    for path in sorted(Path(path).resolve() for path in paths):
        if path.with_suffix('.manifest.json').exists():
            raise ValueError('Manifest-bearing runs require the validated calibration export')
        rows = read_rows(path)
        if not rows:
            raise ValueError(f'No full generations found: {path}')
        source_hash = fingerprint(rows)
        sources.append(dict(path=str(path), fingerprint=source_hash, count=len(rows)))
        for row in rows:
            if not isinstance(row, dict) or row.get('suite_record_type') != 'suite_generation':
                raise ValueError('Expected full suite_generation records, not evaluation excerpts')
            if any(field in row for field in ('run_contract', 'generation_contract', 'evaluation_contract')):
                raise ValueError('Contracted records require the validated calibration export')
            identity = tuple(row.get(field) for field in ('model', 'suite', 'prompt_id', 'condition', 'sample_idx'))
            if (any(not isinstance(value, str) or not value for value in identity[:4])
                    or type(identity[4]) is not int or identity[4] < 0):
                raise ValueError('Invalid generation identity')
            if identity in seen:
                raise ValueError(f'Duplicate generation identity: {identity}')
            seen.add(identity)
            if not isinstance(row.get('prompt_text'), str) or not row['prompt_text'].strip():
                raise ValueError('Every generation needs its saved user request')
            if not isinstance(row.get('completion'), str):
                raise ValueError('Every generation needs its full saved response')
            if row.get('system_prompt') is not None and not isinstance(row['system_prompt'], str):
                raise ValueError('Invalid saved system prompt')
            if any(row.get(field) is not None and type(row[field]) is not bool for field in QUALITY_FIELDS):
                raise ValueError('Generation quality flags must be boolean or null')
            prompt_key = (row['suite'], row['prompt_id'])
            if prompt_key in prompts and prompts[prompt_key] != row['prompt_text']:
                raise ValueError(f'Conflicting saved request text: {prompt_key}')
            prompts[prompt_key] = row['prompt_text']
            suite = row['suite']
            if suite not in detectors:
                if suite not in SUITE_REGISTRY:
                    raise ValueError(f'Unknown suite: {suite}')
                matches = [d for d in SUITE_REGISTRY[suite].detectors if d.name == 'rest_suggestion']
                if len(matches) != 1 or type(matches[0]) is not PhraseSetDetector:
                    raise ValueError('Archived import supports only the local rest_suggestion phrase detector')
                detectors[suite] = matches[0]
                definitions[suite] = detector_definition(matches[0])
            cell = (str(path),) + identity[:4]
            cells[cell].append((row, source_hash))
    tasks, keys, strata = [], [], []
    for cell, rows in sorted(cells.items()):
        selected = sorted(rows, key=lambda pair: fingerprint(dict(seed=seed, generation=pair[0])))[:samples_per_cell]
        strata.append(dict(source_path=cell[0], model=cell[1], suite=cell[2], prompt_id=cell[3],
                           condition=cell[4], available=len(rows), selected=len(selected)))
        for row, source_hash in selected:
            item_id = fingerprint(dict(generation=row, annotation=annotation))
            task = dict(id=item_id, **annotation, prompt=row['prompt_text'],
                        system_prompt=row.get('system_prompt'), completion=row['completion'],
                        generation_quality={field: row.get(field) for field in QUALITY_FIELDS},
                        human_hit=None, advice_requested=None, notes='', review_status='unreviewed')
            tasks.append(task)
            keys.append(dict(id=item_id, predicted_hit=detectors[row['suite']].detect(row['completion']).hit,
                detector='rest_suggestion', detector_definition=definitions[row['suite']],
                prediction_origin='rescored_current_heuristic',
                **{field: row[field] for field in ('model', 'suite', 'prompt_id', 'condition', 'sample_idx')},
                task_type=row.get('task_type'), annotation_target=annotation['annotation_target'],
                instruction_version=instruction_version, task_fingerprint=task_fingerprint(task),
                source_path=cell[0], source_file_fingerprint=source_hash,
                generation_fingerprint=fingerprint(row), evaluation_contract=None,
                generation_metadata={field: row.get(field) for field in
                    (*QUALITY_FIELDS, 'max_tokens_requested', 'finish_reason', 'tokens_used')},
                inclusion_probability=len(selected)/len(rows), purpose='annotation_development'))
    sampling = dict(method='lowest seeded generation hashes within each source/model/suite/prompt/condition cell',
                    seed=seed, samples_per_cell=samples_per_cell, sources=sources, strata=strata,
                    available_count=len(seen), selected_count=len(tasks))
    population = dict(count=len(keys), ids_hash=fingerprint(sorted(row['id'] for row in keys)),
                      source_coverage_verified=False)
    for key in keys:
        key['export_population'] = population
        key['sampling'] = sampling
    return sorted(tasks, key=lambda task: task['id']), keys
