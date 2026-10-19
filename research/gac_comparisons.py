"""Describe all frozen GAC/RLPD contrasts using complete paired terminal evidence.

This is a report auditor, not another raw-log or checkpoint validator. It checks
the lifecycle receipts and recomputes every reported curve metric. Missing or
failed runs remain visible; a four-pair estimate requires all registered seeds.
"""
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import re

import numpy as np

GAC = 'gac_reference_comparison_20261019'
NATIVE = 'action_adapter_reference_20261019'
POSTHOC = 'posthoc_reference_20261019'
SOURCES = {GAC: '371fe5b463d41287b7cde4201d2f08c4b0e5af87',
           NATIVE: '2c2d0c3054aebf6025f13a0836c12feeff2ab776',
           POSTHOC: 'e20834c88c10fc1a6ef3e6b9a2012d41f9a51fe0'}
SEEDS = (10, 11, 12, 13)
ENVIRONMENTS = {'triple2': 'cube-triple-play-singletask-task2-v0',
                'double2': 'cube-double-play-singletask-task2-v0'}
GRID = list(range(0, 1_000_001, 100_000))
UPDATES = dict(critic=994999, actor=994998, target=994999)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def verify_science(root):
    root = Path(root)
    binding, cohort, rows = (read(root / name) for name in ('freeze-binding.json', 'cohort.json', 'tasks.json'))
    if (digest(root/'cohort.json') != binding['cohort_sha256']
            or digest(root/'tasks.json') != binding['tasks_sha256']
            or binding['plans'] != cohort['plan_hashes']):
        raise ValueError('Frozen science binding differs')
    if (cohort['cohort'] != GAC or cohort['source_commit'] != SOURCES[GAC]
            or cohort['seeds'] != list(SEEDS) or cohort['expected_runs'] != 32
            or cohort['budget'] != 1_000_000 or cohort['evaluation_steps'] != GRID
            or cohort['evaluation_episodes'] != 50 or cohort['evaluation_seed'] != 20261019
            or cohort['expected_updates'] != UPDATES or cohort['contrast_count'] != 24
            or len(cohort['contrasts']) != 24 or len({r['name'] for r in cohort['contrasts']}) != 24):
        raise ValueError('Frozen cohort scope differs')
    expected = {(task, profile, protocol, seed) for task in ENVIRONMENTS
                for profile in ('r1', 'r25') for protocol in ('online_only', 'prior_balanced') for seed in SEEDS}
    if (len(rows) != 32 or len({r['name'] for r in rows}) != 32
            or {(r['task'], r['profile'], r['protocol'], r['seed']) for r in rows} != expected
            or len(binding['plans']) != 32):
        raise ValueError('Frozen GAC identity matrix differs')
    for gpu, seed in enumerate(SEEDS):
        lane = sorted((r for r in rows if r['gpu'] == gpu), key=lambda r: r['position'])
        if len(lane) != 8 or [r['position'] for r in lane] != list(range(8)) or {r['seed'] for r in lane} != {seed}:
            raise ValueError('Frozen GPU/seed queue differs')
    for row in rows:
        relative = Path(row['plan'])
        path = (root/relative).resolve(); path.relative_to(root.resolve())
        if relative.is_absolute() or digest(path) != row['plan_sha256'] or row['plan_sha256'] != binding['plans'][row['plan']]:
            raise ValueError('Frozen plan identity differs')
        plan = read(path)
        expected_plan = dict(seed=row['seed'], env_name=ENVIRONMENTS[row['task']], replay_protocol=row['protocol'],
                             online_steps=1_000_000, eval_interval=100_000, eval_episodes=50, eval_seed=20261019,
                             source_commit=SOURCES[GAC], device='cuda',
                             action_execution='identity' if row['profile'] == 'r1' else 'clip')
        if (any(plan.get(k) != v for k, v in expected_plan.items()) or row['env_name'] != plan['env_name']
                or plan['learner']['radius'] != (1. if row['profile'] == 'r1' else 2.5)):
            raise ValueError('Plan disagrees with its frozen identity')
    return cohort, rows


def reference_rows(cohort):
    prefix = 'aa_reference' if cohort == NATIVE else 'aa_posthoc_reference'
    return [dict(name=f'{prefix}_{task}_utd{utd}', seed=seed, task=task, utd=utd)
            for task in ENVIRONMENTS for utd in (1, 20) for seed in SEEDS]


def recompute_curve(row, gac):
    metrics = row['metrics'] if gac else row
    steps, rates = metrics['eval_steps'], np.asarray(metrics['success'], dtype=float)
    if (steps != GRID or any(type(v) is not int for v in steps) or rates.shape != (11,)
            or not np.isfinite(rates).all() or np.any((rates < 0) | (rates > 1))):
        raise ValueError('Expected the complete actual 0--1M success grid')
    reached = [step for step, rate in zip(steps, rates) if rate >= .5]
    recomputed = dict(auc=float(np.trapezoid(rates, steps)/1_000_000), endpoint=float(rates[-1]),
                      tail3=float(rates[-3:].mean()), time_to_50=reached[0] if reached else None,
                      time_to_50_censored=not bool(reached))
    for key in ('auc', 'endpoint', 'tail3'):
        if type(metrics.get(key)) not in (int, float) or not np.isclose(metrics[key], recomputed[key], atol=1e-12, rtol=0):
            raise ValueError('Reported metric differs from the curve: '+key)
    if (type(metrics.get('time_to_50_censored')) is not bool
            or metrics['time_to_50_censored'] != recomputed['time_to_50_censored']
            or (reached and type(metrics.get('time_to_50')) is not int)
            or metrics.get('time_to_50') != recomputed['time_to_50']):
        raise ValueError('Reported threshold time/censoring differs')
    return recomputed


def audit_report(report, cohort_name, definitions):
    if report is None:
        return {}
    expected = {(r['name'], r['seed']): r for r in definitions}
    records = report['records']
    if (report['source_commit'] != SOURCES[cohort_name] or report['expected_runs'] != len(expected)
            or len(records) != len(expected) or {(r['name'], r['seed']) for r in records} != set(expected)):
        raise ValueError('Terminal report source or full identity matrix differs')
    complete = sum(r['status'] == 'complete' for r in records)
    if (type(report['complete_runs']) is not int or report['complete_runs'] != complete
            or report['status'] not in ('complete', 'needs_review')
            or (report['status'] == 'complete') != (complete == len(expected))):
        raise ValueError('Terminal completion count/status differs')
    result = {}
    for row in records:
        key = row['name'], row['seed']; definition = expected[key]
        if row['status'] != 'complete':
            result[key] = dict(status=row['status'], identity=key,
                               error=row.get('validation_error', row.get('error')))
            continue
        if (type(row.get('returncode')) is not int or row['returncode'] != 0
                or row.get('source_commit') != SOURCES[cohort_name]):
            raise ValueError('Complete receipt lacks a verified source/exit')
        gac = cohort_name == GAC
        if gac:
            if (type(row.get('validation_returncode')) is not int or row['validation_returncode'] != 0
                    or row['plan_sha256'] != definition['plan_sha256'] or row['online_steps'] != 1_000_000
                    or row['actual_updates'] != UPDATES or row['exact_checkpoint_fields'] != 179
                    or any(type(value) is not int for value in row['actual_updates'].values())
                    or row['env_name'] != ENVIRONMENTS[definition['task']]
                    or row['replay_protocol'] != definition['protocol']
                    or row['radius'] != (1. if definition['profile'] == 'r1' else 2.5)
                    or row['action_execution'] != ('identity' if definition['profile'] == 'r1' else 'clip')
                    or not re.fullmatch(r'cuda(?::[0-9]+)?', row['runtime']['device'])
                    or row['action_summary']['environment_steps'] != 1_000_000):
                raise ValueError('GAC receipt lacks complete independent CUDA validation')
            clocks = row['actual_updates']
        else:
            clocks = dict(actor=995001, temp=995001, critic=995001*definition['utd'], target_critic=0)
            if (row['budget'] != 1_000_000 or row['actual_optimizer_updates'] != clocks
                    or any(type(value) is not int for value in row['actual_optimizer_updates'].values())
                    or row['observed_eval_episode_counts'] != 'verified'
                    or row.get('cuda_visible_devices') not in ('0', '1', '2', '3')):
                raise ValueError('RLPD receipt lacks the frozen budget, observed counts or update clocks')
        elapsed = row['ended']-row['started']
        if not np.isfinite(elapsed) or elapsed < 0:
            raise ValueError('Invalid observed driver elapsed time')
        result[key] = dict(status='complete', identity=list(key), metrics=recompute_curve(row, gac),
                           source_commit=row['source_commit'], elapsed_seconds=elapsed, actual_updates=clocks,
                           effective_config=row['effective_config'])
    return result


def paired_summary(values):
    values = np.asarray(values, dtype=float)
    if values.shape != (4,) or not np.isfinite(values).all():
        raise ValueError('Exactly four finite paired differences are required')
    indices = np.asarray(list(itertools.product(range(4), repeat=4)))
    return dict(mean=float(values.mean()), median=float(np.median(values)),
                positive_pairs=int((values > 0).sum()), negative_pairs=int((values < 0).sum()),
                zero_pairs=int((values == 0).sum()),
                descriptive_bootstrap_95=np.quantile(values[indices].mean(axis=1), [.025, .975]).tolist())


def summarize(cohort, rows, reports):
    indexed = {name: audit_report(reports.get(name), name, rows if name == GAC else reference_rows(name)) for name in SOURCES}
    groups = {(GAC, r['task'], r['profile'], r['protocol'], r['seed']): (r['name'], r['seed']) for r in rows}
    def arm(spec, seed):
        name = spec['cohort']
        if name == GAC:
            key = groups[name, spec['task'], spec['profile'], spec['protocol'], seed]
        elif name in (NATIVE, POSTHOC):
            key = spec['group'], seed
            if key not in {(r['name'], r['seed']) for r in reference_rows(name)}:
                raise ValueError('Unknown reference comparison arm')
        else:
            raise ValueError('Unknown comparison cohort')
        return indexed[name].get(key), list(key)
    comparisons = []
    for contrast in cohort['contrasts']:
        pairs, unavailable = [], []
        for seed in SEEDS:
            left, left_key = arm(contrast['left'], seed); right, right_key = arm(contrast['right'], seed)
            bad = [dict(side=side, cohort=contrast[side]['cohort'], identity=key,
                        status=value['status'] if value else 'report_missing', error=value.get('error') if value else None)
                   for side, value, key in [('left', left, left_key), ('right', right, right_key)]
                   if value is None or value['status'] != 'complete']
            if bad:
                unavailable.append(dict(seed=seed, reasons=bad)); continue
            differences = {key: left['metrics'][key]-right['metrics'][key] for key in ('auc', 'endpoint', 'tail3')}
            differences['elapsed_seconds'] = left['elapsed_seconds']-right['elapsed_seconds']
            configs = left['effective_config'], right['effective_config']
            changed = {key: [configs[0].get(key), configs[1].get(key)] for key in sorted(set(configs[0]) | set(configs[1]))
                       if configs[0].get(key) != configs[1].get(key)}
            pairs.append(dict(seed=seed, left_identity=left_key, right_identity=right_key, differences=differences,
                              time_to_50={side: {key: value['metrics'][key] for key in ('time_to_50', 'time_to_50_censored')}
                                          for side, value in [('left', left), ('right', right)]},
                              actual_updates=dict(left=left['actual_updates'], right=right['actual_updates']),
                              effective_config_differences=changed))
        comparison = dict(contrast, status='incomplete' if unavailable else 'complete', expected_pairs=4,
                          pairs=pairs, unavailable_seeds=unavailable)
        if not unavailable:
            comparison['summary'] = {key: paired_summary([pair['differences'][key] for pair in pairs])
                                     for key in ('auc', 'endpoint', 'tail3', 'elapsed_seconds')}
        comparisons.append(comparison)
    return dict(status='complete' if all(c['status'] == 'complete' for c in comparisons) else 'incomplete',
                cohort=GAC, source_commit=SOURCES[GAC], comparisons=comparisons, contrast_count=len(comparisons),
                available_complete_runs={name: sum(r['status'] == 'complete' for r in data.values()) for name, data in indexed.items()},
                interpretation='Four-seed development comparisons; 256 empirical paired resamples, descriptive only. '
                'Full algorithm comparisons retain differing networks, clocks, regularization and action supports. '
                'Elapsed times reflect sequential cohorts and load, not a controlled compute-efficiency effect. '
                'Threshold times retain censoring, with no mean over successful runs. '
                'Audits lifecycle reports, not raw logs, checkpoints or historical executed processes again.')


def write_report(output, result):
    output = Path(output); markdown = output.with_suffix('.md')
    if output.exists() or markdown.exists():
        raise FileExistsError('Preserve the existing comparison attempt')
    output.parent.mkdir(parents=True, exist_ok=True)
    lines = ['# Frozen GAC comparisons', '', result['interpretation'], '',
             '| Contrast | Complete pairs | AUC difference (pp) | Descriptive 95% interval (pp) |',
             '|---|---:|---:|---:|']
    for row in result['comparisons']:
        if row['status'] == 'complete':
            value = row['summary']['auc']; low, high = value['descriptive_bootstrap_95']
            lines.append(f"| {row['name']} | 4/4 | {100*value['mean']:+.3f} | [{100*low:+.3f}, {100*high:+.3f}] |")
        else:
            lines.append(f"| {row['name']} | {len(row['pairs'])}/4 | incomplete | not estimated |")
    with output.open('x') as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False)+'\n')
    with markdown.open('x') as stream:
        stream.write('\n'.join(lines)+'\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--science-root', type=Path, required=True)
    for name in ('gac', 'native', 'posthoc'):
        parser.add_argument('--'+name+'-report', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); cohort, rows = verify_science(args.science_root)
    paths = {GAC: args.gac_report, NATIVE: args.native_report, POSTHOC: args.posthoc_report}
    reports = {name: read(path) if path.is_file() else None for name, path in paths.items()}
    result = summarize(cohort, rows, reports)
    inputs = [args.science_root/name for name in ('freeze-binding.json', 'cohort.json', 'tasks.json')]
    inputs += [args.science_root/row['plan'] for row in rows]
    result.update(tool_sha256=digest(__file__), input_hashes={str(p): digest(p) for p in inputs},
                  report_inputs={name: dict(path=str(path), sha256=digest(path) if path.is_file() else None) for name, path in paths.items()})
    write_report(args.output, result)
    print(json.dumps(dict(status=result['status'], contrasts=len(result['comparisons']))))


if __name__ == '__main__':
    main()
