"""Describe all registered actor-response probes without dropping seed pairs."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics


FIELDS = ('q_actor_gradient_norm', 'radial_q_actor_gradient_norm', 'radial_to_total_q_actor_norm',
          'direct_radial_action_to_environment_response_rms', 'radial_parameter_to_environment_response_rms',
          'unit_radial_parameter_environment_response_rms', 'entropy_actor_gradient_norm', 'full_actor_gradient_norm')


def summarize(report, plan):
    if (report['training_source_commit'] != plan['training_source_commit']
            or report['tool_sha256'] != plan['tool_sha256'] or report['expected_cases'] != plan['expected_cases']
            or len(report['records']) != plan['expected_cases']):
        raise ValueError('Actor-response study identity differs')
    records = {r['index']: r for r in report['records']}
    if len(records) != len(report['records']) or set(records) != {c['index'] for c in plan['cases']}:
        raise ValueError('Duplicate or missing registered case identities')
    groups = {}
    for case in plan['cases']:
        row = records[case['index']]
        for key in ('task', 'method', 'seed', 'run_dir'):
            if row.get(key) != case[key]: raise ValueError('Actor-response case identity changed')
        valid = row['status'] == 'complete'
        if valid:
            if (type(row.get('returncode')) is not int or row['returncode'] != 0
                    or row['source_commit'] != plan['training_source_commit'] or row['tool_sha256'] != plan['tool_sha256']):
                raise ValueError('Complete actor response lacks source/exit evidence')
            for key in ('checkpoint_sha256', 'config_sha256', 'flags_sha256'):
                if row[key] != case[key]: raise ValueError('Checkpoint/config identity changed')
            if row['panel'] != plan['panels'][case['task']]: raise ValueError('Fixed state panel changed')
            metrics = row['metrics']
            if (metrics['samples'] != 128 or metrics['seed'] != plan['actor_noise_seed']
                    or metrics['valid_direction_samples']+metrics['small_direction_samples'] != 128
                    or metrics['optimizer_updates_performed'] != 0
                    or metrics['network_step_before'] != metrics['network_step_after'] or metrics['network_step_before'] != case['network_step']):
                raise ValueError('Actor response does not satisfy its fixed derivative contract')
            for key in FIELDS:
                value = metrics[key]
                if value is not None and (not math.isfinite(value) or value < 0):
                    raise ValueError('Invalid gradient norm or response')
        else:
            metrics = {}
        groups.setdefault((case['task'], case['method']), []).append(dict(seed=case['seed'], index=case['index'],
            status=row['status'], values={key: metrics.get(key) for key in FIELDS}))
    output = []
    for (task, method), seeds in groups.items():
        if sorted(r['seed'] for r in seeds) != [10, 11, 12, 13]: raise ValueError('Registered four-seed group changed')
        complete = all(r['status'] == 'complete' for r in seeds)
        summary = {}
        for key in FIELDS:
            values = [r['values'][key] for r in seeds if r['values'][key] is not None]
            summary[key] = dict(defined_seeds=len(values), mean=statistics.fmean(values) if complete and len(values)==4 else None,
                               minimum=min(values) if complete and len(values)==4 else None,
                               maximum=max(values) if complete and len(values)==4 else None)
        output.append(dict(task=task, method=method, status='complete' if complete else 'incomplete', seeds=seeds, summary=summary))
    return dict(status='complete' if all(r['status']=='complete' for r in output) else 'incomplete', groups=output,
        expected_cases=plan['expected_cases'], available_cases=sum(r['status']=='complete' for r in report['records']),
        source_commit=plan['training_source_commit'], probe_tool_sha256=plan['tool_sha256'],
        interpretation='Four-training-seed fixed-panel descriptive summaries. No CI, p-value, optimizer update, or return attribution. '
        'All four seeds required for means; undefined unit responses remain undefined. Direct-action and parameter-direction '
        'responses have different input directions/units; their numerical ratio is not an amplification or waste fraction.')


def write_summary(output, result):
    output = Path(output); markdown = output.with_suffix('.md')
    if output.exists() or markdown.exists(): raise FileExistsError('Preserve existing response summary')
    lines = ['# Fixed-panel actor radial responses', '', result['interpretation'], '',
        '| Task | Method | Complete seeds | Radial actor Q norm | Direct action radial response | Parameter-direction response | Unit parameter response |',
        '|---|---|---:|---:|---:|---:|---:|']
    for row in result['groups']:
        def value(key):
            item=row['summary'][key]
            return f"{item['mean']:.6g}" if item['mean'] is not None else f"N/A ({item['defined_seeds']}/4 defined)"
        count=sum(s['status']=='complete' for s in row['seeds'])
        keys=('radial_q_actor_gradient_norm','direct_radial_action_to_environment_response_rms',
              'radial_parameter_to_environment_response_rms','unit_radial_parameter_environment_response_rms')
        lines.append('| '+row['task']+' | '+row['method']+f' | {count}/4 | '+' | '.join(value(k) for k in keys)+' |')
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
    with markdown.open('x') as f:f.write('\n'.join(lines)+'\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--report',type=Path,required=True)
    parser.add_argument('--plan',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();report=json.loads(args.report.read_text());plan=json.loads(args.plan.read_text())
    digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    if report['plan_sha256']!=digest(args.plan):raise ValueError('Analysis plan identity differs')
    result=summarize(report,plan);result.update(report_sha256=digest(args.report),plan_sha256=digest(args.plan),summary_tool_sha256=digest(Path(__file__)))
    write_summary(args.output,result)
    print(json.dumps(dict(status=result['status'],groups=len(result['groups']),available_cases=result['available_cases'])))


if __name__=='__main__':main()
