"""Gated, single-attempt GAC launch and terminal analysis on the existing job.

Environment/image preparation is separate. Image evidence is a credential-free
projection of actual save, detail and usable-list observations, never a claim
that a later instance has pulled or booted the saved image.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import types

from research import gac_comparisons as comparisons
from research import gac_environment_runtime as environment
from research import gac_execution as execution

GAC = comparisons.GAC
ANALYSIS = '45a0998bad82a5bf95ba7789936d1584d95701df'
TOOL_FILES = {'research/gac_runtime.py', 'research/gac_execution.py',
              'research/gac_comparisons.py', 'research/gac_environment_runtime.py'}
IMAGE_FIELDS = {'id', 'name', 'tag', 'path', 'status', 'jobId', 'clusterId', 'teamId', 'uploadType'}


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return environment.digest(path)


def load_spec(path):
    path = Path(path).resolve(); spec = read(path)
    if spec['format'] != 'gac-runtime-spec-v1' or set(spec['controller_files']) != TOOL_FILES:
        raise ValueError('Unknown controller specification or source set')
    for name, expected in spec['controller_files'].items():
        if digest(path.parent/name) != expected:
            raise ValueError('Controller source differs: '+name)
    if digest(spec['environment_spec']) != spec['environment_spec_sha256']:
        raise ValueError('Environment specification differs')
    if read(spec['environment_spec'])['environment_tool_sha256'] != spec['controller_files']['research/gac_environment_runtime.py']:
        raise ValueError('Prepared environment tool and controller import differ')
    spec.update(_spec_path=str(path), _spec_sha256=digest(path), _tools_root=str(path.parent))
    return spec


def check_imports(spec):
    for name, module in [('gac_runtime', sys.modules[__name__]), ('gac_execution', execution),
                         ('gac_comparisons', comparisons), ('gac_environment_runtime', environment)]:
        if Path(module.__file__).resolve() != Path(spec['_tools_root'])/'research'/(name+'.py'):
            raise ValueError('Controller imported outside its bound release: '+name)


def verified_runtime(spec, env):
    runtime = Path(env['runtime_root']); manifest_path = runtime/'manifest.json'
    if digest(manifest_path) != env['runtime_manifest_sha256'] or digest(env['runtime_archive']) != env['runtime_sha256']:
        raise ValueError('Runtime manifest/archive differs')
    manifest = read(manifest_path); helper = runtime/'verify_bundle.py'
    if (manifest['source_commit'] != execution.SOURCE or manifest['analysis_commit'] != ANALYSIS
            or manifest['dependency_archive_sha256'] != env['dependency_sha256']
            or manifest['dependency_manifest_sha256'] != env['dependency_manifest_sha256']
            or digest(helper) != manifest['files']['verify_bundle.py']):
        raise ValueError('Runtime source, analysis or dependency binding differs')
    module = types.ModuleType('_gac_launch_verified_bundle'); module.__file__ = str(helper)
    exec(compile(helper.read_bytes(), str(helper), 'exec'), module.__dict__)
    module.verify(runtime, 'gac-runtime')
    cohort, rows = comparisons.verify_science(runtime/'science')
    if (cohort['job_id'] != spec['job_id'] or env['job_id'] != spec['job_id']
            or digest(runtime/'science/cohort.json') != env['science_cohort_sha256']):
        raise ValueError('Registered existing-job identity differs')
    return runtime, manifest, cohort, rows


def zero_exit(value):
    return type(value.get('returncode')) is int and value['returncode'] == 0


def verify_engineering(spec, env, runtime, manifest, gate):
    control = Path(env['control']); ready_path = Path(spec['environment_ready'])
    ready = read(ready_path)
    if (ready['status'] != 'cloud_engineering_ready_for_image' or ready['prefix'] != env['prefix']
            or ready['source_commit'] != execution.SOURCE or ready['analysis_commit'] != ANALYSIS
            or ready['spec_sha256'] != digest(spec['environment_spec'])
            or ready['upstream_gate'] != gate or ready['expected_runs'] != 8 or ready['validated_runs'] != 8
            or ready['engineering_steps_per_run'] != 64 or ready['numerical_cases'] != 4):
        raise ValueError('Cloud engineering ready receipt differs')
    expected = {r['name']: r for r in environment.engineering_plans(runtime)}
    records = ready['engineering_runs']
    if len(records) != 8 or {r['name'] for r in records} != set(expected):
        raise ValueError('Cloud engineering lacks all eight registered branches')
    inputs = {str(ready_path): digest(ready_path)}
    initial = {}
    for row in records:
        case = expected[row['name']]; name = row['name']
        driver, validation = row['driver'], row['validation']
        plan_path = control/'plans'/(name+'.json')
        result_path = control/'validations'/(name+'.json')
        driver_process = control/'processes'/(name+'-bound.json')
        validator_process = control/'processes'/(name+'-validation.json')
        if (row['gpu'] != case['gpu'] or row['env_name'] != case['plan']['env_name']
                or not zero_exit(driver) or not zero_exit(validation) or driver['pid'] == validation['pid']
                or any(type(p['pid']) is not int or p['pid'] < 1 for p in (driver, validation))
                or driver['ended'] > validation['started']
                or driver['argv'][0] != str(Path(env['prefix'])/'bin/python')
                or validation['argv'][0] != driver['argv'][0]
                or read(plan_path) != case['plan'] or driver['plan_sha256'] != digest(plan_path)
                or read(driver_process) != driver or read(validator_process) != validation
                or row['report'] != str(result_path) or row['report_sha256'] != digest(result_path)):
            raise ValueError('Cloud engineering process/plan/report identity differs')
        result = read(result_path)
        if (result['status'] != 'complete' or not zero_exit(result) or result['source_commit'] != execution.SOURCE
                or result['seed'] != case['plan']['seed'] or result['env_name'] != row['env_name']
                or result['online_steps'] != 64 or result['plan_sha256'] != digest(plan_path)
                or result['actual_updates'] != dict(critic=31, actor=30, target=31)
                or result['exact_checkpoint_fields'] != 179 or result['runtime']['device'] not in ('cuda', 'cuda:0')):
            raise ValueError('Cloud short run lacks independent complete CUDA validation')
        initial.setdefault(row['env_name'], set()).add(row['initial_model_and_rng_sha256'])
        for path in (plan_path, result_path, driver_process, validator_process):
            inputs[str(path)] = digest(path)
    if set(initial) != set(environment.TASKS) or any(len(values) != 1 for values in initial.values()):
        raise ValueError('Engineering initialization differs within task')
    report_path = control/'environment.json'; report = read(report_path)
    if (report['status'] != 'passed' or report['prefix'] != env['prefix'] or report['torch'] != '2.14.0+cu126'
            or report['compiled_cuda'] != '12.6' or len(report['gpus']) != 4
            or not all('4090' in name for name in report['gpus'])
            or report['inherited_system_site_packages'] is not False or report['jax_installed'] is not False
            or report['flax_installed'] is not False or not report['cuda_libraries']):
        raise ValueError('Prepared four-card environment differs')
    for library in report['cuda_libraries']:
        Path(library).resolve().relative_to(Path(env['prefix']).resolve())
    default, radius = read(control/'numerical-default.json'), read(control/'numerical-radius25.json')
    if (default['status'] != 'passed' or default['backend'] != 'cuda' or len(default['cases']) != 3
            or default['local_source_sha256'] != manifest['training_python_sources']['agents/gac_reference.py']
            or default['oracle_sha256'] != manifest['training_python_sources']['research/check_gac_reference.py']
            or default['upstream_commit'] != manifest['upstream_commit']
            or default['upstream_source_sha256'] != manifest['files']['official/GAC.py']
            or any(r['status'] != 'passed' or r['device'] != 'cuda' or r['max_absolute_state_error'] != 0 for r in default['cases'])
            or radius['status'] != 'passed' or radius['device'] != 'cuda' or radius['max_absolute_state_error'] != 0
            or radius['local_learner_sha256'] != default['local_source_sha256']
            or radius['checker_sha256'] != default['oracle_sha256']
            or radius['upstream_source_sha256'] != default['upstream_source_sha256']
            or radius['config']['radius'] != 2.5):
        raise ValueError('Cloud numerical equivalence evidence differs')
    for name in ('create-venv', 'offline-install', 'pip-check', 'four-gpu-environment', 'source-tests',
                 'analysis-tests', 'numerical-default', 'numerical-radius25'):
        path = control/'processes'/(name+'.json')
        if not zero_exit(read(path)):
            raise ValueError('Environment stage lacks a successful observed exit: '+name)
        inputs[str(path)] = digest(path)
    for path in (report_path, control/'numerical-default.json', control/'numerical-radius25.json'):
        inputs[str(path)] = digest(path)
    return dict(status='engineering_verified', input_hashes=inputs, ready=ready)


def verify_image(receipt, spec, env, ready):
    # Only allowlisted projections are uploaded; API detail can contain passwords.
    detail, usable = receipt['detail'], receipt['usable_record']
    allowed_receipt = {'format', 'status', 'source_job_id', 'prefix', 'environment_ready_sha256', 'save_requested_at',
                       'save_request', 'detail', 'usable_record', 'detail_api_path', 'usable_api_path'}
    request = receipt['save_request']
    if (set(detail)-IMAGE_FIELDS or set(usable)-IMAGE_FIELDS or set(receipt)-allowed_receipt
            or set(request)-{'name', 'tag', 'content', 'uploadType', 'sceneList', 'jobId', 'teamId', 'clusterId'}):
        raise ValueError('Image proof contains fields outside its credential-free projection')
    if (receipt['format'] != 'gac-saved-image-v1' or receipt['status'] != 'usable' or receipt['source_job_id'] != spec['job_id']
            or receipt['prefix'] != env['prefix'] or receipt['environment_ready_sha256'] != digest(spec['environment_ready'])
            or receipt['save_requested_at'] < ready['finished'] or request['jobId'] != spec['job_id']
            or request['uploadType'] != 4 or set(request['sceneList']) != {'training', 'training-debug'}
            or request['teamId'] != spec['team_id'] or request['clusterId'] != spec['cluster_id']
            or detail['teamId'] != request['teamId'] or detail['clusterId'] != request['clusterId']
            or detail['jobId'] != spec['job_id'] or detail['status'] != 2 or usable['status'] != 2
            or not detail['id'] or not detail['path'] or detail['name'] != request['name'] or detail['tag'] != request['tag']
            or not detail['path'].endswith('/'+request['name']+':'+request['tag'])
            or any(c.isspace() for c in detail['path'])
            or any(detail[key] != usable[key] for key in ('id', 'name', 'tag', 'path'))
            or receipt['detail_api_path'] != '/gateway/foundation/api/v1/image/'+detail['id']
            or receipt['usable_api_path'] != '/gateway/foundation/api/v1/image/action/page'):
        raise ValueError('Image save/detail/usable-list evidence differs from this environment')
    return dict(status='image_verified', image_id=detail['id'], image_reference=detail['path'],
                receipt_sha256=digest(spec['image_receipt']), new_instance_pull_verified=False)


def image_projection(detail):
    """Project actual API data in memory; never serialize its credential fields."""
    return {key: value for key, value in detail.items() if key in IMAGE_FIELDS}


def inspect(spec):
    env = read(spec['environment_spec'])
    gate = environment.inspect_gate(env)
    if gate['status'] != 'technical_gate_verified':
        return gate
    if not Path(env['dependency_receipt']).is_file():
        return dict(status='waiting_for_dependency_transfer')
    runtime, manifest, cohort, rows = verified_runtime(spec, env)
    if not Path(spec['environment_ready']).is_file():
        return dict(status='waiting_for_cloud_engineering')
    engineering = verify_engineering(spec, env, runtime, manifest, gate)
    if not Path(spec['image_receipt']).is_file():
        return dict(status='waiting_for_image')
    image = verify_image(read(spec['image_receipt']), spec, env, engineering['ready'])
    inputs = {env[key]: sha for key, sha in gate['input_hashes'].items()}
    inputs.update(engineering['input_hashes'])
    inputs[spec['image_receipt']] = image['receipt_sha256']
    inputs[spec['environment_spec']] = digest(spec['environment_spec'])
    for name, key in ((comparisons.NATIVE, 'native_report'), (comparisons.POSTHOC, 'posthoc_report')):
        report = read(spec[key]); audited = comparisons.audit_report(report, name, comparisons.reference_rows(name))
        if len(audited) != 16 or any(r['status'] != 'complete' for r in audited.values()):
            raise ValueError('Registered RLPD terminal evidence is incomplete')
        inputs[spec[key]] = digest(spec[key])
    return dict(status='ready_to_launch', gate=gate, image=image, input_hashes=inputs,
                runtime_manifest_sha256=digest(runtime/'manifest.json'), expected_runs=len(rows), performance_threshold=None)


def check_layout(spec, env):
    for key, prefix in (('control', '/workspace/control'), ('results', '/workspace/results'), ('reports', '/workspace/reports')):
        if Path(spec[key]).resolve() != Path(prefix)/GAC:
            raise ValueError('Scientific output layout differs: '+key)
    if Path(env['prefix']) != Path('/opt')/('qc-gac-cu126-'+env['dependency_sha256'][:12]):
        raise ValueError('Prepared environment prefix differs')


def command(spec, mode):
    python = str(Path(read(spec['environment_spec'])['prefix'])/'bin/python')
    return [python, '-m', 'research.gac_runtime', mode, '--spec', spec['_spec_path']]


def launch(spec):
    evidence = inspect(spec)
    if evidence['status'] != 'ready_to_launch':
        raise RuntimeError('Scientific launch deferred: '+evidence['status'])
    env = read(spec['environment_spec']); check_layout(spec, env)
    # Rehash the full dependency archive once before committing the launch.
    _, artifacts = environment.verify_artifacts(env)
    if artifacts['status'] != 'artifacts_verified':
        raise RuntimeError('Dependency archive is not complete')
    gpus = subprocess.check_output(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'], text=True).splitlines()
    occupied = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'], text=True).strip()
    if len(gpus) != 4 or not all('4090' in name for name in gpus):
        raise ValueError('Expected four RTX4090 GPUs')
    if occupied:
        return dict(status='waiting_for_idle_gpus', scientific_training_started=False)
    control, results, reports = (Path(spec[key]) for key in ('control', 'results', 'reports'))
    if any(path.exists() for path in (control, results, reports)):
        raise FileExistsError('Preserve the existing scientific launch/attempt')
    control.mkdir(parents=True, exist_ok=False)
    for path in (results, reports): path.mkdir(parents=True, exist_ok=False)
    for name in ('logs', 'processes', 'runs', 'validations', 'tmp'): (control/name).mkdir()
    execution.write(control/'launch.json', dict(spec_sha256=spec['_spec_sha256'], created=time.time(),
                    source_commit=execution.SOURCE, evidence=evidence), exclusive=True)
    argv = command(spec, 'own')
    with (control/'owner.log').open('x') as log:
        process = subprocess.Popen(argv, cwd=spec['_tools_root'],
                    env=environment.clean_environment(env['prefix'], control/'tmp', ''),
                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    record = dict(status='started', pid=process.pid, argv=argv, spec_sha256=spec['_spec_sha256'],
                  source_commit=execution.SOURCE, runs=32, gpus=4, gpu_tasks=1, image=evidence['image'], started=time.time())
    execution.write(control/'started.json', record, exclusive=True)
    return record


def verify_launch(spec):
    control = Path(spec['control']); launch = read(control/'launch.json')
    if launch['spec_sha256'] != spec['_spec_sha256'] or launch['source_commit'] != execution.SOURCE:
        raise ValueError('Existing launch identity differs')
    for path, sha in launch['evidence']['input_hashes'].items():
        if digest(path) != sha:
            raise ValueError('Launch evidence changed: '+path)
    return control, launch


def active_driver(record, plan_path, source):
    pid = record.get('pid')
    if type(pid) is not int:
        return False
    path = Path('/proc')/str(pid)/'cmdline'
    try: args = path.read_bytes().split(bytes([0]))
    except FileNotFoundError: return False
    return str(Path(source)/'main_gac_reference.py').encode() in args and str(plan_path).encode() in args


def collect_records(spec, rows, source):
    control = Path(spec['control']); science = Path(read(spec['environment_spec'])['runtime_root'])/'science'
    records = []
    for row in rows:
        path = control/'runs'/(row['name']+'.json')
        record = read(path) if path.exists() else dict(name=row['name'], seed=row['seed'], status='missing')
        if record['name'] != row['name'] or record['seed'] != row['seed']:
            raise ValueError('Run receipt identity differs')
        if active_driver(record, science/row['plan'], source):
            raise RuntimeError('A driver is still live; do not infer a terminal outcome')
        if record['status'] == 'complete':
            validation = control/'validations'/(row['name']+'.json')
            if (record['validation_report_sha256'] != digest(validation)
                    or any(record.get(key) != value for key, value in read(validation).items())):
                raise ValueError('Independent validation output changed')
        records.append(record)
    return records


def supervise(spec):
    control, launch_record = verify_launch(spec); env = read(spec['environment_spec'])
    runtime, manifest, cohort, rows = verified_runtime(spec, env)
    execution.write(control/'supervisor-started.json', dict(pid=os.getpid(), spec_sha256=spec['_spec_sha256'],
                    started=time.time()), exclusive=True)
    validator = runtime/'analysis/research/gac_reference_validation.py'
    validator_sha = manifest['files']['analysis/research/gac_reference_validation.py']
    def execute(row):
        return execution.execute_case(row, science=runtime/'science', source=runtime/'source', validator=validator,
                    validator_sha256=validator_sha, python=Path(env['prefix'])/'bin/python', control=control,
                    results=Path(spec['results']),
                    environment=environment.clean_environment(env['prefix'], control/'tmp', str(row['gpu'])))
    observed = execution.run_lanes(rows, execute)
    records = collect_records(spec, rows, runtime/'source')
    complete = sum(r['status'] == 'complete' for r in records)
    report = dict(cohort=GAC, source_commit=execution.SOURCE, analysis_commit=ANALYSIS, expected_runs=32,
                  complete_runs=complete, status='complete' if complete == 32 else 'needs_review', records=records,
                  lane_errors=observed['lane_errors'], cohort_sha256=digest(runtime/'science/cohort.json'),
                  tasks_sha256=digest(runtime/'science/tasks.json'), image=launch_record['evidence']['image'])
    reports = Path(spec['reports']); run_path = reports/'gac-run-report.json'
    execution.write(run_path, report, exclusive=True)
    result = comparisons.summarize(cohort, rows, {GAC: report,
                    comparisons.NATIVE: read(spec['native_report']), comparisons.POSTHOC: read(spec['posthoc_report'])})
    result.update(tool_sha256=digest(Path(comparisons.__file__)), gac_run_report_sha256=digest(run_path),
                  launch_evidence=launch_record['evidence'])
    comparisons.write_report(reports/'gac-comparison-report.json', result)
    code = 0 if complete == 32 and result['status'] == 'complete' and not observed['lane_errors'] else 2
    execution.write(control/'supervisor-result.json', dict(returncode=code, source_commit=execution.SOURCE,
                    expected_runs=32, complete_runs=complete, status='complete' if code == 0 else 'needs_review',
                    contrasts=24, comparison_status=result['status'], lane_errors=observed['lane_errors']), exclusive=True)
    return code


def own(spec):
    control, _ = verify_launch(spec); env = read(spec['environment_spec'])
    execution.write(control/'owner-started.json', dict(pid=os.getpid(), started=time.time(),
                    spec_sha256=spec['_spec_sha256']), exclusive=True)
    argv = command(spec, 'supervise'); started = time.time()
    with (control/'supervisor.log').open('x') as log:
        process = subprocess.Popen(argv, cwd=spec['_tools_root'],
                    env=environment.clean_environment(env['prefix'], control/'tmp', ''), stdout=log, stderr=subprocess.STDOUT)
        execution.write(control/'scheduler-process.json', dict(pid=process.pid, argv=argv, started=started), exclusive=True)
        code = process.wait()
    execution.write(control/'scheduler-exit.json', dict(pid=process.pid, returncode=code, started=started,
                    ended=time.time(), observed_by_pid=os.getpid()), exclusive=True)
    result_path = control/'supervisor-result.json'
    if result_path.exists():
        result = read(result_path)
        if result['source_commit'] != execution.SOURCE or result['expected_runs'] != 32:
            raise ValueError('Supervisor terminal identity differs')
        if type(result['returncode']) is not int or result['returncode'] != code:
            result.update(status='needs_review', observed_supervisor_exit=code)
    else:
        result = dict(status='needs_review', source_commit=execution.SOURCE, expected_runs=32,
                      complete_runs=None, returncode=code, reason='Supervisor exited without its terminal analysis')
    result['scheduler_exit_observed'] = True
    execution.write(control/'terminal.json', result, exclusive=True)
    return code if code else (0 if result['status'] == 'complete' else 2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('inspect', 'launch', 'own', 'supervise'))
    parser.add_argument('--spec', type=Path, required=True)
    args = parser.parse_args(); spec = load_spec(args.spec)
    check_imports(spec)
    if args.mode == 'inspect': print(json.dumps(inspect(spec)))
    elif args.mode == 'launch': print(json.dumps(launch(spec)))
    else: raise SystemExit(own(spec) if args.mode == 'own' else supervise(spec))


if __name__ == '__main__': main()
