"""GAC process/validation primitives for a separately gated launch controller.

There is intentionally no command-line launch entry here. The controller must
first verify upstream completion, the prepared CUDA environment and saved image.
These primitives preserve every attempt and never restart a scientific run.
"""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

SOURCE = '371fe5b463d41287b7cde4201d2f08c4b0e5af87'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value, *, exclusive=False):
    path = Path(path)
    if exclusive:
        with path.open('x') as stream:
            stream.write(json.dumps(value, indent=2, allow_nan=False)+'\n')
    else:
        temporary = path.with_suffix(path.suffix+'.tmp')
        with temporary.open('x') as stream:
            stream.write(json.dumps(value, indent=2, allow_nan=False)+'\n')
        temporary.replace(path)


def verify_case(row, science, source, validator, validator_sha256, expected_source_files=125):
    science, source, validator = map(Path, (science, source, validator))
    plan_path = (science/row['plan']).resolve(); plan_path.relative_to(science.resolve())
    if digest(plan_path) != row['plan_sha256']:
        raise ValueError('Frozen scientific plan differs')
    plan = read(plan_path)
    if (plan['source_commit'] != SOURCE or plan['seed'] != row['seed'] or plan['env_name'] != row['env_name']
            or plan['replay_protocol'] != row['protocol'] or plan['device'] != 'cuda'
            or len(plan['source_files']) != expected_source_files):
        raise ValueError('Plan/process source or identity differs')
    for name, expected in plan['source_files'].items():
        path = (source/name).resolve(); path.relative_to(source.resolve())
        if digest(path) != expected:
            raise ValueError('Frozen training source differs: '+name)
    if digest(validator) != validator_sha256:
        raise ValueError('Independent validator source differs')
    return plan_path, plan


VALIDATE = r'''
from pathlib import Path
import importlib.util,json,sys
validator,directory,plan,process,source,output=map(Path,sys.argv[1:])
sys.path.insert(0,str(source))
spec=importlib.util.spec_from_file_location('gac_bound_independent_validation',validator)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
result=module.validate_run(directory,plan,json.loads(process.read_text()),source)
if result['status']!='complete' or type(result['returncode']) is not int or result['returncode']!=0:
 raise ValueError('Independent validator did not complete')
with output.open('x') as stream:stream.write(json.dumps(result,indent=2,allow_nan=False)+'\n')
'''


def execute_case(row, *, science, source, validator, validator_sha256, python, control, results,
                 environment=None, expected_source_files=125):
    """Observe one real driver exit, then validate in another process on that GPU."""
    control, results, source, validator = map(Path, (control, results, source, validator))
    name = row['name']
    if Path(name).name != name or name in ('.', '..'):
        raise ValueError('Unsafe run identity')
    env = dict(os.environ if environment is None else environment)
    if env.get('CUDA_VISIBLE_DEVICES') != str(row['gpu']) or row['gpu'] not in range(4):
        raise ValueError('Expected exactly the assigned physical GPU')
    plan_path, plan = verify_case(row, science, source, validator, validator_sha256, expected_source_files)
    directory = results/name
    receipt, process_path, validated_path = [control/folder/(name+suffix) for folder, suffix in
                                           [('runs', '.json'), ('processes', '.json'), ('validations', '.json')]]
    driver_log, validation_log = [control/'logs'/(name+suffix) for suffix in ('.log', '-validation.log')]
    if any(path.exists() for path in (directory, receipt, process_path, validated_path, driver_log, validation_log)):
        raise FileExistsError('Preserve the existing run attempt: '+name)
    for folder in ('runs', 'processes', 'validations', 'logs'):
        (control/folder).mkdir(parents=True, exist_ok=True)
    results.mkdir(parents=True, exist_ok=True)
    record = dict(name=name, source_commit=SOURCE, seed=row['seed'], env_name=row['env_name'],
                  task=row['task'], profile=row['profile'], protocol=row['protocol'], gpu=row['gpu'],
                  position=row['position'], plan_sha256=digest(plan_path), status='launching', started=time.time(),
                  argv=[str(python), str(source/'main_gac_reference.py'), '--plan', str(plan_path), '--output-dir', str(directory)])
    write(receipt, record, exclusive=True)
    try:
        with driver_log.open('x') as log:
            driver = subprocess.Popen(record['argv'], cwd=source, env=env, stdout=log, stderr=subprocess.STDOUT)
            record.update(pid=driver.pid, status='running')
            write(process_path, record, exclusive=True); write(receipt, record)
            code = driver.wait()
        record.update(returncode=code, ended=time.time(), status='driver_failed' if code else 'driver_finished')
        write(process_path, record)
        final = dict(record, status='failed' if code else 'needs_review')
        if code == 0:
            # Fail before loading a changed trusted validator or training source.
            verify_case(row, science, source, validator, validator_sha256, expected_source_files)
            argv = [str(python), '-c', VALIDATE, str(validator), str(directory), str(plan_path),
                    str(process_path), str(source), str(validated_path)]
            with validation_log.open('x') as log:
                child = subprocess.Popen(argv, cwd=source, env=env, stdout=log, stderr=subprocess.STDOUT)
                final.update(validation_pid=child.pid, validation_started=time.time())
                write(receipt, final)
                validation_code = child.wait()
            final.update(validation_returncode=validation_code, validation_ended=time.time())
            record = final
            if validation_code == 0:
                result = read(validated_path)
                if (result['status'] != 'complete' or result['source_commit'] != SOURCE
                        or result['seed'] != row['seed'] or result['env_name'] != row['env_name']
                        or result['plan_sha256'] != row['plan_sha256'] or result['online_steps'] != plan['online_steps']
                        or result['runtime']['device'] not in ('cuda', 'cuda:0')):
                    raise ValueError('Independent validation result identity/backend differs')
                final.update(result)
                final.update(validation_report_sha256=digest(validated_path), validation_source_sha256=validator_sha256)
                verify_case(row, science, source, validator, validator_sha256, expected_source_files)
            else:
                final['validation_error'] = 'Independent validation failed; preserve its observed exit and log'
        write(receipt, final)
        return final
    except Exception as error:
        # A lost wait is an unknown exit, never a signal to restart the driver.
        record.update(status='needs_review', lifecycle_error=repr(error), observation_ended=time.time())
        write(receipt, record)
        raise


def run_lanes(rows, execute):
    """One sequential queue per physical GPU; errors preserve unstarted cases."""
    if (len(rows) != 32 or len({r['name'] for r in rows}) != 32
            or {r['gpu'] for r in rows} != set(range(4))):
        raise ValueError('Expected the frozen 32 unique four-GPU identities')
    lanes = {}
    for gpu in range(4):
        lane = sorted((r for r in rows if r['gpu'] == gpu), key=lambda r: r['position'])
        if len(lane) != 8 or [r['position'] for r in lane] != list(range(8)) or {r['seed'] for r in lane} != {10+gpu}:
            raise ValueError('GPU lane order or fixed seed differs')
        lanes[gpu] = lane
    def lane(gpu):
        records = []
        try:
            for row in lanes[gpu]:
                records.append(execute(row))
        except Exception as error:
            return records, dict(gpu=gpu, error=repr(error), incident_run=lanes[gpu][len(records)]['name'],
                                 not_started=[r['name'] for r in lanes[gpu][len(records)+1:]])
        return records, None
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(lane, gpu) for gpu in range(4)]
        observed = [future.result() for future in futures]
    return dict(records=[record for records, _ in observed for record in records],
                lane_errors=[error for _, error in observed if error is not None])
