"""Prepare the GAC cloud environment after the frozen FQL technical terminal.

This module never saves an image or launches the scientific GAC matrix.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import copy
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tarfile
import time
import types

FQL_SOURCE = '8e2e8504394623d0d01566358ccd223d54eeb4c9'
TASKS = ('cube-triple-play-singletask-task2-v0', 'cube-double-play-singletask-task2-v0')
SEEDS = (10, 11, 12, 13)


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''): result.update(block)
    return result.hexdigest()


def read(path): return json.loads(Path(path).read_text())


def write(path, value):
    with Path(path).open('x') as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False)+'\n')


def verify_fql_gate(terminal, report, scheduler):
    expected = dict(status='complete', source_commit=FQL_SOURCE, expected_runs=16, complete_runs=16)
    if any(value.get(key) != target for value in (terminal, report) for key, target in expected.items()):
        raise ValueError('FQL technical terminal is incomplete or has a different source')
    if (type(scheduler.get('returncode')) is not int or scheduler['returncode'] != 0
            or scheduler.get('lane_errors') != []):
        raise ValueError('FQL scheduler lacks a successful observed exit')
    records = report['records']
    identities = {(env, method, seed) for env in TASKS for method in ('native', 'posthoc') for seed in SEEDS}
    if len(records) != 16 or {(r['env_name'], r['method'], r['seed']) for r in records} != identities:
        raise ValueError('FQL run identity matrix differs')
    for record in records:
        if (record['status'] != 'complete' or type(record.get('returncode')) is not int or record['returncode'] != 0
                or type(record.get('validation_returncode')) is not int or record['validation_returncode'] != 0
                or record['source_commit'] != FQL_SOURCE
                or record['offline_steps'] != 1_000_000 or record['online_steps'] != 1_000_000
                or record['actual_joint_optimizer_updates'] != 2_000_000
                or record['actual_environment_steps'] != 1_000_000
                or record['observed_evaluation_episode_counts'] != 'verified'
                or record['runtime']['backend'] != 'gpu'):
            raise ValueError('FQL run lacks complete independent GPU validation')
    comparisons = report['comparisons']
    if len(comparisons) != 2 or {r['task'] for r in comparisons} != {'triple2', 'double2'}:
        raise ValueError('FQL comparison identities differ')
    for row in comparisons:
        if (row['status'] != 'complete' or row['unavailable_seeds']
                or len(row['pairs']) != 4 or {p['seed'] for p in row['pairs']} != set(SEEDS)):
            raise ValueError('FQL analysis lacks all four paired seeds')
        if not {'online_auc', 'endpoint', 'online_tail3', 'offline_auc', 'offline_endpoint'} <= set(row.get('summary', {})):
            raise ValueError('FQL analysis lacks its registered summaries')
    return dict(status='technical_gate_verified', source_commit=FQL_SOURCE, runs=16,
                performance_threshold=None, scientific_selection=False)


def inspect_gate(spec):
    paths = {name: Path(spec[name]) for name in ('gate_terminal', 'gate_report', 'gate_scheduler_exit')}
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing: return dict(status='waiting_for_fql', missing=missing)
    result = verify_fql_gate(*(read(paths[key]) for key in ('gate_terminal', 'gate_report', 'gate_scheduler_exit')))
    result['input_hashes'] = {name: digest(path) for name, path in paths.items()}
    return result


def mount_for(path, mountinfo):
    target = str(Path(path).resolve())
    rows = []
    for line in mountinfo.splitlines():
        first, second = line.split(' - ', 1)
        point = first.split()[4].replace('\\040', ' ').replace('\\134', '\\')
        if target == point or target.startswith(point.rstrip('/')+'/'):
            rows.append((point, second.split()[0]))
    if not rows: raise ValueError('No filesystem mount matches path')
    return max(rows, key=lambda row: len(row[0]))


def clean_environment(prefix, temporary, visible_devices, inherited=None):
    result = dict(os.environ if inherited is None else inherited)
    for name in ('PYTHONPATH', 'PYTHONHOME', 'VIRTUAL_ENV'): result.pop(name, None)
    result.update(PYTHONNOUSERSITE='1', PIP_DISABLE_PIP_VERSION_CHECK='1', PIP_NO_CACHE_DIR='1',
                  OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MUJOCO_GL='egl',
                  CUDA_VISIBLE_DEVICES=visible_devices, TMPDIR=str(temporary))
    result['PATH'] = str(Path(prefix)/'bin')+os.pathsep+os.pathsep.join(
        p for p in result.get('PATH', '').split(os.pathsep) if p and not str(Path(p).resolve()).startswith('/workspace/'))
    result['LD_LIBRARY_PATH'] = os.pathsep.join(
        p for p in result.get('LD_LIBRARY_PATH', '').split(os.pathsep) if p and not str(Path(p).resolve()).startswith('/workspace/'))
    return result


def cuda_library_paths(prefix, maps_text):
    prefix = Path(prefix).resolve()
    libraries = set()
    for line in maps_text.splitlines():
        path = line.split()[-1]
        if path.startswith('/') and re.search(r'/lib(cublas|cublasLt|cudnn|cudart|cusparse|curand|cusolver|nvrtc|nvJitLink|nccl|cufft)\.so', path):
            Path(path).resolve().relative_to(prefix)
            libraries.add(path)
    if not libraries: raise ValueError('No loaded CUDA user-library paths were verified')
    return sorted(libraries)


def engineering_plans(runtime):
    runtime = Path(runtime)
    cohort, rows = read(runtime/'science/cohort.json'), read(runtime/'science/tasks.json')
    selected = [r for r in rows if r['seed'] == cohort['seeds'][0]]
    expected = {(env, protocol, profile['name']) for env in TASKS for protocol in cohort['replay_protocols'] for profile in cohort['profiles']}
    if len(selected) != 8 or {(r['env_name'], r['protocol'], r['profile']) for r in selected} != expected:
        raise ValueError('Engineering coverage differs from the frozen task/profile/data branches')
    result = []
    for index, row in enumerate(selected):
        original = read(runtime/'science'/row['plan'])
        plan = copy.deepcopy(original)
        plan.update(seed=9601+TASKS.index(row['env_name']), online_steps=64, replay_capacity=128,
                    eval_interval=64, eval_episodes=1, log_interval=32)
        plan['learner']['learning_starts'] = 32
        result.append(dict(name=row['task']+'_'+row['profile']+'_'+row['protocol'], gpu=index%4,
                           original_science_plan=row['plan'], original_science_sha256=digest(runtime/'science'/row['plan']), plan=plan))
    return result


def verify_artifacts(spec):
    runtime, archive = Path(spec['runtime_root']), Path(spec['runtime_archive'])
    if digest(archive) != spec['runtime_sha256']: raise ValueError('Runtime archive differs')
    if digest(runtime/'manifest.json') != spec['runtime_manifest_sha256']: raise ValueError('Runtime manifest differs')
    manifest = read(runtime/'manifest.json')
    helper = runtime/'verify_bundle.py'
    if digest(helper) != manifest['files']['verify_bundle.py']: raise ValueError('Verifier source differs')
    module = types.ModuleType('_gac_verified_deployment')
    module.__file__ = str(helper)
    exec(compile(helper.read_bytes(), str(helper), 'exec'), module.__dict__)
    module.verify(runtime, 'gac-runtime')
    if (manifest['dependency_archive_sha256'] != spec['dependency_sha256']
            or manifest['dependency_manifest_sha256'] != spec['dependency_manifest_sha256']
            or digest(runtime/'science/cohort.json') != spec['science_cohort_sha256']):
        raise ValueError('Cross-bundle/science binding differs')
    receipt_path = Path(spec['dependency_receipt'])
    if not receipt_path.exists(): return None, dict(status='waiting_for_dependency_transfer')
    receipt = read(receipt_path);dependency = Path(spec['dependency_archive'])
    if (receipt['status'] != 'archive_verified' or receipt['sha256'] != spec['dependency_sha256']
            or receipt['bytes'] != spec['dependency_bytes'] or receipt['readback_verified'] is not True
            or receipt['archive'] != str(dependency) or dependency.stat().st_size != spec['dependency_bytes']):
        raise ValueError('Dependency transfer receipt differs')
    if digest(dependency) != spec['dependency_sha256']: raise ValueError('Dependency archive differs after staging')
    return module, dict(status='artifacts_verified', runtime_manifest_sha256=spec['runtime_manifest_sha256'])


def inspect(spec):
    gate = inspect_gate(spec)
    if gate['status'] != 'technical_gate_verified': return gate
    _, artifacts = verify_artifacts(spec)
    if artifacts['status'] != 'artifacts_verified': return artifacts
    return dict(status='eligible_for_environment_preparation', gate=gate, artifacts=artifacts,
                cloud_environment_installed=False, image_ready=False, scientific_training_started=False)


def execute(argv, stem, control, environment, cwd):
    record = dict(argv=[str(v) for v in argv], cwd=str(cwd), started=time.time())
    with (control/'logs'/(stem+'.log')).open('x') as log:
        child = subprocess.Popen(record['argv'], cwd=cwd, env=environment, stdout=log, stderr=subprocess.STDOUT)
        record['pid'] = child.pid
        record['returncode'] = child.wait()
    record['ended'] = time.time()
    write(control/'processes'/(stem+'.json'), record)
    if record['returncode']: raise RuntimeError('Environment stage failed; preserve its attempt: '+stem)
    return record


CHECK_ENV = r'''
from pathlib import Path
import importlib.metadata as m,importlib.util,json,site,sys
import numpy as np
import torch
dependencies,output,prefix,helper=map(Path,sys.argv[1:])
assert Path(sys.prefix).resolve()==prefix.resolve() and not site.ENABLE_USER_SITE
assert 'include-system-site-packages = false' in (prefix/'pyvenv.cfg').read_text()
assert importlib.util.find_spec('jax') is None and importlib.util.find_spec('flax') is None
manifest=json.loads((dependencies/'wheel-manifest.json').read_text())
versions={}
for row in manifest['wheels']:
 assert m.version(row['name'])==row['version']
 Path(m.distribution(row['name']).locate_file('')).resolve().relative_to(prefix.resolve())
 versions[row['name']]=row['version']
assert torch.__version__=='2.14.0+cu126' and torch.cuda.device_count()==4
gpus=[]
for index in range(4):
 name=torch.cuda.get_device_name(index);assert '4090' in name;gpus.append(name)
 a=torch.arange(16,device='cuda:'+str(index),dtype=torch.float32).reshape(4,4)
 np.testing.assert_array_equal((a@a).cpu().numpy(),np.arange(16,dtype='float32').reshape(4,4)@np.arange(16,dtype='float32').reshape(4,4))
spec=importlib.util.spec_from_file_location('gac_cloud_environment_helper',helper)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
libraries=module.cuda_library_paths(prefix,Path('/proc/self/maps').read_text())
report=dict(status='passed',python=sys.version,prefix=str(prefix),base_prefix=sys.base_prefix,packages=versions,
 torch=torch.__version__,compiled_cuda=torch.version.cuda,gpus=gpus,cuda_libraries=sorted(set(libraries)),
 inherited_system_site_packages=False,jax_installed=False,flax_installed=False)
with output.open('x') as f:f.write(json.dumps(report,indent=2)+'\n')
print(json.dumps(dict(status='passed',gpu_count=4,packages=len(versions))))
'''

VALIDATE_CASE = r'''
from pathlib import Path
import importlib.util,json,sys,torch
torch.set_num_threads(1)
runtime,directory,plan,process,report=map(Path,sys.argv[1:]);source=runtime/'source'
sys.path.insert(0,str(source))
path=runtime/'analysis/research/gac_reference_validation.py'
spec=importlib.util.spec_from_file_location('bound_gac_validation',path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
result=module.validate_run(directory,plan,json.loads(process.read_text()),source)
assert result['status']=='complete' and result['actual_updates']==dict(critic=31,actor=30,target=31)
assert result['exact_checkpoint_fields']==179
with report.open('x') as f:f.write(json.dumps(result,indent=2)+'\n')
print(json.dumps(dict(status='passed',checkpoint_fields=179)))
'''

RADIUS_ORACLE = r'''
from pathlib import Path
import json,sys,torch
from agents.gac_reference import GACReferenceConfig
from research.check_gac_reference import load_upstream,run_case
torch.set_num_threads(1)
root,output=map(Path,sys.argv[1:]);official,code,sha=load_upstream(root)
result=run_case(official,code,GACReferenceConfig(learning_starts=0,radius=2.5),5,'cuda',46)
result['upstream_source_sha256']=sha
assert result['max_absolute_state_error']==0
with output.open('x') as f:f.write(json.dumps(result,indent=2)+'\n')
print(json.dumps(dict(status='passed',max_absolute_state_error=0)))
'''


def prepare(spec):
    # This gate precedes creation of the control directory, lock and venv.
    gate = inspect_gate(spec)
    if gate['status'] != 'technical_gate_verified': raise RuntimeError('FQL is not technically complete; defer environment preparation')
    verifier, artifacts = verify_artifacts(spec)
    if verifier is None: raise RuntimeError('Dependency transfer is incomplete; defer environment preparation')
    control, prefix = Path(spec['control']), Path(spec['prefix'])
    expected_prefix = Path('/opt')/('qc-gac-cu126-'+spec['dependency_sha256'][:12])
    if prefix != expected_prefix or not control.is_relative_to('/workspace/control'):
        raise ValueError('Environment/control locations differ from registered root/PVC layout')
    mountinfo = Path('/proc/self/mountinfo').read_text()
    if mount_for('/opt', mountinfo) != ('/', 'overlay') or mount_for('/workspace', mountinfo)[0] != '/workspace':
        raise ValueError('Image filesystem or persistent workspace layout changed')
    if os.statvfs('/opt').f_bavail*os.statvfs('/opt').f_frsize < 16*1024**3:
        raise RuntimeError('Insufficient reported root filesystem space')
    if os.statvfs('/workspace').f_bavail*os.statvfs('/workspace').f_frsize < spec['dependency_bytes']*2+4*1024**3:
        raise RuntimeError('Insufficient reported workspace space')
    base = Path(spec['base_python']).resolve()
    if mount_for(base, mountinfo)[0] != '/': raise ValueError('Base interpreter is outside the image root filesystem')
    version = json.loads(subprocess.check_output([str(base), '-c', 'import json,sys;print(json.dumps(list(sys.version_info[:3])))'], text=True))
    if version != [3, 12, 13]: raise ValueError('Cloud base interpreter version changed')
    if prefix.exists(): raise FileExistsError('Preserve previous environment initialization')
    control.mkdir(parents=True, exist_ok=False)
    for name in ('logs', 'processes', 'tmp', 'plans', 'runs', 'validations'): (control/name).mkdir()
    write(control/'install.lock', dict(started=time.time(),pid=os.getpid(),spec_sha256=spec['_spec_sha256'],
                                      environment_tool_sha256=digest(Path(__file__).resolve()),gate=gate))
    try:
        dependencies = control/'dependencies';dependencies.mkdir()
        with tarfile.open(spec['dependency_archive']) as archive: archive.extractall(dependencies, filter='data')
        if digest(dependencies/'manifest.json') != spec['dependency_manifest_sha256']: raise ValueError('Dependency manifest differs')
        verifier.verify(dependencies, 'gac-dependencies')
        runtime = Path(spec['runtime_root']);source = runtime/'source'
        upstream = control/'official-reference';verifier.reconstruct_upstream(runtime, upstream)
        env = clean_environment(prefix, control/'tmp', '')
        execute([base, '-m', 'venv', prefix], 'create-venv', control, env, control)
        python = prefix/'bin/python'
        execute([python, '-m', 'pip', 'install', '--no-index', '--find-links', dependencies/'wheelhouse', '--require-hashes',
                 '-r', dependencies/'offline-requirements.txt', '--report', control/'install-report.json'], 'offline-install', control, env, control)
        execute([python, '-m', 'pip', 'check'], 'pip-check', control, env, control)
        all_gpu = clean_environment(prefix, control/'tmp', '0,1,2,3')
        execute([python, '-c', CHECK_ENV, dependencies, control/'environment.json', prefix, Path(__file__).resolve()], 'four-gpu-environment', control, all_gpu, control)
        execute([python, '-m', 'pytest', '-q', 'tests/test_gac_reference.py', 'tests/test_gac_reference_driver.py'],
                'source-tests', control, clean_environment(prefix, control/'tmp', '0'), source)
        execute([python, '-m', 'pytest', '-q', 'tests/test_gac_reference_validation.py'],
                'analysis-tests', control, env, runtime/'analysis')
        cases = engineering_plans(runtime)
        for case in cases:
            plan = control/'plans'/(case['name']+'.json');write(plan, case['plan'])
        def lane(gpu):
            records = []
            for case in [r for r in cases if r['gpu'] == gpu]:
                name = case['name'];plan = control/'plans'/(name+'.json');directory = control/'runs'/name
                environment = clean_environment(prefix, control/'tmp', str(gpu))
                process = execute([python, source/'main_gac_reference.py', '--plan', plan, '--output-dir', directory], name, control, environment, source)
                process['plan_sha256'] = digest(plan)
                process_path = control/'processes'/(name+'-bound.json');write(process_path, process)
                output = control/'validations'/(name+'.json')
                validated = execute([python, '-c', VALIDATE_CASE, runtime, directory, plan, process_path, output], name+'-validation', control, environment, source)
                initial = read(directory/'receipt.json')['initial_model_and_rng_sha256']
                records.append(dict(name=name,gpu=gpu,env_name=case['plan']['env_name'],driver=process,validation=validated,
                                    initial_model_and_rng_sha256=initial,report=str(output),report_sha256=digest(output)))
            return records
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = [pool.submit(lane, gpu) for gpu in range(4)]
            records = [record for future in futures for record in future.result()]
        for task in TASKS:
            group = [r for r in records if r['env_name'] == task]
            if len(group) != 4 or len({r['initial_model_and_rng_sha256'] for r in group}) != 1:
                raise ValueError('Engineering model/RNG initialization differs within task')
        environment = clean_environment(prefix, control/'tmp', '3')
        execute([python, '-m', 'research.check_gac_reference', '--upstream-root', upstream, '--device', 'cuda', '--observation-dim', '46',
                 '--output', control/'numerical-default.json'], 'numerical-default', control, environment, source)
        execute([python, '-c', RADIUS_ORACLE, upstream, control/'numerical-radius25.json'], 'numerical-radius25', control, environment, source)
        default, radius = read(control/'numerical-default.json'), read(control/'numerical-radius25.json')
        if (default['status'] != 'passed' or len(default['cases']) != 3 or radius['status'] != 'passed'
                or any(c['max_absolute_state_error'] != 0 for c in default['cases']) or radius['max_absolute_state_error'] != 0):
            raise ValueError('Official numerical check differs')
        verifier.verify(runtime, 'gac-runtime')
        write(control/'ready-for-image.json', dict(status='cloud_engineering_ready_for_image',finished=time.time(),prefix=str(prefix),
              source_commit=read(runtime/'manifest.json')['source_commit'],analysis_commit=read(runtime/'manifest.json')['analysis_commit'],
              spec_sha256=spec['_spec_sha256'],upstream_gate=gate,engineering_runs=records,expected_runs=8,validated_runs=len(records),
              engineering_steps_per_run=64,numerical_cases=4,image_ready=False,scientific_training_started=False))
        return dict(status='cloud_engineering_ready_for_image',prefix=str(prefix),engineering_runs=8,image_ready=False,scientific_training_started=False)
    except Exception as error:
        write(control/'incident.json',dict(error=repr(error),time=time.time(),scientific_training_started=False))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('inspect', 'prepare'));parser.add_argument('--spec',type=Path,required=True)
    args = parser.parse_args();spec = read(args.spec);spec['_spec_sha256'] = digest(args.spec)
    if digest(Path(__file__).resolve()) != spec['environment_tool_sha256']:
        raise ValueError('Environment preparation tool differs from its registered source')
    result = inspect(spec) if args.command == 'inspect' else prepare(spec)
    print(json.dumps(result))


if __name__ == '__main__': main()
