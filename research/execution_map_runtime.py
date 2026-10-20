"""One persistent, source-bound continuation after the existing actor-Q queue.

Waiting uses no JAX/CUDA context. Engineering and science are single attempts;
lost SSH observations never invoke arm again. The frozen training plan is read,
never rewritten, and scientific outcomes stay sealed until every child exits.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import importlib.metadata
import os
from pathlib import Path
import subprocess
import sys
import time

from research import actor_q_entry as entry
from research import actor_q_validation as common
from research import execution_map_study as study
from research import execution_map_validation as technical

read, digest, write_once = common.read, common.digest, common.write_once
FILES = {'research/execution_map_runtime.py', 'research/execution_map_study.py',
    'research/execution_map_validation.py', 'research/actor_q_entry.py',
    'research/actor_q_validation.py', 'research/__init__.py'}
UPSTREAM_SPEC_SHA = 'f14cb1b0a7746a43f6ff42cdcafe6a423b296cc62521d607bc40297a62b8c757'
VERSIONS = dict(numpy='2.4.4', jax='0.9.2', jaxlib='0.9.2', flax='0.12.6',
    optax='0.2.8', mujoco='3.9.0', ogbench='1.2.1')


def atomic(path, value):
    path = Path(path); temporary = path.with_name(path.name+'.'+str(os.getpid())+'.tmp')
    write_once(temporary, value); temporary.replace(path)


def load_spec(path):
    path = Path(path).resolve(); spec = read(path)
    if (spec['format'] != 'matched-map-runtime-v1' or spec['job_id'] != 'a21566001929515008267857'
            or set(spec['implementation_files']) != FILES or spec['plan_sha256'] != study.PLAN_SHA
            or spec['expected_gpu'] != '4090' or spec['python'] != '/workspace/venv/bin/python'
            or spec['source_binding_sha256'] != study.SOURCE_BINDING_SHA
            or spec['upstream_runtime_spec_sha256'] != UPSTREAM_SPEC_SHA
            or spec['package_versions'] != VERSIONS
            or Path(spec['control']) != Path('/workspace/control')/study.COHORT
            or Path(spec['reports']) != Path('/workspace/reports')/study.COHORT):
        raise ValueError('Runtime layout or frozen specification differs')
    spec.update(_spec_path=str(path), _spec_sha256=digest(path), _root=str(path.parent))
    check(spec)
    for module in (sys.modules[__name__], entry, common, study, technical):
        if Path(module.__file__).resolve() != path.parent/'research'/Path(module.__file__).name:
            raise ValueError('Runtime imported outside its bound release')
    return spec


def check(spec):
    if digest(spec['_spec_path']) != spec['_spec_sha256']:
        raise ValueError('Runtime specification changed')
    for name, sha in spec['implementation_files'].items():
        if digest(Path(spec['_root'])/name) != sha:
            raise ValueError('Runtime implementation changed: '+name)
    if digest(spec['plan']) != study.PLAN_SHA or digest(spec['source_binding']) != spec['source_binding_sha256']:
        raise ValueError('Frozen scientific plan/source binding changed')
    plan = study.verify_plan(read(spec['plan']))
    if Path(spec['source']) != Path(plan['intended_remote_release'])/'source':
        raise ValueError('Source root differs from the registered release')
    entry.verify_source(spec['source'], spec['source_binding'])
    for package, version in spec['package_versions'].items():
        if importlib.metadata.version(package) != version:
            raise ValueError('Runtime package changed: '+package)
    return plan


def verify_data(spec):
    plan = read(spec['plan']); result = {}
    for row in plan['data']:
        path = Path(spec['dataset_root'])/Path(row['path']).name
        if path.stat().st_size != row['bytes'] or digest(path) != row['sha256']:
            raise ValueError('Frozen dataset differs: '+path.name)
        result[str(path)] = dict(sha256=row['sha256'], bytes=row['bytes'], mtime_ns=path.stat().st_mtime_ns)
    return result


def gate(spec):
    if digest(spec['upstream_runtime_spec']) != spec['upstream_runtime_spec_sha256']:
        raise ValueError('Existing actor-Q controller specification changed')
    upstream_spec = read(spec['upstream_runtime_spec'])
    if upstream_spec['job_id'] != spec['job_id'] or upstream_spec['control'] != spec['upstream_control']:
        raise ValueError('Upstream job/control identity differs')
    if (upstream_spec['plan_sha256'] != common.PLAN_SHA
            or digest(upstream_spec['plan']) != common.PLAN_SHA):
        raise ValueError('Existing actor-Q scientific plan changed')
    for name, sha in upstream_spec['implementation_files'].items():
        if digest(Path(spec['upstream_runtime_spec']).parent/name) != sha:
            raise ValueError('Existing upstream controller source changed')
    root = Path(spec['upstream_control'])/'continuation'; terminal = root/'terminal.json'
    if not terminal.exists():
        if (root/'started.json').exists():
            started = read(root/'started.json')
            if (started['spec_sha256'] != spec['upstream_runtime_spec_sha256']
                    or not alive(started['pid'], spec['upstream_runtime_spec'], 'own')):
                raise RuntimeError('Upstream owner missing without terminal; do not restart it')
        return dict(stage='waiting_for_existing_actor_q_queue')
    value = read(terminal); observed = read(root/'supervisor-exit.json')
    if (value['source_commit'] != common.SOURCE or value['expected_runs'] != 40
            or value['spec_sha256'] != spec['upstream_runtime_spec_sha256']
            or value['status'] not in ('complete', 'needs_review') or value.get('scheduler_exit_observed') is not True
            or type(value.get('returncode')) is not int or type(observed.get('returncode')) is not int
            or value['returncode'] != observed['returncode'] or observed['ended'] < observed['started']):
        raise ValueError('Upstream terminal lacks observed source-bound scheduler exit')
    # Status/exit integrity only; no reward, success, or superiority requirement.
    return dict(stage='upstream_terminal', terminal_sha256=digest(terminal),
                scheduler_exit_sha256=digest(root/'supervisor-exit.json'),
                source_commit=common.SOURCE, upstream_returncode=value['returncode'])


def idle_gpus():
    names = subprocess.check_output(['nvidia-smi','--query-gpu=name','--format=csv,noheader'], text=True).splitlines()
    if len(names) != 4 or not all('4090' in name for name in names):
        raise ValueError('Expected the existing four RTX4090 GPUs')
    occupied = subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'], text=True).strip()
    return not bool(occupied)


def alive(pid, script, token=None):
    try:
        args = (Path('/proc')/str(pid)/'cmdline').read_bytes().split(bytes([0]))
    except FileNotFoundError:
        return False
    return str(script).encode() in args and (token is None or str(token).encode() in args)


def command(spec, mode, *extra):
    return [spec['python'], '-m', 'research.execution_map_runtime', mode, '--spec', spec['_spec_path'], *map(str, extra)]


def environment(spec, gpu=None):
    return dict(os.environ, PYTHONPATH=spec['_root'], PYTHONDONTWRITEBYTECODE='1', WANDB_MODE='disabled',
                QC_DATA_ROOT='/workspace', CUDA_VISIBLE_DEVICES='' if gpu is None else str(gpu),
                JAX_PLATFORMS='cpu' if gpu is None else 'cuda', MUJOCO_GL='osmesa' if gpu is None else 'egl',
                XLA_PYTHON_CLIENT_PREALLOCATE='false', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')


def arm(spec):
    check(spec)
    root = Path(spec['control'])/'continuation'
    root.mkdir(parents=True, exist_ok=True)
    write_once(root/'arm-intent.json', dict(spec_sha256=spec['_spec_sha256'], created=time.time(), automatic_resubmission=False))
    with (root/'owner.log').open('x') as log:
        process = subprocess.Popen(command(spec, 'own'), cwd=spec['_root'], env=environment(spec),
                                   stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    value = dict(pid=process.pid, spec_sha256=spec['_spec_sha256'], argv=command(spec,'own'), started=time.time())
    write_once(root/'started.json', value)
    return value


def engineering_cases(plan):
    cases=[]
    overrides=dict(seed='9826', online_steps='64', start_training='32', buffer_size='128', eval_interval='64',
                   eval_episodes='2', save_interval='64', log_interval='32')
    for task in plan['tasks']:
        for mapping in plan['maps']:
            original=next(c for c in plan['cases'] if c['task']==task and c['execution_map']==mapping)
            row=dict(original, index=len(cases), seed=9826, run_group='amm_engineering_'+task+'_'+mapping)
            replacement=dict(overrides, run_group=row['run_group'],
                save_dir='/workspace/results/'+study.COHORT+'/engineering-gpu-r1')
            row['argv']=[*original['argv'][:2], *['--'+k+'='+replacement.get(k,v) for k,v in common.flags_from_argv(original['argv']).items()]]
            cases.append(row)
    return cases


def execute_case(spec, row, root, gpu, *, engineering=False):
    """One driver wait and one independent validation; no retry path."""
    check(spec); root=Path(root); tag=f"case-{row['index']:03d}"
    for name in ('cases','logs','initial','processes','validations','runs'):
        (root/name).mkdir(parents=True,exist_ok=True)
    case=root/'cases'/(tag+'.json'); write_once(case,row)
    initial=root/'initial'/(tag+'.json'); started=time.time()
    record={k:row[k] for k in ('index','task','method','seed','execution_map')}
    record.update(source_commit=common.SOURCE, status='launching', gpu=gpu, case_sha256=digest(case))
    atomic(root/'runs'/(tag+'.json'),record)
    argv=[spec['python'],'-m','research.actor_q_entry','--case',str(case),'--source',spec['source'],
          '--source-binding',spec['source_binding'],'--initial-receipt',str(initial),'--expected-device',spec['expected_gpu']]
    with (root/'logs'/(tag+'.log')).open('x') as log:
        child=subprocess.Popen(argv,cwd=spec['_root'],env=environment(spec,gpu),stdout=log,stderr=subprocess.STDOUT)
        write_once(root/'processes'/(tag+'-started.json'),dict(pid=child.pid,argv=argv,started=started))
        record.update(pid=child.pid,status='running');atomic(root/'runs'/(tag+'.json'),record)
        code=child.wait()
    context=dict(case_index=row['index'],case_sha256=digest(case),source_commit=common.SOURCE)
    process=dict(context,pid=child.pid,returncode=code,started=started,ended=time.time(),observed_by_pid=os.getpid())
    driver_exit=root/'processes'/(tag+'-exit.json');write_once(driver_exit,process)
    record.update(returncode=code,driver_exit_observed=True,status='failed' if code else 'driver_finished',driver_process=process)
    record.update(driver_exit_path=str(driver_exit),driver_exit_sha256=digest(driver_exit))
    atomic(root/'runs'/(tag+'.json'),record)
    if digest(case)!=record['case_sha256']:
        raise ValueError('Operational case changed during the actual driver')
    if code==0:
        check(spec); output=root/'validations'/(tag+'.json')
        args=command(spec,'validate','--case',case,'--initial',initial,'--output',output)
        validation_started=time.time()
        with (root/'logs'/(tag+'-validation.log')).open('x') as log:
            validator=subprocess.Popen(args,cwd=spec['_root'],env=environment(spec),
                                       stdout=log,stderr=subprocess.STDOUT)
            write_once(root/'processes'/(tag+'-validator-started.json'),dict(pid=validator.pid,argv=args,started=validation_started))
            validation_code=validator.wait()
        validator_exit=root/'processes'/(tag+'-validator-exit.json')
        write_once(validator_exit,dict(context,pid=validator.pid,returncode=validation_code,
            started=validation_started,ended=time.time(),observed_by_pid=os.getpid()))
        record.update(validation_returncode=validation_code, validation_pid=validator.pid,
            validator_exit_path=str(validator_exit),validator_exit_sha256=digest(validator_exit),
            status='needs_review' if validation_code else 'validation_finished')
        if validation_code==0:
            value=read(output)
            if any(value[k]!=row[k] for k in ('index','task','method','seed','execution_map')) or value['source_commit']!=common.SOURCE:
                raise ValueError('Independent validation identity differs')
            record.update(value,validation_report_sha256=digest(output),
                initial_receipt_path=str(initial),initial_receipt_sha256=digest(initial))
        else:
            record['validation_error']='Independent validation failed; preserve observed exit'
    check(spec);atomic(root/'runs'/(tag+'.json'),record)
    return record


def run_lanes(rows, operation):
    """Every identity assigned once, with one sequential lane per GPU."""
    if len(rows)!=60 or [r['index'] for r in rows]!=list(range(60)):
        raise ValueError('Expected sixty ordered unique cases')
    def lane(gpu):
        output=[]
        for row in rows[gpu::4]:
            try:output.append(operation(row,gpu))
            except Exception as error:
                return output,dict(gpu=gpu,incident_index=row['index'],error=str(error))
        return output,None
    with ThreadPoolExecutor(max_workers=4) as pool:
        observed=list(pool.map(lane,range(4)))
    return [r for records,_ in observed for r in records],[error for _,error in observed if error]


def supervise(spec):
    plan=check(spec); evidence=gate(spec)
    if evidence['stage']!='upstream_terminal' or not idle_gpus():
        raise RuntimeError('Prerequisites changed before engineering; no science started')
    control=Path(spec['control']); data=verify_data(spec)
    eng=control/'engineering-gpu-r1'; eng.mkdir(exist_ok=False)
    write_once(eng/'launch.json',dict(spec_sha256=spec['_spec_sha256'],gate=evidence,data=data))
    engineering=[]
    for case in engineering_cases(plan):
        result=execute_case(spec,case,eng,case['index']%4,engineering=True);engineering.append(result)
        if result['status']!='complete':
            raise RuntimeError('4090 engineering failed; retain attempt, do not launch science')
    technical.validate_groups([dict(r,driver_returncode=r['returncode']) for r in engineering])
    if verify_data(spec)!=data or not idle_gpus():
        raise ValueError('Data or idle GPU state changed before science')
    write_once(eng/'ready.json',dict(status='complete',spec_sha256=spec['_spec_sha256'],
        cloud_4090_validation=True,tested_gpu_indices=sorted({r['gpu'] for r in engineering}),records=engineering))
    science=control/'science';science.mkdir(exist_ok=False)
    write_once(science/'launch.json',dict(spec_sha256=spec['_spec_sha256'],plan_sha256=study.PLAN_SHA,
               engineering_ready_sha256=digest(eng/'ready.json'),gate=evidence,data=data,started=time.time()))
    _,errors=run_lanes(plan['cases'],lambda row,gpu:execute_case(spec,row,science,gpu))
    records=[]
    for row in plan['cases']:
        path=science/'runs'/f"case-{row['index']:03d}.json"
        records.append(read(path) if path.exists() else dict(index=row['index'],task=row['task'],method=row['method'],execution_map=row['execution_map'],
            seed=row['seed'],source_commit=common.SOURCE,status='not_started',returncode=None,driver_exit_observed=False))
    check(spec)
    if verify_data(spec)!=data:
        raise ValueError('Dataset identity/stat changed during science')
    result=study.collect(plan,records);result.update(plan_sha256=study.PLAN_SHA,spec_sha256=spec['_spec_sha256'],lane_errors=errors)
    report=Path(spec['reports'])/'matched-map-report.json';write_once(report,result)
    return 0 if result['status']=='complete' and not errors else 2


def own(spec, sleep=time.sleep):
    root=Path(spec['control'])/'continuation';root.mkdir(parents=True,exist_ok=True)
    with (root/'owner.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        previous=None
        while True:
            check(spec); evidence=gate(spec)
            stage=evidence['stage']
            if stage=='upstream_terminal':stage='ready' if idle_gpus() else 'waiting_for_idle_gpus'
            atomic(root/'status.json',dict(stage=stage,time=time.time(),spec_sha256=spec['_spec_sha256']))
            if stage!=previous:print(stage,flush=True);previous=stage
            if stage=='ready':break
            sleep(60)
        write_once(root/'supervisor-intent.json',dict(spec_sha256=spec['_spec_sha256'],time=time.time(),gate=evidence))
        atomic(root/'status.json',dict(stage='supervising_engineering_and_science',time=time.time(),spec_sha256=spec['_spec_sha256']))
        started=time.time();argv=command(spec,'supervise')
        with (root/'supervisor.log').open('x') as log:
            child=subprocess.Popen(argv,cwd=spec['_root'],env=environment(spec),stdout=log,stderr=subprocess.STDOUT)
            write_once(root/'supervisor-started.json',dict(pid=child.pid,argv=argv,started=started))
            code=child.wait()
        observed=dict(pid=child.pid,returncode=code,started=started,ended=time.time(),observed_by_pid=os.getpid())
        write_once(root/'supervisor-exit.json',observed)
        report=Path(spec['reports'])/'matched-map-report.json'
        result=read(report) if report.exists() else {}
        complete=(code==0 and result.get('status')=='complete' and result.get('source_commit')==common.SOURCE
            and result.get('expected_runs')==result.get('complete_runs')==60 and result.get('outcomes_released') is True
            and result.get('plan_sha256')==study.PLAN_SHA and result.get('spec_sha256')==spec['_spec_sha256'])
        status='complete' if complete else 'needs_review'
        terminal=dict(status=status,returncode=code,scheduler_exit_observed=True,source_commit=common.SOURCE,
            expected_runs=60,spec_sha256=spec['_spec_sha256'],report_sha256=digest(report) if report.exists() else None)
        write_once(root/'terminal.json',terminal);atomic(root/'status.json',dict(stage='terminal',**terminal))
        return code


def inspect(spec):
    check(spec);root=Path(spec['control'])/'continuation';started=root/'started.json';terminal=root/'terminal.json'
    value=dict(gate=gate(spec),owner=None,terminal=read(terminal) if terminal.exists() else None,
               status=read(root/'status.json') if (root/'status.json').exists() else None)
    if started.exists():
        r=read(started);value['owner']=dict(pid=r['pid'],live=alive(r['pid'],spec['_spec_path'],'own'),spec_sha256=r['spec_sha256'])
    progress={}
    for phase in ('engineering-gpu-r1','science'):
        rows=[read(p) for p in (Path(spec['control'])/phase/'runs').glob('*.json')]
        progress[phase]=dict(receipts=len(rows),complete=sum(r['status']=='complete' for r in rows),
            failed=sum(r['status'] in ('failed','needs_review') for r in rows),
            live_drivers=[r['pid'] for r in rows if 'pid' in r and alive(r['pid'],
                Path(spec['control'])/phase/'cases'/f"case-{r['index']:03d}.json",'research.actor_q_entry')])
    value['progress']=progress
    return value


def validate_case(spec, case, initial, output):
    """Independent process entry; restore via the original env5 replay interface."""
    if os.environ.get('JAX_PLATFORMS')=='cpu':
        os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[-2:])
    sys.path.insert(0,spec['source'])
    row=read(case)
    result=technical.validate_run(row,read(initial),spec['source'],spec['source_binding'],
        expected_device=spec['expected_gpu'])
    if read(initial)['source_binding_sha256']!=spec['source_binding_sha256']:
        raise ValueError('Initial receipt is not bound to this learner release')
    directory=Path(result['run_dir']);budget=int(common.flags_from_argv(row['argv'])['online_steps'])
    result.update({k:row[k] for k in ('index','task','method','seed','execution_map')})
    result.update(source_binding_sha256=spec['source_binding_sha256'],
        effective_config=read(directory/'effective_agent_config.json'),
        identity_hashes={name:digest(directory/name) for name in ('flags.json','effective_agent_config.json',
            'env.csv','online_agent.csv','eval.csv',f'params_{budget}.pkl')})
    write_once(output,result)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=('inspect','arm','own','supervise','validate'))
    p.add_argument('--spec',type=Path,required=True)
    for key in ('case','initial','output'):p.add_argument('--'+key,type=Path)
    a=p.parse_args();spec=load_spec(a.spec)
    if a.mode=='validate':
        if not all((a.case,a.initial,a.output)):p.error('Validation paths required')
        validate_case(spec,a.case,a.initial,a.output);return
    if a.mode in ('own','supervise'):raise SystemExit(own(spec) if a.mode=='own' else supervise(spec))
    print(__import__('json').dumps(arm(spec) if a.mode=='arm' else inspect(spec)))


if __name__=='__main__':main()
