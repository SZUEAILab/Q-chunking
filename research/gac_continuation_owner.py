"""Run the local GAC image/science observer against its bound release."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import sys
import time

from research import gac_continuation_transport as transport
from research import gac_image_continuation as continuation
from research import observation_retry


def load_spec(path):
    path=Path(path).resolve();spec=continuation.read(path)
    if spec['format']!='gac-image-continuation-v1':raise ValueError('Unknown continuation specification')
    modules=[transport,continuation,observation_retry,sys.modules[__name__]]
    expected={'research/'+Path(module.__file__).name for module in modules}
    if set(spec['implementation_files'])!=expected:raise ValueError('Continuation source set differs')
    for name,sha in spec['implementation_files'].items():
        if continuation.digest(path.parent/name)!=sha:raise ValueError('Continuation source changed')
    for module in modules:
        if Path(module.__file__).resolve()!=path.parent/'research'/Path(module.__file__).name:
            raise ValueError('Continuation imported outside its bound release')
    if continuation.digest(spec['frontend_proof_file'])!=spec['frontend_proof_sha256']:
        raise ValueError('Verified frontend evidence changed')
    spec.update(_spec_sha256=continuation.digest(path),_spec_path=str(path))
    return spec


def check_upstream(spec):
    root=Path(spec['upstream_local_control']);state=continuation.read(root/'watch-status.json')
    if state['stage']=='needs_attention':raise RuntimeError('FQL observation owner needs attention')
    if state['stage']=='fql_terminal':return
    metadata=continuation.read(root/'watch-process.json');script=root/'watch_fql.py'
    command=Path('/proc')/str(metadata['pid'])/'cmdline'
    if (continuation.digest(script)!=metadata['script_sha256'] or not command.exists()
            or str(script).encode() not in command.read_bytes().split(bytes([0]))):
        raise RuntimeError('FQL observation owner is absent without terminal evidence')


def check_bindings(spec):
    if continuation.digest(spec['_spec_path'])!=spec['_spec_sha256']:
        raise ValueError('Continuation specification changed while waiting')
    base=Path(spec['_spec_path']).parent
    for name,sha in spec['implementation_files'].items():
        if continuation.digest(base/name)!=sha:raise ValueError('Continuation source changed while waiting')
    if continuation.digest(spec['frontend_proof_file'])!=spec['frontend_proof_sha256']:
        raise ValueError('Frontend proof changed while waiting')


def state(root,stage,**fields):
    value=dict(stage=stage,time=time.time(),**fields)
    temporary=root/('status.'+str(os.getpid())+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temporary.replace(root/'status.json')
    return value


def own(spec,*,sleep=time.sleep):
    root=Path(spec['local_control']);root.mkdir(parents=True,exist_ok=True)
    with (root/'owner.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        owners=root/'owners';owners.mkdir(exist_ok=True)
        started=dict(pid=os.getpid(),started=time.time(),spec_sha256=spec['_spec_sha256'],
                     source_commit=spec['implementation_commit'],job_id=spec['job_id'])
        continuation.write_once(owners/(str(time.time_ns())+'-'+str(os.getpid())+'.json'),started)
        connection=continuation.read(spec['ssh_command_file'])
        remote=transport.RemoteController(spec,connection)
        schema=lambda:transport.verify_frontend(continuation.read(spec['frontend_proof_file']),root/'frontend-rechecks')
        controller=continuation.Continuation(root,spec,remote,transport.ImagePlatform(),schema)
        previous=None
        while True:
            try:
                check_bindings(spec)
                check_upstream(spec)
                outcome=controller.tick()
            except Exception as error:
                if isinstance(error,transport.ObservationUnavailable) or observation_retry.transient_observation_error(error):
                    # Keep the same handles and intents. A read outage neither
                    # restarts science nor rewrites another owner's phase.
                    outcome=dict(stage='observation_unavailable',error_type=type(error).__name__)
                else:
                    value=state(root,'needs_attention',job_id=spec['job_id'],error_type=type(error).__name__,error=str(error),automatic_retry=False)
                    print(json.dumps(value),flush=True);raise
            value=state(root,job_id=spec['job_id'],**outcome)
            if outcome['stage']!=previous:print(json.dumps(value),flush=True);previous=outcome['stage']
            if outcome['stage'] in ('gac_terminal','needs_attention'):return outcome
            sleep(spec['poll_seconds'])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=('inspect','own'));parser.add_argument('--spec',type=Path,required=True)
    args=parser.parse_args();spec=load_spec(args.spec)
    if args.mode=='inspect':
        check_upstream(spec)
        print(json.dumps(transport.RemoteController(spec,continuation.read(spec['ssh_command_file'])).observe()))
    else:own(spec)


if __name__=='__main__':main()
