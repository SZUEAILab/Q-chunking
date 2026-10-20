import json
from pathlib import Path

import pytest

from research import gac_continuation_owner as owner
from research import gac_image_continuation as core


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value)+'\n')


def test_observation_timeout_keeps_same_owner_and_does_not_touch_upstream_phase(tmp_path,monkeypatch):
    root=tmp_path/'owner';upstream=tmp_path/'upstream-status.json';upstream.write_text('{"stage":"authoritative"}\n')
    connection=tmp_path/'ssh.json';save(connection,['fixture-ssh'])
    calls=[]
    class Controller:
        def __init__(self,*args):calls.append('constructed')
        def tick(self):
            calls.append('observed')
            if calls.count('observed')==1:raise owner.transport.ObservationUnavailable('synthetic')
            return dict(stage='gac_terminal')
    monkeypatch.setattr(owner,'check_upstream',lambda _:None)
    monkeypatch.setattr(owner,'check_bindings',lambda _:None)
    monkeypatch.setattr(owner.transport,'RemoteController',lambda *args:object())
    monkeypatch.setattr(owner.transport,'ImagePlatform',lambda:object())
    monkeypatch.setattr(owner.continuation,'Continuation',Controller)
    spec=dict(local_control=str(root),_spec_sha256='spec',implementation_commit='commit',job_id='job',
              ssh_command_file=str(connection),poll_seconds=60)
    sleeps=[];result=owner.own(spec,sleep=lambda duration:sleeps.append(duration))
    assert result['stage']=='gac_terminal' and calls==['constructed','observed','observed'] and sleeps==[60]
    assert core.read(root/'status.json')['stage']=='gac_terminal'
    assert upstream.read_text()=='{"stage":"authoritative"}\n'
    assert len(list((root/'owners').glob('*.json')))==1


@pytest.mark.parametrize('stage',['needs_attention','waiting_for_rlpd_posthoc'])
def test_missing_or_failed_upstream_owner_requires_inspection(tmp_path,stage):
    save(tmp_path/'watch-status.json',dict(stage=stage))
    save(tmp_path/'watch-process.json',dict(pid=-999,script_sha256='wrong'))
    (tmp_path/'watch_fql.py').write_text('fixture')
    with pytest.raises(RuntimeError):owner.check_upstream(dict(upstream_local_control=str(tmp_path)))


def test_upstream_terminal_does_not_require_a_live_observer(tmp_path):
    save(tmp_path/'watch-status.json',dict(stage='fql_terminal'))
    owner.check_upstream(dict(upstream_local_control=str(tmp_path)))


@pytest.fixture
def bound_release(tmp_path,monkeypatch):
    modules=[owner.transport,owner.continuation,owner.observation_retry,owner]
    files={}
    for module in modules:
        path=tmp_path/'research'/Path(module.__file__).name;path.parent.mkdir(exist_ok=True)
        path.write_text('synthetic bound module '+path.name);files[str(path.relative_to(tmp_path))]=core.digest(path)
        monkeypatch.setattr(module,'__file__',str(path))
    frontend=tmp_path/'frontend.json';save(frontend,dict(fixture=True))
    spec=dict(format='gac-image-continuation-v1',implementation_files=files,frontend_proof_file=str(frontend),frontend_proof_sha256=core.digest(frontend))
    path=tmp_path/'spec.json';save(path,spec)
    return path,spec


def test_bound_release_checks_all_operational_sources_and_frontend_proof(bound_release):
    path,spec=bound_release
    assert owner.load_spec(path)['_spec_sha256']==core.digest(path)
    helper=path.parent/'research/observation_retry.py';helper.write_text('changed')
    with pytest.raises(ValueError,match='source changed'):owner.load_spec(path)


@pytest.mark.parametrize('change',['spec','frontend','source'])
def test_waiting_owner_rechecks_immutable_local_bindings(bound_release,change):
    path,_=bound_release;spec=owner.load_spec(path)
    owner.check_bindings(spec)
    target={'spec':path,'frontend':Path(spec['frontend_proof_file']),'source':path.parent/'research/observation_retry.py'}[change]
    target.write_text('changed while waiting')
    with pytest.raises(ValueError,match='changed while waiting'):owner.check_bindings(spec)


def test_imports_from_other_worktrees_cannot_run_the_owner(bound_release,monkeypatch):
    path,spec=bound_release
    monkeypatch.setattr(owner.transport,'__file__','/another/research/gac_continuation_transport.py')
    with pytest.raises(ValueError,match='outside its bound release'):owner.load_spec(path)
