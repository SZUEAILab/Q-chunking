import base64
import contextlib
import copy
import hashlib
import io
import json
from pathlib import Path
from types import SimpleNamespace
import urllib.error

import pytest

from research import gac_continuation_transport as transport
from research.gac_image_continuation import RequestRejected, digest, write_once
from research.observation_retry import retry_observation


@pytest.fixture
def auth(tmp_path):
    path=tmp_path/'authentication.json'
    path.write_text(json.dumps(dict(auths={transport.API:dict(auth=base64.b64encode(b'synthetic-token').decode())})))
    path.chmod(0o600)
    return path


def response(value): return io.BytesIO(json.dumps(dict(message=dict(code=0),data=value)).encode())


def test_only_read_observations_retry_and_save_never_retries(auth,monkeypatch):
    calls=[]
    class Opener:
        def open(self,request,timeout):
            calls.append(request.get_method())
            assert request.get_header('Token')=='synthetic-token'
            if len(calls)<3: raise urllib.error.URLError(TimeoutError())
            return response(dict(state='Running'))
    monkeypatch.setattr(transport,'retry_observation',lambda call,**_:retry_observation(call,attempts=3,sleep=lambda _:None))
    api=transport.ImagePlatform(opener=Opener(),auth_path=auth)
    assert api.request('GET','/fixture',readonly=True)==dict(state='Running')
    assert calls==['GET']*3
    calls.clear()
    with pytest.raises(urllib.error.URLError):api.save(dict(name='fixture'))
    assert calls==['POST']


def test_pagination_retains_only_matching_safe_fields_and_usable_filter(auth):
    payloads=[]
    class Opener:
        def open(self,request,timeout):
            payload=json.loads(request.data);payloads.append(payload)
            if payload['pageNum']==1:
                return response(dict(total=101,list=[dict(id=str(i),name='unrelated',tag='v1',password='private') for i in range(100)]))
            return response(dict(total=101,list=[dict(id='wanted',name='qc-gac',tag='version',password='private',username='private')]))
    api=transport.ImagePlatform(opener=Opener(),auth_path=auth)
    found=api.images(dict(name='qc-gac',tag='version',clusterId='cluster'),usable=True)
    assert found==[dict(id='wanted',name='qc-gac',tag='version')]
    assert [p['pageNum'] for p in payloads]==[1,2]
    assert all(p['imageStatusList']==[2] and p['clusterId']=='cluster' for p in payloads)


def test_wrong_job_or_platform_rejection_is_not_a_save(auth):
    class Opener:
        def open(self,request,timeout):return response(dict(id='other',state='Running',teamId='team',clusterId='cluster'))
    api=transport.ImagePlatform(opener=Opener(),auth_path=auth)
    with pytest.raises(ValueError):api.check_job(dict(job_id='job',team_id='team',cluster_id='cluster'))


def test_save_returns_only_identifier_and_detail_never_exposes_credentials(auth):
    class Opener:
        def open(self,request,timeout):return response(dict(id='image-id',password='secret',username='secret',name='safe'))
    api=transport.ImagePlatform(opener=Opener(),auth_path=auth)
    assert api.save(dict(name='safe'))=='image-id'
    assert api.detail('image-id')==dict(id='image-id',name='safe')
    with pytest.raises(ValueError):api.detail('../other')


@pytest.mark.parametrize('mode',['http','business'])
def test_definitive_api_errors_are_sanitized_and_not_retried(auth,mode):
    calls=[]
    class Opener:
        def open(self,request,timeout):
            calls.append(1)
            if mode=='http':raise urllib.error.HTTPError(request.full_url,403,'private-response',{},None)
            return io.BytesIO(b'{"message":{"code":403,"private":"secret"},"data":{"password":"secret"}}')
    api=transport.ImagePlatform(opener=Opener(),auth_path=auth)
    with pytest.raises(RequestRejected) as error:api.request('GET','/fixture',readonly=True)
    assert 'secret' not in str(error.value) and 'private-response' not in str(error.value) and len(calls)==1


def test_local_authentication_permissions_are_checked_before_request(auth):
    auth.chmod(0o644)
    with pytest.raises(ValueError,match='mode 600'):transport.ImagePlatform(auth_path=auth).request('GET','/fixture')


@pytest.mark.parametrize('changed',['index','body',None])
def test_frontend_build_and_source_are_rechecked_without_mutation(tmp_path,changed):
    body=b'frontend-fixture'
    proof=dict(index_path='/console/assets/index-fixture.js',files=[dict(url=transport.API+'/console/assets/component.js',sha256=hashlib.sha256(body).hexdigest())])
    calls=[]
    class Opener:
        def open(self,url,timeout):
            calls.append(url)
            if url.endswith('/console/'):
                return io.BytesIO(b'<script src="/console/assets/index-other.js">' if changed=='index' else b'<script src="/console/assets/index-fixture.js">')
            return io.BytesIO(b'changed' if changed=='body' else body)
    if changed:
        with pytest.raises(ValueError):transport.verify_frontend(proof,tmp_path,opener=Opener())
        assert not list(tmp_path.iterdir())
    else:
        transport.verify_frontend(proof,tmp_path,opener=Opener())
        assert len(list(tmp_path.iterdir()))==1
    assert all(isinstance(c,str) for c in calls)


def test_remote_image_publish_uses_exact_json_bytes_and_preserves_existing_proof(tmp_path):
    path=tmp_path/'remote-image.json';proof=dict(format='fixture',detail=dict(id='image'))
    calls=[]
    runtime=SimpleNamespace(read=lambda _:dict(finished=1),verify_image=lambda p,*_:calls.append(copy.deepcopy(p)))
    namespace=dict(runtime=runtime,spec=dict(environment_ready='ready',image_receipt=str(path)),env={},
                   sys=SimpleNamespace(stdin=io.StringIO(json.dumps(proof))),Path=Path,json=json,sha=digest)
    with contextlib.redirect_stdout(io.StringIO()):exec(transport.REMOTE_PUBLISH,namespace)
    local=tmp_path/'local-image.json';write_once(local,proof)
    assert path.read_bytes()==local.read_bytes() and calls==[proof]
    namespace['sys'].stdin=io.StringIO(json.dumps(proof))
    with contextlib.redirect_stdout(io.StringIO()):exec(transport.REMOTE_PUBLISH,namespace)
    namespace['sys'].stdin=io.StringIO(json.dumps(dict(format='other')))
    with pytest.raises(AssertionError,match='Preserve existing'):exec(transport.REMOTE_PUBLISH,namespace)
    assert path.read_bytes()==local.read_bytes()


@pytest.mark.parametrize('name',['REMOTE_OBSERVE','REMOTE_PUBLISH','REMOTE_LAUNCH','REMOTE_COLLECT'])
def test_all_remote_programs_compile(name):
    compile('BIND={}\n'+transport.REMOTE_PREAMBLE+'\n'+getattr(transport,name),name,'exec')


def test_remote_failure_does_not_copy_unprojected_output_to_exception(monkeypatch):
    fields=('controller_root','controller_manifest_sha256','runtime_spec_sha256','job_id','environment_spec_sha256',
            'environment_owner_control','environment_owner_script','environment_owner_sha256')
    remote=transport.RemoteController(dict.fromkeys(fields,'fixture'),['ssh','fixture'])
    monkeypatch.setattr(transport.subprocess,'run',lambda *a,**k:SimpleNamespace(returncode=1,stdout=b'password secret',stderr=b'secret'))
    with pytest.raises(RuntimeError) as error:remote.observe()
    assert 'secret' not in str(error.value)


@pytest.mark.parametrize('mode',['ssh_exit','timeout','partial_json'])
def test_lost_transport_is_not_reported_as_remote_termination(monkeypatch,mode):
    fields=('controller_root','controller_manifest_sha256','runtime_spec_sha256','job_id','environment_spec_sha256',
            'environment_owner_control','environment_owner_script','environment_owner_sha256')
    remote=transport.RemoteController(dict.fromkeys(fields,'fixture'),['ssh','fixture'])
    def run(*args,**kwargs):
        if mode=='timeout':raise transport.subprocess.TimeoutExpired('fixture-command',90)
        return SimpleNamespace(returncode=255 if mode=='ssh_exit' else 0,stdout=b'{',stderr=b'private')
    monkeypatch.setattr(transport.subprocess,'run',run)
    with pytest.raises(transport.ObservationUnavailable):remote.observe()


def test_terminal_transfer_checks_hashes_all_runs_and_all_contrasts(tmp_path,monkeypatch):
    remote=transport.RemoteController(dict(science_source_commit='source'),[])
    run=dict(source_commit='source',status='complete',expected_runs=32,complete_runs=32,
             records=[dict(status='complete',returncode=0)]*32)
    comparison=dict(source_commit='source',status='complete',contrast_count=24,comparisons=[dict(status='complete')]*24)
    values={'terminal.json':{},'scheduler-exit.json':{},'started.json':{},'launch.json':{},'gac-run-report.json':run,
            'gac-comparison-report.json':comparison,'gac-comparison-report.md':'markdown'}
    def payload():
        files={}
        for name,value in values.items():
            data=json.dumps(value).encode();files[name]=dict(data=base64.b64encode(data).decode(),sha256=hashlib.sha256(data).hexdigest())
        return dict(files=files)
    monkeypatch.setattr(remote,'call',lambda _:payload())
    assert remote.collect(tmp_path/'complete')['status']=='complete'
    run['records']=[dict(status='complete',returncode=0)]*31
    assert remote.collect(tmp_path/'incomplete')['status']=='needs_review'
    bad=payload();bad['files']['terminal.json']['sha256']='bad'
    monkeypatch.setattr(remote,'call',lambda _:bad)
    with pytest.raises(ValueError,match='transfer changed'):remote.collect(tmp_path/'corrupt')
