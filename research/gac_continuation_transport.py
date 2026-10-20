"""Credential-local API and source-bound SSH adapters for GAC continuation."""
import base64
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import stat
import subprocess
import time
import urllib.error
import urllib.request

from research.gac_image_continuation import PAGE, RequestRejected, digest, read, safe_image, write_once
from research.observation_retry import retry_observation


API = 'https://console.aicloud.szu.edu.cn'


class ObservationUnavailable(RuntimeError):
    """A transport observation failed; it is not proof of remote termination."""


class ImagePlatform:
    def __init__(self, *, opener=None, auth_path=None):
        self.opener = opener or urllib.request.build_opener(urllib.request.ProxyHandler({}))
        self.auth_path = Path(auth_path or Path.home()/'.bita/config.json')

    def request(self, method, path, payload=None, *, readonly=False):
        if stat.S_IMODE(self.auth_path.stat().st_mode) != 0o600:
            raise ValueError('Local authentication file must have mode 600')
        auth = read(self.auth_path)
        token = base64.b64decode(auth['auths'][API]['auth'], validate=True).decode()
        if not token.strip(): raise ValueError('Empty local authentication')
        request = urllib.request.Request(API+path, method=method,
            data=None if payload is None else json.dumps(payload).encode(),
            headers={'token': token, 'Content-Type': 'application/json'})

        def observe():
            try:
                with self.opener.open(request, timeout=25) as response:
                    value = json.load(response)
            except urllib.error.HTTPError as error:
                if 400 <= error.code < 500 and error.code not in (408, 425, 429):
                    raise RequestRejected('Platform HTTP rejection '+str(error.code)) from None
                raise
            if value.get('message', {}).get('code') != 0:
                raise RequestRejected('Platform business rejection')
            return value['data']
        # Creation is exactly one request. Only declared read observations retry.
        return retry_observation(observe, attempts=3) if readonly else observe()

    def check_job(self, spec):
        value = self.request('GET', '/gateway/training/api/v1/job/debug/'+spec['job_id'], readonly=True)
        if any(value.get(key) != expected for key, expected in
               [('id', spec['job_id']), ('teamId', spec['team_id']), ('clusterId', spec['cluster_id']), ('state', 'Running')]):
            raise ValueError('Existing platform job identity/state changed')

    def images(self, request, *, usable=False):
        matches, page, seen = [], 1, set()
        while True:
            payload = dict(pageNum=page, pageSize=100, clusterId=request['clusterId'])
            if usable: payload['imageStatusList'] = [2]
            result = self.request('POST', PAGE, payload, readonly=True)
            if type(result.get('total')) is not int or not isinstance(result.get('list'), list):
                raise ValueError('Image pagination schema changed')
            for raw in result['list']:
                item = safe_image(raw)
                if item.get('name') == request['name'] and item.get('tag') == request['tag']:
                    if item.get('id') in seen: raise ValueError('Duplicate image across list pages')
                    seen.add(item.get('id')); matches.append(item)
            if page*100 >= result['total']: return matches
            if not result['list'] or page >= 1000: raise ValueError('Image pagination did not terminate')
            page += 1

    def save(self, request):
        result = self.request('POST', '/gateway/foundation/api/v1/image', request)
        value = result.get('id') if isinstance(result, dict) else result
        if isinstance(value, (str, int)) and not isinstance(value, bool) and re.fullmatch(r'[A-Za-z0-9_-]+', str(value)):
            return str(value)
        return None  # The frontend does not require an ID in the save response.

    def detail(self, image_id):
        if not re.fullmatch(r'[A-Za-z0-9_-]+', image_id): raise ValueError('Invalid image ID')
        return safe_image(self.request('GET', '/gateway/foundation/api/v1/image/'+image_id, readonly=True))


def verify_frontend(proof, directory, *, opener=None):
    opener = opener or urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(API+'/console/', timeout=25) as response: html = response.read()
    if proof['index_path'].encode() not in html:
        raise ValueError('Platform frontend build changed; inspect API before image creation')
    observed = []
    for row in proof['files']:
        if not row['url'].startswith(API+'/console/assets/'):
            raise ValueError('Frontend proof points outside the verified platform')
        with opener.open(row['url'], timeout=25) as response: data = response.read()
        sha = hashlib.sha256(data).hexdigest()
        if sha != row['sha256']: raise ValueError('Platform frontend source changed')
        observed.append(dict(url=row['url'], sha256=sha, bytes=len(data)))
    directory = Path(directory); directory.mkdir(exist_ok=True)
    write_once(directory/('check-'+str(time.time_ns())+'.json'),
               dict(verified_at=time.time(), index_path=proof['index_path'], files=observed, mutations=0))


REMOTE_PREAMBLE = r'''
from pathlib import Path
import base64,hashlib,json,os,sys
root=Path(BIND['controller_root'])
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
assert sha(root/'manifest.json')==BIND['controller_manifest_sha256'],'Controller manifest changed'
manifest=json.loads((root/'manifest.json').read_text())
for name,expected in manifest['files'].items():assert sha(root/name)==expected,'Controller payload changed'
assert sha(root/'runtime-spec.json')==BIND['runtime_spec_sha256'],'Runtime specification changed'
os.chdir(root);sys.path.insert(0,str(root))
from research import gac_runtime as runtime
spec=runtime.load_spec(root/'runtime-spec.json');runtime.check_imports(spec)
assert spec['job_id']==BIND['job_id'] and spec['environment_spec_sha256']==BIND['environment_spec_sha256']
env=runtime.read(spec['environment_spec'])
'''

REMOTE_OBSERVE = r'''
inspection=runtime.inspect(spec)
control=Path(spec['control']);environment_control=Path(BIND['environment_owner_control'])
started=runtime.read(control/'started.json') if (control/'started.json').exists() else None
terminal=runtime.read(control/'terminal.json') if (control/'terminal.json').exists() else None
owner=runtime.read(environment_control/'owner-started.json')
owner_script=Path(BIND['environment_owner_script'])
assert sha(owner_script)==BIND['environment_owner_sha256'] and owner['source_sha256']==BIND['environment_owner_sha256']
def live(pid,arguments):
 path=Path('/proc')/str(pid)/'cmdline'
 if not path.exists():return False
 actual=path.read_bytes().split(bytes([0]));return all(str(a).encode() in actual for a in arguments)
owner_live=live(owner['pid'],[owner_script,'own'])
owner_stage=runtime.read(environment_control/'status.json')['stage']
result=dict(status=inspection['status'],job_id=spec['job_id'],prefix=env['prefix'],
 environment_spec_sha256=sha(spec['environment_spec']),runtime_spec_sha256=sha(root/'runtime-spec.json'),
 environment_owner_live=owner_live,environment_owner_stage=owner_stage,
 science_started=started,science_terminal=terminal,
 science_owner_live=bool(started and live(started['pid'],started['argv'])))
ready_path=Path(spec['environment_ready'])
if ready_path.exists():
 ready=runtime.read(ready_path)
 result.update(environment_ready_sha256=sha(ready_path),environment_ready_finished=ready['finished'])
image=Path(spec['image_receipt'])
if image.exists():result['image_receipt_sha256']=sha(image)
if inspection['status'] in ('waiting_for_image','ready_to_launch'):
 import subprocess
 names=subprocess.check_output(['nvidia-smi','--query-gpu=name','--format=csv,noheader'],text=True).splitlines()
 assert len(names)==4 and all('4090' in n for n in names),'Existing GPU allocation changed'
 result['gpus_idle']=not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip()
 if not image.exists():
  mountinfo=Path('/proc/self/mountinfo').read_text()
  assert runtime.environment.mount_for(env['prefix'],mountinfo)[0]=='/','Environment is not in image filesystem'
  assert runtime.environment.mount_for(env['base_python'],mountinfo)[0]=='/','Base Python is outside image filesystem'
  private=[Path('/root/.bita/config.json'),Path('/root/.netrc'),*Path('/root/.ssh').glob('id_*')]
  assert not any(p.exists() and not p.name.endswith('.pub') for p in private),'Inspect user credentials before saving image'
print(json.dumps(result))
'''

REMOTE_PUBLISH = r'''
proof=json.load(sys.stdin);ready=runtime.read(spec['environment_ready'])
runtime.verify_image(proof,spec,env,ready)
path=Path(spec['image_receipt']);data=(json.dumps(proof,indent=2,allow_nan=False)+'\n').encode()
if path.exists():assert path.read_bytes()==data,'Preserve existing image proof'
else:
 with path.open('xb') as stream:stream.write(data)
print(json.dumps(dict(status='image_proof_verified_and_published',sha256=sha(path))))
'''

REMOTE_LAUNCH = r'''
print(json.dumps(runtime.launch(spec)))
'''

REMOTE_COLLECT = r'''
control=Path(spec['control']);reports=Path(spec['reports'])
assert (control/'terminal.json').exists(),'No terminal science receipt'
files={}
for path in [control/'terminal.json',control/'scheduler-exit.json',control/'started.json',control/'launch.json',
 reports/'gac-run-report.json',reports/'gac-comparison-report.json',reports/'gac-comparison-report.md']:
 if path.exists():files[path.name]=dict(data=base64.b64encode(path.read_bytes()).decode(),sha256=sha(path))
print(json.dumps(dict(files=files)))
'''


class RemoteController:
    def __init__(self, spec, connection):
        self.spec, self.connection = spec, connection

    def call(self, program, payload=None):
        bind = {key: self.spec[key] for key in ('controller_root','controller_manifest_sha256','runtime_spec_sha256',
            'job_id','environment_spec_sha256','environment_owner_control','environment_owner_script','environment_owner_sha256')}
        source = 'BIND='+repr(bind)+'\n'+REMOTE_PREAMBLE+'\n'+program
        command = 'env -u PYTHONPATH -u PYTHONHOME PYTHONDONTWRITEBYTECODE=1 /workspace/venv/bin/python -c '+shlex.quote(source)
        try:
            result = subprocess.run([*self.connection, command], input=None if payload is None else json.dumps(payload).encode(),
                stdin=subprocess.DEVNULL if payload is None else None, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=90)
        except subprocess.TimeoutExpired:
            raise ObservationUnavailable('SSH observation timed out; remote state remains unverified') from None
        if result.returncode == 255:
            raise ObservationUnavailable('SSH transport unavailable; remote state remains unverified')
        if result.returncode:
            # Retain no credential-bearing remote output in exception strings.
            raise RuntimeError('Source-bound remote controller command failed (exit '+str(result.returncode)+')')
        try: return json.loads(result.stdout)
        except json.JSONDecodeError:
            raise ObservationUnavailable('Remote observation is not yet readable JSON') from None

    def observe(self): return self.call(REMOTE_OBSERVE)

    def publish_image(self, proof): return self.call(REMOTE_PUBLISH, proof)

    def launch(self): return self.call(REMOTE_LAUNCH)

    def collect(self, destination):
        response = self.call(REMOTE_COLLECT); destination = Path(destination); destination.mkdir(exist_ok=True)
        allowed = {'terminal.json','scheduler-exit.json','started.json','launch.json',
                   'gac-run-report.json','gac-comparison-report.json','gac-comparison-report.md'}
        if set(response['files'])-allowed: raise ValueError('Unexpected terminal artifact')
        for name, item in response['files'].items():
            data = base64.b64decode(item['data'], validate=True)
            if hashlib.sha256(data).hexdigest() != item['sha256']: raise ValueError('Terminal artifact transfer changed')
            path = destination/name
            if path.exists():
                if path.read_bytes() != data: raise ValueError('Preserve existing local terminal artifact')
            else:
                with path.open('xb') as stream: stream.write(data)
        complete = set(response['files']) == allowed
        if complete:
            run = read(destination/'gac-run-report.json'); comparison = read(destination/'gac-comparison-report.json')
            complete = (run.get('source_commit') == self.spec['science_source_commit'] and run.get('status') == 'complete'
                and run.get('expected_runs') == run.get('complete_runs') == len(run.get('records', [])) == 32
                and all(r.get('status') == 'complete' and type(r.get('returncode')) is int and r['returncode'] == 0 for r in run['records'])
                and comparison.get('source_commit') == self.spec['science_source_commit'] and comparison.get('status') == 'complete'
                and comparison.get('contrast_count') == len(comparison.get('comparisons', [])) == 24
                and all(c.get('status') == 'complete' for c in comparison['comparisons']))
        return dict(status='complete' if complete else 'needs_review', files={n:v['sha256'] for n,v in response['files'].items()})
