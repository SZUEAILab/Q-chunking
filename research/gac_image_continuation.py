"""Single-save image and single-launch continuation for the frozen GAC cohort.

The transports keep credentials in memory. Remote readiness is independently
checked by the already frozen controller before an image can be saved. An
ambiguous mutation is reconciled through observations, never submitted again.
"""
import hashlib
import json
from pathlib import Path
import re
import time


IMAGE_FIELDS = {'id', 'name', 'tag', 'path', 'status', 'jobId', 'clusterId', 'teamId', 'uploadType'}
PAGE = '/gateway/foundation/api/v1/image/action/page'


class RequestRejected(RuntimeError):
    """A definitive API rejection; callers must inspect without resubmission."""


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_once(path, value):
    with Path(path).open('x') as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False)+'\n')


def safe_image(value):
    return {key: item for key, item in value.items() if key in IMAGE_FIELDS}


def verify_record(record, request, *, usable=False):
    expected = {key: request[key] for key in ('name', 'tag', 'teamId', 'clusterId', 'jobId')}
    if any(record.get(key) != value for key, value in expected.items()):
        raise ValueError('Image identity differs from the prepared environment')
    if not isinstance(record.get('id'), str) or not re.fullmatch(r'[A-Za-z0-9_-]+', record['id']):
        raise ValueError('Image ID is absent or malformed')
    if record.get('uploadType') != 4 or (usable and record.get('status') != 2):
        raise ValueError('Image type or usable state differs')
    if type(record.get('status')) is not int:
        raise ValueError('Image state must be a scalar status code')
    if usable:
        reference = record.get('path')
        if not isinstance(reference, str) or not reference.endswith('/'+request['name']+':'+request['tag']) or any(c.isspace() for c in reference):
            raise ValueError('Image reference differs from its requested version')


class Continuation:
    """Advance one observation at a time; never retry a scientific launch.

    remote.observe() must use the source-bound GAC runtime verifier and confirm
    live PIDs. No reward or success statistic is read while science runs.
    """

    def __init__(self, root, spec, remote, platform, verify_schema, *, clock=time.time):
        self.root, self.spec = Path(root), spec
        self.remote, self.platform, self.verify_schema, self.clock = remote, platform, verify_schema, clock
        self.root.mkdir(parents=True, exist_ok=True)

    def request(self, observation):
        sha = observation['environment_ready_sha256']
        if not re.fullmatch(r'[0-9a-f]{64}', sha):
            raise ValueError('Environment readiness hash is malformed')
        return dict(name=self.spec['image_name'], tag=self.spec['image_tag_prefix']+'-'+sha,
                    content='GAC CUDA environment after all eight frozen engineering branches; readiness sha256 '+sha,
                    uploadType=4, sceneList=['training', 'training-debug'], jobId=self.spec['job_id'],
                    teamId=self.spec['team_id'], clusterId=self.spec['cluster_id'])

    def check_environment(self, observation):
        if (observation.get('job_id') != self.spec['job_id']
                or observation.get('prefix') != self.spec['prefix']
                or observation.get('environment_spec_sha256') != self.spec['environment_spec_sha256']
                or observation.get('runtime_spec_sha256') != self.spec['runtime_spec_sha256']):
            raise ValueError('Remote environment/controller binding changed')

    def tick(self):
        observation = self.remote.observe()
        self.check_environment(observation)
        terminal = observation.get('science_terminal')
        started = observation.get('science_started')
        proof_path = self.root/'image-ready.json'
        if terminal is not None or started is not None:
            if not proof_path.exists() or observation.get('image_receipt_sha256') != digest(proof_path):
                raise ValueError('Existing science lacks this continuation image proof')
            if not started or started.get('source_commit') != self.spec['science_source_commit'] or started.get('runs') != 32:
                raise ValueError('Existing science identity differs')
            if terminal is not None:
                if terminal.get('source_commit') != self.spec['science_source_commit'] or terminal.get('expected_runs') != 32:
                    raise ValueError('Terminal scientific identity differs')
                collection = self.remote.collect(Path(self.spec.get('local_reports', self.root/'reports')))
                complete = (terminal.get('status') == 'complete' and terminal.get('complete_runs') == 32
                            and terminal.get('returncode') == 0 and type(terminal.get('returncode')) is int
                            and terminal.get('scheduler_exit_observed') is True
                            and terminal.get('contrasts') == 24 and terminal.get('comparison_status') == 'complete'
                            and collection.get('status') == 'complete')
                return dict(stage='gac_terminal' if complete else 'needs_attention', terminal=terminal, collection=collection)
            if not observation.get('science_owner_live'):
                raise RuntimeError('Scientific owner is absent without terminal evidence; preserve attempt')
            return dict(stage='gac_running', remote_pid=started['pid'], expected_runs=32)

        status = observation['status']
        if status not in ('waiting_for_image', 'ready_to_launch'):
            if observation.get('environment_owner_stage') == 'needs_attention':
                raise RuntimeError('Cloud environment preparation requires inspection')
            if not observation.get('environment_owner_live'):
                raise RuntimeError('Environment owner is absent without validated readiness')
            return dict(stage=status)

        request = self.request(observation)
        intent_path = self.root/'save-intent.json'
        if not intent_path.exists():
            if status != 'waiting_for_image' or proof_path.exists() or observation.get('image_receipt_sha256'):
                raise FileExistsError('Preserve pre-existing image evidence; inspect its owner')
            if not observation.get('gpus_idle'):
                return dict(stage='waiting_for_idle_gpus_before_image')
            self.verify_schema()
            self.platform.check_job(self.spec)
            if self.platform.images(request):
                raise FileExistsError('Image name/tag already exists before this save intent')
            requested = self.clock()
            if requested < observation['environment_ready_finished']:
                raise ValueError('Image request would precede engineering completion')
            write_once(intent_path, dict(request=request, requested_at=requested,
                       environment_ready_sha256=observation['environment_ready_sha256']))
            try:
                image_id = self.platform.save(request)
            except RequestRejected:
                raise
            except Exception as error:
                # The service may have accepted the single request. Keep the
                # committed intent and reconcile its unique version through GET/list.
                write_once(self.root/'save-response.json', dict(response_known=False, error_type=type(error).__name__))
                return dict(stage='reconciling_image_save')
            write_once(self.root/'save-response.json', dict(response_known=True, image_id=image_id))
            return dict(stage='waiting_for_saved_image')

        intent = read(intent_path)
        if intent['request'] != request or intent['environment_ready_sha256'] != observation['environment_ready_sha256']:
            raise ValueError('Engineering readiness changed after image intent')
        if self.clock()-intent['requested_at'] > self.spec['image_wait_seconds'] and not proof_path.exists():
            raise RuntimeError('Image save observation window expired; do not resubmit')
        if not proof_path.exists():
            matches = self.platform.images(request)
            if len(matches) > 1:
                raise ValueError('Multiple images match the single saved version')
            if not matches:
                return dict(stage='reconciling_image_save')
            candidate = safe_image(matches[0]); verify_record(candidate, request)
            response_path = self.root/'save-response.json'
            response_id = read(response_path).get('image_id') if response_path.exists() else None
            if response_id is not None and response_id != candidate['id']:
                raise ValueError('Save response and observed image ID differ')
            detail = safe_image(self.platform.detail(candidate['id'])); verify_record(detail, request)
            if detail['id'] != candidate['id']:
                raise ValueError('Image detail returned another identity')
            if detail.get('status') != 2:
                return dict(stage='waiting_for_saved_image', image_id=detail['id'], image_status=detail.get('status'))
            verify_record(detail, request, usable=True)
            usable = [safe_image(r) for r in self.platform.images(request, usable=True) if r.get('id') == detail['id']]
            if not usable:
                return dict(stage='waiting_for_usable_image_list', image_id=detail['id'])
            if len(usable) != 1:
                raise ValueError('Duplicate image in usable list')
            verify_record(usable[0], request, usable=True)
            if any(usable[0][key] != detail[key] for key in ('id', 'name', 'tag', 'path')):
                raise ValueError('Image detail and usable-list reference differ')
            proof = dict(format='gac-saved-image-v1', status='usable', source_job_id=self.spec['job_id'],
                         prefix=self.spec['prefix'], environment_ready_sha256=intent['environment_ready_sha256'],
                         save_requested_at=intent['requested_at'], save_request=request, detail=detail,
                         usable_record=usable[0], detail_api_path='/gateway/foundation/api/v1/image/'+detail['id'],
                         usable_api_path=PAGE)
            write_once(proof_path, proof)

        if observation.get('image_receipt_sha256') != digest(proof_path):
            self.remote.publish_image(read(proof_path))
            return dict(stage='image_proof_published', image_id=read(proof_path)['detail']['id'])
        if status != 'ready_to_launch':
            raise ValueError('Published image did not pass frozen scientific verifier')
        if not observation.get('gpus_idle'):
            return dict(stage='waiting_for_idle_gpus_before_science')
        launch_intent = self.root/'science-launch-intent.json'
        if launch_intent.exists():
            # The launch may have succeeded without returning an SSH result.
            # observe() above is the sole way to adopt its authoritative start.
            if self.clock()-read(launch_intent)['created'] > self.spec['launch_observation_seconds']:
                raise RuntimeError('Scientific launch remains unobserved; inspect without retry')
            return dict(stage='reconciling_science_launch')
        write_once(launch_intent, dict(created=self.clock(), image_receipt_sha256=digest(proof_path),
                   source_commit=self.spec['science_source_commit'], expected_runs=32))
        try:
            result = self.remote.launch()
        except Exception as error:
            write_once(self.root/'science-launch-response.json', dict(response_known=False, error_type=type(error).__name__))
            return dict(stage='reconciling_science_launch')
        write_once(self.root/'science-launch-response.json', dict(response_known=True, result=result))
        if result.get('status') != 'started' or result.get('source_commit') != self.spec['science_source_commit'] or result.get('runs') != 32:
            raise RuntimeError('Scientific launch did not return its registered start; inspect without retry')
        return dict(stage='gac_started', remote_pid=result['pid'], expected_runs=32)
