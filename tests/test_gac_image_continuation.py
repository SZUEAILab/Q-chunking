import copy
import json

import pytest

from research import gac_image_continuation as owner
from research import gac_runtime


@pytest.fixture
def flow(tmp_path):
    spec = dict(job_id='job-1', team_id='team-1', cluster_id='cluster-1', prefix='/opt/fixture',
                image_name='qc-gac', image_tag_prefix='20261020', environment_spec_sha256='a'*64,
                runtime_spec_sha256='b'*64, science_source_commit='c'*40,
                image_wait_seconds=86400, launch_observation_seconds=600)

    class Remote:
        observation = dict(job_id=spec['job_id'], prefix=spec['prefix'], environment_spec_sha256='a'*64,
            runtime_spec_sha256='b'*64, status='waiting_for_image', environment_owner_live=True,
            environment_owner_stage='ready_for_image', environment_ready_sha256='d'*64,
            environment_ready_finished=100., gpus_idle=True)
        publishes = launches = collections = 0
        fail_launch = False

        def observe(self): return copy.deepcopy(self.observation)

        def publish_image(self, proof):
            self.publishes += 1
            self.observation.update(status='ready_to_launch', image_receipt_sha256=owner.digest(tmp_path/'image-ready.json'))

        def launch(self):
            self.launches += 1
            result = dict(status='started', source_commit=spec['science_source_commit'], runs=32, pid=1234)
            if self.fail_launch: raise TimeoutError('lost transport')
            self.observation.update(science_started=result, science_owner_live=True)
            return result

        def collect(self, destination):
            self.collections += 1
            return dict(status='complete', contrasts=24)

    class Platform:
        saves = checks = 0
        records = []
        usable = False
        uncertain = rejected = False

        def check_job(self, current): self.checks += 1

        def images(self, request, usable=False):
            return copy.deepcopy(self.records) if not usable or self.usable else []

        def save(self, request):
            self.saves += 1
            if self.rejected: raise owner.RequestRejected('rejected')
            self.records = [dict(id='image-1', path='registry/team/'+request['name']+':'+request['tag'], status=1,
                **{k: request[k] for k in ['name', 'tag', 'jobId', 'teamId', 'clusterId', 'uploadType']},
                password='must-never-persist', username='secret-user')]
            if self.uncertain: raise TimeoutError('response lost after acceptance')
            return 'image-1'

        def detail(self, image_id): return copy.deepcopy(self.records[0])

    schema = []
    remote, platform = Remote(), Platform()
    continuation = owner.Continuation(tmp_path, spec, remote, platform, lambda: schema.append('verified'), clock=lambda: 200.)
    return continuation, remote, platform, schema


def make_usable(flow):
    continuation, remote, platform, _ = flow
    continuation.tick()
    platform.records[0]['status'] = 2; platform.usable = True
    assert continuation.tick()['stage'] == 'image_proof_published'
    return continuation, remote, platform


def test_full_lifecycle_saves_once_launches_once_and_collects_only_at_terminal(flow):
    c, remote, platform, schema = flow
    assert c.tick()['stage'] == 'waiting_for_saved_image'
    assert c.tick()['stage'] == 'waiting_for_saved_image'
    platform.records[0]['status'] = 2
    assert c.tick()['stage'] == 'waiting_for_usable_image_list'
    platform.usable = True
    assert c.tick()['stage'] == 'image_proof_published'
    assert c.tick()['stage'] == 'gac_started'
    assert c.tick()['stage'] == 'gac_running'
    assert platform.saves == remote.launches == remote.publishes == 1 and remote.collections == 0
    remote.observation['science_terminal'] = dict(source_commit=c.spec['science_source_commit'], expected_runs=32,
        complete_runs=32, status='complete', returncode=0, scheduler_exit_observed=True,
        contrasts=24, comparison_status='complete')
    assert c.tick()['stage'] == 'gac_terminal' and remote.collections == 1
    assert schema == ['verified']


def test_uncertain_save_and_restarted_observer_only_reconcile(flow):
    c, remote, platform, schema = flow; platform.uncertain = True
    assert c.tick()['stage'] == 'reconciling_image_save'
    restarted = owner.Continuation(c.root, c.spec, remote, platform, lambda: pytest.fail('schema should not trigger new save'), clock=lambda: 201.)
    assert restarted.tick()['stage'] == 'waiting_for_saved_image'
    platform.records[0]['status'] = 2; platform.usable = True
    assert restarted.tick()['stage'] == 'image_proof_published'
    assert platform.saves == 1


def test_definitive_save_rejection_preserves_intent_and_never_resubmits(flow):
    c, _, platform, _ = flow; platform.rejected = True
    with pytest.raises(owner.RequestRejected): c.tick()
    assert c.tick()['stage'] == 'reconciling_image_save'
    assert platform.saves == 1 and (c.root/'save-intent.json').exists()


@pytest.mark.parametrize('stage', ['waiting_for_fql', 'waiting_for_cloud_engineering', 'waiting_for_dependency_transfer'])
def test_no_platform_mutation_before_frozen_remote_gate(flow, stage):
    c, remote, platform, schema = flow; remote.observation['status'] = stage
    assert c.tick()['stage'] == stage
    assert platform.saves == platform.checks == 0 and schema == []
    assert not list(c.root.iterdir())


def test_dead_environment_owner_is_not_treated_as_a_verified_wait(flow):
    c, remote, _, _ = flow; remote.observation.update(status='waiting_for_fql', environment_owner_live=False)
    with pytest.raises(RuntimeError, match='owner is absent'): c.tick()


@pytest.mark.parametrize('field,value', [('job_id','other'), ('prefix','/workspace/venv'),
    ('environment_spec_sha256','e'*64), ('runtime_spec_sha256','e'*64)])
def test_changed_environment_or_controller_cannot_save(flow, field, value):
    c, remote, platform, _ = flow; remote.observation[field] = value
    with pytest.raises(ValueError, match='binding changed'): c.tick()
    assert platform.saves == 0


def test_idle_resource_gate_precedes_image_and_science_mutations(flow):
    c, remote, platform, _ = flow; remote.observation['gpus_idle'] = False
    assert c.tick()['stage'] == 'waiting_for_idle_gpus_before_image' and platform.saves == 0
    remote.observation['gpus_idle'] = True; make_usable(flow)
    remote.observation['gpus_idle'] = False
    assert c.tick()['stage'] == 'waiting_for_idle_gpus_before_science' and remote.launches == 0


def test_preexisting_or_duplicate_image_cannot_be_overwritten_or_selected(flow):
    c, _, platform, _ = flow; platform.records = [dict(id='preexisting')]
    with pytest.raises(FileExistsError): c.tick()
    assert platform.saves == 0
    platform.records = []; c.tick(); platform.records *= 2
    with pytest.raises(ValueError, match='Multiple images'): c.tick()
    assert platform.saves == 1


@pytest.mark.parametrize('field,value', [('jobId','other'),('teamId','other'),('clusterId','other'),
    ('uploadType',5),('tag','other'),('id','bad/id'),('status',{'password':'untrusted'})])
def test_mismatched_image_is_never_published(flow, field, value):
    c, remote, platform, _ = flow; c.tick(); platform.records[0][field] = value
    with pytest.raises(ValueError): c.tick()
    assert remote.publishes == remote.launches == 0


def test_credentials_are_removed_and_proof_matches_frozen_verifier_contract(flow, monkeypatch):
    c, _, _ = make_usable(flow)
    assert owner.IMAGE_FIELDS == gac_runtime.IMAGE_FIELDS
    proof = owner.read(c.root/'image-ready.json')
    text = json.dumps(proof)
    assert 'password' not in text and 'must-never-persist' not in text and 'secret-user' not in text
    spec = dict(job_id=c.spec['job_id'], team_id=c.spec['team_id'], cluster_id=c.spec['cluster_id'],
                environment_ready='synthetic-ready', image_receipt='synthetic-receipt')
    monkeypatch.setattr(gac_runtime, 'digest', lambda path: 'd'*64 if path=='synthetic-ready' else 'proof-sha')
    assert gac_runtime.verify_image(proof, spec, dict(prefix=c.spec['prefix']), dict(finished=100.))['status'] == 'image_verified'


def test_save_or_launch_response_loss_never_causes_a_second_launch(flow):
    c, remote, platform = make_usable(flow); remote.fail_launch = True
    assert c.tick()['stage'] == 'reconciling_science_launch'
    assert c.tick()['stage'] == 'reconciling_science_launch'
    assert platform.saves == remote.launches == 1
    c.clock = lambda: 1000.
    with pytest.raises(RuntimeError, match='unobserved'): c.tick()
    assert remote.launches == 1


def test_ready_receipt_change_and_expired_image_observation_do_not_retry(flow):
    c, remote, platform, _ = flow; c.tick()
    remote.observation['environment_ready_sha256'] = 'e'*64
    with pytest.raises(ValueError, match='changed after image intent'): c.tick()
    remote.observation['environment_ready_sha256'] = 'd'*64; c.clock = lambda: 100000.
    with pytest.raises(RuntimeError, match='window expired'): c.tick()
    assert platform.saves == 1


def test_lost_scientific_owner_is_not_completion_or_a_restart(flow):
    c, remote, platform = make_usable(flow); c.tick(); remote.observation['science_owner_live'] = False
    with pytest.raises(RuntimeError, match='owner is absent'): c.tick()
    assert platform.saves == remote.launches == 1


def test_existing_science_without_our_image_proof_is_not_adopted(flow):
    c, remote, platform, _ = flow; remote.observation['science_started'] = dict(runs=32)
    with pytest.raises(ValueError, match='lacks this continuation'): c.tick()
    assert platform.saves == 0


@pytest.mark.parametrize('field,value', [('complete_runs',31), ('returncode',2), ('returncode',False),
    ('contrasts',23), ('comparison_status','needs_review'), ('scheduler_exit_observed',False)])
def test_incomplete_or_failed_terminal_is_retained_without_scientific_retry(flow, field, value):
    c, remote, platform = make_usable(flow); c.tick()
    terminal=dict(source_commit=c.spec['science_source_commit'], expected_runs=32, complete_runs=32,
        status='complete', returncode=0, scheduler_exit_observed=True, contrasts=24, comparison_status='complete')
    terminal[field]=value; remote.observation['science_terminal']=terminal
    assert c.tick()['stage']=='needs_attention' and remote.collections==1
    assert platform.saves==remote.launches==1
