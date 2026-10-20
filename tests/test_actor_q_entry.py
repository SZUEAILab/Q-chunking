import hashlib
from pathlib import Path
from types import SimpleNamespace

import flax.serialization
import jax
import pytest

from research import actor_q_entry as entry
from research.actor_q_validation import SOURCE, write_once, read
from tests.test_actor_q_gradient import make_agent


def test_actual_create_result_is_returned_unchanged_and_recorded_once(tmp_path,monkeypatch):
    agent=make_agent();before=flax.serialization.to_bytes(agent);calls=[]
    binding=tmp_path/'binding.json';binding.write_text('bound release')
    case=dict(index=0,seed=29,method='full',argv=['python','main_online.py','--seed=29'])
    def original(cls,seed):calls.append(seed);return agent
    monkeypatch.setattr(jax,'default_backend',lambda:'gpu')
    monkeypatch.setattr(jax,'devices',lambda:[SimpleNamespace(device_kind='NVIDIA RTX 4090')])
    function=entry.observe_creation(original,case,tmp_path/'initial.json',binding,expected_device='4090')
    assert function(object,29) is agent
    assert flax.serialization.to_bytes(agent)==before
    assert read(tmp_path/'initial.json')['serialized_agent_sha256']==hashlib.sha256(before).hexdigest()
    with pytest.raises(FileExistsError):function(object,29)
    assert calls==[29]


def test_constructor_seed_mismatch_is_rejected_before_creation(tmp_path):
    called=[]
    function=entry.observe_creation(lambda *args:called.append(1),dict(seed=29),tmp_path/'initial.json',tmp_path/'binding',expected_device='4090')
    with pytest.raises(ValueError,match='constructor seed'):function(object,30)
    assert not called


def test_source_binding_requires_all_registered_files_and_safe_paths(tmp_path):
    root=tmp_path/'source';root.mkdir();files={}
    for i in range(153):
        name=f'file{i}.py';(root/name).write_text(str(i));files[name]=entry.digest(root/name)
    binding=tmp_path/'binding.json';write_once(binding,dict(source_commit=SOURCE,python_files=files))
    assert entry.verify_source(root,binding)['source_commit']==SOURCE
    (root/'file2.py').write_text('changed')
    with pytest.raises(ValueError,match='learner changed'):entry.verify_source(root,binding)
