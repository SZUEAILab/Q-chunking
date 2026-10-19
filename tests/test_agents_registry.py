import json
import os
import subprocess
import sys

import pytest


def test_package_and_torch_learner_do_not_preload_jax_agents():
    script="""import sys,json
import agents
assert list(agents.agents)==['acfql','acrlpd']
assert 'acrlpd' in agents.agents
assert 'agents.acrlpd' not in sys.modules and 'agents.acfql' not in sys.modules
from agents.gac_reference import GACReferenceConfig
assert 'agents.acrlpd' not in sys.modules and 'agents.acfql' not in sys.modules
print(json.dumps({'independent':True}))
"""
    result=subprocess.run([sys.executable,'-c',script],capture_output=True,text=True,check=True)
    assert json.loads(result.stdout)['independent']


def test_runtime_registration_keeps_mapping_mutation_and_dictionary_views():
    import agents
    registry=type(agents.agents)()
    marker=object();registry['reference']=marker
    assert registry['reference'] is marker and registry.get('reference') is marker
    assert list(registry.keys())==['acfql','acrlpd','reference'] and len(registry)==3
    del registry['reference'];assert 'reference' not in registry
    del registry['acrlpd'];assert list(registry)==['acfql']
    with pytest.raises(KeyError):registry['absent']


def test_legacy_exports_and_lookup_resolve_original_classes():
    from agents import ACFQLAgent, ACRLPDAgent, agents
    from agents.acfql import ACFQLAgent as flow
    from agents.acrlpd import ACRLPDAgent as rlpd
    assert ACFQLAgent is flow and ACRLPDAgent is rlpd
    assert agents['acfql'] is flow and agents['acrlpd'] is rlpd
