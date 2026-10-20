"""Observe the frozen driver's actual initialization without changing its state."""
import argparse
import hashlib
from pathlib import Path
import runpy
import sys
import time

from research.actor_q_validation import SOURCE, digest, read, write_once


def verify_source(root, binding):
    root = Path(root); manifest = read(binding)
    if manifest['source_commit'] != SOURCE or len(manifest['python_files']) != 153:
        raise ValueError('Unexpected frozen learner source')
    for name, sha in manifest['python_files'].items():
        path = (root/name).resolve(); path.relative_to(root.resolve())
        if digest(path) != sha:
            raise ValueError('Frozen learner changed: '+name)
    return manifest


def observe_creation(original, case, output, binding, *, expected_device):
    """The returned object is the original create result, with identical bytes."""
    import flax.serialization
    import jax
    def create(cls, *args, **kwargs):
        if Path(output).exists():
            raise FileExistsError('Preserve the first actual initialization')
        if (args[0] if args else kwargs['seed']) != case['seed']:
            raise ValueError('Actual constructor seed differs')
        agent = original(cls, *args, **kwargs)
        if (jax.default_backend() != 'gpu' or len(jax.devices()) != 1
                or expected_device not in jax.devices()[0].device_kind
                or agent.config['adapter_actor_q_gradient'] != case['method']):
            raise ValueError('Driver backend or gradient path differs')
        before = flax.serialization.to_bytes(agent)
        record = dict(source_commit=SOURCE, seed=case['seed'], method=case['method'],
            case_index=case['index'], serialized_agent_sha256=hashlib.sha256(before).hexdigest(),
            source_binding_sha256=digest(binding), network_step=int(agent.network.step),
            backend=jax.default_backend(), device_kind=jax.devices()[0].device_kind,
            training_flags=case['argv'][2:], observed_at=time.time())
        if before != flax.serialization.to_bytes(agent):
            raise ValueError('Initialization observer mutated the agent')
        write_once(output, record)
        return agent
    return create


def run(case_path, source, binding, output, expected_device):
    source = Path(source).resolve(); verify_source(source, binding); case = read(case_path)
    if case['argv'][1] != 'main_online.py' or Path(output).exists():
        raise ValueError('Unexpected driver or existing initialization')
    sys.path.insert(0, str(source))
    from agents.acrlpd import ACRLPDAgent
    import agents.acrlpd as module
    if Path(module.__file__).resolve() != source/'agents/acrlpd.py':
        raise ValueError('Learner imported outside the frozen release')
    descriptor = ACRLPDAgent.__dict__['create']; previous = sys.argv
    ACRLPDAgent.create = classmethod(observe_creation(descriptor.__func__, case, output, binding, expected_device=expected_device))
    sys.argv = [str(source/'main_online.py'), *case['argv'][2:]]
    try:
        runpy.run_path(str(source/'main_online.py'), run_name='__main__')
    finally:
        ACRLPDAgent.create = descriptor; sys.argv = previous


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('case', 'source', 'source-binding', 'initial-receipt'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--expected-device', default='4090')
    a = p.parse_args(); run(a.case, a.source, a.source_binding, a.initial_receipt, a.expected_device)


if __name__ == '__main__':
    main()
