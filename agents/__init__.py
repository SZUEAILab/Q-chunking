"""Lazy agent registration keeps optional learner dependencies independent."""
from collections.abc import MutableMapping
from importlib import import_module

_DEFAULT_TARGETS = {
    'acfql': ('agents.acfql', 'ACFQLAgent'),
    'acrlpd': ('agents.acrlpd', 'ACRLPDAgent'),
}


class _AgentRegistry(MutableMapping):
    def __init__(self):
        self._targets = dict(_DEFAULT_TARGETS)
        self._loaded = {}

    def __getitem__(self, key):
        if key not in self._loaded:
            module, name = self._targets[key]
            self._loaded[key] = getattr(import_module(module), name)
        return self._loaded[key]

    def __setitem__(self, key, value):
        self._loaded[key] = value

    def __delitem__(self, key):
        if key not in self:
            raise KeyError(key)
        self._loaded.pop(key, None)
        self._targets.pop(key, None)

    def __iter__(self):
        return iter(dict.fromkeys((*self._targets, *self._loaded)))

    def __len__(self):
        return len(set(self._targets) | set(self._loaded))

    def __contains__(self, key):
        return key in self._targets or key in self._loaded

    def copy(self):
        return dict(self)


agents = _AgentRegistry()
__all__ = ['agents', 'ACFQLAgent', 'ACRLPDAgent']


def __getattr__(name):
    for module, exported in _DEFAULT_TARGETS.values():
        if name == exported:
            value = getattr(import_module(module), exported)
            globals()[name] = value
            return value
    raise AttributeError(name)
