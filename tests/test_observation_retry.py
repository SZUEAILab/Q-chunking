import errno
import socket
import urllib.error

import pytest

from research.observation_retry import retry_observation, transient_observation_error


@pytest.mark.parametrize('error', [
    urllib.error.URLError(socket.gaierror(socket.EAI_AGAIN, 'temporary DNS')),
    urllib.error.URLError(socket.gaierror(socket.EAI_NONAME, 'resolver outage')),
    urllib.error.URLError(socket.gaierror(getattr(socket, 'EAI_NODATA', -5), 'no address')),
    ConnectionResetError(errno.ECONNRESET, 'reset'), TimeoutError(),
    urllib.error.HTTPError('https://fixture.invalid', 503, 'unavailable', {}, None),
    urllib.error.HTTPError('https://fixture.invalid', 429, 'rate limited', {}, None),
])
def test_transient_read_recovers_without_claiming_remote_termination(error):
    calls, waits, events = [], [], []
    def read():
        calls.append(1)
        if len(calls) == 1:
            raise error
        return {'state': 'Running'}
    assert retry_observation(read, sleep=waits.append, on_retry=events.append) == {'state': 'Running'}
    assert len(calls) == 2 and waits == [5]
    assert events == [dict(attempt=1, maximum_attempts=12, retry_after_seconds=5,
                           error_type=type(error).__name__)]


@pytest.mark.parametrize('error', [
    ValueError('source/schema mismatch'), FileNotFoundError('missing credentials'),
    PermissionError('credentials inaccessible'), RuntimeError('platform business error'),
    urllib.error.URLError('certificate verify failed'),
    *[urllib.error.HTTPError('https://fixture.invalid', code, 'permanent', {}, None)
      for code in (400, 401, 403, 404)],
])
def test_authentication_identity_and_business_errors_are_not_retried(error):
    calls, waits = [], []
    def read():
        calls.append(1)
        raise error
    with pytest.raises(type(error)) as caught:
        retry_observation(read, sleep=waits.append)
    assert caught.value is error and calls == [1] and not waits
    assert not transient_observation_error(error)


def test_exhaustion_preserves_original_failure_and_bounded_delays():
    error = TimeoutError('read unavailable'); waits, events = [], []
    def read(): raise error
    with pytest.raises(TimeoutError) as caught:
        retry_observation(read, attempts=7, sleep=waits.append, on_retry=events.append)
    assert caught.value is error
    assert waits == [5, 10, 20, 30, 30, 30]
    assert len(events) == 6 and events[-1]['attempt'] == 6


@pytest.mark.parametrize('attempts', [0, -1, False, 1.5])
def test_invalid_retry_limit_never_calls_read(attempts):
    with pytest.raises(ValueError):
        retry_observation(lambda: pytest.fail('Read should not run'), attempts=attempts)
