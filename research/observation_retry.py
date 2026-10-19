"""Bounded retries for read-only observations, never launch or mutation calls."""
import errno
import socket
import time
import urllib.error


def transient_observation_error(error):
    if isinstance(error, urllib.error.HTTPError):
        return error.code in {408, 425, 429, 500, 502, 503, 504}
    if isinstance(error, urllib.error.URLError):
        return transient_observation_error(error.reason)
    if isinstance(error, socket.gaierror):
        # The local resolver has returned EAI_NONAME/EAI_NODATA during a
        # temporary campus-network outage for this previously verified host.
        return error.errno in {socket.EAI_AGAIN, socket.EAI_NONAME,
                               getattr(socket, 'EAI_NODATA', socket.EAI_NONAME)}
    if isinstance(error, TimeoutError):
        return True
    return isinstance(error, OSError) and error.errno in {
        errno.ECONNABORTED, errno.ECONNREFUSED, errno.ECONNRESET,
        errno.EHOSTUNREACH, errno.ENETDOWN, errno.ENETUNREACH,
        errno.ETIMEDOUT, errno.EPIPE,
    }


def retry_observation(read, *, attempts=12, sleep=time.sleep, on_retry=None):
    """Retry a caller-declared pure read; surface the final original error.

    Callers must keep authentication parsing, schema/business validation and
    every write/submission outside this function. Exhaustion is not evidence
    that the observed remote process has stopped.
    """
    if type(attempts) is not int or attempts < 1:
        raise ValueError('Expected a positive integer observation attempt limit')
    for attempt in range(1, attempts + 1):
        try:
            return read()
        except Exception as error:
            if not transient_observation_error(error) or attempt == attempts:
                raise
            delay = min(5 * 2 ** (attempt - 1), 30)
            if on_retry:
                on_retry(dict(attempt=attempt, maximum_attempts=attempts,
                              retry_after_seconds=delay, error_type=type(error).__name__))
            sleep(delay)
