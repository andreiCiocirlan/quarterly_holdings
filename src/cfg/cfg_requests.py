import requests
from ratelimit import limits, sleep_and_retry

ONE_SECOND = 1
MAX_CALLS_PER_SECOND = 5

# Create a global session to reuse TCP connections and headers
session = requests.Session()
session.headers.update({"User-Agent": "Your Name your.email@example.com"})


@sleep_and_retry
@limits(calls=MAX_CALLS_PER_SECOND, period=ONE_SECOND)
def limited_get(url):
    resp = session.get(url, timeout=1800)
    resp.raise_for_status()
    return resp

@sleep_and_retry
@limits(calls=1, period=ONE_SECOND)
def limited_get_one_per_sec(url):
    resp = session.get(url, timeout=1800)
    resp.raise_for_status()
    return resp