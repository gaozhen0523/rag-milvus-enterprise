# tests/test_query_cache.py
import time
from libs.caching.query_cache import QueryCache


def test_cache_set_get_and_expire():
    qc = QueryCache()
    key = qc.make_key("hello world", False, 5, 5, 5, 1, 10, False)

    qc.set(key, {"a": 123}, ttl=1)
    assert qc.get(key) == {"a": 123}

    time.sleep(1.1)
    assert qc.get(key) is None


def test_key_diff_when_params_diff():
    qc = QueryCache()

    k1 = qc.make_key("hello", False, 5, 5, 5, 1, 10, False)
    k2 = qc.make_key("hello", False, 5, 5, 5, 2, 10, False)   # page 不同
    k3 = qc.make_key("hello", False, 5, 5, 5, 1, 20, False)   # page_size 不同
    k4 = qc.make_key("hello", False, 5, 5, 5, 1, 10, True)    # rerank 不同

    assert k1 != k2
    assert k1 != k3
    assert k1 != k4
