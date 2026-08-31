from __future__ import annotations

import asyncio
import importlib
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PROXY = ROOT / "_hf_spaces_proxy"
if str(PROXY) not in sys.path:
    sys.path.insert(0, str(PROXY))
RATE = ROOT / "_hf_spaces_proxy" / "_utils" / "_rate_limit.py"
APP = ROOT / "_hf_spaces_proxy" / "app.py"
WORKER = ROOT / "_cf_worker" / "index.js"
WRANGLER = ROOT / "_cf_worker" / "wrangler.toml"
REQ = ROOT / "_hf_spaces_proxy" / "requirements.txt"


def _load_rate_module():
    return importlib.import_module("_utils._rate_limit")




def _assert_hf_shared_authority_contract(src: str) -> None:
    assert 'if RATE_LIMIT_REQUIRE_SHARED and RATE_LIMIT_BACKEND == "local":' in src
    assert 'if RATE_LIMIT_BACKEND != "local":' in src
    start = src.index('if RATE_LIMIT_BACKEND != "local":')
    end = src.index('now = _time.time()', start)
    shared_branch = src[start:end]
    # Keep the assertion semantic: formatter changes may wrap HTTPException over
    # multiple lines without changing the fail-closed 503 contract.
    assert 'status_code=503' in shared_branch
    assert 'detail="Shared rate limiter unavailable."' in shared_branch
    assert '_chat_rl' not in shared_branch
    assert '_share_rl' not in shared_branch
    assert '_feedback_rl' not in shared_branch
    assert '_contrib_rl' not in shared_branch


def _assert_worker_shared_authority_contract(worker: str, wrangler: str) -> None:
    assert 'export class RateLimitBucket extends DurableObject' in worker
    assert "if (env.RATE_LIMIT_DO)" in worker
    assert "const identityHash = await _hmacSha256Hex(env.RATE_LIMIT_IDENTITY_SECRET, identity || 'unknown')" in worker
    assert "await stub.consume" in worker
    assert 'name = "RATE_LIMIT_DO"' in wrangler
    assert 'class_name = "RateLimitBucket"' in wrangler
    assert '[exports.RateLimitBucket]' in wrangler
    assert 'storage = "sqlite"' in wrangler
    assert 'RATE_LIMIT_REQUIRE_AUTHORITATIVE = "true"' in wrangler


class FakeRedis:
    def __init__(self) -> None:
        self.counts: dict[str, int] = {}
        self.ttl: dict[str, int] = {}
        self.keys_seen: list[str] = []
        self._lock = asyncio.Lock()
        self.closed = False

    async def ping(self):
        return True

    async def eval(self, _script, numkeys, key, window_seconds):
        assert numkeys == 1
        async with self._lock:
            self.keys_seen.append(str(key))
            self.counts[str(key)] = self.counts.get(str(key), 0) + 1
            self.ttl[str(key)] = int(window_seconds)
            return [self.counts[str(key)], self.ttl[str(key)]]

    async def aclose(self):
        self.closed = True


@pytest.mark.asyncio
async def test_redis_rate_limiter_is_shared_atomic_and_identity_private():
    mod = _load_rate_module()
    fake = FakeRedis()
    secret = "s" * 48
    a = mod.RedisRateLimiter("redis://shared.invalid/0", identity_secret=secret, client=fake)
    b = mod.RedisRateLimiter("redis://shared.invalid/0", identity_secret=secret, client=fake)
    await a.initialize()
    await b.initialize()

    assert (await a.consume("203.0.113.44", scope="chat", limit=2))[0] is True
    assert (await b.consume("203.0.113.44", scope="chat", limit=2))[0] is True
    allowed, count, retry = await a.consume("203.0.113.44", scope="chat", limit=2)
    assert allowed is False
    assert count == 3
    assert retry == 3600
    assert len(set(fake.keys_seen)) == 1
    assert all("203.0.113.44" not in key for key in fake.keys_seen)
    assert fake.keys_seen[0].startswith("sphinx-ai-assistant:rl:chat:")


@pytest.mark.asyncio
async def test_redis_scope_separation_and_manifest():
    mod = _load_rate_module()
    fake = FakeRedis()
    limiter = mod.RedisRateLimiter(
        "rediss://shared.invalid/0", identity_secret="k" * 64, key_prefix="demo", client=fake
    )
    await limiter.initialize()
    assert (await limiter.consume("198.51.100.5", scope="chat", limit=1))[0]
    assert (await limiter.consume("198.51.100.5", scope="share", limit=1))[0]
    assert len(set(fake.keys_seen)) == 2
    manifest = limiter.manifest()
    assert manifest["shared"] is True
    assert manifest["authoritative"] is True
    assert manifest["consistency_scope"] == "single_redis_consistency_domain"
    assert manifest["identity_externalized"] == "hmac_sha256"


def test_redis_mode_requires_nontrivial_identity_secret():
    mod = _load_rate_module()
    with pytest.raises(mod.RateLimitBackendError) as exc:
        mod.RedisRateLimiter("redis://shared.invalid/0", identity_secret="short")
    assert exc.value.code == "IDENTITY_SECRET_TOO_SHORT"


def test_hf_proxy_redis_mode_is_fail_closed_not_local_fallback():
    src = APP.read_text(encoding="utf-8")
    # Formatting may wrap the assignment; assert the environment contract rather
    # than one historical single-line rendering of it.
    assert 'RATE_LIMIT_BACKEND: str = (' in src
    assert 'os.environ.get("RATE_LIMIT_BACKEND", "local").strip().lower() or "local"' in src
    _assert_hf_shared_authority_contract(src)
    for scope in ('scope="chat"', 'scope="share"', 'scope="feedback"', 'scope="contribution"'):
        assert scope in src


def test_worker_bundles_durable_object_authority_and_keeps_kv_fallback_explicit():
    worker = WORKER.read_text(encoding="utf-8")
    wrangler = WRANGLER.read_text(encoding="utf-8")
    _assert_worker_shared_authority_contract(worker, wrangler)
    assert "backend: 'durable_object'" in worker
    assert "backend: 'kv'" in worker
    assert 'RATE_LIMIT_IDENTITY_SECRET' in wrangler


def test_worker_does_not_substitute_permissive_rate_limit_binding_for_authority():
    wrangler = WRANGLER.read_text(encoding="utf-8")
    assert "[[ratelimits]]" not in wrangler
    assert "Rate Limiting binding" in wrangler
    assert "intentionally local/permissive" in wrangler


def test_hf_fresh_deploy_installs_redis_client_but_local_mode_is_lazy():
    req = REQ.read_text(encoding="utf-8")
    rate = RATE.read_text(encoding="utf-8")
    assert "redis==8.1.0" in req
    assert "import redis.asyncio as redis_async" in rate
    assert rate.index("import redis.asyncio as redis_async") > rate.index("async def initialize")


def test_run15_positive_control_mutant_hf_shared_requirement_removed_is_caught():
    src = APP.read_text(encoding="utf-8")
    anchor = 'if RATE_LIMIT_REQUIRE_SHARED and RATE_LIMIT_BACKEND == "local":'
    assert src.count(anchor) == 1
    mutated = src.replace(anchor, 'if False and RATE_LIMIT_BACKEND == "local":', 1)
    with pytest.raises(AssertionError):
        _assert_hf_shared_authority_contract(mutated)


def test_run15_positive_control_mutant_worker_hmac_removed_is_caught():
    worker = WORKER.read_text(encoding="utf-8")
    wrangler = WRANGLER.read_text(encoding="utf-8")
    anchor = "const identityHash = await _hmacSha256Hex(env.RATE_LIMIT_IDENTITY_SECRET, identity || 'unknown')"
    assert worker.count(anchor) == 1
    mutated = worker.replace(anchor, "const identityHash = await _sha256Hex(identity || 'unknown')", 1)
    with pytest.raises(AssertionError):
        _assert_worker_shared_authority_contract(mutated, wrangler)


def test_run15_positive_control_mutant_worker_do_binding_removed_is_caught():
    worker = WORKER.read_text(encoding="utf-8")
    wrangler = WRANGLER.read_text(encoding="utf-8")
    anchor = 'name = "RATE_LIMIT_DO"'
    assert wrangler.count(anchor) == 1
    mutated = wrangler.replace(anchor, 'name = "RATE_LIMIT_DO_DISABLED"', 1)
    with pytest.raises(AssertionError):
        _assert_worker_shared_authority_contract(worker, mutated)


def test_run15_positive_control_mutant_authoritative_default_disabled_is_caught():
    worker = WORKER.read_text(encoding="utf-8")
    wrangler = WRANGLER.read_text(encoding="utf-8")
    anchor = 'RATE_LIMIT_REQUIRE_AUTHORITATIVE = "true"'
    assert wrangler.count(anchor) == 1
    mutated = wrangler.replace(anchor, 'RATE_LIMIT_REQUIRE_AUTHORITATIVE = "false"', 1)
    with pytest.raises(AssertionError):
        _assert_worker_shared_authority_contract(worker, mutated)
