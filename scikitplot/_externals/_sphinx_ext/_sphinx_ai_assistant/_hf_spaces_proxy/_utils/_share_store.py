"""
Mutable Global Share storage control plane.

Global Share is a capability-bearing lifecycle, not a cache.  This module keeps
that lifecycle behind one bounded store interface so a deployment can choose:

``memory``
    Compatibility/development only. Process-local and lost at restart.
``sqlite``
    Restart-durable transactional storage for one local filesystem authority.
``redis``
    Shared transactional storage for multiple replicas in one Redis consistency
    domain. Redis durability is reported only when the operator explicitly
    confirms it; shared is not synonymous with durable.

Public Share identifiers are never stored as SQLite/Redis keys verbatim.  Their
SHA-256 digest is sufficient for lookup because generated identifiers carry at
least 128 bits of entropy, while keeping bearer read capabilities out of routine
backend key listings.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import sqlite3
import time
from pathlib import Path
from typing import Any

from ._redis_security import RedisSecurityError, redis_connection_kwargs


class ShareStoreError(RuntimeError):
    """Stable, non-sensitive Share control-plane error."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def _copy(entry: dict[str, Any] | None) -> dict[str, Any] | None:
    return None if entry is None else json.loads(json.dumps(entry, ensure_ascii=False))


def _key(share_id: str) -> str:
    return hashlib.sha256(str(share_id).encode("utf-8")).hexdigest()


def _now() -> float:
    return time.time()


class MemoryShareStore:
    backend = "memory"
    durability = "process_local"
    durable = False
    shared = False
    authoritative = False
    consistency_scope = "process_local"

    def __init__(self, *, max_entries: int, max_total_bytes: int) -> None:
        self.max_entries = int(max_entries)
        self.max_total_bytes = int(max_total_bytes)
        self.entries: dict[str, dict[str, Any]] = {}
        self._lock = asyncio.Lock()

    async def initialize(self) -> None:
        return None

    async def close(self) -> None:
        return None

    def manifest(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "durability": self.durability,
            "durable": self.durable,
            "shared": self.shared,
            "authoritative": self.authoritative,
            "consistency_scope": self.consistency_scope,
        }

    def _sweep(self, now: float) -> None:
        for sid in [
            sid
            for sid, e in self.entries.items()
            if float(e.get("expiresAt_ts") or 0) <= now
        ]:
            self.entries.pop(sid, None)

    async def create(self, share_id: str, entry: dict[str, Any]) -> None:
        async with self._lock:
            self._sweep(_now())
            if share_id in self.entries:
                raise ShareStoreError("DUPLICATE_SHARE")
            if len(self.entries) >= self.max_entries:
                raise ShareStoreError("ENTRY_CAPACITY")
            total = sum(int(e.get("bytes") or 0) for e in self.entries.values())
            if total + int(entry.get("bytes") or 0) > self.max_total_bytes:
                raise ShareStoreError("BYTE_CAPACITY")
            self.entries[share_id] = _copy(entry) or {}

    async def get(self, share_id: str) -> dict[str, Any] | None:
        async with self._lock:
            entry = self.entries.get(share_id)
            if entry is None:
                return None
            if float(entry.get("expiresAt_ts") or 0) <= _now():
                self.entries.pop(share_id, None)
                raise ShareStoreError("EXPIRED")
            return _copy(entry)

    async def replace_authorized(
        self, share_id: str, edit_hash: str, entry: dict[str, Any]
    ) -> None:
        async with self._lock:
            current = self.entries.get(share_id)
            if current is None:
                raise ShareStoreError("NOT_FOUND")
            if float(current.get("expiresAt_ts") or 0) <= _now():
                self.entries.pop(share_id, None)
                raise ShareStoreError("EXPIRED")
            if str(current.get("edit_hash") or "") != str(edit_hash or ""):
                raise ShareStoreError("AUTH")
            total = sum(int(e.get("bytes") or 0) for e in self.entries.values())
            proposed = (
                total - int(current.get("bytes") or 0) + int(entry.get("bytes") or 0)
            )
            if proposed > self.max_total_bytes:
                raise ShareStoreError("BYTE_CAPACITY")
            self.entries[share_id] = _copy(entry) or {}

    async def delete_authorized(self, share_id: str, edit_hash: str) -> None:
        async with self._lock:
            current = self.entries.get(share_id)
            if current is None:
                raise ShareStoreError("NOT_FOUND")
            if float(current.get("expiresAt_ts") or 0) <= _now():
                self.entries.pop(share_id, None)
                raise ShareStoreError("EXPIRED")
            if str(current.get("edit_hash") or "") != str(edit_hash or ""):
                raise ShareStoreError("AUTH")
            self.entries.pop(share_id, None)

    async def delete_unchecked(self, share_id: str) -> None:
        async with self._lock:
            self.entries.pop(share_id, None)


class SQLiteShareStore:
    backend = "sqlite"
    durability = "restart_durable_local"
    durable = True
    shared = False
    authoritative = True
    consistency_scope = "single_sqlite_file"

    def __init__(self, path: str, *, max_entries: int, max_total_bytes: int) -> None:
        if not str(path or "").strip():
            raise ShareStoreError("SQLITE_PATH_REQUIRED")
        self.path = str(path)
        self.max_entries = int(max_entries)
        self.max_total_bytes = int(max_total_bytes)
        self._lock = asyncio.Lock()

    def manifest(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "durability": self.durability,
            "durable": self.durable,
            "shared": self.shared,
            "authoritative": self.authoritative,
            "consistency_scope": self.consistency_scope,
            "public_id_at_rest": "sha256",
        }

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=5.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA busy_timeout=5000")
        conn.execute("PRAGMA secure_delete=ON")
        return conn

    def _init_sync(self) -> None:
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        conn = self._connect()
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=FULL")
            conn.execute("""CREATE TABLE IF NOT EXISTS global_shares (
                    share_key TEXT PRIMARY KEY,
                    entry_json TEXT NOT NULL,
                    bytes INTEGER NOT NULL,
                    expires_at REAL NOT NULL,
                    edit_hash TEXT NOT NULL,
                    updated_at REAL NOT NULL
                )
            """)
            conn.execute(
                "CREATE INDEX IF NOT EXISTS ix_global_shares_expires ON global_shares(expires_at)"
            )
            conn.execute("DELETE FROM global_shares WHERE expires_at <= ?", (_now(),))
            conn.commit()
        finally:
            conn.close()

    async def initialize(self) -> None:
        await asyncio.to_thread(self._init_sync)

    async def close(self) -> None:
        return None

    async def create(self, share_id: str, entry: dict[str, Any]) -> None:
        async with self._lock:

            def op() -> None:
                conn = self._connect()
                try:
                    conn.execute("BEGIN IMMEDIATE")
                    now = _now()
                    conn.execute(
                        "DELETE FROM global_shares WHERE expires_at <= ?", (now,)
                    )
                    count, total = conn.execute(
                        "SELECT COUNT(*), COALESCE(SUM(bytes),0) FROM global_shares"
                    ).fetchone()
                    if int(count) >= self.max_entries:
                        raise ShareStoreError("ENTRY_CAPACITY")
                    if int(total) + int(entry.get("bytes") or 0) > self.max_total_bytes:
                        raise ShareStoreError("BYTE_CAPACITY")
                    try:
                        conn.execute(
                            "INSERT INTO global_shares(share_key,entry_json,bytes,expires_at,edit_hash,updated_at) VALUES(?,?,?,?,?,?)",
                            (
                                _key(share_id),
                                json.dumps(
                                    entry, ensure_ascii=False, separators=(",", ":")
                                ),
                                int(entry.get("bytes") or 0),
                                float(entry.get("expiresAt_ts") or 0),
                                str(entry.get("edit_hash") or ""),
                                now,
                            ),
                        )
                    except sqlite3.IntegrityError as exc:
                        raise ShareStoreError("DUPLICATE_SHARE") from exc
                    conn.commit()
                except Exception:
                    conn.rollback()
                    raise
                finally:
                    conn.close()

            await asyncio.to_thread(op)

    async def get(self, share_id: str) -> dict[str, Any] | None:
        async with self._lock:

            def op() -> dict[str, Any] | None:
                conn = self._connect()
                try:
                    conn.execute("BEGIN IMMEDIATE")
                    row = conn.execute(
                        "SELECT entry_json,expires_at FROM global_shares WHERE share_key=?",
                        (_key(share_id),),
                    ).fetchone()
                    if row is None:
                        conn.commit()
                        return None
                    if float(row["expires_at"] or 0) <= _now():
                        conn.execute(
                            "DELETE FROM global_shares WHERE share_key=?",
                            (_key(share_id),),
                        )
                        conn.commit()
                        raise ShareStoreError("EXPIRED")
                    conn.commit()
                    return json.loads(row["entry_json"])
                finally:
                    conn.close()

            return await asyncio.to_thread(op)

    async def replace_authorized(
        self, share_id: str, edit_hash: str, entry: dict[str, Any]
    ) -> None:
        async with self._lock:

            def op() -> None:
                conn = self._connect()
                try:
                    conn.execute("BEGIN IMMEDIATE")
                    row = conn.execute(
                        "SELECT bytes,expires_at,edit_hash FROM global_shares WHERE share_key=?",
                        (_key(share_id),),
                    ).fetchone()
                    if row is None:
                        raise ShareStoreError("NOT_FOUND")
                    if float(row["expires_at"] or 0) <= _now():
                        conn.execute(
                            "DELETE FROM global_shares WHERE share_key=?",
                            (_key(share_id),),
                        )
                        conn.commit()
                        raise ShareStoreError("EXPIRED")
                    if str(row["edit_hash"] or "") != str(edit_hash or ""):
                        raise ShareStoreError("AUTH")
                    total = int(
                        conn.execute(
                            "SELECT COALESCE(SUM(bytes),0) FROM global_shares"
                        ).fetchone()[0]
                    )
                    proposed = (
                        total - int(row["bytes"] or 0) + int(entry.get("bytes") or 0)
                    )
                    if proposed > self.max_total_bytes:
                        raise ShareStoreError("BYTE_CAPACITY")
                    conn.execute(
                        "UPDATE global_shares SET entry_json=?,bytes=?,expires_at=?,edit_hash=?,updated_at=? WHERE share_key=? AND edit_hash=?",
                        (
                            json.dumps(
                                entry, ensure_ascii=False, separators=(",", ":")
                            ),
                            int(entry.get("bytes") or 0),
                            float(entry.get("expiresAt_ts") or 0),
                            str(entry.get("edit_hash") or ""),
                            _now(),
                            _key(share_id),
                            edit_hash,
                        ),
                    )
                    conn.commit()
                except Exception:
                    conn.rollback()
                    raise
                finally:
                    conn.close()

            await asyncio.to_thread(op)

    async def delete_authorized(self, share_id: str, edit_hash: str) -> None:
        async with self._lock:

            def op() -> None:
                conn = self._connect()
                try:
                    conn.execute("BEGIN IMMEDIATE")
                    row = conn.execute(
                        "SELECT expires_at,edit_hash FROM global_shares WHERE share_key=?",
                        (_key(share_id),),
                    ).fetchone()
                    if row is None:
                        raise ShareStoreError("NOT_FOUND")
                    if float(row["expires_at"] or 0) <= _now():
                        conn.execute(
                            "DELETE FROM global_shares WHERE share_key=?",
                            (_key(share_id),),
                        )
                        conn.commit()
                        raise ShareStoreError("EXPIRED")
                    if str(row["edit_hash"] or "") != str(edit_hash or ""):
                        raise ShareStoreError("AUTH")
                    conn.execute(
                        "DELETE FROM global_shares WHERE share_key=?", (_key(share_id),)
                    )
                    conn.commit()
                except Exception:
                    conn.rollback()
                    raise
                finally:
                    conn.close()

            await asyncio.to_thread(op)

    async def delete_unchecked(self, share_id: str) -> None:
        async with self._lock:

            def op() -> None:
                conn = self._connect()
                try:
                    conn.execute(
                        "DELETE FROM global_shares WHERE share_key=?", (_key(share_id),)
                    )
                    conn.commit()
                finally:
                    conn.close()

            await asyncio.to_thread(op)


_REDIS_CREATE = r"""
local now=tonumber(ARGV[1]); local member=ARGV[2]; local raw=ARGV[3]; local exp=tonumber(ARGV[4]);
local max_entries=tonumber(ARGV[5]); local max_bytes=tonumber(ARGV[6]); local bytes=tonumber(ARGV[7]); local ttl=tonumber(ARGV[8]); local prefix=ARGV[9]
local expired=redis.call('ZRANGEBYSCORE',KEYS[1],'-inf',now)
for _,m in ipairs(expired) do
  local old=redis.call('GET',prefix..m); if old then local e=cjson.decode(old); redis.call('DECRBY',KEYS[2],tonumber(e.bytes or 0)) end
  redis.call('DEL',prefix..m); redis.call('ZREM',KEYS[1],m)
end
if redis.call('EXISTS',KEYS[3]) == 1 then return {0,'DUPLICATE_SHARE'} end
if redis.call('ZCARD',KEYS[1]) >= max_entries then return {0,'ENTRY_CAPACITY'} end
local total=tonumber(redis.call('GET',KEYS[2]) or '0'); if total+bytes > max_bytes then return {0,'BYTE_CAPACITY'} end
redis.call('SET',KEYS[3],raw,'EX',ttl); redis.call('ZADD',KEYS[1],exp,member); redis.call('INCRBY',KEYS[2],bytes); return {1,'OK'}
""".strip()

_REDIS_GET = r"""
local now=tonumber(ARGV[1]); local member=ARGV[2]
local old=redis.call('GET',KEYS[3]); if not old then return {0,'NOT_FOUND'} end
local e=cjson.decode(old)
if tonumber(e.expiresAt_ts or 0) <= now then
  redis.call('DEL',KEYS[3]); redis.call('ZREM',KEYS[1],member)
  local n=tonumber(e.bytes or 0); if n > 0 then redis.call('DECRBY',KEYS[2],n) end
  return {0,'EXPIRED'}
end
return {1,old}
""".strip()

_REDIS_REPLACE = r"""
local now=tonumber(ARGV[1]); local raw=ARGV[2]; local exp=tonumber(ARGV[3]); local bytes=tonumber(ARGV[4]); local ttl=tonumber(ARGV[5]); local expected=ARGV[6]; local max_bytes=tonumber(ARGV[7])
local old=redis.call('GET',KEYS[3]); if not old then return {0,'NOT_FOUND'} end
local e=cjson.decode(old); if tonumber(e.expiresAt_ts or 0) <= now then redis.call('DEL',KEYS[3]); redis.call('ZREM',KEYS[1],ARGV[8]); redis.call('DECRBY',KEYS[2],tonumber(e.bytes or 0)); return {0,'EXPIRED'} end
if tostring(e.edit_hash or '') ~= expected then return {0,'AUTH'} end
local total=tonumber(redis.call('GET',KEYS[2]) or '0'); local proposed=total-tonumber(e.bytes or 0)+bytes; if proposed > max_bytes then return {0,'BYTE_CAPACITY'} end
redis.call('SET',KEYS[3],raw,'EX',ttl); redis.call('ZADD',KEYS[1],exp,ARGV[8]); redis.call('SET',KEYS[2],proposed); return {1,'OK'}
""".strip()

_REDIS_DELETE = r"""
local now=tonumber(ARGV[1]); local expected=ARGV[2]; local member=ARGV[3]
local old=redis.call('GET',KEYS[3]); if not old then return {0,'NOT_FOUND'} end
local e=cjson.decode(old); if tonumber(e.expiresAt_ts or 0) <= now then redis.call('DEL',KEYS[3]); redis.call('ZREM',KEYS[1],member); redis.call('DECRBY',KEYS[2],tonumber(e.bytes or 0)); return {0,'EXPIRED'} end
if tostring(e.edit_hash or '') ~= expected then return {0,'AUTH'} end
redis.call('DEL',KEYS[3]); redis.call('ZREM',KEYS[1],member); redis.call('DECRBY',KEYS[2],tonumber(e.bytes or 0)); return {1,'OK'}
""".strip()


class RedisShareStore:
    backend = "redis"
    shared = True
    authoritative = True
    consistency_scope = "single_redis_consistency_domain"

    def __init__(
        self,
        url: str,
        *,
        key_prefix: str,
        max_entries: int,
        max_total_bytes: int,
        durable_confirmed: bool = False,
        socket_timeout_seconds: float = 2.0,
        client: Any | None = None,
        require_tls: bool = False,
    ) -> None:
        if not str(url or "").strip():
            raise ShareStoreError("REDIS_URL_REQUIRED")
        self.url = str(url).strip()
        self.max_entries = int(max_entries)
        self.max_total_bytes = int(max_total_bytes)
        self.require_tls = bool(require_tls)
        try:
            self._transport, self._connection_kwargs = redis_connection_kwargs(
                self.url,
                require_tls=self.require_tls,
                socket_timeout_seconds=socket_timeout_seconds,
            )
        except RedisSecurityError as exc:
            raise ShareStoreError(exc.code) from exc
        self.durable = bool(durable_confirmed)
        self.durability = (
            "shared_external_persistence_confirmed"
            if self.durable
            else "shared_external_persistence_unverified"
        )
        safe = "".join(
            c
            for c in str(key_prefix or "sphinx-ai-assistant").lower()
            if c.isalnum() or c in "_-:"
        )[:64]
        self.key_prefix = safe or "sphinx-ai-assistant"
        tag = f"{self.key_prefix}:{{share}}"
        self._all = f"{tag}:all"
        self._bytes = f"{tag}:bytes"
        self._prefix = f"{tag}:entry:"
        self.socket_timeout_seconds = max(
            0.25, min(float(socket_timeout_seconds), 10.0)
        )
        self._client = client
        self._owns = client is None
        self._lock = asyncio.Lock()

    def manifest(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "durability": self.durability,
            "durable": self.durable,
            "shared": True,
            "authoritative": True,
            "consistency_scope": self.consistency_scope,
            "public_id_at_rest": "sha256",
            **self._transport.manifest(),
        }

    async def initialize(self) -> None:
        async with self._lock:
            if self._client is None:
                try:
                    import redis.asyncio as redis_async  # type: ignore[import-not-found]  # ruff: ignore[import-outside-top-level]
                except Exception as exc:
                    raise ShareStoreError("REDIS_DEPENDENCY_UNAVAILABLE") from exc
                self._client = redis_async.from_url(self.url, **self._connection_kwargs)
            try:
                await self._client.ping()
            except Exception as exc:
                raise ShareStoreError("REDIS_UNAVAILABLE") from exc

    async def close(self) -> None:
        if self._client is None or not self._owns:
            return
        closer = getattr(self._client, "aclose", None) or getattr(
            self._client, "close", None
        )
        if closer:
            result = closer()
            if hasattr(result, "__await__"):
                await result
        self._client = None

    def _keys(self, share_id: str) -> tuple[str, str]:
        member = _key(share_id)
        return member, self._prefix + member

    async def _eval(
        self, script: str, keys: list[str], args: list[Any]
    ) -> tuple[int, str]:
        if self._client is None:
            raise ShareStoreError("REDIS_NOT_INITIALIZED")
        try:
            out = await self._client.eval(script, len(keys), *keys, *args)
        except Exception as exc:
            raise ShareStoreError("REDIS_OPERATION_FAILED") from exc
        if not isinstance(out, (list, tuple)) or len(out) < (
            2  # ruff: ignore[magic-value-comparison]
        ):
            raise ShareStoreError("REDIS_PROTOCOL_ERROR")
        val = out[1].decode() if isinstance(out[1], bytes) else str(out[1])
        return int(out[0]), val

    @staticmethod
    def _encode(entry: dict[str, Any]) -> str:
        return json.dumps(entry, ensure_ascii=False, separators=(",", ":"))

    async def create(self, share_id: str, entry: dict[str, Any]) -> None:
        member, key = self._keys(share_id)
        now = _now()
        exp = float(entry.get("expiresAt_ts") or 0)
        ttl = max(1, int(exp - now + 0.999))
        ok, val = await self._eval(
            _REDIS_CREATE,
            [self._all, self._bytes, key],
            [
                now,
                member,
                self._encode(entry),
                exp,
                self.max_entries,
                self.max_total_bytes,
                int(entry.get("bytes") or 0),
                ttl,
                self._prefix,
            ],
        )
        if not ok:
            raise ShareStoreError(val)

    async def get(self, share_id: str) -> dict[str, Any] | None:
        member, key = self._keys(share_id)
        ok, val = await self._eval(
            _REDIS_GET, [self._all, self._bytes, key], [_now(), member]
        )
        if not ok:
            if val == "NOT_FOUND":
                return None
            raise ShareStoreError(val)
        try:
            return json.loads(val)
        except Exception as exc:
            raise ShareStoreError("REDIS_PROTOCOL_ERROR") from exc

    async def replace_authorized(
        self, share_id: str, edit_hash: str, entry: dict[str, Any]
    ) -> None:
        member, key = self._keys(share_id)
        now = _now()
        exp = float(entry.get("expiresAt_ts") or 0)
        ttl = max(1, int(exp - now + 0.999))
        ok, val = await self._eval(
            _REDIS_REPLACE,
            [self._all, self._bytes, key],
            [
                now,
                self._encode(entry),
                exp,
                int(entry.get("bytes") or 0),
                ttl,
                edit_hash,
                self.max_total_bytes,
                member,
            ],
        )
        if not ok:
            raise ShareStoreError(val)

    async def delete_authorized(self, share_id: str, edit_hash: str) -> None:
        member, key = self._keys(share_id)
        ok, val = await self._eval(
            _REDIS_DELETE, [self._all, self._bytes, key], [_now(), edit_hash, member]
        )
        if not ok:
            raise ShareStoreError(val)

    async def delete_unchecked(self, share_id: str) -> None:
        if self._client is None:
            raise ShareStoreError("REDIS_NOT_INITIALIZED")
        member, key = self._keys(share_id)
        try:
            raw = await self._client.get(key)
            n = 0
            if raw:
                if isinstance(raw, bytes):
                    raw = raw.decode("utf-8")
                try:
                    n = int(json.loads(str(raw)).get("bytes") or 0)
                except Exception:  # ruff: ignore[blind-except]
                    n = 0
            pipe = self._client.pipeline(transaction=True)
            pipe.delete(key)
            pipe.zrem(self._all, member)
            if n:
                pipe.decrby(self._bytes, n)
            await pipe.execute()
        except Exception as exc:
            raise ShareStoreError("REDIS_OPERATION_FAILED") from exc


def build_share_store(
    backend: str,
    *,
    sqlite_path: str,
    redis_url: str = "",
    redis_key_prefix: str = "sphinx-ai-assistant",
    redis_timeout_seconds: float = 2.0,
    redis_durable_confirmed: bool = False,
    max_entries: int,
    max_total_bytes: int,
    redis_client: Any | None = None,
    require_redis_tls: bool = False,
):
    name = str(backend or "memory").strip().lower()
    if name == "memory":
        return MemoryShareStore(
            max_entries=max_entries, max_total_bytes=max_total_bytes
        )
    if name == "sqlite":
        return SQLiteShareStore(
            sqlite_path, max_entries=max_entries, max_total_bytes=max_total_bytes
        )
    if name == "redis":
        return RedisShareStore(
            redis_url,
            key_prefix=redis_key_prefix,
            max_entries=max_entries,
            max_total_bytes=max_total_bytes,
            durable_confirmed=redis_durable_confirmed,
            socket_timeout_seconds=redis_timeout_seconds,
            client=redis_client,
            require_tls=require_redis_tls,
        )
    raise ShareStoreError("UNSUPPORTED_BACKEND")
