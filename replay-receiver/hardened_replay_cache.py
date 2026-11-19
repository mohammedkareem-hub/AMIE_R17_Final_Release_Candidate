"""Opt-in, standalone SQLite replay defense for AMIE-BakerChen receivers.

Not a replacement for HMAC verification. Authenticate the complete package with
its expected master key BEFORE calling reserve. The package must bind the nonce.

Security boundary: protect the DB and sidecar with OS access control. A trusted
external monotonic high-water mark (when configured) improves rollback detection;
restoring *all* local state, including any anchor stored on the same device, is
not detectable cryptographically. Rotation requires a NEW master key and
preservation of state for old keys while they are accepted.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import secrets
import sqlite3
import time
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
HEX_64 = re.compile(r"[0-9a-f]{64}\Z")
HEX_32 = re.compile(r"[0-9a-f]{32}\Z")

class ReplayCacheError(RuntimeError):
    """Base for operational replay-cache failures. Fail closed: do not decrypt."""

class ReplayDetectedError(ReplayCacheError):
    pass

class CacheStateUnavailable(ReplayCacheError):
    pass

class CacheCorruptError(ReplayCacheError):
    pass

class CacheCapacityExceeded(ReplayCacheError):
    pass

class InvalidReplayInput(ReplayCacheError):
    pass

class ReservationStateError(ReplayCacheError):
    pass


def replay_key_namespace(master_key: bytes) -> str:
    """Use CFP9's existing namespace scheme; caller must bind authentic key."""
    if not isinstance(master_key, bytes) or len(master_key) != 32:
        raise InvalidReplayInput("master_key must be exactly 32 bytes")
    return hashlib.sha256(b"ReplayKeyNamespace-v1|" + master_key).hexdigest()


def _validate(receiver_id: str, key_namespace: str | None = None,
              nonce: bytes | None = None, package_fingerprint: str | None = None) -> None:
    if (not isinstance(receiver_id, str) or not 1 <= len(receiver_id) <= 128
            or not re.fullmatch(r"[A-Za-z0-9_.:-]+", receiver_id)):
        raise InvalidReplayInput("receiver_id must be 1..128 safe ASCII chars")
    if key_namespace is not None and (not isinstance(key_namespace, str) or
                                         not HEX_64.fullmatch(key_namespace)):
        raise InvalidReplayInput("key_namespace must be lower-case 64-digit SHA-256 hex")
    if nonce is not None and (not isinstance(nonce, bytes) or len(nonce) != 16):
        raise InvalidReplayInput("nonce must be exactly 16 raw bytes (128 bits)")
    if package_fingerprint is not None and (not isinstance(package_fingerprint, str) or
                                            not HEX_64.fullmatch(package_fingerprint)):
        raise InvalidReplayInput("fingerprint must be lower-case 64-digit SHA-256 hex")


def _configure(db: sqlite3.Connection) -> None:
    mode = db.execute("PRAGMA journal_mode=WAL").fetchone()[0]
    if str(mode).lower() != "wal":
        raise CacheStateUnavailable("WAL mode unavailable")
    db.execute("PRAGMA synchronous=FULL")
    db.execute("PRAGMA busy_timeout=30000")
    db.execute("PRAGMA foreign_keys=ON")
    db.execute("PRAGMA trusted_schema=OFF")


def _sidecar_path(path: Path) -> Path:
    return path.with_name(path.name + ".identity")


class HardenedReplayCache:
    def __init__(self, db: sqlite3.Connection, database_path: Path,
                 receiver_id: str, instance_uuid: str, max_entries: int):
        self.connection = db
        self.database_path = database_path
        self.receiver_id = receiver_id
        self.instance_uuid = instance_uuid
        self.max_entries = max_entries
        self._closed = False

    @classmethod
    def provision(cls, database_path: str | Path, *, receiver_id: str,
                  max_entries: int = 1_000_000) -> "HardenedReplayCache":
        """Explicit one-time provisioning. NEVER auto-initialize missing state."""
        _validate(receiver_id)
        if isinstance(max_entries, bool) or not isinstance(max_entries, int) or max_entries < 1:
            raise InvalidReplayInput("max_entries must be positive integer")
        path = Path(database_path).absolute()
        marker = _sidecar_path(path)
        if not path.parent.exists() or not path.parent.is_dir():
            raise CacheStateUnavailable("Create dedicated protected parent directory first")
        if path.exists() or marker.exists() or any(path.parent.glob(path.name + '-*')):
            raise CacheStateUnavailable("Existing state or identity marker: provision refuses overwrite")
        inst = secrets.token_hex(16)
        envelope = {"instance_uuid": inst, "receiver_id": receiver_id,
                    "schema_version": SCHEMA_VERSION, "max_entries": max_entries}
        try:
            fd = os.open(marker, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError as exc:
            raise CacheStateUnavailable("Already provisioned or concurrent provisioning") from exc
        try:
            with os.fdopen(fd, 'wb') as out:
                out.write(json.dumps(envelope, sort_keys=True, separators=(',', ':')).encode('ascii'))
                out.flush()
                os.fsync(out.fileno())
            # DB is created only during operator-approved explicit provisioning.
            db = sqlite3.connect(path, timeout=30, isolation_level=None)
            try:
                _configure(db)
                db.execute('BEGIN IMMEDIATE')
                db.execute("""CREATE TABLE replay_meta (
                  singleton INTEGER PRIMARY KEY CHECK(singleton=1),
                  instance_uuid TEXT NOT NULL, receiver_id TEXT NOT NULL,
                  schema_version INTEGER NOT NULL, max_entries INTEGER NOT NULL
                )""")
                db.execute("""CREATE TABLE replay_nonces (
                  key_namespace TEXT NOT NULL CHECK(length(key_namespace)=64),
                  nonce_hex TEXT NOT NULL CHECK(length(nonce_hex)=32),
                  state TEXT NOT NULL CHECK(state IN ('pending','accepted')),
                  package_fingerprint TEXT NOT NULL CHECK(length(package_fingerprint)=64),
                  created_utc INTEGER NOT NULL,
                  PRIMARY KEY(key_namespace,nonce_hex)
                )""")
                db.execute("INSERT INTO replay_meta VALUES (1,?,?,?,?)",
                           (inst, receiver_id, SCHEMA_VERSION, max_entries))
                db.execute('COMMIT')
            except Exception:
                if db.in_transaction:
                    db.execute('ROLLBACK')
                db.close()
                raise
            db.close()
            return cls.open(path, receiver_id=receiver_id)
        except Exception:
            # Do not silently delete marker or DB. Incomplete setup MUST fail closed.
            raise

    @classmethod
    def open(cls, database_path: str | Path, *, receiver_id: str,
             trusted_minimum_records: int | None = None,
             expected_instance_uuid: str | None = None) -> "HardenedReplayCache":
        _validate(receiver_id)
        if trusted_minimum_records is not None and (
                isinstance(trusted_minimum_records, bool) or
                not isinstance(trusted_minimum_records, int) or trusted_minimum_records < 0):
            raise InvalidReplayInput("trusted_minimum_records must be nonnegative")
        path = Path(database_path).absolute()
        marker = _sidecar_path(path)
        if not path.is_file() or not marker.is_file():
            raise CacheStateUnavailable("Missing SQLite state or identity marker: STOP. No auto-create")
        try:
            guard = json.loads(marker.read_text(encoding="ascii"))
            if not isinstance(guard, dict) or set(guard) != {
                    'instance_uuid', 'receiver_id', 'schema_version', 'max_entries'}:
                raise ValueError("malformed identity marker")
            inst = guard['instance_uuid']
            if not isinstance(inst, str) or not HEX_32.fullmatch(inst):
                raise ValueError("invalid UUID")
            if guard['receiver_id'] != receiver_id or guard['schema_version'] != SCHEMA_VERSION:
                raise ValueError("wrong receiver / schema version")
            max_entries = guard['max_entries']
            if type(max_entries) is not int or max_entries < 1:
                raise ValueError("invalid capacity")
            if expected_instance_uuid is not None and inst != expected_instance_uuid:
                raise ValueError("wrong trusted receiver instance UUID")
        except (OSError, ValueError, TypeError, KeyError, UnicodeError) as exc:
            raise CacheStateUnavailable("Identity marker unavailable or inconsistent") from exc
        try:
            # uri mode=rw never silently creates a missing DB.
            db = sqlite3.connect(path.as_uri() + '?mode=rw', uri=True,
                                 timeout=30, isolation_level=None)
            _configure(db)
            result = db.execute('PRAGMA quick_check').fetchone()
            if not result or result[0] != 'ok':
                raise CacheCorruptError("SQLite quick_check failed")
            cols = [row[1] for row in db.execute('PRAGMA table_info(replay_nonces)')]
            if cols != ['key_namespace', 'nonce_hex', 'state', 'package_fingerprint', 'created_utc']:
                raise CacheCorruptError("Unexpected replay_nonces schema")
            meta = db.execute('SELECT instance_uuid,receiver_id,schema_version,max_entries FROM replay_meta WHERE singleton=1').fetchone()
            if meta != (inst, receiver_id, SCHEMA_VERSION, max_entries):
                raise CacheCorruptError("Metadata does not match identity marker")
            if db.execute('SELECT COUNT(*) FROM replay_meta').fetchone()[0] != 1:
                raise CacheCorruptError("Unexpected metadata cardinality")
            total = db.execute('SELECT COUNT(*) FROM replay_nonces').fetchone()[0]
            if total > max_entries:
                raise CacheCorruptError("Cache exceeded configured capacity")
            if trusted_minimum_records is not None and total < trusted_minimum_records:
                raise CacheStateUnavailable("DB behind independently trusted record checkpoint: possible rollback")
            if db.execute("""SELECT COUNT(*) FROM replay_nonces
                WHERE length(key_namespace)!=64 OR key_namespace GLOB '*[^0-9a-f]*'
                OR length(nonce_hex)!=32 OR nonce_hex GLOB '*[^0-9a-f]*'
                OR length(package_fingerprint)!=64 OR package_fingerprint GLOB '*[^0-9a-f]*'
                OR state NOT IN ('pending','accepted')""").fetchone()[0]:
                raise CacheCorruptError("Malformed replay rows")
            return cls(db, path, receiver_id, inst, max_entries)
        except Exception as exc:
            try:
                db.close()
            except Exception:
                pass
            if isinstance(exc, ReplayCacheError):
                raise
            raise CacheCorruptError("Cannot safely verify or open replay database") from exc

    def _ensure_open(self):
        if self._closed:
            raise CacheStateUnavailable("Closed replay cache")
        # Catch obvious state deletion/replacement while process is running.
        if not self.database_path.is_file() or not _sidecar_path(self.database_path).is_file():
            raise CacheStateUnavailable("State missing: reception is blocked")

    def reserve(self, *, key_namespace: str, nonce: bytes, package_fingerprint: str):
        if nonce is None or package_fingerprint is None:
            raise InvalidReplayInput("nonce and fingerprint are required")
        _validate(self.receiver_id, key_namespace, nonce, package_fingerprint)
        self._ensure_open()
        try:
            self.connection.execute('BEGIN IMMEDIATE')
            entries = self.connection.execute('SELECT COUNT(*) FROM replay_nonces').fetchone()[0]
            if entries >= self.max_entries:
                # Intentionally no pruning: old authenticated packages must stay rejected.
                raise CacheCapacityExceeded("Replay state full: stop and rotate key under operator policy")
            self.connection.execute("""INSERT INTO replay_nonces
                (key_namespace,nonce_hex,state,package_fingerprint,created_utc)
                VALUES (?,?,'pending',?,?)""",
                (key_namespace, nonce.hex(), package_fingerprint, int(time.time())))
            self.connection.execute('COMMIT')
        except sqlite3.IntegrityError as exc:
            if self.connection.in_transaction:
                self.connection.execute('ROLLBACK')
            raise ReplayDetectedError("Authenticated nonce already seen") from exc
        except Exception:
            if self.connection.in_transaction:
                self.connection.execute('ROLLBACK')
            raise

    def mark_accepted(self, *, key_namespace: str, nonce: bytes):
        if nonce is None:
            raise InvalidReplayInput("nonce is required")
        _validate(self.receiver_id, key_namespace, nonce)
        self._ensure_open()
        try:
            self.connection.execute('BEGIN IMMEDIATE')
            cursor = self.connection.execute("""UPDATE replay_nonces SET state='accepted'
                WHERE key_namespace=? AND nonce_hex=? AND state='pending'""",
                (key_namespace, nonce.hex()))
            if cursor.rowcount != 1:
                raise ReservationStateError("No matching pending reservation")
            self.connection.execute('COMMIT')
        except Exception:
            if self.connection.in_transaction:
                self.connection.execute('ROLLBACK')
            raise

    def count_entries(self) -> int:
        self._ensure_open()
        return int(self.connection.execute('SELECT COUNT(*) FROM replay_nonces').fetchone()[0])

    def get_state(self, *, key_namespace: str, nonce: bytes) -> str | None:
        if nonce is None:
            raise InvalidReplayInput("nonce is required")
        _validate(self.receiver_id, key_namespace, nonce)
        self._ensure_open()
        row = self.connection.execute('SELECT state FROM replay_nonces WHERE key_namespace=? AND nonce_hex=?',
                                      (key_namespace, nonce.hex())).fetchone()
        return row[0] if row else None

    def close(self):
        if not self._closed:
            self.connection.close()
            self._closed = True

    def __enter__(self):
        self._ensure_open()
        return self

    def __exit__(self, *_):
        self.close()
