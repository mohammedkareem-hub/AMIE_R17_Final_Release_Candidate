"""Phase 6 TRUE Go receiver / actual authenticated .amiedcm integration tests.

Fixtures supplied externally; no DICOM or patient-identifying bytes in this kit.
Each test creates its own dedicated TemporaryDirectory + explicitly provisioned
SQLite database. The original dataset, production DBs, and frozen experiments
are never modified.
"""
import hashlib
import json
import os
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from hardened_replay_cache import (
    HardenedReplayCache, CacheCorruptError, CacheStateUnavailable,
    ReplayDetectedError, replay_key_namespace,
)
from real_go_receiver_with_replay import (
    ReceiverVerificationError, receive_with_persistent_replay_protection,
)


def flip_payload(data: bytes, label: bytes) -> bytes:
    marker = len(label).to_bytes(2, 'big') + label
    start = data.index(marker) + len(marker)
    n = int.from_bytes(data[start:start+8], 'big')
    payload_start = start + 8
    assert n > 0 and payload_start + n <= len(data)
    result = bytearray(data)
    result[payload_start] ^= 0x01
    return bytes(result)


class LiveGoReceiverReplayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        req = ['AMIE_PHASE6_GO_BIN', 'AMIE_PHASE6_PACKAGE',
               'AMIE_PHASE6_VECTOR']
        missing = [x for x in req if x not in os.environ]
        if missing:
            raise unittest.SkipTest('External real Go/DICOM fixtures missing: ' + ', '.join(missing))
        cls.go = Path(os.environ[req[0]]).resolve()
        cls.raw = Path(os.environ[req[1]]).read_bytes()
        cls.vector = json.loads(Path(os.environ[req[2]]).read_text(encoding='utf-8'))
        cls.original = (Path(os.environ['AMIE_PHASE6_ORIGINAL_DICOM']).read_bytes()
                        if os.environ.get('AMIE_PHASE6_ORIGINAL_DICOM') else None)
        cls.key = bytes.fromhex(cls.vector['validation_master_key_hex'])
        cls.expected = cls.vector['expected_hashes']['original_dicom_sha256']
        assert hashlib.sha256(cls.raw).hexdigest() == cls.vector['expected_hashes']['serialized_package_sha256']
        if cls.original is not None:
            assert hashlib.sha256(cls.original).hexdigest() == cls.expected

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='amie-phase6-test-')
        self.addCleanup(self.temp.cleanup)
        self.db = Path(self.temp.name) / 'cache.sqlite3'
        self.cache = HardenedReplayCache.provision(self.db, receiver_id='research-receiver-1', max_entries=50)
        self.addCleanup(self.cache.close)

    def receive(self, *, data=None, key=None, expected=None, cache=None):
        return receive_with_persistent_replay_protection(
            raw_package=self.raw if data is None else data,
            public_validation_key=self.key if key is None else key,
            go_receiver_binary=self.go,
            cache=self.cache if cache is None else cache,
            expected_original_sha256=self.expected if expected is None else expected,
        )

    def test_01_real_Go_full_recovery_is_byte_exact(self):
        rec = self.receive()
        if self.original is not None:
            self.assertEqual(rec.recovered_dicom, self.original)
        self.assertEqual(rec.recovered_sha256, self.expected)
        self.assertEqual(self.cache.count_entries(), 1)
        self.assertEqual(self.cache.get_state(key_namespace=replay_key_namespace(self.key),
                                              nonce=bytes.fromhex(rec.nonce_hex)), 'accepted')

    def test_02_same_authenticated_package_rejected(self):
        self.receive()
        with self.assertRaises(ReplayDetectedError):
            self.receive()
        self.assertEqual(self.cache.count_entries(), 1)

    def test_03_duplicate_rejected_after_reopen(self):
        self.receive()
        self.cache.close()
        with HardenedReplayCache.open(self.db, receiver_id='research-receiver-1') as reopened:
            with self.assertRaises(ReplayDetectedError):
                self.receive(cache=reopened)
            self.assertEqual(reopened.count_entries(), 1)

    def test_04_wrong_key_rejected_without_poisoning(self):
        wrong = bytes([self.key[0] ^ 1]) + self.key[1:]
        with self.assertRaises(ReceiverVerificationError):
            self.receive(key=wrong)
        self.assertEqual(self.cache.count_entries(), 0)
        self.assertEqual(self.receive().recovered_sha256, self.expected)

    def test_05_outer_HMAC_tamper_rejected_without_poisoning(self):
        with self.assertRaises(ReceiverVerificationError):
            self.receive(data=flip_payload(self.raw, b'dicom-outer-tag'))
        self.assertEqual(self.cache.count_entries(), 0)
        self.assertEqual(self.receive().recovered_sha256, self.expected)

    def test_06_ciphertext_tamper_rejected_without_poisoning(self):
        with self.assertRaises(ReceiverVerificationError):
            self.receive(data=flip_payload(self.raw, b'amie-ciphertext'))
        self.assertEqual(self.cache.count_entries(), 0)
        self.assertEqual(self.receive().recovered_sha256, self.expected)

    def test_07_DICOM_prefix_tamper_rejected_without_poisoning(self):
        with self.assertRaises(ReceiverVerificationError):
            self.receive(data=flip_payload(self.raw, b'dicom-prefix'))
        self.assertEqual(self.cache.count_entries(), 0)
        self.assertEqual(self.receive().recovered_sha256, self.expected)

    def test_08_noncanonical_package_rejected_without_poisoning(self):
        with self.assertRaises(ReceiverVerificationError):
            self.receive(data=self.raw + b'\x00')
        self.assertEqual(self.cache.count_entries(), 0)

    def test_09_after_reservation_decryption_failure_keeps_pending(self):
        with self.assertRaises(ReceiverVerificationError):
            self.receive(expected='0'*64)
        self.assertEqual(self.cache.count_entries(), 1)
        nonce = bytes.fromhex(self.vector['validation_session_nonce_hex'])
        self.assertEqual(self.cache.get_state(key_namespace=replay_key_namespace(self.key), nonce=nonce),
                         'pending')
        with self.assertRaises(ReplayDetectedError):
            self.receive()

    def test_10_missing_database_fails_closed_after_reopen(self):
        self.receive()
        self.cache.close()
        self.db.unlink()
        with self.assertRaises(CacheStateUnavailable):
            HardenedReplayCache.open(self.db, receiver_id='research-receiver-1')
        self.assertFalse(self.db.exists())

    def test_11_corrupt_database_fails_closed(self):
        self.cache.close()
        self.db.write_bytes(b'NOT SQLITE')
        with self.assertRaises((CacheCorruptError, CacheStateUnavailable)):
            HardenedReplayCache.open(self.db, receiver_id='research-receiver-1')

    def test_12_concurrent_full_receives_single_winner(self):
        self.cache.close()
        def task(_):
            with HardenedReplayCache.open(self.db, receiver_id='research-receiver-1') as cache:
                try:
                    result = self.receive(cache=cache)
                    return ('accepted', result.recovered_sha256)
                except ReplayDetectedError:
                    return ('replayed', None)
        with ThreadPoolExecutor(max_workers=8) as exe:
            observations = list(exe.map(task, range(8)))
        self.assertEqual(sum(x[0]=='accepted' for x in observations), 1)
        self.assertEqual(sum(x[0]=='replayed' for x in observations), 7)
        self.assertEqual([x[1] for x in observations if x[0]=='accepted'], [self.expected])
        with HardenedReplayCache.open(self.db, receiver_id='research-receiver-1') as cache:
            self.assertEqual(cache.count_entries(), 1)

    def test_13_nonce_meta_mutation_is_rejected_without_poisoning(self):
        orig = self.vector['validation_session_nonce_hex'].encode()
        idx = self.raw.index(orig)
        bad = bytearray(self.raw)
        bad[idx] = ord('0') if bad[idx] != ord('0') else ord('1')
        with self.assertRaises(ReceiverVerificationError):
            self.receive(data=bytes(bad))
        self.assertEqual(self.cache.count_entries(), 0)

    def test_14_recovery_reference_integrity_verified(self):
        res = self.receive()
        self.assertEqual(hashlib.sha256(res.recovered_dicom).hexdigest(), self.expected)
        if self.original is not None:
            self.assertEqual(len(res.recovered_dicom), len(self.original))
        self.assertTrue(res.authentication_verified_before_reservation)
        self.assertEqual(res.package_sha256, hashlib.sha256(self.raw).hexdigest())


if __name__ == '__main__':
    unittest.main(verbosity=2)
