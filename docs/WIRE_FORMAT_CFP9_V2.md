# AMIE-BakerChen CFP9-v2 — implementation-derived byte-format specification

**Status:** Private, source-derived interoperability documentation; not a stable public standard or third-party cryptographic review. The authoritative implementation is `native_dicom_package_serialization_CFP9.py`, `native_dicom_authenticated_package_CFP9.py`, `v4_secure_pipeline.py`, `key_schedule_v4.py`, `chaotic_generator.py` and independent Go `go-receiver/*.go`.

## Serialized package layout (in order)

```
ASCII magic + version: b"AMIE-NATIVE-DICOM-PACKAGE\\x00\\x01"
field_count:  uint16 big-endian, exactly 7
for each of seven ordered fields:
    label_length: uint16 big-endian, nonzero
    label:        label_length ASCII bytes, fixed name below
    value_length: uint64 big-endian
    value:        exactly value_length bytes
```

The seven labels, in strict order, are:

1. `dicom-prefix` — *unchanged* DICOM bytes before Pixel Data value, **authenticated but visible**.
2. `dicom-suffix` — *unchanged* DICOM bytes following Pixel Data value, **authenticated but visible**.
3. `dicom-context` — UTF-8 canonical JSON describing native pixel layout and byte offsets, **authenticated but visible**.
4. `amie-metadata` — UTF-8 canonical JSON containing cipher/padding/session fields, **authenticated but visible**.
5. `amie-ciphertext` — flattened C-contiguous `uint8` ciphertext array (shape from authenticated `padded_shape` metadata).
6. `amie-inner-tag` — 32 raw bytes of inner HMAC-SHA256.
7. `dicom-outer-tag` — 32 raw bytes of outer HMAC-SHA256.

Canonical JSON in Python is `json.dumps(object, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")`. Parsing enforces this representation; never substitute pretty-printed JSON. `uint16`/`uint64` field lengths are big-endian. The Python serializer configures a 2 GiB package cap; Go receiver defaults to 32 MiB (an implementation resource cap, *not* a DICOM standard limit). The actual supported interoperability range is the intersection of those bounds, and presently tested synthetic profile.

## Keys, domains and HMAC messages

- Master key: exactly 32 bytes (256 bits).
- Public session nonce: exactly 16 bytes (128 bits); fresh per encryption is an operational **requirement**, not enforced uniqueness by the core function.
- HKDF-SHA256: `PRK = HMAC_SHA256(salt=session_nonce, IKM=master_key)`; context-separated `HKDF-Expand` with `info = UTF8("MedicalImageEncryption-v1|" + context)`; one expansion block for 32-byte contexts.
- Inner authentication subkey context: `ciphertext-authentication-key`.
- Outer envelope authentication subkey context: `native-dicom-envelope-authentication-key`.
- Inner authentication message = `b"Ciphertext-v1|" || canonical_json(amie-metadata) || b"|" || amie-ciphertext`; tag = HMAC-SHA256 under the derived inner key.
- Outer authentication message = `b"AMIE-Native-DICOM-Envelope-v1|" || frame(dicom-prefix) || frame(dicom-suffix) || frame(dicom-context) || frame(amie-metadata) || frame(amie-ciphertext) || frame(amie-inner-tag)`; tag = HMAC-SHA256 under the derived outer key.
- The outer tag does **not** self-authenticate its own bytes. Both tags are checked using constant-time comparisons by the relevant receiver implementation. The outer envelope tag is verified before any inverse AMIE transform; the inner tag is also verified during authenticated decryption.
- Other key contexts used by the Python key schedule include `chaotic-initial-conditions`, `global-permutation-salt`, `parallel-baker-salt`, `global-diffusion-salt`, and `plaintext-context-key`. See source for numerical state derivation and full algorithms.

## Solver, input profiles and exactness boundaries

- `chaotic_generator.py` uses `CANONICAL_STATE_DECIMALS = 9` and `CANONICAL_STATE_PROFILE = "CFP9-v1"`. The solver runs in IEEE-754 double precision; the serialized retained Chen state profile is part of compatibility and must not change silently. Refer to generator and Go source for authoritative solver update steps and rounding behavior.
- AMIE processing accepts a two-dimensional `uint8` matrix; the native profile reinterprets original Pixel Data bytes as a reversible byte matrix rather than normalizing medical intensity values.
- Demonstrated native profile: supported uncompressed, single-frame grayscale configurations; the frozen 128×128 synthetic case uses Explicit VR Little Endian, 16 bits allocated / 12 bits stored, plus a nonempty DICOM suffix.
- The byte-exact original DICOM recovery claim relies on reconstructing the original Pixel Data and concatenating the preserved, authenticated prefix and suffix; it does not claim DICOM metadata confidentiality.
- Package `session_nonce_hex` is untrusted until the outer HMAC verifies; it is read only to derive the authentication key. Replay prevention is outside this framing and requires trustworthy persistent receiver state.

## Test vectors / independent reproduction

Run from repo root: `python verify_staging.py` for the frozen synthetic vector and Go recovery. For fresh Python-to-Go synthetic regeneration, follow `README.md` and `synthetic-public-vector/README.md` after installing `requirements-synthetic.txt`. The `reproduce_r17.py` script is not included in this snapshot; full 100-image medical-data reproduction is not available from this synthetic-only release candidate. The sample vector lives in `synthetic-public-vector/` and includes the deliberately public test master key/nonce plus expected package and recovered-DICOM SHA-256 values. **Never use these test values for real clinical data.**

**Caution:** This document is a concise source-derived framing summary, not a formal proof or exhaustive normative specification for every internal permutation/Baker/Chen sub-step. The Python and Go sources are required to reproduce the algorithm end to end.
