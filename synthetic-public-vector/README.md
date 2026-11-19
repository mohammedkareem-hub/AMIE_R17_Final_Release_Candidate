# Entirely synthetic CFP9-v2 native-DICOM interoperability vector (PRE-RELEASE)

**This is a fabricated 128×128 image, not a patient scan.** It uses a deterministic
integer pattern and fabricated DICOM UIDs. The stored pixels have 16 bits
allocated and 12 bits stored; the one-frame transfer syntax is uncompressed
Explicit VR Little Endian. A synthetic private element follows Pixel Data so
that the **nonempty DICOM suffix** is also authenticated by the outer tag.

The published **research-only** 256-bit master key and deterministic 128-bit
nonce are included in the JSON vector. They must **never** be used on real data.
In this mode, the DICOM header is authenticated, **not encrypted**.

## Frozen files

- `synthetic_128x128_original.dcm`: fabricated DICOM original.
- `synthetic_128x128.amiedcm`: Python-generated encrypted & dual-authenticated binary package.
- `synthetic_128x128_test_vector.json`: test key, nonce, SHA-256, HMAC expected values and byte framing.
- `generate_public_synthetic_vector.py`: regenerates the vector using the user's unmodified CFP9-v2 Python implementation.
- `verify_public_synthetic_vector.py`: checks framing, canonical JSON and SHA-256 using only Python standard library (NOT cryptographic decryption).

## Check the frozen vector

At repository root:

```bash
python synthetic-public-vector/verify_public_synthetic_vector.py --directory synthetic-public-vector
cd go-receiver
go test ./...
go run . -package ../synthetic-public-vector/synthetic_128x128.amiedcm -vector ../synthetic-public-vector/synthetic_128x128_test_vector.json
```

**Go must report** `inner_hmac_verified: true`, `outer_hmac_verified: true`,
`native_dicom_decrypted: true`, and
`recovered_dicom_sha256: ee548d56fcea1114f4443b5307d22bc718b9e17caa5691aaac141bfd3ec59ca1`.

## Recreate vector via Python CFP9-v2

Run from the root of the *frozen* Python source repository (Python modules at root):

```bash
python -m pip install numpy==2.3.5 pydicom==3.0.2 pillow matplotlib
python synthetic-public-vector/generate_public_synthetic_vector.py --source-root . --out synthetic-public-vector/generated_ci
```

The generator uses the **existing Python encrypt/decrypt and package-serialization
functions**; it constructs synthetic-only DICOM bytes independently instead
of reading or decoding patient files. Because the execution environment used
for this initial local audit did not include pydicom, the *only* local run
used a clearly declared `AMIE_SYNTHETIC_IMPORT_SHIM=1` to satisfy an unused
import dependency, **not** to replace encryption, decryption, or packet
serialization. The GitHub workflow installs and checks with real `pydicom`.

## Status/limits

Local Linux: original Python encryption and decryption **PASS**, independent
Go authentication and native DICOM recovery **PASS**, binary equality **PASS**.
In the new private repository, a user-provided screenshot of GitHub Actions run 442235320682 on 18 November 2026 shows frozen-vector passes on Ubuntu, Windows and macOS Intel, plus a fresh-vector pass on Ubuntu (4/4 jobs). These checks are limited to synthetic input and do not constitute an independent 100-image clinical rerun.
Reproduction on real medical DICOM still depends on the real dataset and
privacy/licensing permissions. The current repository is not yet a public
reproducibility release and this folder alone does not satisfy all of R1-7.
