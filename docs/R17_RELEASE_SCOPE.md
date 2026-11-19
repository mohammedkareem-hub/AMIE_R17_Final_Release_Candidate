# AMIE-BakerChen - Release Candidate Scope

Status: PRIVATE REVIEW - PUBLIC RELEASE NOT YET AUTHORIZED

## Included

- Python encryption and authenticated-decryption source.
- Independent Go receiver.
- Fabricated synthetic interoperability vectors.
- Controlled baseline reimplementations.
- Security evaluation source for the 100-image MR study.
- Private-data evaluation runner.
- Implementation-derived protocol documentation.

## Evidence recorded on 19 November 2026

The project operator reported a local 100-image MR evaluation
using the release-candidate runner and original implementation.

The recorded results include:

- Correct-key exact recovery: 100/100.
- Wrong-key rejection: 100/100.
- Ciphertext tampering rejection: 100/100.
- Authentication-tag tampering rejection: 100/100.
- Plaintext-context metadata tampering rejection: 100/100.
- Session-nonce metadata tampering rejection: 100/100.

These are operator-recorded experimental results.
They are not a formal cryptographic security proof.

The original patient-derived DICOM files and restricted
per-image research records are not distributed.

## Reproduction limitations

Exact recreation of the original MR selection requires
authorized access to the original source data and selection
provenance.

Synthetic interoperability tests do not independently
reproduce the clinical 100-image experiment.

The native-DICOM profile supports tested uncompressed,
single-frame monochrome configurations.

DICOM non-pixel metadata remains authenticated but visible.

The construction is not claimed to provide formally proven
IND-CPA or IND-CCA security.

## Publication gates

Before public release:

1. Complete source-code rights and third-party license review.
2. Complete privacy and dataset-provenance review.
3. Run the final GitHub Actions workflow on this exact snapshot.
4. Verify release contents and SHA-256 manifest.
5. Finalize author and citation metadata.
6. Obtain explicit approval from the responsible rights holders.

Only after approval should a public repository, release tag,
or archival DOI be created.
