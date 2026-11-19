\# AMIE-BakerChen - Authenticated Medical Image Encryption



\## Reproducibility Release Candidate



\*\*Status: PRIVATE REVIEW - NOT YET APPROVED FOR PUBLIC RELEASE\*\*



This repository contains the source-code release candidate for AMIE-BakerChen, an authenticated medical-image encryption framework integrating a fractional-order Chen system, key-dependent permutation, Baker-map processing, bidirectional diffusion, and HMAC-SHA256 authentication.



The implementation includes Python encryption and decryption components, native-DICOM processing, an independently implemented Go receiver, synthetic interoperability tests, and controlled baseline reimplementations.



The release candidate does not contain original clinical DICOM images or private per-image research records.



Public release remains subject to source-licensing, data-privacy, institutional, and final publication approvals.



\## 1. Repository Contents



The release candidate includes:



\- Core Python encryption and authenticated-decryption implementation.

\- Fractional-order Chen chaotic generator.

\- Key-dependent Fisher-Yates permutation.

\- Baker-map block processing and diffusion.

\- HKDF-SHA256 key derivation and HMAC-SHA256 authentication.

\- Native-DICOM serialization and authenticated package handling.

\- Independent Go receiver implementation.

\- Synthetic interoperability test vectors.

\- Controlled Kamal, Afzal, and Liu baseline reimplementations.

\- Security and replay-protection test utilities.

\- Source code for the 100-image security evaluation.

\- A dedicated runner for private-data evaluation.



The controlled baseline implementations are adapted research reconstructions and are not claimed to be byte-identical reproductions of the original authors' software.



\## 2. Software Environment



The 100-image local evaluation performed on 19 November 2026 used:



| Component | Version |

|---|---|

| Python | 3.14.6 |

| NumPy | 2.5.0 |

| pydicom | 3.0.2 |

| Cryptography | 50.0.0 |



The synthetic GitHub Actions workflow is configured to use Python 3.13 and Go 1.23.



These configurations represent the documented evaluation environments, not an exhaustive compatibility guarantee.



\## 3. Synthetic Reproducibility



The repository includes a deliberately fabricated 128x128 DICOM test vector.



The synthetic test material is intended for interoperability verification and contains no original clinical images.



Synthetic test keys and session nonces must never be used for real medical information.



The repository provides:



\- Frozen synthetic vector verification.

\- Fresh synthetic package generation.

\- Python-to-Go interoperability tests.

\- Authentication verification.

\- Byte-exact synthetic recovery tests.



The existing `verify\_staging.py` requires the repository's finalized `SHA256SUMS.txt` manifest.



That manifest must be regenerated and verified after the release files are finalized.



\## 4. Medical Dataset and Selection



The research evaluation used 100 selected MR DICOM images.



The original local selection was reconstructed from a source subset containing 1,985 MR DICOM files using deterministic selection with seed 2026.



A read-only local verification confirmed:



| Verification | Result |

|---|---|

| Local source objects | 1985/1985 |

| Deterministic selection | 100/100 |

| Source-to-manifest SHA-256 | 100/100 |

| Selected-copy SHA-256 | 100/100 |



These results establish consistency with the preserved local source subset.



The complete public acquisition procedure required to reconstruct that exact 1,985-file subset from the external dataset is not yet finalized.



The clinical source images and restricted provenance records are not included in this repository.



Researchers must obtain the underlying dataset through lawful, authorized access and comply with its applicable terms.



\## 5. Reproducing the 100-Image Security Evaluation



The release candidate contains:



`src/final\_security\_evaluation\_100\_images.py`



and:



`run\_r17\_100\_image\_evaluation.py`



The evaluation requires an externally supplied directory containing 100 selected DICOM files and the corresponding `selected\_images.csv` manifest.



\### Step 1 - Verify the dataset



```powershell

py .\\run\_r17\_100\_image\_evaluation.py --selected-dir "<AUTHORIZED\_SELECTED\_DATA\_DIRECTORY>"

```



This performs a read-only preflight verification.



\### Step 2 - Run the security evaluation



```powershell

py -u .\\run\_r17\_100\_image\_evaluation.py --selected-dir "<AUTHORIZED\_SELECTED\_DATA\_DIRECTORY>" --output-dir "<PRIVATE\_OUTPUT\_DIRECTORY>" --run

```



The output directory must be outside the release repository and must not already exist.



The evaluation produces per-image security metrics, aggregate summaries, and experimental authentication results.



The original research data, experimental keys, and restricted result files must not be uploaded without the necessary publication clearance.



\## 6. Local 100-Image Evaluation Results



A local evaluation was completed on 19 November 2026 using the release-candidate runner and the original Python implementation.



All 100 selected MR images passed manifest verification.



\### Aggregate Security Metrics



| Metric | Mean |

|---|---:|

| Entropy | 7.999147 |

| NPCR (%) | 99.608718 |

| UACI (%) | 33.474644 |

| Key-NPCR (%) | 99.607977 |

| Ciphertext Bit Hamming Distance (%) | 50.001198 |



\### Authentication and Recovery



| Evaluation | Result |

|---|---|

| Correct-key exact recovery | 100/100 |

| Wrong-key authentication rejection | 100/100 |

| Ciphertext tampering rejection | 100/100 |

| Authentication-tag tampering rejection | 100/100 |

| Protected plaintext-context tampering rejection | 100/100 |

| Session-nonce metadata tampering rejection | 100/100 |



These results were obtained by rerunning the original Python implementation.



They support experimental repeatability within the tested configuration.



They do not constitute an independent third-party implementation, a formal cryptographic security proof, or a guarantee of equivalent results under every environment.



The original manuscript results remain separate from these additional evaluation runs.



\## 7. Native-DICOM Support



The native-DICOM processing layer preserves original DICOM byte structure within the supported profiles.



Pixel Data is encrypted, while preserved DICOM prefix and suffix bytes are authenticated but remain visible.



Validated native-DICOM interoperability is limited to supported uncompressed, single-frame monochrome configurations.



The byte-format specification is documented in:



`docs/WIRE\_FORMAT\_CFP9\_V2.md`



This implementation-derived description is not presented as a formally standardized cryptographic protocol.



\## 8. Security Limitations



AMIE-BakerChen is a bespoke authenticated image-encryption construction.



The reported statistical results do not establish formal IND-CPA or IND-CCA security.



The encryption system requires appropriate management of session nonces and master keys.



Replay protection depends on trustworthy receiver-side persistent state and does not independently guarantee protection against arbitrary state rollback.



The measured properties of Chen and Baker components do not establish superiority over standard authenticated encryption algorithms.



The current implementation should not be represented as a certified production-ready healthcare security system.



\## 9. Reproducibility Scope



The available evidence includes:



\- Source-code implementations.

\- Synthetic interoperability vectors.

\- Python and Go interoperability testing.

\- Controlled baseline implementation tests.

\- A documented local 100-image security rerun.

\- Read-only source-dataset selection verification.



The original clinical dataset and restricted source identifiers are not distributed.



Exact third-party reconstruction of the original 100-image dataset requires an approved data-access and source-selection protocol.



\## 10. Publication and Licensing



The repository includes an MIT license for the source code covered by the stated rights-holder authorization.



Third-party dependencies, datasets, and externally sourced assets remain subject to their own licenses and applicable obligations.



The final public release requires completion of privacy, rights, attribution, and publication reviews.



A public release tag, verified repository URL, and optional archival DOI will be documented after an authorized release has been created.



\*\*This release candidate must remain private until all required publication approvals have been completed.\*\*

