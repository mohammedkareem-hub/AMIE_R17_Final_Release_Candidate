#!/usr/bin/env python3
"""Research-only paired controls for AMIE-BakerChen reviewer R1-4 / R2-1.

Uses unmodified original CFP9-v2 Python primitives from the repository root.
Tests the *mechanism* of Baker spatial rearrangement against an in-block,
SHAKE256/Fisher-Yates control and a spatial identity control, holding each
block's original Chen-derived cumulative-feedback keystream fixed.

DO NOT deploy the experimental variants.  The in-memory per-variant monkey-
patch is limited to max_workers=1 and never serializes/enrolls variant packages.
The original cipher source and on-disk packages are never modified.
No DICOM or patient data is used. Keys and nonces are deliberately public test
fixtures; a reused nonce between one-pixel differential inputs is TEST ONLY.

Usage: python novelty-controls/run_novelty_controls.py --out /tmp/phase15
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from baker_map import (
    cumulative_feedback_diffusion,
    inverse_cumulative_feedback_diffusion,
    discrete_baker_permutation,
    inverse_discrete_baker_permutation,
    generate_baker_diffusion_key,
)
from permutation_utils import Shake256Stream

N = 64
MODES = ("baker", "fisher_yates", "identity")
PUBLIC_DOMAIN = b"AMIE-BakerChen|PHASE15-RESEARCH-ONLY|NOT-PRODUCTION|"


def inblock_fisher_indices(key: np.ndarray) -> np.ndarray:
    """Destination-to-source permutation using project's SHAKE stream and randbelow.

    Seed is domain-separated from feedback key bytes.  This is an explicitly
    *experimental* comparator, not a claim to reproduce a prior published cipher.
    """
    seed = hashlib.sha256(b"Phase15|InBlockFisherYatesControl-v1|" + key.tobytes(order="C")).digest()
    stream = Shake256Stream(seed)
    perm = np.arange(key.size, dtype=np.int64)
    for i in range(perm.size - 1, 0, -1):
        j = stream.randbelow(i + 1)
        perm[i], perm[j] = perm[j], perm[i]
    return perm


def spatial_forward(block: np.ndarray, key: np.ndarray, mode: str) -> np.ndarray:
    if mode == "baker":
        return discrete_baker_permutation(block)
    if mode == "identity":
        return block.copy()
    if mode == "fisher_yates":
        return block.reshape(-1)[inblock_fisher_indices(key)].reshape(block.shape)
    raise ValueError(mode)


def spatial_inverse(permuted: np.ndarray, key: np.ndarray, mode: str) -> np.ndarray:
    if mode == "baker":
        return inverse_discrete_baker_permutation(permuted)
    if mode == "identity":
        return permuted.copy()
    if mode == "fisher_yates":
        indices = inblock_fisher_indices(key)
        original = np.empty(indices.size, dtype=np.uint8)
        original[indices] = permuted.ravel()
        return original.reshape(permuted.shape)
    raise ValueError(mode)


def entropy(a: np.ndarray) -> float:
    hist = np.bincount(a.ravel(), minlength=256).astype(float)
    p = hist[hist > 0] / hist.sum()
    return float(-np.sum(p * np.log2(p)))


def correlation_abs(a: np.ndarray) -> float:
    x = np.asarray(a, dtype=np.float64)
    pairs = [(x[:, :-1].ravel(), x[:, 1:].ravel()),
             (x[:-1, :].ravel(), x[1:, :].ravel())]
    r = []
    for left, right in pairs:
        if np.std(left) == 0 or np.std(right) == 0:
            continue
        r.append(abs(float(np.corrcoef(left, right)[0, 1])))
    return float(np.mean(r)) if r else float("nan")


def difference_metrics(left: np.ndarray, right: np.ndarray) -> tuple[float, float]:
    """NPCR percentage, UACI percentage for uint8 data."""
    assert left.shape == right.shape
    a = left.astype(np.int16)
    b = right.astype(np.int16)
    npcr = 100.0 * np.count_nonzero(a != b) / a.size
    uaci = 100.0 * np.mean(np.abs(a - b)) / 255.0
    return float(npcr), float(uaci)


def images() -> list[tuple[str, np.ndarray]]:
    """Eight deterministic 128x128 *nonmedical* synthetic images."""
    x, y = np.meshgrid(np.arange(128), np.arange(128))
    rng = np.random.default_rng(20261104)
    arr = [
        ("zeros", np.zeros((128, 128), dtype=np.uint8)),
        ("constant_127", np.full((128, 128), 127, dtype=np.uint8)),
        ("horizontal_gradient", np.asarray((2*x) % 256, dtype=np.uint8)),
        ("checkerboard", np.where((x // 8 + y // 8) % 2, 255, 0).astype(np.uint8)),
        ("diagonal", np.asarray((x + y) % 256, dtype=np.uint8)),
        ("rings", np.asarray(((x-63)**2 + (y-63)**2)//21 % 256, dtype=np.uint8)),
        ("sparse", np.where(((x*37+y*19) % 97) == 0, 255, 0).astype(np.uint8)),
        ("pseudorandom", rng.integers(0, 256, (128, 128), dtype=np.uint8)),
    ]
    return arr


def theoretical_and_stage_checks(chaos_params: dict, block_salt: int) -> list[dict]:
    """Check exhaustive permutation-coverage theorem and actual block inverses."""
    key = generate_baker_diffusion_key(length=N*N, chaos_params=chaos_params, salt=block_salt)
    orig = np.random.default_rng(4215).integers(0, 256, (N, N), dtype=np.uint8)
    index_grid = np.arange(N*N, dtype=np.int64).reshape((N, N))
    records = []
    n = N*N
    expected = 100.0*(n+1)/(2*n)
    positions = [0,1,63,64,511,1023,2048,3072,4095]
    for mode in MODES:
        scrambled = spatial_forward(index_grid, key, mode)
        assert np.array_equal(np.sort(scrambled.ravel()), np.arange(n))
        src_to_dest = np.empty(n, dtype=np.int64)
        src_to_dest[scrambled.ravel()] = np.arange(n)
        coverage = (n - src_to_dest) / n * 100.0
        assert np.isclose(coverage.mean(), expected, rtol=0, atol=1e-12)
        candidate = spatial_forward(orig, key, mode)
        assert np.array_equal(np.bincount(orig.ravel(), minlength=256),
                              np.bincount(candidate.ravel(), minlength=256))
        assert abs(entropy(orig) - entropy(candidate)) < 1e-12
        ciphertext = cumulative_feedback_diffusion(candidate, key)
        recovered = spatial_inverse(inverse_cumulative_feedback_diffusion(ciphertext, key), key, mode)
        assert np.array_equal(orig, recovered), f"block round-trip failed in {mode}"
        # Verify actual feedback propagation for selected source locations.
        for pos in positions:
            modified = orig.copy()
            modified.ravel()[pos] ^= 1
            changed = cumulative_feedback_diffusion(spatial_forward(modified, key, mode), key)
            actual = float(100.0*np.count_nonzero(ciphertext != changed)/n)
            predicted = float(100.0*(n-src_to_dest[pos])/n)
            assert abs(actual - predicted) < 1e-12, (mode,pos,actual,predicted)
        records.append({
            "variant":mode,
            "n_block_bytes":n,
            "theorem_mean_coverage_percent":float(coverage.mean()),
            "mean_displacement_positions":float(np.mean(np.abs(src_to_dest - np.arange(n)))),
            "permuted_pixel_entropy":entropy(candidate),
            "original_pixel_entropy":entropy(orig),
            "tested_injected_positions":len(positions),
            "block_roundtrip_ok":True,
            "histogram_invariant":True,
        })
    return records


def run_paired_full_pipeline() -> list[dict]:
    """In-memory controlled variants using the user's frozen full pipeline.

    All modes share original key material, upstream global permutation, block
    feedback keystream, downstream plaintext-sensitive global diffusion, and
    HMAC. Only the spatial transform inside each block is changed.
    """
    import v4_secure_pipeline as pipeline
    from baker_map import apply_baker_substitution as original_baker_encrypt
    from baker_map import inverse_baker_substitution as original_baker_decrypt

    if pipeline.ALGORITHM_VERSION != "AMIE-BakerChen-CFP9-v2":
        raise RuntimeError("Expected frozen CFP9-v2 source")

    original_apply = pipeline.apply_baker_substitution
    original_inverse = pipeline.inverse_baker_substitution
    out = []
    try:
        for case_index, (case_name, plain) in enumerate(images()):
            master = hashlib.sha256(PUBLIC_DOMAIN + b"test-key|" + bytes([case_index])).digest()
            nonce = hashlib.sha256(PUBLIC_DOMAIN + b"test-nonce|" + bytes([case_index])).digest()[:16]
            modified = plain.copy()
            modified[case_index*13 % 128, (case_index*23+7) % 128] ^= 1
            first_global_perm = None
            full_original_cipher = None
            for mode in MODES:
                def experimental_apply(block, chaos_params, salt=0, partition=None):
                    if mode == "baker":
                        return original_baker_encrypt(block, chaos_params, salt, partition)
                    key = generate_baker_diffusion_key(length=block.size, chaos_params=chaos_params, salt=salt)
                    return cumulative_feedback_diffusion(spatial_forward(block, key, mode), key), key

                def experimental_inverse(encrypted_block, key, partition=None):
                    if mode == "baker":
                        return original_baker_decrypt(encrypted_block, key, partition)
                    permuted = inverse_cumulative_feedback_diffusion(encrypted_block, key)
                    return spatial_inverse(permuted, key, mode)

                pipeline.apply_baker_substitution = experimental_apply
                pipeline.inverse_baker_substitution = experimental_inverse
                enc = pipeline.encrypt_image_secure(plain, master, session_nonce=nonce,
                                                    block_size=N, max_workers=1)
                dec = pipeline.decrypt_image_secure(enc["encrypted_image"], enc["metadata"],
                                                    enc["authentication_tag"], master, max_workers=1)
                assert np.array_equal(dec["decrypted_image"], plain), (case_name,mode)
                alt = pipeline.encrypt_image_secure(modified, master, session_nonce=nonce,
                                                    block_size=N, max_workers=1)
                # Keeping master key + nonce fixed is *diagnostic only*; nonce
                # reuse is NOT recommended for actual operational encryption.
                diff_npcr, diff_uaci = difference_metrics(enc["encrypted_image"], alt["encrypted_image"])
                stage_npcr, stage_uaci = difference_metrics(enc["stage_block_diffusion"], alt["stage_block_diffusion"])
                if first_global_perm is None:
                    first_global_perm = enc["stage_global_permutation"].copy()
                else:
                    assert np.array_equal(first_global_perm, enc["stage_global_permutation"])
                if mode == "baker":
                    full_original_cipher = enc["encrypted_image"].copy()
                out.append({
                    "case": case_name,
                    "variant":mode,
                    "pixels":int(plain.size),
                    "base_plain_entropy":entropy(plain),
                    "block_stage_entropy_bits":entropy(enc["stage_block_diffusion"]),
                    "block_stage_abs_adjacent_correlation":correlation_abs(enc["stage_block_diffusion"]),
                    "final_entropy_bits":entropy(enc["encrypted_image"]),
                    "final_abs_adjacent_correlation":correlation_abs(enc["encrypted_image"]),
                    "block_stage_one_pixel_npcr_percent":stage_npcr,
                    "block_stage_one_pixel_uaci_percent":stage_uaci,
                    "final_one_pixel_npcr_percent":diff_npcr,
                    "final_one_pixel_uaci_percent":diff_uaci,
                    "roundtrip_ok":True,
                })
            assert full_original_cipher is not None
    finally:
        pipeline.apply_baker_substitution = original_apply
        pipeline.inverse_baker_substitution = original_inverse
    return out


def source_hashes():
    relevant = ["baker_map.py", "chaotic_generator.py", "permutation_utils.py",
                "key_schedule_v4.py", "v4_secure_pipeline.py"]
    return {s:hashlib.sha256((ROOT/s).read_bytes()).hexdigest() for s in relevant}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("novelty-controls/results-local"))
    args=ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    import v4_secure_pipeline as pipeline
    from key_schedule_v4 import derive_key_material
    master = hashlib.sha256(PUBLIC_DOMAIN + b"stage-check-key").digest()
    nonce = hashlib.sha256(PUBLIC_DOMAIN + b"stage-check-nonce").digest()[:16]
    derived = derive_key_material(master, nonce, pipeline.CHAOS_PARAMS)
    stage = theoretical_and_stage_checks(derived.chaos_params, derived.baker_salt)
    paired = run_paired_full_pipeline()
    assert len(stage)==3 and len(paired)==24
    for mode in MODES:
        assert sum(z["variant"]==mode and z["roundtrip_ok"] for z in paired)==8
    metrics = ["block_stage_entropy_bits","block_stage_abs_adjacent_correlation",
               "final_entropy_bits","final_abs_adjacent_correlation",
               "block_stage_one_pixel_npcr_percent","final_one_pixel_npcr_percent"]
    summary = {
        "experiment":"AMIE-Phase15-Synthetic-Controlled-Spatial-Ablation",
        "security_proof":False,
        "claim_hyperchaos_proved":False,
        "clinical_images_reprocessed":0,
        "samples":len(images()),
        "experimental_variants_only":list(MODES),
        "nonce_reuse_is_for_differential_diagnostics_only":True,
        "source_sha256":source_hashes(),
        "theorem": "For any bijective n-byte spatial permutation and uniformly sampled single source-byte change, expected feedback-suffix coverage equals (n+1)/(2n), independent of the permutation.",
        "theorem_exact_64x64_percent":100.0*((N*N)+1)/(2*N*N),
        "mean_over_8_synthetic_cases":{
            mode:{name:float(np.mean([r[name] for r in paired if r["variant"]==mode]))
                  for name in metrics}
            for mode in MODES
        },
        "stage_checks":stage,
        "limits":[
            "No proof of IND-CPA/IND-CCA or novelty",
            "No tests of 100 medical MR images here; existing dataset results remain unmodified",
            "The original code is not modified; variant monkeypatch is in-memory single-worker only",
            "A different plaintext changes the plaintext-sensitive global-diffusion key even under fixed nonce; final NPCR does not isolate a Baker benefit",
            "Test fixtures deliberately use public deterministic keys and one nonce for a differential pair; never deploy them",
            "All in-block Fisher-Yates candidates use the original Chen-dependent feedback keystream as input to a domain-separated SHA256/SHAKE permutation seed",
        ],
    }
    with (args.out/"phase15_paired_8synthetic_x3.csv").open("w", newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f, fieldnames=list(paired[0]))
        writer.writeheader(); writer.writerows(paired)
    with (args.out/"phase15_stage_exact_coverage_3controls.csv").open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=list(stage[0]))
        writer.writeheader(); writer.writerows(stage)
    (args.out/"phase15_summary.json").write_text(json.dumps(summary,indent=2,ensure_ascii=False)+"\n", encoding="utf-8")
    print("PASS: 3/3 exact 64x64 permutation/coverage identities")
    print("PASS: 9/9 sampled actual feedback effects for every mode")
    print("PASS: 24/24 original-pipeline synthetic encryption/decryption controls")
    print("WARNING: result is a functional/diagnostic comparison, not a cryptographic security proof")
    print("OUTPUT:", args.out)
    for mode,m in summary["mean_over_8_synthetic_cases"].items():
        print(mode, "final entropy",round(m["final_entropy_bits"],6),
              "final NPCR",round(m["final_one_pixel_npcr_percent"],4),
              "block NPCR",round(m["block_stage_one_pixel_npcr_percent"],4))


if __name__=="__main__":
    main()
