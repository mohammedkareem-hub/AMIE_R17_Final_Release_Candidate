from __future__ import annotations

import hashlib
import hmac
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from numpy.typing import NDArray
from PIL import Image
import matplotlib.pyplot as plt

from chaotic_generator import (
    CHAOS_PARAMS,
    generate_chaotic_byte_sequence,
)
from dicom_loader import load_dicom_as_uint8
from permutation_utils import (
    shuffle_image,
    generate_permutation_indices,
    inverse_shuffle_image,
)
from block_utils import (
    pad_image_to_block_size,
    split_into_blocks,
    merge_blocks,
    remove_padding,
)
from baker_map import (
    apply_baker_substitution,
    inverse_baker_substitution,
    generate_baker_diffusion_key,
)
from metrics import (
    calculate_entropy,
    calculate_chi_square_uniformity,
    calculate_correlations,
    calculate_npcr_uaci,
    create_one_pixel_modified_image,
    calculate_decryption_quality,
    get_image_size_info,
    calculate_throughput,
    calculate_cpb,
    save_metrics_to_csv,
)
from key_schedule_v4 import (
    DerivedKeyMaterial,
    derive_key_material,
    generate_master_key,
    generate_session_nonce,
    master_key_to_hex,
    master_key_from_hex,
)


# =========================================================
# PATHS AND SETTINGS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "selected"

OUTPUT_ORIGINAL = PROJECT_ROOT / "output" / "original"
OUTPUT_ENCRYPTED = PROJECT_ROOT / "output" / "encrypted"
OUTPUT_DECRYPTED = PROJECT_ROOT / "output" / "decrypted"
OUTPUT_PACKAGES = PROJECT_ROOT / "output" / "packages"
OUTPUT_METRICS = PROJECT_ROOT / "output" / "metrics"
OUTPUT_STAGE_ANALYSIS = PROJECT_ROOT / "output" / "stage_analysis"

for directory in (
    OUTPUT_ORIGINAL,
    OUTPUT_ENCRYPTED,
    OUTPUT_DECRYPTED,
    OUTPUT_PACKAGES,
    OUTPUT_METRICS,
    OUTPUT_STAGE_ANALYSIS,
):
    directory.mkdir(parents=True, exist_ok=True)

ALGORITHM_VERSION = "AMIE-BakerChen-CFP9-v2"
BLOCK_SIZE = 64
CPU_FREQ_HZ = 2.40e9
MAX_WORKERS: int | None = None
PERSISTENT_WORKERS = 4
NUM_IMAGES = 100
GLOBAL_DIFFUSION_SEED_BYTES = 64
ENABLE_STAGE_ANALYSIS = True
STAGE_ANALYSIS_IMAGE_NAME = "image_001"


# =========================================================
# VALIDATION
# =========================================================

def _validate_uint8_image(
    image: NDArray,
    name: str = "image",
) -> NDArray[np.uint8]:
    array = np.asarray(image)

    if array.ndim != 2:
        raise ValueError(f"{name} must be a 2D grayscale image.")

    if array.size == 0:
        raise ValueError(f"{name} must not be empty.")

    if array.dtype != np.uint8:
        raise ValueError(
            f"{name} must use uint8. Received dtype={array.dtype}."
        )

    return array


def _validate_nonce(session_nonce: bytes) -> bytes:
    if not isinstance(session_nonce, bytes):
        raise TypeError("session_nonce must be bytes.")

    if len(session_nonce) != 16:
        raise ValueError("session_nonce must contain exactly 16 bytes.")

    return session_nonce


# =========================================================
# PARALLEL BLOCK WORKERS
# =========================================================

def _encrypt_block_worker(
    args: tuple[
        NDArray[np.uint8],
        dict[str, Any],
        int,
        Sequence[int] | None,
    ],
) -> NDArray[np.uint8]:
    block, chaos_params, block_salt, partition = args

    encrypted_block, _ = apply_baker_substitution(
        block=block,
        chaos_params=chaos_params,
        salt=block_salt,
        partition=partition,
    )

    return encrypted_block


def _decrypt_block_worker(
    args: tuple[
        NDArray[np.uint8],
        dict[str, Any],
        int,
        Sequence[int] | None,
    ],
) -> NDArray[np.uint8]:
    encrypted_block, chaos_params, block_salt, partition = args

    key = generate_baker_diffusion_key(
        length=encrypted_block.size,
        chaos_params=chaos_params,
        salt=block_salt,
    )

    return inverse_baker_substitution(
        encrypted_block=encrypted_block,
        key=key,
        partition=partition,
    )


def _parallel_map(
    function,
    arguments,
    max_workers: int | None,
    executor: ProcessPoolExecutor | None = None,
):
    arguments = list(arguments)

    if not arguments:
        return []

    # True serial execution. No process pool is created.
    if max_workers == 1:
        return [
            function(argument)
            for argument in arguments
        ]

    if len(arguments) == 1:
        return [function(arguments[0])]

    # Preferred path: reuse an already-running persistent pool.
    if executor is not None:
        return list(
            executor.map(
                function,
                arguments,
            )
        )

    # Backward-compatible path: create a temporary process pool.
    # This is correct but slower when called repeatedly.
    with ProcessPoolExecutor(
        max_workers=max_workers
    ) as local_executor:
        return list(
            local_executor.map(
                function,
                arguments,
            )
        )


# =========================================================
# PLAINTEXT-SENSITIVE PROTECTED CONTEXT + GLOBAL KEY STREAMS
# =========================================================

def _normalize_salt(salt: int | str | bytes) -> bytes:
    if isinstance(salt, bytes):
        return salt

    return str(salt).encode("utf-8")


def _expand_sha256_stream(
    seed_material: bytes,
    length: int,
    domain: bytes,
) -> NDArray[np.uint8]:
    if length <= 0:
        raise ValueError("length must be greater than zero.")

    output = bytearray()
    counter = 0

    while len(output) < length:
        output.extend(
            hashlib.sha256(
                domain
                + counter.to_bytes(8, "big", signed=False)
                + seed_material
            ).digest()
        )
        counter += 1

    return np.frombuffer(
        bytes(output[:length]),
        dtype=np.uint8,
    ).copy()


def calculate_plaintext_digest(image: NDArray) -> bytes:
    image_u8 = _validate_uint8_image(image)
    height, width = image_u8.shape

    return hashlib.sha256(
        b"MedicalImagePlaintext-v1|"
        + height.to_bytes(8, "big", signed=False)
        + width.to_bytes(8, "big", signed=False)
        + image_u8.tobytes(order="C")
    ).digest()


def derive_protected_plaintext_context(
    plaintext_digest: bytes,
    plaintext_context_key: bytes,
) -> bytes:
    """
    Convert the raw plaintext SHA-256 digest into a keyed 256-bit
    context value. Only this protected context is transmitted in
    authenticated public metadata; the raw plaintext digest is never
    stored in the encryption package.

    The context remains plaintext-sensitive while preventing the
    metadata from exposing a direct public SHA-256 fingerprint of the
    padded plaintext image.
    """
    if not isinstance(plaintext_digest, bytes) or len(plaintext_digest) != 32:
        raise ValueError(
            "plaintext_digest must be a 32-byte SHA-256 digest."
        )

    if (
        not isinstance(plaintext_context_key, bytes)
        or len(plaintext_context_key) != 32
    ):
        raise ValueError(
            "plaintext_context_key must be a 32-byte secret key."
        )

    return hmac.new(
        plaintext_context_key,
        b"PlaintextContext-v1|" + plaintext_digest,
        hashlib.sha256,
    ).digest()


def generate_global_diffusion_keys(
    image_shape: tuple[int, int],
    chaos_params: dict[str, Any],
    salt: int | str | bytes,
    plaintext_context: bytes,
) -> tuple[NDArray[np.uint8], NDArray[np.uint8]]:
    height, width = image_shape

    if height <= 0 or width <= 0:
        raise ValueError("Image dimensions must be positive.")

    if not isinstance(plaintext_context, bytes) or len(plaintext_context) != 32:
        raise ValueError(
            "plaintext_context must be a 32-byte protected context."
        )

    length = height * width
    salt_bytes = _normalize_salt(salt)

    chaotic_seed = generate_chaotic_byte_sequence(
        length=GLOBAL_DIFFUSION_SEED_BYTES,
        chaos_params=chaos_params,
        salt=(
            b"global-diffusion-seed|"
            + salt_bytes
            + b"|"
            + plaintext_context
        ),
        discard=100,
    )

    context = (
        chaotic_seed.tobytes()
        + height.to_bytes(8, "big", signed=False)
        + width.to_bytes(8, "big", signed=False)
        + b"|"
        + salt_bytes
        + b"|"
        + plaintext_context
    )

    forward_key = _expand_sha256_stream(
        context,
        length,
        b"GlobalForwardDiffusion-v2|",
    )

    backward_key = _expand_sha256_stream(
        context,
        length,
        b"GlobalBackwardDiffusion-v2|",
    )

    return forward_key, backward_key


# =========================================================
# GLOBAL BIDIRECTIONAL DIFFUSION
# =========================================================

def global_bidirectional_diffusion(
    image: NDArray,
    forward_key: NDArray,
    backward_key: NDArray,
) -> NDArray[np.uint8]:
    image_u8 = _validate_uint8_image(image)

    plaintext = image_u8.reshape(-1).astype(np.int64)
    key_forward = np.asarray(
        forward_key,
        dtype=np.uint8,
    ).reshape(-1).astype(np.int64)
    key_backward = np.asarray(
        backward_key,
        dtype=np.uint8,
    ).reshape(-1).astype(np.int64)

    if (
        key_forward.size != plaintext.size
        or key_backward.size != plaintext.size
    ):
        raise ValueError(
            "Global key lengths must equal the number of image pixels."
        )

    forward = np.empty_like(plaintext)

    forward[0] = (
        plaintext[0] + key_forward[0]
    ) % 256

    for index in range(1, plaintext.size):
        forward[index] = (
            plaintext[index]
            + key_forward[index]
            + forward[index - 1]
        ) % 256

    ciphertext = np.empty_like(forward)
    last = forward.size - 1

    ciphertext[last] = (
        forward[last] + key_backward[last]
    ) % 256

    for index in range(last - 1, -1, -1):
        ciphertext[index] = (
            forward[index]
            + key_backward[index]
            + ciphertext[index + 1]
        ) % 256

    return ciphertext.astype(np.uint8).reshape(image_u8.shape)


def inverse_global_bidirectional_diffusion(
    encrypted_image: NDArray,
    forward_key: NDArray,
    backward_key: NDArray,
) -> NDArray[np.uint8]:
    encrypted_u8 = _validate_uint8_image(
        encrypted_image,
        "encrypted_image",
    )

    ciphertext = encrypted_u8.reshape(-1).astype(np.int64)
    key_forward = np.asarray(
        forward_key,
        dtype=np.uint8,
    ).reshape(-1).astype(np.int64)
    key_backward = np.asarray(
        backward_key,
        dtype=np.uint8,
    ).reshape(-1).astype(np.int64)

    if (
        key_forward.size != ciphertext.size
        or key_backward.size != ciphertext.size
    ):
        raise ValueError(
            "Global key lengths must equal the number of image pixels."
        )

    forward = np.empty_like(ciphertext)
    last = ciphertext.size - 1

    forward[last] = (
        ciphertext[last] - key_backward[last]
    ) % 256

    for index in range(last - 1, -1, -1):
        forward[index] = (
            ciphertext[index]
            - key_backward[index]
            - ciphertext[index + 1]
        ) % 256

    plaintext = np.empty_like(forward)

    plaintext[0] = (
        forward[0] - key_forward[0]
    ) % 256

    for index in range(1, forward.size):
        plaintext[index] = (
            forward[index]
            - key_forward[index]
            - forward[index - 1]
        ) % 256

    return plaintext.astype(np.uint8).reshape(encrypted_u8.shape)


# =========================================================
# AUTHENTICATED METADATA
# =========================================================

def _canonical_metadata_bytes(metadata: dict[str, Any]) -> bytes:
    """
    Serialize metadata deterministically for HMAC calculation.
    """
    return json.dumps(
        metadata,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")


def compute_authentication_tag(
    encrypted_image: NDArray,
    metadata: dict[str, Any],
    authentication_key: bytes,
) -> bytes:
    encrypted_u8 = _validate_uint8_image(
        encrypted_image,
        "encrypted_image",
    )

    if not isinstance(authentication_key, bytes):
        raise TypeError("authentication_key must be bytes.")

    return hmac.new(
        authentication_key,
        b"Ciphertext-v1|"
        + _canonical_metadata_bytes(metadata)
        + b"|"
        + encrypted_u8.tobytes(order="C"),
        hashlib.sha256,
    ).digest()


def verify_authentication_tag(
    encrypted_image: NDArray,
    metadata: dict[str, Any],
    authentication_key: bytes,
    supplied_tag: bytes,
) -> None:
    expected = compute_authentication_tag(
        encrypted_image,
        metadata,
        authentication_key,
    )

    if not hmac.compare_digest(expected, supplied_tag):
        raise ValueError(
            "Authentication failed: wrong key or modified "
            "ciphertext/metadata."
        )


# =========================================================
# SECURE ENCRYPTION
# =========================================================

def encrypt_image_secure(
    image: NDArray,
    master_key: bytes,
    *,
    session_nonce: bytes | None = None,
    block_size: int = BLOCK_SIZE,
    baker_partition: Sequence[int] | None = None,
    max_workers: int | None = MAX_WORKERS,
    executor: ProcessPoolExecutor | None = None,
) -> dict[str, Any]:
    """
    Encrypt a uint8 grayscale image using one 256-bit master key.

    The returned package contains ciphertext, public metadata,
    authentication tag, and timing information. It does not contain
    the master key or expanded secret key streams.
    """
    original = _validate_uint8_image(image)

    if session_nonce is None:
        session_nonce = generate_session_nonce()

    session_nonce = _validate_nonce(session_nonce)

    derived: DerivedKeyMaterial = derive_key_material(
        master_key,
        session_nonce,
        CHAOS_PARAMS,
    )

    start = time.perf_counter()

    padded, original_shape = pad_image_to_block_size(
        original,
        block_size,
    )

    plaintext_digest = calculate_plaintext_digest(padded)
    plaintext_context = derive_protected_plaintext_context(
        plaintext_digest,
        derived.plaintext_context_key,
    )

    permuted, _ = shuffle_image(
        padded,
        derived.chaos_params,
        salt=derived.permutation_salt,
    )

    blocks, positions = split_into_blocks(
        permuted,
        block_size,
    )

    encryption_arguments = [
        (
            block,
            derived.chaos_params,
            derived.baker_salt + index,
            baker_partition,
        )
        for index, block in enumerate(blocks)
    ]

    encrypted_blocks = _parallel_map(
        _encrypt_block_worker,
        encryption_arguments,
        max_workers,
        executor,
    )

    block_encrypted = merge_blocks(
        encrypted_blocks,
        positions,
        padded.shape,
    )

    forward_key, backward_key = generate_global_diffusion_keys(
        block_encrypted.shape,
        derived.chaos_params,
        derived.global_diffusion_salt,
        plaintext_context,
    )

    encrypted = global_bidirectional_diffusion(
        block_encrypted,
        forward_key,
        backward_key,
    )

    metadata: dict[str, Any] = {
        "algorithm_version": ALGORITHM_VERSION,
        "session_nonce_hex": session_nonce.hex(),
        "plaintext_context_hex": plaintext_context.hex(),
        "original_shape": [
            int(original_shape[0]),
            int(original_shape[1]),
        ],
        "padded_shape": [
            int(padded.shape[0]),
            int(padded.shape[1]),
        ],
        "block_size": int(block_size),
        "baker_partition": (
            list(map(int, baker_partition))
            if baker_partition is not None
            else None
        ),
        "dtype": "uint8",
    }

    tag = compute_authentication_tag(
        encrypted,
        metadata,
        derived.authentication_key,
    )

    elapsed = time.perf_counter() - start

    return {
        "encrypted_image": encrypted,
        "stage_original": original.copy(),
        "stage_padded": padded.copy(),
        "stage_global_permutation": permuted.copy(),
        "stage_block_diffusion": block_encrypted.copy(),
        "stage_global_diffusion": encrypted.copy(),
        "metadata": metadata,
        "authentication_tag": tag,
        "encryption_time_sec": float(elapsed),
    }


# =========================================================
# SECURE DECRYPTION
# =========================================================

def decrypt_image_secure(
    encrypted_image: NDArray,
    metadata: dict[str, Any],
    authentication_tag: bytes,
    master_key: bytes,
    *,
    max_workers: int | None = MAX_WORKERS,
    executor: ProcessPoolExecutor | None = None,
) -> dict[str, Any]:
    """
    Authenticate and decrypt using ciphertext + metadata + master key.

    No stored permutation vector, block key list, or global key streams
    are required. All secret values are regenerated from the master key
    and public session nonce.
    """
    encrypted = _validate_uint8_image(
        encrypted_image,
        "encrypted_image",
    )

    if metadata.get("algorithm_version") != ALGORITHM_VERSION:
        raise ValueError("Unsupported algorithm version.")

    session_nonce = bytes.fromhex(
        str(metadata["session_nonce_hex"])
    )
    session_nonce = _validate_nonce(session_nonce)

    derived = derive_key_material(
        master_key,
        session_nonce,
        CHAOS_PARAMS,
    )

    verify_authentication_tag(
        encrypted,
        metadata,
        derived.authentication_key,
        authentication_tag,
    )

    start = time.perf_counter()

    plaintext_context = bytes.fromhex(
        str(metadata["plaintext_context_hex"])
    )

    padded_shape = tuple(
        int(value)
        for value in metadata["padded_shape"]
    )
    original_shape = tuple(
        int(value)
        for value in metadata["original_shape"]
    )
    block_size = int(metadata["block_size"])

    partition_value = metadata.get("baker_partition")
    partition = (
        tuple(int(value) for value in partition_value)
        if partition_value is not None
        else None
    )

    if encrypted.shape != padded_shape:
        raise ValueError(
            "Encrypted image dimensions do not match metadata."
        )

    forward_key, backward_key = generate_global_diffusion_keys(
        encrypted.shape,
        derived.chaos_params,
        derived.global_diffusion_salt,
        plaintext_context,
    )

    recovered_block_encrypted = (
        inverse_global_bidirectional_diffusion(
            encrypted,
            forward_key,
            backward_key,
        )
    )

    encrypted_blocks, positions = split_into_blocks(
        recovered_block_encrypted,
        block_size,
    )

    decryption_arguments = [
        (
            block,
            derived.chaos_params,
            derived.baker_salt + index,
            partition,
        )
        for index, block in enumerate(encrypted_blocks)
    ]

    recovered_blocks = _parallel_map(
        _decrypt_block_worker,
        decryption_arguments,
        max_workers,
        executor,
    )

    recovered_permuted = merge_blocks(
        recovered_blocks,
        positions,
        padded_shape,
    )

    permutation = generate_permutation_indices(
        padded_shape,
        derived.chaos_params,
        salt=derived.permutation_salt,
    )

    recovered_padded = inverse_shuffle_image(
        recovered_permuted,
        permutation,
    )

    recovered = remove_padding(
        recovered_padded,
        original_shape,
    )

    elapsed = time.perf_counter() - start

    return {
        "decrypted_image": recovered,
        "decryption_time_sec": float(elapsed),
    }


# =========================================================
# SAVE / LOAD SECURE PACKAGE
# =========================================================

def save_secure_package(
    package: dict[str, Any],
    image_name: str,
) -> dict[str, Path]:
    encrypted_path = (
        OUTPUT_ENCRYPTED / f"{image_name}_encrypted.png"
    )
    metadata_path = (
        OUTPUT_PACKAGES / f"{image_name}_metadata.json"
    )
    tag_path = (
        OUTPUT_PACKAGES / f"{image_name}_auth_tag.txt"
    )

    Image.fromarray(
        package["encrypted_image"]
    ).save(encrypted_path)

    metadata_path.write_text(
        json.dumps(
            package["metadata"],
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    tag_path.write_text(
        package["authentication_tag"].hex(),
        encoding="ascii",
    )

    return {
        "encrypted": encrypted_path,
        "metadata": metadata_path,
        "authentication_tag": tag_path,
    }


def load_secure_package(
    encrypted_path: str | Path,
    metadata_path: str | Path,
    tag_path: str | Path,
) -> tuple[NDArray[np.uint8], dict[str, Any], bytes]:
    encrypted = np.asarray(
        Image.open(encrypted_path).convert("L"),
        dtype=np.uint8,
    )

    metadata = json.loads(
        Path(metadata_path).read_text(
            encoding="utf-8",
        )
    )

    authentication_tag = bytes.fromhex(
        Path(tag_path).read_text(
            encoding="ascii",
        ).strip()
    )

    return encrypted, metadata, authentication_tag


def decrypt_saved_package(
    encrypted_path: str | Path,
    metadata_path: str | Path,
    tag_path: str | Path,
    master_key: bytes,
    *,
    save_path: str | Path | None = None,
) -> NDArray[np.uint8]:
    encrypted, metadata, tag = load_secure_package(
        encrypted_path,
        metadata_path,
        tag_path,
    )

    result = decrypt_image_secure(
        encrypted,
        metadata,
        tag,
        master_key,
    )

    decrypted = result["decrypted_image"]

    if save_path is not None:
        output_path = Path(save_path)
        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        Image.fromarray(decrypted).save(output_path)

    return decrypted


# =========================================================
# STAGE ANALYSIS FOR PAPER FIGURES
# =========================================================

def _save_stage_image(image: NDArray, save_path: Path) -> None:
    image_u8 = _validate_uint8_image(image)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image_u8).save(save_path)

def _save_histogram(image: NDArray, save_path: Path, title: str) -> None:
    image_u8 = _validate_uint8_image(image)
    histogram = np.bincount(image_u8.reshape(-1), minlength=256)
    figure = plt.figure(figsize=(7.0, 4.5))
    axis = figure.add_subplot(111)
    axis.plot(np.arange(256), histogram)
    axis.set_xlim(0, 255)
    axis.set_xlabel("Pixel intensity")
    axis.set_ylabel("Frequency")
    axis.set_title(title)
    axis.grid(True, linewidth=0.3, alpha=0.4)
    figure.tight_layout()
    figure.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(figure)

def _adjacent_pairs(image: NDArray, direction: str) -> tuple[NDArray[np.uint8], NDArray[np.uint8]]:
    image_u8 = _validate_uint8_image(image)
    if direction == "horizontal":
        return image_u8[:, :-1], image_u8[:, 1:]
    if direction == "vertical":
        return image_u8[:-1, :], image_u8[1:, :]
    if direction == "diagonal":
        return image_u8[:-1, :-1], image_u8[1:, 1:]
    raise ValueError("direction must be horizontal, vertical, or diagonal.")

def _save_correlation_scatter(image: NDArray, save_path: Path, title: str, direction: str, *, max_pairs: int = 5000, random_seed: int = 2026) -> None:
    first, second = _adjacent_pairs(image, direction)
    x = first.reshape(-1)
    y = second.reshape(-1)
    if x.size > max_pairs:
        rng = np.random.default_rng(random_seed)
        indices = rng.choice(x.size, size=max_pairs, replace=False)
        x = x[indices]
        y = y[indices]
    figure = plt.figure(figsize=(5.2, 5.2))
    axis = figure.add_subplot(111)
    axis.scatter(x, y, s=5, alpha=0.35)
    axis.set_xlim(0, 255)
    axis.set_ylim(0, 255)
    axis.set_xlabel("Pixel value")
    axis.set_ylabel("Adjacent pixel value")
    axis.set_title(title)
    axis.grid(True, linewidth=0.3, alpha=0.4)
    figure.tight_layout()
    figure.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(figure)

def _stage_metric_row(stage_name: str, image: NDArray) -> dict[str, Any]:
    image_u8 = _validate_uint8_image(image)
    correlations = calculate_correlations(image_u8)
    chi_square = calculate_chi_square_uniformity(image_u8)
    return {
        "stage": stage_name,
        "height": int(image_u8.shape[0]),
        "width": int(image_u8.shape[1]),
        "entropy": calculate_entropy(image_u8),
        "chi_square": chi_square["chi_square"],
        "reduced_chi_square": chi_square["reduced_chi_square"],
        "chi_square_degrees_of_freedom": chi_square["degrees_of_freedom"],
        "expected_histogram_frequency": chi_square["expected_frequency"],
        "corr_horizontal": correlations["corr_horizontal"],
        "corr_vertical": correlations["corr_vertical"],
        "corr_diagonal": correlations["corr_diagonal"],
    }

def save_stage_analysis(package: dict[str, Any], recovered_image: NDArray, image_name: str) -> dict[str, Path]:
    required_fields = {
        "stage_original",
        "stage_global_permutation",
        "stage_block_diffusion",
        "stage_global_diffusion",
    }
    missing = required_fields.difference(package.keys())
    if missing:
        raise ValueError("Encryption package lacks stage-analysis fields: " + ", ".join(sorted(missing)))

    analysis_dir = OUTPUT_STAGE_ANALYSIS / image_name
    analysis_dir.mkdir(parents=True, exist_ok=True)
    stages = [
        ("original", "Stage 0: Original image", package["stage_original"]),
        ("global_permutation", "Stage 1: Global permutation", package["stage_global_permutation"]),
        ("block_diffusion", "Stage 2: Baker block diffusion", package["stage_block_diffusion"]),
        ("final_cipher", "Stage 3: Global bidirectional diffusion", package["stage_global_diffusion"]),
        ("recovered", "Stage 4: Recovered image", recovered_image),
    ]
    saved_paths: dict[str, Path] = {}
    metric_rows: list[dict[str, Any]] = []
    for index, (short_name, title, stage_image) in enumerate(stages):
        image_path = analysis_dir / f"stage{index}_{short_name}.png"
        histogram_path = analysis_dir / f"hist_stage{index}_{short_name}.png"
        _save_stage_image(stage_image, image_path)
        _save_histogram(stage_image, histogram_path, title=f"{title} histogram")
        saved_paths[f"stage_{short_name}"] = image_path
        saved_paths[f"hist_{short_name}"] = histogram_path
        metric_rows.append(_stage_metric_row(title, stage_image))

    for source_name, source_title, source_image in (
        ("original", "Original image", package["stage_original"]),
        ("encrypted", "Final encrypted image", package["stage_global_diffusion"]),
    ):
        for direction in ("horizontal", "vertical", "diagonal"):
            scatter_path = analysis_dir / f"{source_name}_{direction}_correlation.png"
            _save_correlation_scatter(
                source_image,
                scatter_path,
                title=f"{source_title}: {direction.capitalize()} correlation",
                direction=direction,
            )
            saved_paths[f"{source_name}_{direction}_correlation"] = scatter_path

    metrics_path = analysis_dir / "stage_metrics.csv"
    save_metrics_to_csv(metric_rows, metrics_path)
    saved_paths["stage_metrics"] = metrics_path

    original = _validate_uint8_image(package["stage_original"])
    recovered = _validate_uint8_image(recovered_image)
    permutation = _validate_uint8_image(package["stage_global_permutation"])
    verification = {
        "recovered_equals_original": bool(np.array_equal(original, recovered)),
        "original_entropy_equals_recovered": bool(np.isclose(calculate_entropy(original), calculate_entropy(recovered), atol=0.0, rtol=0.0)),
        "original_histogram_equals_permutation": bool(np.array_equal(np.bincount(original.reshape(-1), minlength=256), np.bincount(permutation.reshape(-1), minlength=256))),
        "original_histogram_equals_recovered": bool(np.array_equal(np.bincount(original.reshape(-1), minlength=256), np.bincount(recovered.reshape(-1), minlength=256))),
    }
    verification_path = analysis_dir / "stage_verification.json"
    verification_path.write_text(json.dumps(verification, indent=2, sort_keys=True), encoding="utf-8")
    saved_paths["stage_verification"] = verification_path
    return saved_paths


# =========================================================
# FULL EXPERIMENTAL PIPELINE
# =========================================================

def run_full_secure_pipeline(
    image: NDArray,
    master_key: bytes,
    *,
    image_name: str = "image",
    session_nonce: bytes | None = None,
    block_size: int = BLOCK_SIZE,
    baker_partition: Sequence[int] | None = None,
    analyze_stages: bool = False,
    max_workers: int | None = MAX_WORKERS,
    executor: ProcessPoolExecutor | None = None,
) -> dict[str, Any]:
    original = _validate_uint8_image(image)

    package = encrypt_image_secure(
        original,
        master_key,
        session_nonce=session_nonce,
        block_size=block_size,
        baker_partition=baker_partition,
        max_workers=max_workers,
        executor=executor,
    )

    decrypted_result = decrypt_image_secure(
        package["encrypted_image"],
        package["metadata"],
        package["authentication_tag"],
        master_key,
        max_workers=max_workers,
        executor=executor,
    )

    decrypted = decrypted_result["decrypted_image"]

    quality = calculate_decryption_quality(
        original,
        decrypted,
    )

    entropy_original = calculate_entropy(original)
    entropy_encrypted = calculate_entropy(
        package["encrypted_image"]
    )
    chi_square_original = calculate_chi_square_uniformity(original)
    chi_square_encrypted = calculate_chi_square_uniformity(
        package["encrypted_image"]
    )
    corr_original = calculate_correlations(original)
    corr_encrypted = calculate_correlations(
        package["encrypted_image"]
    )

    modified = create_one_pixel_modified_image(
        original,
        row=0,
        col=0,
        delta=1,
    )

    # Same master key and same session nonce.
    modified_package = encrypt_image_secure(
        modified,
        master_key,
        session_nonce=bytes.fromhex(
            package["metadata"]["session_nonce_hex"]
        ),
        block_size=block_size,
        baker_partition=baker_partition,
        max_workers=max_workers,
        executor=executor,
    )

    npcr, uaci = calculate_npcr_uaci(
        package["encrypted_image"],
        modified_package["encrypted_image"],
    )

    image_info = get_image_size_info(
        package["encrypted_image"]
    )

    throughput = calculate_throughput(
        image_info["size_mb"],
        package["encryption_time_sec"],
    )

    estimated_cpb = calculate_cpb(
        CPU_FREQ_HZ,
        package["encryption_time_sec"],
        image_info["size_bytes"],
    )

    saved = save_secure_package(
        package,
        image_name,
    )

    original_path = (
        OUTPUT_ORIGINAL / f"{image_name}_original.png"
    )
    decrypted_path = (
        OUTPUT_DECRYPTED / f"{image_name}_decrypted.png"
    )

    Image.fromarray(original).save(original_path)
    Image.fromarray(decrypted).save(decrypted_path)

    stage_analysis_paths: dict[str, Path] = {}
    if analyze_stages:
        stage_analysis_paths = save_stage_analysis(
            package=package,
            recovered_image=decrypted,
            image_name=image_name,
        )

    metrics = {
        "image_name": image_name,
        "entropy_original": entropy_original,
        "entropy_encrypted": entropy_encrypted,
        "chi_square_original": chi_square_original["chi_square"],
        "chi_square_encrypted": chi_square_encrypted["chi_square"],
        "reduced_chi_square_original": chi_square_original["reduced_chi_square"],
        "reduced_chi_square_encrypted": chi_square_encrypted["reduced_chi_square"],
        "corr_h_original": corr_original["corr_horizontal"],
        "corr_h_encrypted": corr_encrypted["corr_horizontal"],
        "corr_v_original": corr_original["corr_vertical"],
        "corr_v_encrypted": corr_encrypted["corr_vertical"],
        "corr_d_original": corr_original["corr_diagonal"],
        "corr_d_encrypted": corr_encrypted["corr_diagonal"],
        "npcr_percent": npcr,
        "uaci_percent": uaci,
        "decryption_identical": quality["identical"],
        "decryption_mse": quality["mse"],
        "decryption_mae": quality["mae"],
        "encryption_time_sec": package["encryption_time_sec"],
        "decryption_time_sec": decrypted_result[
            "decryption_time_sec"
        ],
        "throughput_mb_s": throughput,
        "estimated_cpb": estimated_cpb,
        "authentication_verified": True,
    }

    metrics_path = (
        OUTPUT_METRICS / f"{image_name}_secure_metrics.csv"
    )
    save_metrics_to_csv([metrics], metrics_path)

    return {
        "encrypted_image": package["encrypted_image"],
        "decrypted_image": decrypted,
        "metadata": package["metadata"],
        "authentication_tag": package["authentication_tag"],
        "metrics": metrics,
        "saved_paths": {
            **saved,
            "original": original_path,
            "decrypted": decrypted_path,
            "metrics": metrics_path,
            **stage_analysis_paths,
        },
    }


# =========================================================
# BATCH RUNNER
# =========================================================

def run_batch(
    number_of_images: int = NUM_IMAGES,
    master_key_hex: str | None = None,
) -> None:
    dcm_files = sorted(DATA_DIR.rglob("*.dcm"))

    if not dcm_files:
        raise FileNotFoundError(
            f"No DICOM files were found in: {DATA_DIR}"
        )

    if master_key_hex is None:
        master_key = generate_master_key()
        print("=" * 72)
        print("GENERATED MASTER KEY — STORE IT SECURELY")
        print(master_key_to_hex(master_key))
        print("=" * 72)
    else:
        master_key = master_key_from_hex(
            master_key_hex
        )

    selected_files = dcm_files[:number_of_images]
    all_metrics: list[dict[str, Any]] = []

    print(
        f"Persistent multiprocessing pool: "
        f"{PERSISTENT_WORKERS} worker processes"
    )

    # Create the process pool once and reuse it for encryption,
    # decryption, differential testing, and all selected images.
    with ProcessPoolExecutor(
        max_workers=PERSISTENT_WORKERS
    ) as persistent_executor:

        for index, dicom_file in enumerate(
            selected_files,
            start=1,
        ):
            print(
                f"\nProcessing {index}/{len(selected_files)}: "
                f"{dicom_file.name}"
            )

            image = load_dicom_as_uint8(dicom_file)

            result = run_full_secure_pipeline(
                image,
                master_key,
                image_name=dicom_file.stem,
                analyze_stages=(
                    ENABLE_STAGE_ANALYSIS
                    and dicom_file.stem == STAGE_ANALYSIS_IMAGE_NAME
                ),
                max_workers=PERSISTENT_WORKERS,
                executor=persistent_executor,
            )

            all_metrics.append(result["metrics"])

            print(
                "Exact recovery:",
                result["metrics"]["decryption_identical"],
            )
            print(
                "NPCR/UACI:",
                f"{result['metrics']['npcr_percent']:.6f}% / "
                f"{result['metrics']['uaci_percent']:.6f}%",
            )

            if (
                ENABLE_STAGE_ANALYSIS
                and dicom_file.stem == STAGE_ANALYSIS_IMAGE_NAME
            ):
                print(
                    "Stage analysis saved to:",
                    OUTPUT_STAGE_ANALYSIS / dicom_file.stem,
                )

    batch_path = (
        OUTPUT_METRICS / "batch_secure_results.csv"
    )
    save_metrics_to_csv(
        all_metrics,
        batch_path,
    )

    print(f"\nBatch results saved to: {batch_path}")


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":
    run_batch(NUM_IMAGES)
