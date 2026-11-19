from __future__ import annotations

from typing import Any, Sequence
import hashlib
import numpy as np
from numpy.typing import NDArray

from chaotic_generator import generate_chaotic_byte_sequence


# =========================================================
# DEFAULT BAKER PARTITION
# =========================================================

def default_baker_partition(block_size: int) -> tuple[int, ...]:
    """
    Return a valid default partition for the discrete Baker map.

    For a 64 × 64 block, the partition is:
        (8, 8, 8, 8, 8, 8, 8, 8)

    Every partition value:
    1. must be positive,
    2. must divide the block size,
    3. and all values must sum to the block size.
    """
    if block_size <= 0:
        raise ValueError("block_size must be greater than zero.")

    if block_size % 8 == 0:
        strip_width = block_size // 8
        return tuple([strip_width] * 8)

    # A general fallback valid for every positive block size.
    return tuple([1] * block_size)


def validate_baker_partition(
    block_size: int,
    partition: Sequence[int],
) -> tuple[int, ...]:
    """
    Validate the strip widths used by the discrete Baker map.
    """
    normalized = tuple(int(value) for value in partition)

    if len(normalized) == 0:
        raise ValueError("The Baker partition must not be empty.")

    if any(value <= 0 for value in normalized):
        raise ValueError("Every partition value must be positive.")

    if sum(normalized) != block_size:
        raise ValueError(
            "The sum of the Baker partition values must equal "
            f"the block size ({block_size})."
        )

    for value in normalized:
        if block_size % value != 0:
            raise ValueError(
                f"Partition value {value} does not divide "
                f"the block size {block_size}."
            )

    return normalized


# =========================================================
# DISCRETE BAKER MAP
# =========================================================

def discrete_baker_permutation(
    block: NDArray,
    partition: Sequence[int] | None = None,
) -> NDArray:
    """
    Apply the discrete Baker map to a square image block.

    For a block of size N × N and strip widths:

        n1, n2, ..., nk

    where:

        n1 + n2 + ... + nk = N,

    each source pixel (r, s) in strip i is mapped to:

        r' = (N / ni)(r - Ni) + (s mod (N / ni))

        s' = floor(s / (N / ni)) + Ni

    where Ni is the sum of the widths of all preceding strips.

    The operation changes pixel positions only and preserves
    all pixel values.
    """
    if block.ndim != 2:
        raise ValueError("The input block must be a 2D array.")

    height, width = block.shape

    if height != width:
        raise ValueError("The discrete Baker map requires a square block.")

    block_size = height

    if partition is None:
        partition = default_baker_partition(block_size)

    partition = validate_baker_partition(
        block_size,
        partition,
    )

    permuted = np.empty_like(block)

    strip_start = 0

    for strip_width in partition:
        scale = block_size // strip_width

        for row in range(strip_start, strip_start + strip_width):
            local_row = row - strip_start

            for column in range(block_size):
                new_row = (
                    scale * local_row
                    + column % scale
                )

                new_column = (
                    column // scale
                    + strip_start
                )

                permuted[new_row, new_column] = block[row, column]

        strip_start += strip_width

    return permuted


def inverse_discrete_baker_permutation(
    permuted_block: NDArray,
    partition: Sequence[int] | None = None,
) -> NDArray:
    """
    Recover the original block by applying the inverse mapping
    of the discrete Baker transformation.
    """
    if permuted_block.ndim != 2:
        raise ValueError("The input block must be a 2D array.")

    height, width = permuted_block.shape

    if height != width:
        raise ValueError("The inverse Baker map requires a square block.")

    block_size = height

    if partition is None:
        partition = default_baker_partition(block_size)

    partition = validate_baker_partition(
        block_size,
        partition,
    )

    recovered = np.empty_like(permuted_block)

    strip_start = 0

    for strip_width in partition:
        scale = block_size // strip_width

        for row in range(strip_start, strip_start + strip_width):
            local_row = row - strip_start

            for column in range(block_size):
                new_row = (
                    scale * local_row
                    + column % scale
                )

                new_column = (
                    column // scale
                    + strip_start
                )

                recovered[row, column] = permuted_block[
                    new_row,
                    new_column,
                ]

        strip_start += strip_width

    return recovered


# =========================================================
# CUMULATIVE FEEDBACK DIFFUSION
# =========================================================

def cumulative_feedback_diffusion(
    block: NDArray,
    key: NDArray,
) -> NDArray:
    """
    Apply cumulative feedback diffusion:

        C[0] = (P[0] + K[0]) mod 256

        C[i] = (P[i] + K[i] + C[i-1]) mod 256
    """
    if block.ndim != 2:
        raise ValueError("The input block must be two-dimensional.")

    if block.dtype != np.uint8:
        raise ValueError(
            "Cumulative diffusion currently requires uint8 blocks."
        )

    flat = block.reshape(-1).astype(np.int64)
    key_array = np.asarray(key, dtype=np.uint8).reshape(-1)

    if key_array.size != flat.size:
        raise ValueError(
            "The chaotic key length must equal the number "
            "of pixels in the block."
        )

    key_int = key_array.astype(np.int64)
    cipher = np.empty(flat.size, dtype=np.int64)

    cipher[0] = (flat[0] + key_int[0]) % 256

    for index in range(1, flat.size):
        cipher[index] = (
            flat[index]
            + key_int[index]
            + cipher[index - 1]
        ) % 256

    return cipher.astype(np.uint8).reshape(block.shape)


def inverse_cumulative_feedback_diffusion(
    diffused_block: NDArray,
    key: NDArray,
) -> NDArray:
    """
    Reverse cumulative feedback diffusion:

        P[0] = (C[0] - K[0]) mod 256

        P[i] = (C[i] - K[i] - C[i-1]) mod 256
    """
    if diffused_block.ndim != 2:
        raise ValueError(
            "The diffused block must be two-dimensional."
        )

    if diffused_block.dtype != np.uint8:
        raise ValueError(
            "Inverse diffusion currently requires uint8 blocks."
        )

    cipher = diffused_block.reshape(-1).astype(np.int64)
    key_array = np.asarray(key, dtype=np.uint8).reshape(-1)

    if key_array.size != cipher.size:
        raise ValueError(
            "The chaotic key length must equal the number "
            "of pixels in the block."
        )

    key_int = key_array.astype(np.int64)
    plaintext = np.empty(cipher.size, dtype=np.int64)

    plaintext[0] = (
        cipher[0] - key_int[0]
    ) % 256

    for index in range(1, cipher.size):
        plaintext[index] = (
            cipher[index]
            - key_int[index]
            - cipher[index - 1]
        ) % 256

    return plaintext.astype(np.uint8).reshape(diffused_block.shape)


# =========================================================
# COMPLETE BAKER BLOCK ENCRYPTION
# =========================================================
BAKER_CHAOTIC_SEED_BYTES = 64

def generate_baker_diffusion_key(
    length,
    chaos_params,
    salt=0,
):
    if length <= 0:
        raise ValueError(
            "Baker key length must be positive."
        )

    chaotic_seed = generate_chaotic_byte_sequence(
        length=BAKER_CHAOTIC_SEED_BYTES,
        chaos_params=chaos_params,
        salt=f"baker-diffusion|{salt}",
        discard=100,
    ).astype(np.uint8)

    if isinstance(salt, bytes):
        salt_bytes = salt
    else:
        salt_bytes = str(salt).encode("utf-8")

    shake_input = (
        b"AMIE-BakerChen|BakerKeyExpand-v1|"
        + int(length).to_bytes(
            8,
            "big",
            signed=False,
        )
        + b"|"
        + salt_bytes
        + b"|"
        + chaotic_seed.tobytes(
            order="C"
        )
    )

    expanded = hashlib.shake_256(
        shake_input
    ).digest(length)

    return np.frombuffer(
        expanded,
        dtype=np.uint8,
    ).copy()

def apply_baker_substitution(
    block: NDArray,
    chaos_params: dict[str, Any] | None,
    salt: int | str | bytes = 0,
    partition: Sequence[int] | None = None,
) -> tuple[NDArray, NDArray]:
    """
    Encrypt one block using:

        1. Discrete Baker permutation.
        2. Chaotic cumulative feedback diffusion.

    This function retains the old function name so that the
    existing pipeline can continue importing it.
    """
    if block.ndim != 2:
        raise ValueError("The image block must be two-dimensional.")

    if block.shape[0] != block.shape[1]:
        raise ValueError("The Baker block must be square.")

    if block.dtype != np.uint8:
        raise ValueError("The Baker encryption block must use uint8.")

    if partition is None:
        partition = default_baker_partition(block.shape[0])

    # Step 1: real discrete Baker permutation.
    baker_permuted = discrete_baker_permutation(
        block,
        partition,
    )

    # Step 2: generate a different chaotic key for this block.
    key = generate_baker_diffusion_key(
    length=block.size,
    chaos_params=chaos_params,
    salt=salt,
           )

    # Step 3: cumulative feedback diffusion.
    encrypted_block = cumulative_feedback_diffusion(
        baker_permuted,
        key,
    )

    return encrypted_block, key


def inverse_baker_substitution(
    encrypted_block: NDArray,
    key: NDArray,
    partition: Sequence[int] | None = None,
) -> NDArray:
    """
    Recover the original block using:

        1. Inverse cumulative feedback diffusion.
        2. Inverse discrete Baker permutation.
    """
    if encrypted_block.ndim != 2:
        raise ValueError(
            "The encrypted block must be two-dimensional."
        )

    if encrypted_block.shape[0] != encrypted_block.shape[1]:
        raise ValueError("The encrypted Baker block must be square.")

    if partition is None:
        partition = default_baker_partition(
            encrypted_block.shape[0]
        )

    # Reverse diffusion first.
    baker_permuted = inverse_cumulative_feedback_diffusion(
        encrypted_block,
        key,
    )

    # Then reverse the Baker permutation.
    recovered_block = inverse_discrete_baker_permutation(
        baker_permuted,
        partition,
    )

    return recovered_block


# =========================================================
# SELF-TEST
# =========================================================

if __name__ == "__main__":
    from chaotic_generator import CHAOS_PARAMS

    print("=" * 72)
    print("DISCRETE BAKER MAP + CUMULATIVE DIFFUSION TEST")
    print("=" * 72)

    # The paper's discrete Baker map requires a valid square block.
    test_block = np.arange(
        1,
        65,
        dtype=np.uint8,
    ).reshape(8, 8)

    test_partition = (2, 2, 2, 2)

    print("Original block:")
    print(test_block)

    # -----------------------------------------------------
    # Test 1: Baker map alone
    # -----------------------------------------------------
    baker_block = discrete_baker_permutation(
        test_block,
        test_partition,
    )

    baker_recovered = inverse_discrete_baker_permutation(
        baker_block,
        test_partition,
    )

    print("\nBaker-permuted block:")
    print(baker_block)

    print(
        "\nBaker inverse successful:",
        np.array_equal(test_block, baker_recovered),
    )

    print(
        "Baker preserves pixel values:",
        np.array_equal(
            np.sort(test_block.reshape(-1)),
            np.sort(baker_block.reshape(-1)),
        ),
    )

    # -----------------------------------------------------
    # Test 2: Full Baker + diffusion
    # -----------------------------------------------------
    encrypted, key = apply_baker_substitution(
        test_block,
        CHAOS_PARAMS,
        salt=1,
        partition=test_partition,
    )

    recovered = inverse_baker_substitution(
        encrypted,
        key,
        partition=test_partition,
    )

    print("\nEncrypted block:")
    print(encrypted)

    print("\nRecovered block:")
    print(recovered)

    print(
        "\nFull recovery successful:",
        np.array_equal(test_block, recovered),
    )

    # -----------------------------------------------------
    # Test 3: Key sensitivity
    # -----------------------------------------------------
    encrypted_other_key, _ = apply_baker_substitution(
        test_block,
        CHAOS_PARAMS,
        salt=2,
        partition=test_partition,
    )

    difference_rate = (
        np.mean(encrypted != encrypted_other_key) * 100.0
    )

    print(
        "Different-key ciphertext rate:",
        f"{difference_rate:.4f}%",
    )

    print("=" * 72)
    print("Baker-map test completed.")
