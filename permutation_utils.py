from __future__ import annotations

import hashlib
from typing import Any

import numpy as np
from numpy.typing import NDArray

from chaotic_generator import generate_chaotic_byte_sequence


IntegerArray = NDArray[np.int64]


class Shake256Stream:
    """Deterministic cryptographic byte stream based on SHAKE-256."""

    def __init__(self, seed_material: bytes):
        if not isinstance(seed_material, bytes):
            raise TypeError("seed_material must be bytes.")
        if not seed_material:
            raise ValueError("seed_material must not be empty.")
        self._seed = seed_material
        self._counter = 0
        self._buffer = bytearray()

    def _refill(self, minimum_bytes: int) -> None:
        while len(self._buffer) < minimum_bytes:
            block = hashlib.shake_256(
                b"PermutationSHAKE256-v1|"
                + self._counter.to_bytes(8, "big", signed=False)
                + self._seed
            ).digest(64)
            self._buffer.extend(block)
            self._counter += 1

    def read(self, length: int) -> bytes:
        if length <= 0:
            raise ValueError("length must be greater than zero.")
        self._refill(length)
        result = bytes(self._buffer[:length])
        del self._buffer[:length]
        return result

    def randbelow(self, upper_bound: int) -> int:
        """Return a uniform integer in [0, upper_bound) using rejection sampling."""
        if upper_bound <= 0:
            raise ValueError("upper_bound must be greater than zero.")
        if upper_bound == 1:
            return 0

        number_of_bytes = max(1, (upper_bound.bit_length() + 7) // 8)
        sample_space = 1 << (8 * number_of_bytes)
        acceptance_limit = sample_space - sample_space % upper_bound

        while True:
            candidate = int.from_bytes(
                self.read(number_of_bytes),
                "big",
                signed=False,
            )
            if candidate < acceptance_limit:
                return candidate % upper_bound


def _normalize_salt(salt: int | str | bytes) -> bytes:
    if isinstance(salt, bytes):
        return salt
    return str(salt).encode("utf-8")


def derive_permutation_seed(
    image_shape: tuple[int, int],
    chaos_params: dict[str, Any] | None,
    salt: int | str | bytes,
) -> bytes:
    """Derive a 256-bit seed for the cryptographic permutation stage."""
    if len(image_shape) != 2:
        raise ValueError("image_shape must contain height and width.")

    height, width = image_shape
    if height <= 0 or width <= 0:
        raise ValueError("Image dimensions must be positive.")

    salt_bytes = _normalize_salt(salt)
    chaotic_bytes = generate_chaotic_byte_sequence(
        length=64,
        chaos_params=chaos_params,
        salt=b"permutation-chaos|" + salt_bytes,
        discard=100,
    )

    dimension_bytes = (
        int(height).to_bytes(8, "big", signed=False)
        + int(width).to_bytes(8, "big", signed=False)
    )

    return hashlib.sha256(
        b"GlobalPermutationSeed-v2|"
        + chaotic_bytes.tobytes()
        + b"|"
        + dimension_bytes
        + b"|"
        + salt_bytes
    ).digest()


def generate_permutation_indices(
    image_shape: tuple[int, int],
    chaos_params: dict[str, Any] | None = None,
    salt: int | str | bytes = 0,
) -> IntegerArray:
    """
    Generate a deterministic key-dependent permutation using
    Fisher-Yates driven by a SHAKE-256 cryptographic stream.
    """
    if len(image_shape) != 2:
        raise ValueError("image_shape must contain height and width.")

    height, width = image_shape
    if height <= 0 or width <= 0:
        raise ValueError("Image dimensions must be positive.")

    total_pixels = height * width
    seed = derive_permutation_seed(image_shape, chaos_params, salt)
    stream = Shake256Stream(seed)

    permutation = np.arange(total_pixels, dtype=np.int64)

    for index in range(total_pixels - 1, 0, -1):
        swap_index = stream.randbelow(index + 1)
        if swap_index != index:
            permutation[index], permutation[swap_index] = (
                permutation[swap_index],
                permutation[index],
            )

    return permutation


def shuffle_image(
    image: NDArray,
    chaos_params: dict[str, Any] | None = None,
    salt: int | str | bytes = 0,
) -> tuple[NDArray, IntegerArray]:
    """Apply global key-dependent pixel permutation to a 2D image."""
    array = np.asarray(image)
    if array.ndim != 2:
        raise ValueError("The input image must be a 2D grayscale image.")
    if array.size == 0:
        raise ValueError("The input image must not be empty.")

    permutation = generate_permutation_indices(array.shape, chaos_params, salt)
    shuffled = array.reshape(-1)[permutation].reshape(array.shape)
    return shuffled, permutation


def inverse_shuffle_image(
    shuffled_image: NDArray,
    permutation: IntegerArray,
) -> NDArray:
    """Recover the original image using the permutation vector."""
    array = np.asarray(shuffled_image)
    if array.ndim != 2:
        raise ValueError("The shuffled image must be a 2D grayscale image.")
    if array.size == 0:
        raise ValueError("The shuffled image must not be empty.")

    total_pixels = array.size
    permutation_array = np.asarray(permutation, dtype=np.int64)

    if permutation_array.ndim != 1:
        raise ValueError("The permutation vector must be one-dimensional.")
    if permutation_array.size != total_pixels:
        raise ValueError("Permutation length must equal the number of image pixels.")
    if (
        permutation_array.min(initial=0) < 0
        or permutation_array.max(initial=0) >= total_pixels
    ):
        raise ValueError("Permutation indices are outside valid bounds.")
    if np.unique(permutation_array).size != total_pixels:
        raise ValueError("The permutation vector must contain unique indices.")

    shuffled_flat = array.reshape(-1)
    original_flat = np.empty_like(shuffled_flat)
    original_flat[permutation_array] = shuffled_flat
    return original_flat.reshape(array.shape)


def regenerate_and_inverse_shuffle(
    shuffled_image: NDArray,
    chaos_params: dict[str, Any] | None = None,
    salt: int | str | bytes = 0,
) -> NDArray:
    """Regenerate the same permutation and recover the original image."""
    array = np.asarray(shuffled_image)
    if array.ndim != 2:
        raise ValueError("The shuffled image must be a 2D grayscale image.")

    permutation = generate_permutation_indices(array.shape, chaos_params, salt)
    return inverse_shuffle_image(array, permutation)


if __name__ == "__main__":
    from chaotic_generator import CHAOS_PARAMS

    print("=" * 72)
    print("CRYPTOGRAPHIC SHAKE-256 PERMUTATION TEST")
    print("=" * 72)

    original = np.arange(256 * 256, dtype=np.uint32).reshape(256, 256)

    shuffled_1, permutation_1 = shuffle_image(original, CHAOS_PARAMS, salt=1)
    shuffled_1_repeat, permutation_1_repeat = shuffle_image(
        original, CHAOS_PARAMS, salt=1
    )
    shuffled_2, permutation_2 = shuffle_image(original, CHAOS_PARAMS, salt=2)

    restored = inverse_shuffle_image(shuffled_1, permutation_1)
    regenerated = regenerate_and_inverse_shuffle(shuffled_1, CHAOS_PARAMS, salt=1)
    wrong_key_result = regenerate_and_inverse_shuffle(
        shuffled_1, CHAOS_PARAMS, salt=99
    )

    permutation_difference = float(
        np.mean(permutation_1 != permutation_2) * 100.0
    )
    fixed_points = int(
        np.count_nonzero(
            permutation_1
            == np.arange(permutation_1.size, dtype=np.int64)
        )
    )

    print("Same key deterministic      :", np.array_equal(permutation_1, permutation_1_repeat))
    print("Same shuffled image         :", np.array_equal(shuffled_1, shuffled_1_repeat))
    print("Different salt permutation  :", not np.array_equal(permutation_1, permutation_2))
    print("Permutation difference rate :", f"{permutation_difference:.4f}%")
    print("Stored inverse successful   :", np.array_equal(original, restored))
    print("Regenerated inverse success :", np.array_equal(original, regenerated))
    print("Wrong key restores image    :", np.array_equal(original, wrong_key_result))
    print(
        "Pixel values preserved      :",
        np.array_equal(
            np.sort(original.reshape(-1)),
            np.sort(shuffled_1.reshape(-1)),
        ),
    )
    print("Permutation is unique       :", np.unique(permutation_1).size == permutation_1.size)
    print("Fixed points                :", fixed_points)
    print("=" * 72)
    print("Permutation test completed.")
