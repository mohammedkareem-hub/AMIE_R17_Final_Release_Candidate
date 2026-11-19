from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple
import time

import numpy as np
from PIL import Image


# ============================================================
# Zigzag
# ============================================================

def zigzag_indices(n: int) -> np.ndarray:
    coords = []

    for s in range(2 * n - 1):
        diagonal = []

        r_min = max(0, s - (n - 1))
        r_max = min(n - 1, s)

        for r in range(r_min, r_max + 1):
            c = s - r
            diagonal.append((r, c))

        if s % 2 == 0:
            diagonal.reverse()

        coords.extend(diagonal)

    return np.asarray(
        [r * n + c for r, c in coords],
        dtype=np.int64,
    )


_ZIGZAG_CACHE = {}


def zigzag_permute(block: np.ndarray) -> np.ndarray:

    n = block.shape[0]

    if block.shape != (n, n):
        raise ValueError("Zigzag requires a square block.")

    if n not in _ZIGZAG_CACHE:
        idx = zigzag_indices(n)
        inv = np.argsort(idx)
        _ZIGZAG_CACHE[n] = (idx, inv)

    idx, _ = _ZIGZAG_CACHE[n]

    flat = block.reshape(-1)

    return flat[idx].reshape(n, n)


def zigzag_inverse(block: np.ndarray) -> np.ndarray:

    n = block.shape[0]

    if block.shape != (n, n):
        raise ValueError("Inverse zigzag requires a square block.")

    if n not in _ZIGZAG_CACHE:
        idx = zigzag_indices(n)
        inv = np.argsort(idx)
        _ZIGZAG_CACHE[n] = (idx, inv)

    _, inv = _ZIGZAG_CACHE[n]

    flat = block.reshape(-1)

    return flat[inv].reshape(n, n)


# ============================================================
# Padding
# ============================================================

def pad_to_multiple(
    image: np.ndarray,
    multiple: int,
):

    h, w = image.shape

    hp = ((h + multiple - 1) // multiple) * multiple
    wp = ((w + multiple - 1) // multiple) * multiple

    if hp == h and wp == w:
        return image.copy(), (h, w)

    padded = np.pad(
        image,
        (
            (0, hp - h),
            (0, wp - w),
        ),
        mode="edge",
    )

    return padded.astype(np.uint8), (h, w)


# ============================================================
# Metadata
# ============================================================

@dataclass(frozen=True)
class KamalMetadata:

    original_shape: Tuple[int, int]

    padded_shape: Tuple[int, int]

    y0: float

    block_size: int

    a: float

    n0: int

    rng_seed: int


# ============================================================
# Kamal et al. 2021
# ============================================================

class Kamal2021:

    def __init__(
        self,
        block_size: int = 16,
        a: float = 3.9,
        n0: int = 1000,
        rng_seed: int = 2021,
    ):

        if block_size not in (16, 32, 64):
            raise ValueError(
                "block_size must be 16, 32, or 64"
            )

        n = int(np.log2(block_size))

        if 2 ** n != block_size:
            raise ValueError(
                "block_size must be a power of 2"
            )

        self.block_size = block_size

        self.n = n

        self.a = float(a)

        self.n0 = int(n0)

        self.rng_seed = int(rng_seed)


    # ========================================================
    # Image loading
    # ========================================================

    @staticmethod
    def load_grayscale(
        path: str,
    ) -> np.ndarray:

        image = Image.open(path).convert("L")

        return np.asarray(
            image,
            dtype=np.uint8,
        )


    # ========================================================
    # Random vectors Ri and r
    # ========================================================

    def _random_parameters(
        self,
        image_shape,
    ):

        rows, cols = image_shape

        h = self.block_size

        if rows % h != 0 or cols % h != 0:

            raise ValueError(
                "Image dimensions must be divisible by block size."
            )

        number_blocks = (
            (rows // h)
            *
            (cols // h)
        )

        # Paper does not specify PRNG/seed.
        # Fixed seed used for reproducibility.
        rng = np.random.default_rng(
            self.rng_seed
        )

        # l belongs to {2,3,...,n}
        split_l = rng.integers(
            low=2,
            high=self.n + 1,
            size=number_blocks,
        )

        # Random block permutation r
        block_permutation = rng.permutation(
            number_blocks
        )

        return (
            split_l.astype(np.int64),
            block_permutation.astype(np.int64),
        )


    # ========================================================
    # Confusion
    # ========================================================

    def _confusion(
        self,
        image: np.ndarray,
    ) -> np.ndarray:

        h = self.block_size

        rows, cols = image.shape

        number_block_rows = rows // h

        number_block_cols = cols // h

        (
            split_l,
            block_permutation,
        ) = self._random_parameters(
            image.shape
        )

        processed_blocks = []

        block_index = 0

        for br in range(number_block_rows):

            for bc in range(number_block_cols):

                block = image[
                    br * h:(br + 1) * h,
                    bc * h:(bc + 1) * h,
                ].copy()

                l = int(
                    split_l[block_index]
                )

                sub_size = 2 ** l

                transformed_block = np.empty_like(
                    block
                )

                for sr in range(
                    0,
                    h,
                    sub_size,
                ):

                    for sc in range(
                        0,
                        h,
                        sub_size,
                    ):

                        sub_block = block[
                            sr:sr + sub_size,
                            sc:sc + sub_size,
                        ]

                        # Step 1: zigzag
                        temp = zigzag_permute(
                            sub_block
                        )

                        # Step 2:
                        # 90 degrees counter-clockwise
                        temp = np.rot90(
                            temp,
                            k=1,
                        )

                        transformed_block[
                            sr:sr + sub_size,
                            sc:sc + sub_size,
                        ] = temp

                processed_blocks.append(
                    transformed_block
                )

                block_index += 1


        # Random permutation between blocks
        shuffled_blocks = [

            processed_blocks[int(i)]

            for i in block_permutation
        ]


        scrambled = np.empty_like(
            image
        )

        index = 0

        for br in range(
            number_block_rows
        ):

            for bc in range(
                number_block_cols
            ):

                scrambled[
                    br * h:(br + 1) * h,
                    bc * h:(bc + 1) * h,
                ] = shuffled_blocks[index]

                index += 1


        return scrambled


    # ========================================================
    # Inverse confusion
    # ========================================================

    def _inverse_confusion(
        self,
        scrambled: np.ndarray,
    ) -> np.ndarray:

        h = self.block_size

        rows, cols = scrambled.shape

        number_block_rows = rows // h

        number_block_cols = cols // h

        number_blocks = (
            number_block_rows
            *
            number_block_cols
        )

        (
            split_l,
            block_permutation,
        ) = self._random_parameters(
            scrambled.shape
        )


        shuffled_blocks = []

        for br in range(
            number_block_rows
        ):

            for bc in range(
                number_block_cols
            ):

                block = scrambled[
                    br * h:(br + 1) * h,
                    bc * h:(bc + 1) * h,
                ].copy()

                shuffled_blocks.append(
                    block
                )


        # Reverse block permutation
        processed_blocks = [
            None
        ] * number_blocks

        for output_index, source_index in enumerate(
            block_permutation
        ):

            processed_blocks[
                int(source_index)
            ] = shuffled_blocks[
                output_index
            ]


        recovered = np.empty_like(
            scrambled
        )

        block_index = 0

        for br in range(
            number_block_rows
        ):

            for bc in range(
                number_block_cols
            ):

                block = processed_blocks[
                    block_index
                ]

                l = int(
                    split_l[block_index]
                )

                sub_size = 2 ** l

                recovered_block = np.empty_like(
                    block
                )


                for sr in range(
                    0,
                    h,
                    sub_size,
                ):

                    for sc in range(
                        0,
                        h,
                        sub_size,
                    ):

                        sub_block = block[
                            sr:sr + sub_size,
                            sc:sc + sub_size,
                        ]


                        # Reverse rotation first
                        temp = np.rot90(
                            sub_block,
                            k=-1,
                        )

                        # Then inverse zigzag
                        temp = zigzag_inverse(
                            temp
                        )


                        recovered_block[
                            sr:sr + sub_size,
                            sc:sc + sub_size,
                        ] = temp


                recovered[
                    br * h:(br + 1) * h,
                    bc * h:(bc + 1) * h,
                ] = recovered_block


                block_index += 1


        return recovered


    # ========================================================
    # Logistic map initial condition
    # Eq. (2)
    # ========================================================

    @staticmethod
    def _calculate_y0(
        plain_image: np.ndarray,
    ) -> float:

        M, N = plain_image.shape

        pixel_sum = np.sum(
            plain_image,
            dtype=np.float64,
        )

        y0 = (
            pixel_sum
            /
            (M * N * 255.0)
        )

        return float(y0)


    # ========================================================
    # Logistic map key generation
    # Eq. (1) and Eq. (3)
    # ========================================================

    def _generate_key(
        self,
        image_shape,
        y0: float,
    ) -> np.ndarray:

        M, N = image_shape

        number_pixels = M * N

        total_length = (
            self.n0
            +
            number_pixels
        )

        sequence = np.empty(
            total_length,
            dtype=np.float64,
        )

        # Y(1) = Y0
        sequence[0] = y0


        for i in range(
            1,
            total_length,
        ):

            sequence[i] = (

                self.a
                *
                sequence[i - 1]
                *
                (
                    1.0
                    -
                    sequence[i - 1]
                )
            )


        # Remove first N0 values
        S = sequence[
            self.n0:
            self.n0 + number_pixels
        ]


        # Eq. (3)
        key = np.mod(

            np.floor(
                S * 1e14
            ),

            256,

        ).astype(
            np.uint8
        )


        return key


    # ========================================================
    # Encryption
    # ========================================================

    def encrypt_array(
        self,
        plain_image: np.ndarray,
    ):

        if plain_image.ndim != 2:

            raise ValueError(
                "Expected 2-D grayscale image."
            )


        plain_image = np.asarray(
            plain_image,
            dtype=np.uint8,
        )


        start_time = time.perf_counter()


        padded_image, original_shape = (
            pad_to_multiple(
                plain_image,
                self.block_size,
            )
        )


        # Eq. (2)
        y0 = self._calculate_y0(
            padded_image
        )


        # Confusion
        scrambled = self._confusion(
            padded_image
        )


        # Logistic key
        key = self._generate_key(
            padded_image.shape,
            y0,
        )


        # MATLAB X(:) convention:
        # column-major ordering
        scrambled_vector = scrambled.reshape(
            -1,
            order="F",
        )


        # XOR diffusion
        encrypted_vector = np.bitwise_xor(
            scrambled_vector,
            key,
        )


        ciphertext = encrypted_vector.reshape(
            padded_image.shape,
            order="F",
        ).astype(
            np.uint8
        )


        encryption_time = (
            time.perf_counter()
            -
            start_time
        )


        metadata = KamalMetadata(

            original_shape=original_shape,

            padded_shape=padded_image.shape,

            y0=y0,

            block_size=self.block_size,

            a=self.a,

            n0=self.n0,

            rng_seed=self.rng_seed,

        )


        return (
            ciphertext,
            metadata,
            encryption_time,
        )


    # ========================================================
    # Decryption
    # ========================================================

    def decrypt_array(
        self,
        ciphertext: np.ndarray,
        metadata: KamalMetadata,
    ):

        ciphertext = np.asarray(
            ciphertext,
            dtype=np.uint8,
        )


        if tuple(
            ciphertext.shape
        ) != tuple(
            metadata.padded_shape
        ):

            raise ValueError(
                "Ciphertext dimensions do not match metadata."
            )


        start_time = time.perf_counter()


        # Regenerate same key
        key = self._generate_key(
            metadata.padded_shape,
            metadata.y0,
        )


        cipher_vector = ciphertext.reshape(
            -1,
            order="F",
        )


        # XOR inverse
        scrambled_vector = np.bitwise_xor(
            cipher_vector,
            key,
        )


        scrambled = scrambled_vector.reshape(
            metadata.padded_shape,
            order="F",
        ).astype(
            np.uint8
        )


        # Undo confusion
        padded_plain = self._inverse_confusion(
            scrambled
        )


        H, W = metadata.original_shape


        recovered = padded_plain[
            :H,
            :W,
        ].astype(
            np.uint8
        )


        decryption_time = (
            time.perf_counter()
            -
            start_time
        )


        return (
            recovered,
            decryption_time,
        )