from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def pad_image_to_block_size(
    image: NDArray,
    block_size: int = 64,
) -> tuple[NDArray, tuple[int, int]]:
    """
    Pad a 2D image so that its height and width are divisible
    by block_size.

    Returns
    -------
    padded_image:
        The padded image.
    original_shape:
        Original image shape, used later to remove the padding.
    """
    if image.ndim != 2:
        raise ValueError("The input image must be a 2D grayscale image.")

    if block_size <= 0:
        raise ValueError("block_size must be greater than zero.")

    original_shape = image.shape
    height, width = original_shape

    pad_height = (-height) % block_size
    pad_width = (-width) % block_size

    padded_image = np.pad(
        image,
        ((0, pad_height), (0, pad_width)),
        mode="edge",
    )

    return padded_image, original_shape


def split_into_blocks(
    image: NDArray,
    block_size: int = 64,
) -> tuple[list[NDArray], list[tuple[int, int]]]:
    """
    Split a padded 2D image into non-overlapping square blocks.

    Every produced block has exactly:
        block_size × block_size
    """
    if image.ndim != 2:
        raise ValueError("The input image must be a 2D grayscale image.")

    if block_size <= 0:
        raise ValueError("block_size must be greater than zero.")

    height, width = image.shape

    if height % block_size != 0 or width % block_size != 0:
        raise ValueError(
            "Image dimensions must be divisible by block_size. "
            "Call pad_image_to_block_size() before splitting."
        )

    blocks: list[NDArray] = []
    positions: list[tuple[int, int]] = []

    for row in range(0, height, block_size):
        for column in range(0, width, block_size):
            block = image[
                row : row + block_size,
                column : column + block_size,
            ].copy()

            blocks.append(block)
            positions.append((row, column))

    return blocks, positions


def merge_blocks(
    blocks: list[NDArray],
    positions: list[tuple[int, int]],
    padded_shape: tuple[int, int],
) -> NDArray:
    """
    Merge equally sized blocks into the padded image.
    """
    if not blocks:
        raise ValueError("The block list must not be empty.")

    if len(blocks) != len(positions):
        raise ValueError(
            "The number of blocks must equal the number of positions."
        )

    height, width = padded_shape
    dtype = blocks[0].dtype

    merged = np.zeros((height, width), dtype=dtype)

    for block, (row, column) in zip(blocks, positions):
        if block.ndim != 2:
            raise ValueError("Every block must be a 2D array.")

        block_height, block_width = block.shape

        if row + block_height > height or column + block_width > width:
            raise ValueError(
                f"Block at position {(row, column)} exceeds image bounds."
            )

        merged[
            row : row + block_height,
            column : column + block_width,
        ] = block

    return merged


def remove_padding(
    padded_image: NDArray,
    original_shape: tuple[int, int],
) -> NDArray:
    """
    Remove padding and restore the original image dimensions.
    """
    original_height, original_width = original_shape

    if (
        original_height > padded_image.shape[0]
        or original_width > padded_image.shape[1]
    ):
        raise ValueError(
            "The original shape cannot exceed the padded image shape."
        )

    return padded_image[
        :original_height,
        :original_width,
    ].copy()


# =========================================================
# SELF-TEST
# =========================================================

if __name__ == "__main__":
    print("=" * 65)
    print("BLOCK UTILITIES TEST")
    print("=" * 65)

    # Deliberately use dimensions that are not divisible by 64.
    original = np.arange(
        510 * 500,
        dtype=np.uint32,
    ).reshape(510, 500)

    block_size = 64

    padded, original_shape = pad_image_to_block_size(
        original,
        block_size,
    )

    blocks, positions = split_into_blocks(
        padded,
        block_size,
    )

    merged = merge_blocks(
        blocks,
        positions,
        padded.shape,
    )

    restored = remove_padding(
        merged,
        original_shape,
    )

    print(f"Original shape        : {original.shape}")
    print(f"Padded shape          : {padded.shape}")
    print(f"Number of blocks      : {len(blocks)}")
    print(f"First block shape     : {blocks[0].shape}")
    print(
        "Restored == Original :",
        np.array_equal(restored, original),
    )

    print("=" * 65)