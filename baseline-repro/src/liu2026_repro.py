from __future__ import annotations

from dataclasses import dataclass, asdict
import hashlib
import time

import numpy as np


# ============================================================
# CONSTANTS
# ============================================================

MASK24 = (1 << 24) - 1

# Paper:
# C(0) = 010001010100010101000101
#      = 0x454545
C0 = 0x454545


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass(frozen=True)
class Liu2026Config:
    """
    Independent controlled reimplementation of:

    Liu et al. (2026)
    A newly image encryption scheme based on 3-D coupled
    map lattice and Baker map.

    IMPORTANT:
    This is NOT claimed to be author-exact source code.

    Source-paper settings:
        3D CML = 4 x 4 x 4
        epsilon = 0.1
        PLM N = 64
        PLM mu = 4
        warm-up = 1000 iterations

    Controlled reproduction choices:
        it_rounds = 1
        deterministic public 64-node initial state
        Baker partition
        24-bit reversible diffusion modulus
        Eq. (10) stream interpretation
    """

    # --------------------------------------------------------
    # Source-paper settings
    # --------------------------------------------------------

    cml_r: int = 4
    cml_l: int = 4
    cml_u: int = 4

    epsilon: float = 0.1

    plm_segments: int = 64
    plm_mu: float = 4.0

    warmup: int = 1000

    # --------------------------------------------------------
    # Controlled reproduction choices
    # --------------------------------------------------------

    # Paper says IT times but does not publish IT.
    it_rounds: int = 1

    # Reversible interpretation of the 24-bit diffusion.
    diffusion_modulus: int = 1 << 24

    # Eq. (10) ambiguity.
    stream_mode: str = "fixed64_low32"

    # Public deterministic seed because the actual
    # 64 author initial conditions are not published.
    public_seed: bytes = (
        b"Liu-2026-controlled-reimplementation-v1"
    )

    # None -> equal four-strip partition.
    baker_partition: tuple[int, ...] | None = None


# ============================================================
# CONFIG VALIDATION
# ============================================================

def validate_config(
    config: Liu2026Config,
) -> None:

    if (
        config.cml_r,
        config.cml_l,
        config.cml_u,
    ) != (
        4,
        4,
        4,
    ):
        raise ValueError(
            "Liu-2026 profile requires a 4x4x4 CML."
        )

    if not (
        0.0
        <= config.epsilon
        <= 1.0
    ):
        raise ValueError(
            "epsilon must be in [0,1]."
        )

    if config.plm_segments <= 0:
        raise ValueError(
            "plm_segments must be positive."
        )

    if not (
        0.0
        < config.plm_mu
        <= 4.0
    ):
        raise ValueError(
            "plm_mu must be in (0,4]."
        )

    if config.warmup < 0:
        raise ValueError(
            "warmup must be >= 0."
        )

    if config.it_rounds < 1:
        raise ValueError(
            "it_rounds must be >= 1."
        )

    if (
        config.diffusion_modulus
        != 1 << 24
    ):
        raise ValueError(
            "Current controlled profile requires "
            "diffusion modulus = 2^24."
        )

    if config.stream_mode not in {
        "fixed64_low32",
        "mantissa_low32",
    }:
        raise ValueError(
            "Unknown stream_mode."
        )


# ============================================================
# REPRODUCTION PROFILE
# ============================================================

def reproduction_profile(
    config: Liu2026Config,
) -> dict:

    return {
        "name":
            "Liu et al. 2026 controlled reimplementation",

        "source_defined": {
            "cml_shape":
                [
                    config.cml_r,
                    config.cml_l,
                    config.cml_u,
                ],

            "epsilon":
                config.epsilon,

            "plm_segments":
                config.plm_segments,

            "plm_mu":
                config.plm_mu,

            "warmup_iterations":
                config.warmup,

            "combined_pixel_bits":
                24,

            "C0_hex":
                f"{C0:06x}",
        },

        "controlled_choices": {
            "it_rounds":
                config.it_rounds,

            "stream_mode":
                config.stream_mode,

            "diffusion_modulus":
                config.diffusion_modulus,

            "public_seed_hex":
                config.public_seed.hex(),

            "baker_partition":
                (
                    list(
                        config.baker_partition
                    )
                    if
                    config.baker_partition
                    is not None
                    else
                    "equal-quarter"
                ),

            "author_64_initial_states_available":
                False,
        },

        "claim_boundary":
            (
                "Independent controlled reimplementation; "
                "not author-exact reproduction."
            ),
    }


# ============================================================
# INITIAL 64 CML STATES
# ============================================================

def initial_states(
    config: Liu2026Config,
) -> np.ndarray:

    validate_config(
        config
    )

    node_count = (
        config.cml_r
        * config.cml_l
        * config.cml_u
    )

    raw = hashlib.shake_256(
        config.public_seed
    ).digest(
        node_count * 8
    )

    denominator = float(
        1 << 64
    )

    values = []

    for i in range(
        node_count
    ):

        integer = int.from_bytes(
            raw[
                i * 8:
                (i + 1) * 8
            ],
            byteorder="big",
            signed=False,
        )

        value = (
            integer
            + 0.5
        ) / denominator

        value = min(
            max(
                value,
                np.nextafter(
                    0.0,
                    1.0,
                ),
            ),
            np.nextafter(
                1.0,
                0.0,
            ),
        )

        values.append(
            value
        )

    return np.asarray(
        values,
        dtype=np.float64,
    ).reshape(
        config.cml_r,
        config.cml_l,
        config.cml_u,
    )


# ============================================================
# PIECE-WISE LOGISTIC MAP
# ============================================================

def plm(
    x: np.ndarray,
    segments: int = 64,
    mu: float = 4.0,
) -> np.ndarray:
    """
    Controlled implementation of Liu Eq. (3).

    The adjacent 1/N intervals alternate between:

        mu*u*(1-u)

    and:

        1 - mu*u*(1-u)
    """

    x = np.asarray(
        x,
        dtype=np.float64,
    )

    low = np.nextafter(
        0.0,
        1.0,
    )

    high = np.nextafter(
        1.0,
        0.0,
    )

    x = np.clip(
        x,
        low,
        high,
    )

    scaled = (
        x
        * float(
            segments
        )
    )

    interval = np.floor(
        scaled
    ).astype(
        np.int64
    )

    interval = np.clip(
        interval,
        0,
        segments - 1,
    )

    local = (
        scaled
        - interval
    )

    logistic = (
        mu
        * local
        * (
            1.0
            - local
        )
    )

    result = np.where(
        (
            interval
            % 2
        )
        == 0,
        logistic,
        1.0 - logistic,
    )

    return np.clip(
        result,
        low,
        high,
    )


# ============================================================
# 3-D CML — LIU EQ. (2)
# ============================================================

def cml_step(
    state: np.ndarray,
    config: Liu2026Config,
) -> np.ndarray:

    f = plm(
        state,
        segments=
            config.plm_segments,
        mu=
            config.plm_mu,
    )

    neighbor_sum = (
        np.roll(
            f,
            1,
            axis=0,
        )
        +
        np.roll(
            f,
            -1,
            axis=0,
        )
        +
        np.roll(
            f,
            1,
            axis=1,
        )
        +
        np.roll(
            f,
            -1,
            axis=1,
        )
        +
        np.roll(
            f,
            1,
            axis=2,
        )
        +
        np.roll(
            f,
            -1,
            axis=2,
        )
    )

    result = (
        (
            1.0
            - config.epsilon
        )
        * f
        +
        (
            config.epsilon
            / 6.0
        )
        * neighbor_sum
    )

    return np.clip(
        result,
        np.nextafter(
            0.0,
            1.0,
        ),
        np.nextafter(
            1.0,
            0.0,
        ),
    )


# ============================================================
# ALGORITHM E — STREAM EXTRACTION
# ============================================================

def extract_u32(
    state: np.ndarray,
    config: Liu2026Config,
) -> np.ndarray:

    values = np.asarray(
        state,
        dtype=np.float64,
    ).reshape(-1)

    # --------------------------------------------------------
    # Controlled interpretation of:
    #
    # x = 0.w1 ... w64
    # take low 32 bits
    # --------------------------------------------------------

    if (
        config.stream_mode
        == "fixed64_low32"
    ):

        two64 = float(
            1 << 64
        )

        safe_max = np.nextafter(
            two64,
            0.0,
        )

        scaled = np.floor(
            values
            * two64
        )

        scaled = np.minimum(
            scaled,
            safe_max,
        )

        integers = scaled.astype(
            np.uint64
        )

        return (
            integers
            & np.uint64(
                0xFFFFFFFF
            )
        ).astype(
            np.uint32
        )

    # --------------------------------------------------------
    # Alternative ambiguity profile
    # --------------------------------------------------------

    elif (
        config.stream_mode
        == "mantissa_low32"
    ):

        values = np.ascontiguousarray(
            values
        )

        bits = values.view(
            np.uint64
        )

        mantissa = (
            bits
            & np.uint64(
                (1 << 52)
                - 1
            )
        )

        return (
            mantissa
            & np.uint64(
                0xFFFFFFFF
            )
        ).astype(
            np.uint32
        )

    raise ValueError(
        "Unknown stream_mode."
    )


# ============================================================
# ALGORITHM E GENERATOR
# ============================================================

class AlgorithmE:

    def __init__(
        self,
        config: Liu2026Config,
    ):

        validate_config(
            config
        )

        self.config = config

        self.state = initial_states(
            config
        )

        # Source-specified transient removal.
        for _ in range(
            config.warmup
        ):

            self.state = cml_step(
                self.state,
                config,
            )

        self.buffer = bytearray()

    def _generate_iteration(
        self,
    ) -> None:

        self.state = cml_step(
            self.state,
            self.config,
        )

        words = extract_u32(
            self.state,
            self.config,
        )

        # Each node contributes one 32-bit word.
        for word in words:

            self.buffer.extend(
                int(
                    word
                ).to_bytes(
                    4,
                    byteorder="big",
                    signed=False,
                )
            )

    def take_bytes(
        self,
        count: int,
    ) -> bytes:

        if count < 0:
            raise ValueError(
                "count must be >= 0."
            )

        while (
            len(
                self.buffer
            )
            < count
        ):

            self._generate_iteration()

        output = bytes(
            self.buffer[
                :count
            ]
        )

        del self.buffer[
            :count
        ]

        return output

    def take_u24(
        self,
        count: int,
    ) -> np.ndarray:

        raw = self.take_bytes(
            count
            * 3
        )

        bytes_matrix = np.frombuffer(
            raw,
            dtype=np.uint8,
        ).reshape(
            count,
            3,
        ).astype(
            np.uint32
        )

        return (
            (
                bytes_matrix[
                    :,
                    0
                ]
                << 16
            )
            |
            (
                bytes_matrix[
                    :,
                    1
                ]
                << 8
            )
            |
            bytes_matrix[
                :,
                2
            ]
        ).astype(
            np.uint32
        )


# ============================================================
# RGB <-> 24-BIT COMBINED PIXEL
# ============================================================

def rgb_to_u24(
    image: np.ndarray,
) -> np.ndarray:

    image = np.asarray(
        image,
        dtype=np.uint8,
    )

    if (
        image.ndim != 3
        or
        image.shape[2] != 3
    ):
        raise ValueError(
            "Expected RGB uint8 image."
        )

    r = image[
        :,
        :,
        0
    ].astype(
        np.uint32
    )

    g = image[
        :,
        :,
        1
    ].astype(
        np.uint32
    )

    b = image[
        :,
        :,
        2
    ].astype(
        np.uint32
    )

    return (
        (
            r
            << 16
        )
        |
        (
            g
            << 8
        )
        |
        b
    ).astype(
        np.uint32
    )


def u24_to_rgb(
    image: np.ndarray,
) -> np.ndarray:

    image = (
        np.asarray(
            image,
            dtype=np.uint32,
        )
        & np.uint32(
            MASK24
        )
    )

    output = np.empty(
        (
            image.shape[0],
            image.shape[1],
            3,
        ),
        dtype=np.uint8,
    )

    output[
        :,
        :,
        0
    ] = (
        (
            image
            >> 16
        )
        & 0xFF
    ).astype(
        np.uint8
    )

    output[
        :,
        :,
        1
    ] = (
        (
            image
            >> 8
        )
        & 0xFF
    ).astype(
        np.uint8
    )

    output[
        :,
        :,
        2
    ] = (
        image
        & 0xFF
    ).astype(
        np.uint8
    )

    return output


# ============================================================
# GRAYSCALE -> RGB ADAPTER
# ============================================================

def grayscale_to_rgb(
    image: np.ndarray,
) -> np.ndarray:
    """
    Fair adapter for the grayscale DICOM benchmark.

        R = G = B = grayscale

    This keeps Liu's original 24-bit RGB processing path.
    """

    image = np.asarray(
        image,
        dtype=np.uint8,
    )

    if image.ndim != 2:
        raise ValueError(
            "Expected grayscale image."
        )

    return np.repeat(
        image[
            :,
            :,
            None
        ],
        3,
        axis=2,
    )


# ============================================================
# REVERSIBLE 24-BIT DIFFUSION
# ============================================================

def diffuse(
    image: np.ndarray,
    h: np.ndarray,
) -> np.ndarray:
    """
    Controlled reversible interpretation of Liu Eq. (15):

        C(k) =
            H(k)
            XOR
            ((I(k)+H(k)) mod 2^24)
            XOR
            C(k-1)

    C(0) = 0x454545
    """

    source = np.asarray(
        image,
        dtype=np.uint32,
    ).reshape(-1)

    h = np.asarray(
        h,
        dtype=np.uint32,
    ).reshape(-1)

    if (
        source.size
        != h.size
    ):
        raise ValueError(
            "Stream length mismatch."
        )

    output = np.empty_like(
        source
    )

    previous = int(
        C0
    )

    for i in range(
        source.size
    ):

        h_value = int(
            h[i]
        )

        plain_value = int(
            source[i]
        )

        added = (
            plain_value
            + h_value
        ) & MASK24

        current = (
            h_value
            ^ added
            ^ previous
        ) & MASK24

        output[i] = np.uint32(
            current
        )

        previous = current

    return output.reshape(
        image.shape
    )


def inverse_diffuse(
    cipher: np.ndarray,
    h: np.ndarray,
) -> np.ndarray:

    # Preserve original 2-D image shape BEFORE flattening.
    original_shape = np.asarray(
        cipher
    ).shape

    cipher_flat = np.asarray(
        cipher,
        dtype=np.uint32,
    ).reshape(-1)

    h_flat = np.asarray(
        h,
        dtype=np.uint32,
    ).reshape(-1)

    if (
        cipher_flat.size
        != h_flat.size
    ):
        raise ValueError(
            "Stream length mismatch."
        )

    output = np.empty_like(
        cipher_flat
    )

    previous = int(
        C0
    )

    for i in range(
        cipher_flat.size
    ):

        current = int(
            cipher_flat[i]
        )

        h_value = int(
            h_flat[i]
        )

        temporary = (
            current
            ^ h_value
            ^ previous
        )

        plain = (
            temporary
            - h_value
        ) & MASK24

        output[i] = np.uint32(
            plain
        )

        previous = current

    # Restore the ORIGINAL H x W shape.
    return output.reshape(
        original_shape
    )

# ============================================================
# BAKER PARTITION
# ============================================================

def default_partition(
    size: int,
) -> tuple[int, ...]:
    """
    Controlled reproduction choice.

    Equal four-part partition:

        64  -> 16,16,16,16
        256 -> 64,64,64,64
        512 -> 128,128,128,128
    """

    if (
        size <= 0
        or
        size % 4 != 0
    ):
        raise ValueError(
            "Default Baker partition requires "
            "image size divisible by 4."
        )

    width = (
        size
        // 4
    )

    return (
        width,
        width,
        width,
        width,
    )


def validate_partition(
    size: int,
    partition,
) -> tuple[int, ...]:

    partition = tuple(
        int(
            value
        )
        for value
        in partition
    )

    if not partition:
        raise ValueError(
            "Baker partition cannot be empty."
        )

    if (
        sum(
            partition
        )
        != size
    ):
        raise ValueError(
            "Baker widths must sum to image size."
        )

    for width in partition:

        if width <= 0:
            raise ValueError(
                "Baker widths must be positive."
            )

        if (
            size
            % width
            != 0
        ):
            raise ValueError(
                "Each Baker width must divide image size."
            )

    return partition


# ============================================================
# CORRECTED GENERALIZED DISCRETE BAKER MAP
# ============================================================

def baker_forward(
    image: np.ndarray,
    partition,
) -> np.ndarray:
    """
    Correct zero-based discrete interpretation of Liu Eq. (6).

    For strip i with width ni and starting row Ni:

        new_row =
            (N/ni)*(r-Ni)
            + s mod (N/ni)

        new_col =
            (ni/N) *
            (s - s mod (N/ni))
            + Ni

    Since:

        multiplier = N/ni

    the second coordinate simplifies exactly to:

        floor(s / multiplier) + Ni

    IMPORTANT:
    There must NOT be an extra multiplication by ni.
    """

    image = np.asarray(
        image
    )

    if (
        image.ndim != 2
        or
        image.shape[0]
        != image.shape[1]
    ):
        raise ValueError(
            "Baker map requires a square 2-D array."
        )

    size = int(
        image.shape[0]
    )

    partition = validate_partition(
        size,
        partition,
    )

    output = np.empty_like(
        image
    )

    columns = np.arange(
        size,
        dtype=np.int64,
    )[
        None,
        :
    ]

    start = 0

    for width in partition:

        multiplier = (
            size
            // width
        )

        local_rows = np.arange(
            width,
            dtype=np.int64,
        )[
            :,
            None
        ]

        # ----------------------------------------------------
        # First coordinate:
        #
        # (N/ni)*(r-Ni)
        # + s mod (N/ni)
        # ----------------------------------------------------

        new_rows = (
            multiplier
            * local_rows
            +
            (
                columns
                % multiplier
            )
        )

        # ----------------------------------------------------
        # CORRECTED SECOND COORDINATE
        #
        # (ni/N) *
        # (s - s mod (N/ni))
        # + Ni
        #
        # =
        #
        # floor(s / (N/ni))
        # + Ni
        #
        # NO multiplication by width here.
        # ----------------------------------------------------

        new_cols = (
            columns
            // multiplier
            + start
        )

        source = image[
            start:
            start + width,
            :
        ]

        output[
            new_rows,
            new_cols
        ] = source

        start += width

    return output


def baker_inverse(
    image: np.ndarray,
    partition,
) -> np.ndarray:
    """
    Exact inverse of baker_forward.
    """

    image = np.asarray(
        image
    )

    if (
        image.ndim != 2
        or
        image.shape[0]
        != image.shape[1]
    ):
        raise ValueError(
            "Baker inverse requires a square 2-D array."
        )

    size = int(
        image.shape[0]
    )

    partition = validate_partition(
        size,
        partition,
    )

    output = np.empty_like(
        image
    )

    columns = np.arange(
        size,
        dtype=np.int64,
    )[
        None,
        :
    ]

    start = 0

    for width in partition:

        multiplier = (
            size
            // width
        )

        local_rows = np.arange(
            width,
            dtype=np.int64,
        )[
            :,
            None
        ]

        new_rows = (
            multiplier
            * local_rows
            +
            (
                columns
                % multiplier
            )
        )

        # Corrected coordinate.
        new_cols = (
            columns
            // multiplier
            + start
        )

        output[
            start:
            start + width,
            :
        ] = image[
            new_rows,
            new_cols
        ]

        start += width

    return output


# ============================================================
# LIU-2026 CONTROLLED PIPELINE
# ============================================================

class Liu2026:

    def __init__(
        self,
        config: Liu2026Config | None = None,
    ):

        self.config = (
            config
            if config is not None
            else Liu2026Config()
        )

        validate_config(
            self.config
        )

    def _partition(
        self,
        size: int,
    ) -> tuple[int, ...]:

        if (
            self.config.baker_partition
            is None
        ):
            return default_partition(
                size
            )

        return validate_partition(
            size,
            self.config.baker_partition,
        )

    def _streams(
        self,
        pixel_count: int,
    ) -> list[np.ndarray]:

        generator = AlgorithmE(
            self.config
        )

        streams = []

        for _ in range(
            self.config.it_rounds
        ):

            streams.append(
                generator.take_u24(
                    pixel_count
                )
            )

        return streams

    # ========================================================
    # ENCRYPT
    # ========================================================

    def encrypt_rgb(
        self,
        image: np.ndarray,
    ):

        image = np.asarray(
            image,
            dtype=np.uint8,
        )

        if (
            image.ndim != 3
            or
            image.shape[2] != 3
            or
            image.shape[0]
            != image.shape[1]
        ):
            raise ValueError(
                "Expected square RGB uint8 image."
            )

        start_time = (
            time.perf_counter()
        )

        size = int(
            image.shape[0]
        )

        partition = self._partition(
            size
        )

        packed = rgb_to_u24(
            image
        )

        streams = self._streams(
            packed.size
        )

        state = packed.copy()

        # ----------------------------------------------------
        # IT rounds:
        # diffusion -> Baker
        # ----------------------------------------------------

        for round_index in range(
            self.config.it_rounds
        ):

            state = diffuse(
                state,
                streams[
                    round_index
                ],
            )

            state = baker_forward(
                state,
                partition,
            )

        # ----------------------------------------------------
        # Paper Eq. (16):
        #
        # v(1,1) =
        # v(1,1) XOR v(N,N)
        #
        # Self-inverse.
        # ----------------------------------------------------

        state[
            0,
            0
        ] = np.uint32(
            (
                int(
                    state[
                        0,
                        0
                    ]
                )
                ^
                int(
                    state[
                        -1,
                        -1
                    ]
                )
            )
            & MASK24
        )

        encrypted = u24_to_rgb(
            state
        )

        elapsed = (
            time.perf_counter()
            - start_time
        )

        metadata = {
            "shape":
                list(
                    image.shape
                ),

            "partition":
                list(
                    partition
                ),

            "it_rounds":
                self.config.it_rounds,

            "stream_mode":
                self.config.stream_mode,

            "profile":
                (
                    "independent-controlled-"
                    "reimplementation"
                ),

            "reproduction_profile":
                reproduction_profile(
                    self.config
                ),
        }

        return (
            encrypted,
            metadata,
            float(
                elapsed
            ),
        )

    # ========================================================
    # DECRYPT
    # ========================================================

    def decrypt_rgb(
        self,
        encrypted: np.ndarray,
        metadata: dict,
    ):

        encrypted = np.asarray(
            encrypted,
            dtype=np.uint8,
        )

        if (
            encrypted.ndim != 3
            or
            encrypted.shape[2] != 3
            or
            encrypted.shape[0]
            != encrypted.shape[1]
        ):
            raise ValueError(
                "Expected square RGB uint8 ciphertext."
            )

        start_time = (
            time.perf_counter()
        )

        partition = validate_partition(
            encrypted.shape[0],
            metadata[
                "partition"
            ],
        )

        state = rgb_to_u24(
            encrypted
        )

        streams = self._streams(
            state.size
        )

        # ----------------------------------------------------
        # Undo Eq. (16)
        # ----------------------------------------------------

        state[
            0,
            0
        ] = np.uint32(
            (
                int(
                    state[
                        0,
                        0
                    ]
                )
                ^
                int(
                    state[
                        -1,
                        -1
                    ]
                )
            )
            & MASK24
        )

        # ----------------------------------------------------
        # Reverse IT rounds:
        # inverse Baker -> inverse diffusion
        # ----------------------------------------------------

        for round_index in reversed(
            range(
                self.config.it_rounds
            )
        ):

            state = baker_inverse(
                state,
                partition,
            )

            state = inverse_diffuse(
                state,
                streams[
                    round_index
                ],
            )

        recovered = u24_to_rgb(
            state
        )

        elapsed = (
            time.perf_counter()
            - start_time
        )

        return (
            recovered,
            float(
                elapsed
            ),
        )


# ============================================================
# SELF TEST
# ============================================================

def self_test() -> dict:

    config = Liu2026Config(
        it_rounds=1,
        baker_partition=(
            16,
            16,
            16,
            16,
        ),
    )

    algorithm = Liu2026(
        config
    )

    rng = np.random.default_rng(
        2026
    )

    # ========================================================
    # RANDOM RGB TEST IMAGE
    # ========================================================

    image = rng.integers(
        0,
        256,
        size=(
            64,
            64,
            3,
        ),
        dtype=np.uint8,
    )

    packed = rgb_to_u24(
        image
    )

    # ========================================================
    # TEST 1 — PLM RANGE
    # ========================================================

    values = np.linspace(
        0.001,
        0.999,
        1000,
        dtype=np.float64,
    )

    mapped = plm(
        values,
        config.plm_segments,
        config.plm_mu,
    )

    plm_valid = bool(
        np.all(
            mapped > 0.0
        )
        and
        np.all(
            mapped < 1.0
        )
    )

    # ========================================================
    # TEST 2 — BAKER EXACT INVERSE
    # ========================================================

    baker_cipher = baker_forward(
        packed,
        config.baker_partition,
    )

    baker_plain = baker_inverse(
        baker_cipher,
        config.baker_partition,
    )

    baker_exact = bool(
        np.array_equal(
            packed,
            baker_plain,
        )
    )

    # ========================================================
    # TEST 3 — DIFFUSION EXACT INVERSE
    # ========================================================

    stream = AlgorithmE(
        config
    ).take_u24(
        packed.size
    )

    diffused = diffuse(
        packed,
        stream,
    )

    undiffused = inverse_diffuse(
        diffused,
        stream,
    )

    diffusion_exact = bool(
        np.array_equal(
            packed,
            undiffused,
        )
    )

    # ========================================================
    # TEST 4 — FULL PIPELINE
    # ========================================================

    (
        encrypted,
        metadata,
        encryption_time,
    ) = algorithm.encrypt_rgb(
        image
    )

    (
        recovered,
        decryption_time,
    ) = algorithm.decrypt_rgb(
        encrypted,
        metadata,
    )

    full_exact = bool(
        np.array_equal(
            image,
            recovered,
        )
    )

    # ========================================================
    # TEST 5 — DETERMINISTIC REPEAT
    # ========================================================

    (
        encrypted_2,
        _,
        _,
    ) = algorithm.encrypt_rgb(
        image
    )

    deterministic = bool(
        np.array_equal(
            encrypted,
            encrypted_2,
        )
    )

    # ========================================================
    # TEST 6 — GRAYSCALE ADAPTER
    # ========================================================

    gray = rng.integers(
        0,
        256,
        size=(
            64,
            64,
        ),
        dtype=np.uint8,
    )

    gray_rgb = grayscale_to_rgb(
        gray
    )

    adapter_exact = bool(
        np.array_equal(
            gray,
            gray_rgb[
                :,
                :,
                0
            ],
        )
        and
        np.array_equal(
            gray,
            gray_rgb[
                :,
                :,
                1
            ],
        )
        and
        np.array_equal(
            gray,
            gray_rgb[
                :,
                :,
                2
            ],
        )
    )

    # ========================================================
    # RESULT
    # ========================================================

    all_pass = bool(
        plm_valid
        and
        baker_exact
        and
        diffusion_exact
        and
        full_exact
        and
        deterministic
        and
        adapter_exact
    )

    ciphertext_hash = hashlib.sha256(
        encrypted.tobytes(
            order="C"
        )
    ).hexdigest()

    result = {
        "plm_valid":
            plm_valid,

        "baker_exact":
            baker_exact,

        "diffusion_exact":
            diffusion_exact,

        "full_exact":
            full_exact,

        "deterministic":
            deterministic,

        "adapter_exact":
            adapter_exact,

        "all_pass":
            all_pass,

        "encryption_time":
            encryption_time,

        "decryption_time":
            decryption_time,

        "ciphertext_sha256":
            ciphertext_hash,
    }

    # ========================================================
    # PRINT
    # ========================================================

    print(
        "=" * 96
    )

    print(
        "LIU 2026 CONTROLLED REIMPLEMENTATION — SELF TEST"
    )

    print(
        "=" * 96
    )

    print(
        f"PLM valid range:              {plm_valid}"
    )

    print(
        f"Baker inverse exact:          {baker_exact}"
    )

    print(
        f"Diffusion inverse exact:      {diffusion_exact}"
    )

    print(
        f"Full RGB exact recovery:      {full_exact}"
    )

    print(
        f"Deterministic repeat:         {deterministic}"
    )

    print(
        f"Grayscale->RGB adapter exact: {adapter_exact}"
    )

    print()

    print(
        f"Encryption time:              "
        f"{encryption_time:.6f} s"
    )

    print(
        f"Decryption time:              "
        f"{decryption_time:.6f} s"
    )

    print()

    print(
        "Ciphertext SHA-256:"
    )

    print(
        ciphertext_hash
    )

    print()

    print(
        f"ALL REQUIRED TESTS PASS:      {all_pass}"
    )

    print()

    print(
        "IMPORTANT:"
    )

    print(
        "This is an independent controlled reimplementation,"
    )

    print(
        "not an author-exact reproduction."
    )

    return result


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    self_test()