from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray


# =========================================================
# DEFAULT PARAMETERS
# =========================================================

CHAOS_PARAMS: dict[str, float] = {
    "x0": 0.3,
    "y0": 0.3,
    "z0": 0.3,
    "u0": 0.3,
    "a": 35.0,
    "b": 3.0,
    "c": 12.0,
    "gamma": 28.0,
    "d": 0.5,
    "alpha": 0.97,
    "h": 0.001,
}


FloatArray = NDArray[np.float64]
UInt8Array = NDArray[np.uint8]


# =========================================================
# CROSS-PLATFORM CANONICAL PROFILE
# =========================================================

CANONICAL_STATE_DECIMALS = 9
CANONICAL_STATE_PROFILE = "CFP9-v1"


# =========================================================
# FRACTIONAL-ORDER 4D CHEN SYSTEM
# =========================================================

@dataclass(frozen=True)
class ChenParameters:
    """Parameters of the fractional-order 4D Chen system."""

    x0: float = 0.3
    y0: float = 0.3
    z0: float = 0.3
    u0: float = 0.3

    a: float = 35.0
    b: float = 3.0
    c: float = 12.0
    gamma: float = 28.0
    d: float = 0.5

    alpha: float = 0.97
    h: float = 0.001

    def validate(self) -> None:
        if not 0.0 < self.alpha <= 1.0:
            raise ValueError(
                "alpha must satisfy 0 < alpha <= 1."
            )

        if self.h <= 0.0:
            raise ValueError(
                "Integration step h must be positive."
            )

        values = (
            self.x0,
            self.y0,
            self.z0,
            self.u0,
            self.a,
            self.b,
            self.c,
            self.gamma,
            self.d,
            self.alpha,
            self.h,
        )

        if not all(
            math.isfinite(value)
            for value in values
        ):
            raise ValueError(
                "All system parameters must be finite."
            )


class FractionalChen4DGenerator:
    """
    Fractional-order 4D Chen-system generator.

    System:

        D^alpha x = a(y - x) + u
        D^alpha y = gamma*x - x*z + c*y
        D^alpha z = x*y - b*z
        D^alpha u = y*z + d*u

    The numerical solution uses the fractional
    Adams-Bashforth-Moulton predictor-corrector method.
    """

    def __init__(
        self,
        parameters: ChenParameters,
    ) -> None:
        parameters.validate()
        self.params = parameters

    def _system_function(
        self,
        state: FloatArray,
    ) -> FloatArray:
        """Evaluate the Chen-system right-hand side."""

        x, y, z, u = state
        p = self.params

        return np.array(
            [
                p.a * (y - x) + u,
                p.gamma * x - x * z + p.c * y,
                x * y - p.b * z,
                y * z + p.d * u,
            ],
            dtype=np.float64,
        )

    def generate(
        self,
        n: int = 1000,
        discard: int = 100,
        divergence_limit: float = 1.0e12,
    ) -> tuple[
        FloatArray,
        FloatArray,
        FloatArray,
        FloatArray,
    ]:
        """
        Generate the four fractional Chen state sequences.

        Parameters
        ----------
        n:
            Number of retained samples.

        discard:
            Number of transient samples discarded.

        divergence_limit:
            Maximum permitted absolute state value.
        """

        if n <= 0:
            raise ValueError(
                "n must be greater than zero."
            )

        if discard < 0:
            raise ValueError(
                "discard must be non-negative."
            )

        total = n + discard

        p = self.params
        alpha = p.alpha

        h_alpha = (
            p.h ** alpha
        )

        states = np.zeros(
            (total, 4),
            dtype=np.float64,
        )

        derivatives = np.zeros(
            (total, 4),
            dtype=np.float64,
        )

        states[0] = np.array(
            [
                p.x0,
                p.y0,
                p.z0,
                p.u0,
            ],
            dtype=np.float64,
        )

        derivatives[0] = (
            self._system_function(
                states[0]
            )
        )

        initial_state = (
            states[0].copy()
        )

        predictor_scale = (
            h_alpha
            / math.gamma(
                alpha + 1.0
            )
        )

        corrector_scale = (
            h_alpha
            / math.gamma(
                alpha + 2.0
            )
        )

        for step_index in range(
            total - 1
        ):
            n_index = step_index

            # ---------------------------------------------
            # Predictor
            # ---------------------------------------------

            j = np.arange(
                n_index + 1,
                dtype=np.float64,
            )

            predictor_weights = (
                (
                    n_index
                    + 1.0
                    - j
                )
                ** alpha
                -
                (
                    n_index
                    - j
                )
                ** alpha
            )

            predictor_sum = (
                predictor_weights
                @ derivatives[
                    : n_index + 1
                ]
            )

            predicted_state = (
                initial_state
                + predictor_scale
                * predictor_sum
            )

            predicted_derivative = (
                self._system_function(
                    predicted_state
                )
            )

            # ---------------------------------------------
            # Corrector
            # ---------------------------------------------

            corrector_weights = np.empty(
                n_index + 1,
                dtype=np.float64,
            )

            corrector_weights[0] = (
                n_index
                ** (
                    alpha + 1.0
                )
                -
                (
                    n_index - alpha
                )
                * (
                    n_index + 1.0
                )
                ** alpha
            )

            if n_index >= 1:
                j_inner = np.arange(
                    1,
                    n_index + 1,
                    dtype=np.float64,
                )

                distance = (
                    n_index
                    - j_inner
                )

                corrector_weights[1:] = (
                    (
                        distance + 2.0
                    )
                    ** (
                        alpha + 1.0
                    )
                    +
                    distance
                    ** (
                        alpha + 1.0
                    )
                    -
                    2.0
                    * (
                        distance + 1.0
                    )
                    ** (
                        alpha + 1.0
                    )
                )

            corrector_sum = (
                corrector_weights
                @ derivatives[
                    : n_index + 1
                ]
            )

            corrected_state = (
                initial_state
                + corrector_scale
                * (
                    predicted_derivative
                    + corrector_sum
                )
            )

            if not np.all(
                np.isfinite(
                    corrected_state
                )
            ):
                raise FloatingPointError(
                    "The fractional Chen system produced "
                    "a non-finite value at iteration "
                    f"{step_index + 1}."
                )

            if (
                np.max(
                    np.abs(
                        corrected_state
                    )
                )
                > divergence_limit
            ):
                raise FloatingPointError(
                    "The fractional Chen system diverged at "
                    f"iteration {step_index + 1}."
                )

            states[
                step_index + 1
            ] = corrected_state

            derivatives[
                step_index + 1
            ] = self._system_function(
                corrected_state
            )

        retained = states[
            discard:
            discard + n
        ]

        return (
            retained[:, 0].copy(),
            retained[:, 1].copy(),
            retained[:, 2].copy(),
            retained[:, 3].copy(),
        )


# =========================================================
# PARAMETER AND SALT HANDLING
# =========================================================

def _parameters_from_dictionary(
    chaos_params: dict[str, Any] | None,
    salt: int | str | bytes = 0,
) -> ChenParameters:
    """
    Build Chen parameters and deterministic salt-dependent
    perturbations of the four initial conditions.
    """

    source = (
        CHAOS_PARAMS
        if chaos_params is None
        else chaos_params
    )

    def get_float(
        name: str,
        default: float,
    ) -> float:
        return float(
            source.get(
                name,
                default,
            )
        )

    if isinstance(
        salt,
        bytes,
    ):
        salt_bytes = salt
    else:
        salt_bytes = str(
            salt
        ).encode(
            "utf-8"
        )

    digest = hashlib.sha256(
        b"FractionalChen4D-Salt-v1|"
        + salt_bytes
    ).digest()

    offsets: list[float] = []

    for start in range(
        0,
        32,
        8,
    ):
        integer = int.from_bytes(
            digest[
                start:
                start + 8
            ],
            byteorder="big",
            signed=False,
        )

        normalized = (
            integer
            / float(
                2**64 - 1
            )
        )

        offsets.append(
            normalized - 0.5
        )

    perturbation_scale = 1.0e-8

    return ChenParameters(
        x0=(
            get_float(
                "x0",
                0.3,
            )
            + perturbation_scale
            * offsets[0]
        ),

        y0=(
            get_float(
                "y0",
                0.3,
            )
            + perturbation_scale
            * offsets[1]
        ),

        z0=(
            get_float(
                "z0",
                0.3,
            )
            + perturbation_scale
            * offsets[2]
        ),

        u0=(
            get_float(
                "u0",
                0.3,
            )
            + perturbation_scale
            * offsets[3]
        ),

        a=get_float(
            "a",
            35.0,
        ),

        b=get_float(
            "b",
            3.0,
        ),

        c=get_float(
            "c",
            12.0,
        ),

        gamma=get_float(
            "gamma",
            28.0,
        ),

        d=get_float(
            "d",
            0.5,
        ),

        alpha=get_float(
            "alpha",
            0.97,
        ),

        h=get_float(
            "h",
            0.001,
        ),
    )


def _key_context(
    parameters: ChenParameters,
    salt: int | str | bytes,
) -> bytes:
    """
    Create the deterministic domain-separated
    key-stream whitening context.
    """

    if isinstance(
        salt,
        bytes,
    ):
        salt_bytes = salt
    else:
        salt_bytes = str(
            salt
        ).encode(
            "utf-8"
        )

    parameter_text = (
        f"{parameters.x0:.17g}|"
        f"{parameters.y0:.17g}|"
        f"{parameters.z0:.17g}|"
        f"{parameters.u0:.17g}|"
        f"{parameters.a:.17g}|"
        f"{parameters.b:.17g}|"
        f"{parameters.c:.17g}|"
        f"{parameters.gamma:.17g}|"
        f"{parameters.d:.17g}|"
        f"{parameters.alpha:.17g}|"
        f"{parameters.h:.17g}"
    ).encode(
        "ascii"
    )

    return hashlib.sha256(
        b"FractionalChen4D-Key-v1|"
        + parameter_text
        + b"|"
        + salt_bytes
    ).digest()


# =========================================================
# CROSS-PLATFORM CANONICAL STATE SERIALIZATION
# =========================================================

def _canonical_state_bytes(
    state,
) -> bytes:
    """
    Serialize one retained 4-D Chen state using the
    cross-platform CFP10 canonical profile.

    Raw IEEE-754 state bytes are not used because the
    cross-platform tests showed very small numerical
    differences between Windows, Linux, and macOS.

    Each state component is formatted with exactly
    10 digits after the decimal point.
    """

    components: list[str] = []

    for value in state:
        numeric_value = float(
            value
        )

        if not math.isfinite(
            numeric_value
        ):
            raise FloatingPointError(
                "Non-finite Chen state encountered "
                "during canonical serialization."
            )

        formatted = (
            f"{numeric_value:.{CANONICAL_STATE_DECIMALS}f}"
        )

        negative_zero = (
            "-"
            + "0."
            + (
                "0"
                * CANONICAL_STATE_DECIMALS
            )
        )

        if formatted == negative_zero:
            formatted = (
                "0."
                + (
                    "0"
                    * CANONICAL_STATE_DECIMALS
                )
            )

        components.append(
            formatted
        )

    canonical_text = (
        "ChenState-"
        + CANONICAL_STATE_PROFILE
        + "|"
        + "|".join(
            components
        )
    )

    return canonical_text.encode(
        "ascii"
    )


# =========================================================
# CHAOTIC BYTE STREAM
# =========================================================

def generate_chaotic_byte_sequence(
    length: int,
    chaos_params: dict[str, Any] | None = None,
    salt: int | str | bytes = 0,
    discard: int = 100,
) -> UInt8Array:
    """
    Generate a deterministic Chen-based byte stream.

    Each retained 4-D state is first serialized using
    CFP10-v1 and is then whitened using SHA-256.
    """

    if length <= 0:
        raise ValueError(
            "length must be greater than zero."
        )

    parameters = (
        _parameters_from_dictionary(
            chaos_params,
            salt,
        )
    )

    generator = (
        FractionalChen4DGenerator(
            parameters
        )
    )

    state_count = math.ceil(
        length / 32
    )

    x, y, z, u = generator.generate(
        n=state_count,
        discard=discard,
    )

    context = _key_context(
        parameters,
        salt,
    )

    output = bytearray()

    for index, state in enumerate(
        zip(
            x,
            y,
            z,
            u,
        )
    ):
        canonical_state = (
            _canonical_state_bytes(
                state
            )
        )

        digest = hashlib.sha256(
            context
            + index.to_bytes(
                8,
                byteorder="big",
                signed=False,
            )
            + canonical_state
        ).digest()

        output.extend(
            digest
        )

    return np.frombuffer(
        bytes(
            output[:length]
        ),
        dtype=np.uint8,
    ).copy()


def generate_chaotic_binary_sequence(
    length: int,
    chaos_params: dict[str, Any] | None = None,
    salt: int | str | bytes = 0,
    discard: int = 100,
) -> UInt8Array:
    """
    Generate exactly `length` chaotic bits.
    """

    if length <= 0:
        raise ValueError(
            "length must be greater than zero."
        )

    byte_length = math.ceil(
        length / 8
    )

    byte_sequence = (
        generate_chaotic_byte_sequence(
            length=byte_length,
            chaos_params=chaos_params,
            salt=salt,
            discard=discard,
        )
    )

    bits = np.unpackbits(
        byte_sequence
    )

    return bits[
        :length
    ].astype(
        np.uint8,
        copy=True,
    )


# =========================================================
# VALIDATION UTILITIES
# =========================================================

def byte_entropy(
    sequence: UInt8Array,
) -> float:
    """Compute Shannon entropy of a byte sequence."""

    histogram = np.bincount(
        sequence,
        minlength=256,
    ).astype(
        np.float64
    )

    probabilities = (
        histogram
        / histogram.sum()
    )

    probabilities = probabilities[
        probabilities > 0.0
    ]

    return float(
        -np.sum(
            probabilities
            * np.log2(
                probabilities
            )
        )
    )


def sequence_difference_rate(
    first: UInt8Array,
    second: UInt8Array,
) -> float:
    """
    Percentage of byte positions that differ.
    """

    if first.shape != second.shape:
        raise ValueError(
            "Sequences must have the same shape."
        )

    return float(
        np.mean(
            first != second
        )
        * 100.0
    )


# =========================================================
# SELF TEST
# =========================================================

if __name__ == "__main__":
    print(
        "=" * 78
    )

    print(
        "FRACTIONAL-ORDER 4D CHEN "
        "CROSS-PLATFORM GENERATOR TEST"
    )

    print(
        "=" * 78
    )

    print(
        "Canonical profile      :",
        CANONICAL_STATE_PROFILE,
    )

    print(
        "Canonical decimals     :",
        CANONICAL_STATE_DECIMALS,
    )

    test_length = 4096

    key_1 = generate_chaotic_byte_sequence(
        length=test_length,
        chaos_params=CHAOS_PARAMS,
        salt=1,
    )

    key_1_repeat = (
        generate_chaotic_byte_sequence(
            length=test_length,
            chaos_params=CHAOS_PARAMS,
            salt=1,
        )
    )

    key_2 = generate_chaotic_byte_sequence(
        length=test_length,
        chaos_params=CHAOS_PARAMS,
        salt=2,
    )

    sensitive_params = dict(
        CHAOS_PARAMS
    )

    sensitive_params[
        "x0"
    ] += 1.0e-12

    key_sensitive = (
        generate_chaotic_byte_sequence(
            length=test_length,
            chaos_params=sensitive_params,
            salt=1,
        )
    )

    print(
        "Generated bytes       :",
        len(
            key_1
        ),
    )

    print(
        "First 20 bytes        :",
        key_1[:20],
    )

    print(
        "Deterministic repeat  :",
        np.array_equal(
            key_1,
            key_1_repeat,
        ),
    )

    print(
        "Different salt equal  :",
        np.array_equal(
            key_1,
            key_2,
        ),
    )

    print(
        "Salt difference rate  :",
        f"{sequence_difference_rate(key_1, key_2):.4f}%",
    )

    print(
        "x0 + 1e-12 equal      :",
        np.array_equal(
            key_1,
            key_sensitive,
        ),
    )

    print(
        "Key sensitivity rate  :",
        f"{sequence_difference_rate(key_1, key_sensitive):.4f}%",
    )

    print(
        "Byte entropy          :",
        f"{byte_entropy(key_1):.6f} bits",
    )

    binary = (
        generate_chaotic_binary_sequence(
            length=100,
            chaos_params=CHAOS_PARAMS,
            salt=3,
        )
    )

    print(
        "First 100 bits        :",
        binary,
    )

    print(
        "=" * 78
    )

    print(
        "Generator test completed."
    )
