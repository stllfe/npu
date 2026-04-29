"""
exp2 LUT generator/approximation for RK3588-style NVDLA tables.

- Table: 1026 entries covering [X_MIN, X_MAX] (default [-2, 2]).
- Indexing: linear in x using a scale+shift (INDEX_SCALE, INDEX_SHIFT).
- Output encoding: Q14 biased by default (raw = (y + 1) * 16384).

Hardware note:
The NPU path quantizes multiple stages (input fp16, BN scale/bias, LUT index, and
LUT interpolation on 16-bit integer codes). The output is effectively limited to
~1/128 resolution even with a perfect LUT. This script does float-domain
interpolation over decoded LUT values, so it can report smaller errors than the
hardware path. To match hardware error, simulate fp16 input/scale and integer
LUT interpolation instead of float interpolation.
"""

from __future__ import annotations

import numpy as np

LUT_SIZE = 1026
X_MIN = -2.0
X_MAX = 2.0

INDEX_SHIFT = 5
INDEX_SCALE = (1 << INDEX_SHIFT) * (LUT_SIZE - 1) / (X_MAX - X_MIN)

# Options: "q14_biased", "q015", "fp16"
OUTPUT_FORMAT = "fp16" if X_MAX > 0.0 else "q14_biased"


def exp2_fn(x: float) -> float:
    return float(np.exp2(x))


def _encode_q14_biased(y: float) -> int:
    q = int(np.round((y + 1.0) * 16384.0))
    if q < 0:
        q = 0
    if q > 32767:
        q = 32767
    return q


def _decode_q14_biased(raw: int) -> float:
    return float(raw) / 16384.0 - 1.0


def _encode_q015(y: float) -> int:
    q = int(np.round(y * (1 << 15)))
    if q < 0:
        q = 0
    if q > 32767:
        q = 32767
    return q


def _decode_q015(raw: int) -> float:
    return float(raw) / float(1 << 15)


def _encode_fp16(y: float) -> int:
    return int(np.float16(y).view(np.uint16))


def _decode_fp16(raw: int) -> float:
    return float(np.array(raw, dtype=np.uint16).view(np.float16))


def _encode(y: float) -> int:
    if OUTPUT_FORMAT == "q14_biased":
        return _encode_q14_biased(y)
    if OUTPUT_FORMAT == "q015":
        return _encode_q015(y)
    if OUTPUT_FORMAT == "fp16":
        return _encode_fp16(y)
    raise ValueError(f"Unknown OUTPUT_FORMAT: {OUTPUT_FORMAT}")


def _decode(raw: int) -> float:
    if OUTPUT_FORMAT == "q14_biased":
        return _decode_q14_biased(raw)
    if OUTPUT_FORMAT == "q015":
        return _decode_q015(raw)
    if OUTPUT_FORMAT == "fp16":
        return _decode_fp16(raw)
    raise ValueError(f"Unknown OUTPUT_FORMAT: {OUTPUT_FORMAT}")


def _generate_lut() -> np.ndarray:
    step = (X_MAX - X_MIN) / (LUT_SIZE - 1)
    vals = np.zeros(LUT_SIZE, dtype=np.uint16)
    for i in range(LUT_SIZE):
        x = X_MIN + i * step
        y = exp2_fn(x)
        vals[i] = np.uint16(_encode(y))
    return vals


exp2_lut = _generate_lut()
exp2_lut_float = np.array([_decode(int(v)) for v in exp2_lut], dtype=np.float64)


def _index_and_frac(x: float) -> tuple[int, float]:
    t = float(np.clip(x, X_MIN, X_MAX))
    idx_f = (t - X_MIN) * INDEX_SCALE / float(1 << INDEX_SHIFT)
    if idx_f >= LUT_SIZE - 1:
        return LUT_SIZE - 2, 1.0
    if idx_f <= 0.0:
        return 0, 0.0
    base = int(np.floor(idx_f))
    frac = float(idx_f - base)
    return base, frac


def hardware_exp2(x_float: float) -> float:
    base, frac = _index_and_frac(x_float)
    y0 = exp2_lut_float[base]
    y1 = exp2_lut_float[base + 1]
    return float(y0 + (y1 - y0) * frac)


if __name__ == "__main__":
    print(
        f"Generated exp2 LUT with {len(exp2_lut)} entries, "
        f"range [{X_MIN}, {X_MAX}], OUTPUT_FORMAT={OUTPUT_FORMAT}"
    )
    print(f"INDEX_SCALE={INDEX_SCALE:.4f} INDEX_SHIFT={INDEX_SHIFT}")

    samples = np.linspace(X_MIN, X_MAX, num=4096, dtype=np.float64)
    expected = np.exp2(samples)
    hw_vals = np.array([hardware_exp2(float(x)) for x in samples], dtype=np.float64)

    diffs = np.abs(hw_vals - expected)
    rel = diffs / np.maximum(np.abs(expected), 1e-12)
    atol = 1e-6
    rtol = 1e-3
    try:
        np.testing.assert_allclose(hw_vals, expected, atol=atol, rtol=rtol)
        passes = True
    except AssertionError:
        passes = False

    print(f"Max abs error: {diffs.max():.8f}")
    print(f"Max rel error: {rel.max():.8f}")
    print(f"All within atol={atol} rtol={rtol}: {'YES' if passes else 'NO'}")
