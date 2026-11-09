"""Spectral QA gate.

Physiological hand tremor concentrates energy in the 8–12 Hz band. Because
its amplitude is small, time-domain filters typically leave it alone, but it
excites low-level controllers on real hardware and shows up as jitter in
end-effector traces. This module performs two operations:

1. **PSD computation** — Welch estimator on each channel of a trajectory.
2. **Spectral matching** — design a zero-phase FIR filter whose magnitude
   response pushes the synthetic trajectory's PSD toward the median PSD of a
   reference bank (``D_real``), with an explicit notch covering the tremor
   band.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.signal import firwin2, filtfilt, welch

# Tremor band for typical humans performing manipulation tasks
TREMOR_BAND_HZ = (8.0, 12.0)


def compute_psd(
    trajectory: NDArray, fs: float, nperseg: int | None = None
) -> tuple[NDArray, NDArray]:
    """Welch PSD per channel.

    Returns ``(freqs, psd)`` where ``freqs`` has shape ``(F,)`` and ``psd``
    has shape ``(F, C)``.
    """
    x = np.asarray(trajectory, dtype=np.float64)
    if x.ndim == 1:
        x = x[:, None]
    if nperseg is None:
        nperseg = min(128, x.shape[0])
    freqs, psds = welch(x, fs=fs, nperseg=nperseg, axis=0)
    return freqs, psds


def aggregate_reference_psd(
    trajectories: list[NDArray], fs: float, nperseg: int | None = None
) -> tuple[NDArray, NDArray]:
    """Median PSD across a bank of reference trajectories.

    Trajectories may have different lengths; we compute per-trajectory PSD
    then take the elementwise median.
    """
    psds = []
    freqs_ref = None
    for traj in trajectories:
        freqs, psd = compute_psd(traj, fs=fs, nperseg=nperseg)
        if freqs_ref is None:
            freqs_ref = freqs
        elif len(freqs) != len(freqs_ref):
            continue  # skip shapes that don't match nperseg
        psds.append(psd)
    if not psds:
        raise ValueError("empty reference bank")
    stack = np.stack(psds, axis=0)  # (N, F, C)
    return freqs_ref, np.median(stack, axis=0)


def _design_match_filter(
    freqs_target: NDArray,
    psd_target: NDArray,
    freqs_source: NDArray,
    psd_source: NDArray,
    fs: float,
    ntaps: int = 65,
    tremor_band: tuple[float, float] = TREMOR_BAND_HZ,
    tremor_attenuation_db: float = 20.0,
) -> NDArray:
    """Design an FIR filter whose magnitude approximates
    ``sqrt(psd_target / psd_source)`` with an explicit tremor notch.
    """
    # Align on the source frequency grid
    psd_t_on_src = np.interp(freqs_source, freqs_target, psd_target)
    ratio = np.sqrt(np.clip(psd_t_on_src, 1e-12, None) / np.clip(psd_source, 1e-12, None))
    ratio = np.clip(ratio, 0.1, 10.0)  # safety

    # Apply tremor notch in linear gain
    lo, hi = tremor_band
    notch_gain = 10.0 ** (-tremor_attenuation_db / 20.0)
    band_mask = (freqs_source >= lo) & (freqs_source <= hi)
    ratio = np.where(band_mask, np.minimum(ratio, notch_gain), ratio)

    # Build the filter via firwin2 on normalized frequency
    nyq = fs / 2.0
    fn = np.clip(freqs_source / nyq, 0.0, 1.0)
    # firwin2 requires strictly monotonic frequencies and 0, 1 endpoints
    fn, idx = np.unique(fn, return_index=True)
    gain = ratio[idx]
    if fn[0] > 0.0:
        fn = np.concatenate([[0.0], fn])
        gain = np.concatenate([[gain[0]], gain])
    if fn[-1] < 1.0:
        fn = np.concatenate([fn, [1.0]])
        gain = np.concatenate([gain, [gain[-1]]])

    return firwin2(ntaps, fn, gain)


def spectral_match(
    trajectory: NDArray,
    reference_bank: list[NDArray],
    fs: float,
    ntaps: int = 65,
    tremor_band: tuple[float, float] = TREMOR_BAND_HZ,
    tremor_attenuation_db: float = 20.0,
) -> NDArray:
    """Apply spectral matching + tremor notching to an input trajectory.

    The filter is designed once from the aggregate reference PSD and then
    applied via ``filtfilt`` (zero-phase) per channel.
    """
    x = np.asarray(trajectory, dtype=np.float64)
    if x.ndim == 1:
        x = x[:, None]
    if x.shape[0] < ntaps * 3:
        # Filter is too large for this sequence — skip matching, just notch
        return _notch_only(x, fs, tremor_band, tremor_attenuation_db)

    # Reference
    nperseg = min(128, x.shape[0])
    fr, psd_ref = aggregate_reference_psd(reference_bank, fs=fs, nperseg=nperseg)
    fs_f, psd_src = compute_psd(x, fs=fs, nperseg=nperseg)

    # One filter per channel because PSDs differ across axes
    out = np.empty_like(x)
    for c in range(x.shape[1]):
        taps = _design_match_filter(
            fr,
            psd_ref[:, c] if psd_ref.ndim > 1 else psd_ref,
            fs_f,
            psd_src[:, c],
            fs=fs,
            ntaps=ntaps,
            tremor_band=tremor_band,
            tremor_attenuation_db=tremor_attenuation_db,
        )
        out[:, c] = filtfilt(taps, [1.0], x[:, c], padlen=min(len(taps) * 3, x.shape[0] - 1))
    return out if out.shape[1] > 1 else out[:, 0]


def _notch_only(
    x: NDArray, fs: float, band: tuple[float, float], atten_db: float
) -> NDArray:
    """Simple band-stop FIR when the sequence is too short for full matching."""
    nyq = fs / 2.0
    lo = max(band[0] / nyq, 0.01)
    hi = min(band[1] / nyq, 0.99)
    # Bandstop via firwin2
    freqs = np.array([0.0, max(lo - 0.02, 0.01), lo, hi, min(hi + 0.02, 0.99), 1.0])
    freqs = np.unique(np.clip(freqs, 0.0, 1.0))
    gain_at_freqs = np.interp(
        freqs,
        [0.0, lo, hi, 1.0],
        [1.0, 10.0 ** (-atten_db / 20.0), 10.0 ** (-atten_db / 20.0), 1.0],
    )
    ntaps = 31
    taps = firwin2(ntaps, freqs, gain_at_freqs)
    if x.shape[0] < ntaps * 3:
        return x
    out = np.empty_like(x)
    for c in range(x.shape[1]):
        out[:, c] = filtfilt(taps, [1.0], x[:, c], padlen=min(ntaps * 3, x.shape[0] - 1))
    return out


def tremor_band_energy(
    trajectory: NDArray, fs: float, band: tuple[float, float] = TREMOR_BAND_HZ
) -> float:
    """Fraction of total signal energy falling in the tremor band.

    Useful as a diagnostic: values near 0 indicate a clean trajectory; human
    hand motion typically shows values around 0.05–0.15.
    """
    freqs, psd = compute_psd(trajectory, fs=fs)
    band_mask = (freqs >= band[0]) & (freqs <= band[1])
    total = np.trapz(psd.sum(axis=-1), freqs)
    band_e = np.trapz(psd[band_mask].sum(axis=-1), freqs[band_mask])
    return float(band_e / max(total, 1e-12))
