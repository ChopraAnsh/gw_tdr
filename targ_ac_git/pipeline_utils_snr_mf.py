import os
import numpy as np
import subprocess
import sys
import h5py
from gwpy.timeseries import TimeSeries

try:
    from .gwosc_utils_snr_mf import _write_temp_segment_hdf5
except ImportError:
    from gwosc_utils_snr_mf import _write_temp_segment_hdf5


SCAN_STEP = 32.0
SCAN_NEG_EXT = 0.0    # GRB scan start; overridable by --scan-start

class PSDWrapper:
    def __init__(self, arr):
        self.sample_frequencies = arr[:, 0]
        self.data = arr[:, 1]

    def __array__(self, dtype=None):
        return np.asarray(self.data, dtype=dtype)


def _offset_label(off):
    sign = "m" if off < 0 else "p"
    value = f"{abs(float(off)):.9f}".rstrip("0").rstrip(".")
    integer, dot, fraction = value.partition(".")
    return "scan_t0" + sign + integer.zfill(4) + ("p" + fraction if dot else "")


def _build_offsets(scan_end=0.0, scan_start=SCAN_NEG_EXT, scan_step=SCAN_STEP):
    import math
    if not all(np.isfinite(x) for x in (scan_start, scan_end, scan_step)):
        raise ValueError("Scan start, end and step must be finite")
    if scan_start > scan_end or scan_step <= 0:
        raise ValueError("Require scan-start <= scan-end and scan-step > 0")
    first, last = math.ceil(scan_start / scan_step), math.floor(scan_end / scan_step)
    offsets = sorted(set([float(scan_start), float(scan_end)] +
                         [float(k * scan_step) for k in range(first, last + 1)]))
    if len({_offset_label(off) for off in offsets}) != len(offsets):
        raise ValueError("Scan offsets are too close for the output directory labels")
    return offsets


def compute_psd_from_hdf5(input_hdf5, t0, output_psd, sample_rate):
    """
    Read a strain segment from HDF5 and compute the PSD using Welch averaging.

    Assumptions:
    - input_hdf5 contains dataset: strain/Strain
    - segment length for Welch = 16 s
    - overlap = 8 s
    """
    import pycbc
    import pycbc.psd

    with h5py.File(input_hdf5, "r") as f:
        data = f["strain"]["Strain"][:].astype(np.float64)

    # Construct a GWPy TimeSeries first
    ts = TimeSeries(data, sample_rate=sample_rate, t0=t0 - 128)

    # Convert to PyCBC TimeSeries
    pycbc_ts = pycbc.types.TimeSeries(
        ts.value,
        delta_t=1.0 / sample_rate,
        epoch=ts.t0.value
    )

    # Apply dynamic range factor, same convention as before
    pycbc_ts *= pycbc.DYN_RANGE_FAC

    seg_len = int(16 * sample_rate)
    seg_stride = seg_len // 2

    psd = pycbc.psd.welch(
        pycbc_ts,
        seg_len=seg_len,
        seg_stride=seg_stride
    ).astype(np.float64)

    psd /= pycbc.DYN_RANGE_FAC**2

    freqs = np.array(psd.sample_frequencies, dtype=np.float64)
    vals = np.array(psd, dtype=np.float64)

    arr = np.column_stack([freqs, vals])
    arr = arr[np.isfinite(arr[:, 0]) & np.isfinite(arr[:, 1])]
    arr = arr[arr[:, 0] >= 0]
    arr = arr[arr[:, 1] > 0]

    np.savetxt(output_psd, arr)

    return arr


def _make_psd_from_segment(ifo, segment, t_center, outdir):
    temp_hdf5 = _write_temp_segment_hdf5(segment, ifo, t_center, outdir)
    psd_path = os.path.join(outdir, f"{ifo.lower()}_psd.txt")
    sample_rate = int(round(segment.sample_rate.value))

    subprocess.run(
        [
            sys.executable,
            "-m",
            "targ_ac_git.make_psd_pycbc",
            "--input-hdf5",
            temp_hdf5,
            "--t0",
            str(t_center),
            "--output-psd",
            psd_path,
            "--sample-rate",
            str(sample_rate),
        ],
        check=True,
    )

    arr = np.loadtxt(psd_path)
    arr = arr[arr[:, 1] > 0]
    np.savetxt(psd_path, arr)
    return PSDWrapper(arr), psd_path, temp_hdf5


def _validate_psd_file(psd_path, ifo):
    arr = np.loadtxt(psd_path)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise RuntimeError(f"Bad PSD file format for {ifo}: {psd_path}")
    if not np.all(np.isfinite(arr)):
        raise RuntimeError(f"Non-finite values in PSD file for {ifo}: {psd_path}")
    if np.any(arr[:, 0] < 0):
        raise RuntimeError(f"Negative frequencies in PSD file for {ifo}: {psd_path}")
    if np.any(arr[:, 1] <= 0):
        raise RuntimeError(f"Non-positive PSD values in PSD file for {ifo}: {psd_path}")
