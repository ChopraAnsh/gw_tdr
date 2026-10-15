import argparse
import os
import h5py
import numpy as np
from gwpy.timeseries import TimeSeries

import pycbc
import pycbc.psd
from pycbc.types import TimeSeries as PyCBCTimeSeries


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-hdf5", required=True)
    parser.add_argument("--t0", type=float, required=True)
    parser.add_argument("--output-psd", required=True)
    parser.add_argument("--sample-rate", type=int, required=True)
    args = parser.parse_args()

    with h5py.File(args.input_hdf5, "r") as f:
        data = f["strain"]["Strain"][:].astype(np.float32)
        file_start = float(f.attrs.get("gps_start", args.t0 - 128.0))

    if len(data) != 256 * args.sample_rate:
        raise ValueError("Expected a complete 256-second PSD segment")

    ts = TimeSeries(
        data,
        sample_rate=args.sample_rate,
        t0=file_start,
    )

    segment = ts  # already the exact 256-second segment; do not crop a rounded epoch

    ts_pycbc = PyCBCTimeSeries(
        segment.value,
        delta_t=1 / segment.sample_rate.value,
        epoch=segment.t0.value,
    )

    ts_pycbc *= pycbc.DYN_RANGE_FAC

    seg_len = 16 * int(ts_pycbc.sample_rate)
    seg_stride = seg_len // 2

    psd = pycbc.psd.welch(
        ts_pycbc,
        seg_len=seg_len,
        seg_stride=seg_stride,
    ).astype(np.float64)

    psd /= pycbc.DYN_RANGE_FAC ** 2
    psd.save(args.output_psd)


if __name__ == "__main__":
    main()