"""Real PyCBC integration using synthetic strain instead of GWOSC downloads."""
import json
import os

import healpy as hp
import numpy as np
import pytest


@pytest.mark.skipif(os.environ.get("RUN_GRB_INTEGRATION") != "1", reason="Optional waveform integration")
def test_waveforms_both_priors_metadata_and_automatic_plot(tmp_path, monkeypatch):
    from gwpy.timeseries import TimeSeries
    from targ_ac_git import run_grb
    import targ_ac_git.targ_range_snr_mf as runner

    def synthetic(ifo, config, center, cache):
        if ifo == "V1":
            raise runner.GWOSCNoDataError("Synthetic test: V1 unavailable")
        data = np.random.default_rng(1 if ifo == "H1" else 2).normal(0, 8e-22, 256 * 4096)
        return TimeSeries(data, t0=center - 128, sample_rate=4096), ["synthetic"]

    monkeypatch.setattr(runner, "_load_segment_from_cache", synthetic)
    probability = np.zeros(192)
    probability[20:23] = [0.7, 0.25, 0.05]
    skymap = tmp_path / "synthetic_fermi.fit"
    hp.write_map(skymap, probability, nest=True, dtype=float, column_names=["PROBABILITY"])
    output = tmp_path / "GRB200228291"
    args = runner.build_parser().parse_args([
        "--grb-name", "GRB200228291", "--output-dir", str(output),
        "--t0", "2020-02-28T06:58:33.806", "--ra", "333.89292", "--dec=-42.944167",
        "--skymap-file", str(skymap), "--iota-min", "10", "--iota-max", "35",
    ])
    returned = run_grb(**vars(args))
    assert returned["user_prior"] == "10-35"
    bin_dir = output / "scan_t0p0000"
    for family in ("bns", "nsbh"):
        payload = json.loads((bin_dir / f"results_{family}.json").read_text())[family]
        assert len(payload) == 3
        for entry in payload.values():
            assert entry["snr_threshold"] == 9
            assert set(entry["tdr"]) == {"10-35", "0-90"}
            assert entry["antenna_factor"]["credible_level"] == 0.9
            assert entry["antenna_factor"]["averaging"] == "arithmetic_mean_over_equal_area_pixels"
            assert entry["antenna_factor"]["D90_sky_treatment"] == "probability_weighted_sky_samples"
        assert (bin_dir / f"{family}_targeted_range.pdf").stat().st_size > 1000
    assert (output / "GRB200228291_D90_and_IFO_windows_combined.pdf").stat().st_size > 1000
    summary = json.loads((output / "scan_summary.json").read_text())
    assert summary["bins"][0]["online_ifos"] == ["H1", "L1"]
