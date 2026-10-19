import ast
import hashlib
import json
from pathlib import Path

import healpy as hp
import numpy as np


def test_numerical_functions_are_unchanged_from_uploaded_baseline():
    import targ_ac_git.aux_snr_mf as aux
    source = Path(aux.__file__).read_text()
    expected = json.loads(Path(__file__).with_name("baseline_hashes.json").read_text())
    functions = {node.name: ast.get_source_segment(source, node)
                 for node in ast.parse(source).body if isinstance(node, ast.FunctionDef)}
    for name, digest in expected.items():
        assert hashlib.sha256(functions[name].encode()).hexdigest() == digest, name


def test_defaults_and_always_saved_isotropic_prior():
    from targ_ac_git.targ_range_snr_mf import build_parser, parse_iota_ranges
    from targ_ac_git.pipeline_utils_snr_mf import _build_offsets, _offset_label
    args = build_parser().parse_args(["--grb-name", "GRB200228291", "--output-dir", "out",
                                     "--t0", "2020-02-28T06:58:33.806", "--ra", "333.89292",
                                     "--dec=-42.944167"])
    assert (args.scan_start, args.scan_end, args.scan_step) == (0, 0, 32)
    assert (args.iota_min, args.iota_max, args.snr_threshold) == (0, 45, 9)
    assert [p["label"] for p in parse_iota_ranges()] == ["0-45", "0-90"]
    assert [p["label"] for p in parse_iota_ranges(10, 35)] == ["10-35", "0-90"]
    assert [p["label"] for p in parse_iota_ranges(0, 90)] == ["0-90"]
    assert _build_offsets(3.584, -100, 32) == [-100, -96, -64, -32, 0, 3.584]
    assert _offset_label(3.584) == "scan_t0p0003p584"


def test_all_three_range_contours_and_legend_patterns_are_retained(tmp_path, monkeypatch):
    from matplotlib.lines import Line2D
    from matplotlib.legend import Legend
    import targ_ac_git.aux_snr_mf as aux
    values = np.linspace(20, 520, 48)
    path = tmp_path / "range_map_bns_m1_1.4_m2_1.4.fits"
    hp.write_map(path, values, nest=True, dtype=float)
    monkeypatch.setattr(aux.pp, "close", lambda *args: None)
    aux.plot_final(str(tmp_path), str(path), None, [[1.0], [0.1]], 0, np.radians(30))
    fig = aux.pp.gcf()
    ax = fig.axes[0]
    red_legend = next(item for item in ax.artists if isinstance(item, Legend))
    handles = red_legend.get_lines()
    assert len(handles) == 3 and all(isinstance(handle, Line2D) for handle in handles)
    assert len({str(handle._unscaled_dash_pattern) for handle in handles}) == 3
    assert (tmp_path / "range_map_bns_m1_1.4_m2_1.4.pdf").stat().st_size > 1000
    aux.pp.close = monkeypatch.undo() or aux.pp.close


def test_targeted_range_legend_uses_both_iota_bounds(monkeypatch):
    import targ_ac_git.aux_snr_mf as aux
    from targ_ac_git.targ_range_snr_mf import parse_iota_ranges
    aux.pp.figure()
    aux._add_tdr_legend([(1.4, 1.4)], parse_iota_ranges(10, 35))
    legend = aux.pp.gca().get_legend()
    labels = [text.get_text() for text in legend.get_texts()]
    assert any("10" in label and "35" in label for label in labels)
    assert any("0" in label and "90" in label for label in labels)
    assert all(isinstance(handle, aux.Line2D) for handle in legend.get_lines())
    assert legend.get_lines()[-2].is_dashed()
    assert not legend.get_lines()[-1].is_dashed()
    aux.pp.close()
