"""TDR targeted detectability range package."""

__all__ = ["run_grb"]


def __getattr__(name):
    # Load the driver on demand so python -m can execute it without a
    # pre-imported-module warning, and importing the package stays lightweight.
    if name == "run_grb":
        from .targ_range_snr_mf import run_grb
        return run_grb
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
