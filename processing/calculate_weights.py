#!/usr/bin/env python3
"""
Calculate the BBH model weights and the SFH convolution for population files.

Script version of processing/get_BBH_populations.ipynb: for each file it
1. calculates the 'extended_IMF' model weights, reweighting the simulated
   initial conditions to POP_PARAMS, and
2. convolves them with the star-formation history (calculate_cosmic_weights)
   and computes the intrinsic rate density,
so all populations are reweighted to the same initial conditions and SFH.

The POSYDON checkout in software/POSYDON has a Jacobian bug in
norm_pop.get_period_pdf for orbital_scheme='separation': it returns a density
in linear separation (∝ 1/a), while Sana12Period.pdf is a density per dex of
log10(P). Reweighting a period-sampled run (the kick runs) to the separation
target is then wrong by a factor ∝ 1/a. This script patches that function in
memory; the POSYDON source is not modified.

Usage
-----
    # all population files under data/
    software/v2.3/bin/python processing/calculate_weights.py

    # selected files
    software/v2.3/bin/python processing/calculate_weights.py data/kicks_conservative/*.h5

This overwrites '/transients/BBH/weights/extended_IMF' and
'/transients/BBH/rates/<SFH>/' in each file.
"""
import argparse
import copy
import os
import warnings
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

# posydon.config raises if PATH_TO_POSYDON is not a valid directory
if not os.path.isdir(os.environ.get("PATH_TO_POSYDON", "")):
    os.environ["PATH_TO_POSYDON"] = str(REPO / "software" / "POSYDON")

import numpy as np

import posydon.popsyn.norm_pop as norm_pop
import posydon.popsyn.rate_calculation as rate_calculation
from posydon.popsyn.distributions import Sana12Period
from posydon.popsyn.synthetic_population import TransientPopulation
from posydon.utils.common_functions import orbital_separation_from_period

warnings.filterwarnings("ignore")

TRANSIENT = "BBH"
MODEL_WEIGHTS = "extended_IMF"

# target population, identical to processing/get_BBH_populations.ipynb
POP_PARAMS = {
    "binary_fraction_scheme": "const",
    "binary_fraction_const": 0.7,
    # extending the IMF to higher and lower masses
    "primary_mass_scheme": "Kroupa2001",
    "primary_mass_min": 0.01,
    "primary_mass_max": 300,
    # overwriting q_min and q_max for flat mass ratio
    "secondary_mass_scheme": "flat_mass_ratio",
    "secondary_mass_min": 0.01,
    "secondary_mass_max": 300,
    "q_min": 0,
    "q_max": 1,
    # orbital parameters
    "orbital_scheme": "separation",
    "orbital_separation_scheme": "log_uniform",
    "orbital_separation_min": 5.0,
    "orbital_separation_max": 1e5,
}

SFH_MODELS = {
    "IllustrisTNG": {
        "delta_t": 100,  # Myr
        "SFR": "IllustrisTNG",
        "sigma_SFR": None,
        "Z_min": 1e-11,
        "normalise": True,
    },
}

# calculate_cosmic_weights updates DEFAULT_SFH_MODEL in place, so keys of one
# SFH model leak into the next. Keep a pristine copy to reset it.
PRISTINE_DEFAULT_SFH_MODEL = copy.deepcopy(rate_calculation.DEFAULT_SFH_MODEL)


# POSYDON patch: period pdf per dex of log10(P) for both orbital schemes
_original_get_period_pdf = norm_pop.get_period_pdf


def _get_period_pdf_per_dex(kwargs):
    """Sana12Period.pdf is already per dex of log10(P). For a log-uniform
    separation, d log a / d log P = 2/3, so
    p(log10 P) = (2/3) / log10(a_max / a_min) inside the separation bounds.
    """
    if (kwargs["orbital_scheme"] == "separation"
            and kwargs["orbital_separation_scheme"] == "log_uniform"):
        a_min = kwargs["orbital_separation_min"]
        a_max = kwargs["orbital_separation_max"]
        density = (2.0 / 3.0) / np.log10(a_max / a_min)

        def period_pdf(P, m1, q):
            a = orbital_separation_from_period(P, m1, q * m1)
            return np.where((a >= a_min) & (a <= a_max), density, 0.0)

        return period_pdf
    return _original_get_period_pdf(kwargs)


norm_pop.get_period_pdf = _get_period_pdf_per_dex


def check_period_pdfs():
    """Both period pdfs must integrate to 1 in log10(P), else the weights are off."""
    logP = np.linspace(-3, 8, 200001)
    P = 10**logP
    m1 = np.full_like(P, 40.0)
    q = np.full_like(P, 0.8)
    pdfs = {
        "log-uniform separation": _get_period_pdf_per_dex(POP_PARAMS)(P, m1, q),
        "Sana+12": Sana12Period(0.75, 6000).pdf(P, m1),
    }
    for name, pdf in pdfs.items():
        integral = np.trapz(pdf, logP)
        if abs(integral - 1) > 1e-3:
            raise RuntimeError(f"{name} period pdf integrates to {integral}, not 1")


def simulation_parameters(ini_params):
    # the separation runs do not store their separation distribution in the
    # ini parameters; they were sampled log-uniform in 5-1e5 Rsun
    sim = dict(ini_params)
    if sim["orbital_scheme"] == "separation":
        sim["orbital_separation_scheme"] = "log_uniform"
        sim["orbital_separation_min"] = 5.0
        sim["orbital_separation_max"] = 1e5
    return sim


def calculate_weights(filename, sfh_models):
    transient_pop = TransientPopulation(filename, TRANSIENT)
    transient_pop.calculate_model_weights(
        model_weights_identifier=MODEL_WEIGHTS,
        population_parameters=POP_PARAMS,
        simulation_parameters=simulation_parameters(transient_pop.ini_params),
    )
    for model in sfh_models:
        print(f"  {model}", flush=True)
        rate_calculation.DEFAULT_SFH_MODEL.clear()
        rate_calculation.DEFAULT_SFH_MODEL.update(copy.deepcopy(PRISTINE_DEFAULT_SFH_MODEL))
        rates = transient_pop.calculate_cosmic_weights(
            model, model_weights=MODEL_WEIGHTS, MODEL_in=copy.deepcopy(SFH_MODELS[model]))
        rates.calculate_intrinsic_rate_density(channels=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("files", nargs="*", help="population files (default: all under data/)")
    parser.add_argument("--sfh", nargs="+", default=list(SFH_MODELS), choices=list(SFH_MODELS),
                        help="SFH models to convolve with (default: all in SFH_MODELS)")
    args = parser.parse_args()

    files = args.files or sorted(
        str(p) for p in (REPO / "data").rglob("*.h5") if p.parent != REPO / "data")

    check_period_pdfs()
    for f in files:
        print(f"Processing {f}", flush=True)
        calculate_weights(f, args.sfh)


if __name__ == "__main__":
    main()
