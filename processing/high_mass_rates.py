"""Write the high-mass BBH rate density of every population file to a CSV file."""
from pathlib import Path

import numpy as np
import pandas as pd
from posydon.popsyn.rate_calculation import get_shell_comoving_volume
from posydon.popsyn.synthetic_population import Rates

MASS_CUTOFF = 39.76734837
Z_MIN = 0.15
Z_MAX = 0.25
SFH_type = "IllustrisTNG"

data_dir = Path(__file__).resolve().parents[1] / "data"
volume = get_shell_comoving_volume(Z_MIN, Z_MAX)

rows = []
for file in sorted(f for f in data_dir.rglob("*.h5") if f.parent != data_dir):
    data = Rates(str(file), "BBH", SFH_type)
    max_mass = np.maximum(data.population["S1_mass"].values, data.population["S2_mass"].values)
    mask = max_mass >= MASS_CUTOFF
    z_event_mask = (data.z_events >= Z_MIN) & (data.z_events <= Z_MAX)
    rates = np.nansum(data.weights[z_event_mask][mask], axis=1) / volume
    negative = data.population["chi_eff"].values[mask] < 0
    rows.append({
        "file": str(file.relative_to(data_dir)),
        "rate_Gpc-3_yr-1": rates.sum(),
        "frac_negative_chi_eff": rates[negative].sum() / rates.sum(),
    })

pd.DataFrame(rows).to_csv(Path(__file__).with_name("high_mass_rates.csv"), index=False)
