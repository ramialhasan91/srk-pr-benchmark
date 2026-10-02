"""
Optional helper (not part of the run order): extracts the component-specific
consistent Twu-91 parameters (L, M, N) and the fitted volume-translation
constants of the tc-PR and tc-RK equations of state for the benchmark fluids
from the 1800-fluid compilation of

    Pina-Martinez, A.; Privat, R.; Jaubert, J.-N. Use of 300,000
    pseudo-experimental data over 1800 pure fluids to assess the performance of
    four cubic equations of state: SRK, PR, tc-RK, and tc-PR. AIChE J. 2022,
    68, e17518 (Supporting Information),

as redistributed, keyed by CAS number, in the open-source `thermo` library
(C. I. Bell, https://github.com/CalebBell/thermo; tables PRTwu_PinaMartinez,
SRKTwu_PinaMartinez). The extracted subset is written to
../data/twu_fit_params.csv, which is committed to the repository so that the
benchmark itself (01_run_benchmark.py) does not depend on `thermo`.

Usage:  pip install thermo && python code/00_extract_twu_fit_params.py
"""

import os
import sys

import pandas as pd
from CoolProp.CoolProp import get_fluid_param_string
from thermo.interaction_parameters import SPDB

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fluid_set import FLUIDS  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")

rows = []
for cp_name, (disp, family, subgroup) in FLUIDS.items():
    cas = get_fluid_param_string(cp_name, "CAS")
    pr = SPDB.tables["PRTwu_PinaMartinez"][cas]
    sr = SPDB.tables["SRKTwu_PinaMartinez"][cas]
    rows.append(dict(
        fluid=disp, coolprop=cp_name, CAS=cas,
        L_PR=pr["TwuPRL"], M_PR=pr["TwuPRM"], N_PR=pr["TwuPRN"],
        c_PR_cm3mol=pr["TwuPRc"] * 1e6,
        L_SRK=sr["TwuSRKL"], M_SRK=sr["TwuSRKM"], N_SRK=sr["TwuSRKN"],
        c_SRK_cm3mol=sr["TwuSRKc"] * 1e6,
    ))
df = pd.DataFrame(rows)
out = os.path.join(DATA, "twu_fit_params.csv")
with open(out, "w") as f:
    f.write("# Component-specific consistent Twu-91 parameters (L, M, N) and fitted "
            "volume-translation constants c (cm3/mol) of tc-PR and tc-RK,\n"
            "# from Pina-Martinez, Privat & Jaubert, AIChE J. 68 (2022) e17518 "
            "(Supporting Information), via the thermo library tables\n"
            "# PRTwu_PinaMartinez and SRKTwu_PinaMartinez (keyed by CAS).\n")
    df.to_csv(f, index=False)
print(df.to_string())
print("wrote", out)
