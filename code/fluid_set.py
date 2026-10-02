"""
Fluid set of the benchmark: 37 fluids in five chemical families.

CoolProp names on the left; (display name, family, subgroup) on the right.
The polar/associating family is subdivided into hydrogen-bonding
(self-associating) fluids and polar aprotic fluids for the dedicated
analysis of Section 3.3 of the paper. Hydrogen and helium are excluded
(quantum fluids for which classical corresponding-states cubics are not
intended).
"""

FLUIDS = {
    # Inorganic / light gases
    "Nitrogen":        ("nitrogen",         "Inorganic gases", ""),
    "Oxygen":          ("oxygen",           "Inorganic gases", ""),
    "Argon":           ("argon",            "Inorganic gases", ""),
    "CarbonMonoxide":  ("carbon monoxide",  "Inorganic gases", ""),
    "CarbonDioxide":   ("carbon dioxide",   "Inorganic gases", ""),
    "HydrogenSulfide": ("hydrogen sulfide", "Inorganic gases", ""),
    # Alkanes (n-, iso-, cyclo-)
    "Methane":     ("methane",     "Alkanes", ""),
    "Ethane":      ("ethane",      "Alkanes", ""),
    "n-Propane":   ("propane",     "Alkanes", ""),
    "n-Butane":    ("n-butane",    "Alkanes", ""),
    "IsoButane":   ("isobutane",   "Alkanes", ""),
    "n-Pentane":   ("n-pentane",   "Alkanes", ""),
    "n-Hexane":    ("n-hexane",    "Alkanes", ""),
    "n-Heptane":   ("n-heptane",   "Alkanes", ""),
    "n-Octane":    ("n-octane",    "Alkanes", ""),
    "n-Decane":    ("n-decane",    "Alkanes", ""),
    "CycloHexane": ("cyclohexane", "Alkanes", ""),
    # Aromatics
    "Benzene":      ("benzene",      "Aromatics", ""),
    "Toluene":      ("toluene",      "Aromatics", ""),
    "EthylBenzene": ("ethylbenzene", "Aromatics", ""),
    "o-Xylene":     ("o-xylene",     "Aromatics", ""),
    "m-Xylene":     ("m-xylene",     "Aromatics", ""),
    "p-Xylene":     ("p-xylene",     "Aromatics", ""),
    # Polar / associating: hydrogen-bonding (self-associating)
    "Water":    ("water",    "Polar/associating", "H-bonding"),
    "Methanol": ("methanol", "Polar/associating", "H-bonding"),
    "Ethanol":  ("ethanol",  "Polar/associating", "H-bonding"),
    "Ammonia":  ("ammonia",  "Polar/associating", "H-bonding"),
    # Polar / associating: polar aprotic
    "Acetone":         ("acetone",         "Polar/associating", "polar aprotic"),
    "DimethylEther":   ("dimethyl ether",  "Polar/associating", "polar aprotic"),
    "SulfurDioxide":   ("sulfur dioxide",  "Polar/associating", "polar aprotic"),
    "EthyleneOxide":   ("ethylene oxide",  "Polar/associating", "polar aprotic"),
    "R40":             ("methyl chloride", "Polar/associating", "polar aprotic"),
    "Tetrahydrofuran": ("tetrahydrofuran", "Polar/associating", "polar aprotic"),
    # Refrigerants (HFC / HFO)
    "R32":     ("R-32",     "Refrigerants", ""),
    "R125":    ("R-125",    "Refrigerants", ""),
    "R134a":   ("R-134a",   "Refrigerants", ""),
    "R1234yf": ("R-1234yf", "Refrigerants", ""),
}

FAMILY_ORDER = ["Inorganic gases", "Alkanes", "Aromatics", "Polar/associating", "Refrigerants"]
