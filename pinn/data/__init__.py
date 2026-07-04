"""Per-head dataset loaders.

Every loader returns numpy arrays plus the *raw physical quantities* the
physics losses need (unscaled temperatures, rpm, timestamps, band energies...),
so `pinn.losses` never has to un-normalize anything.

Data provenance labels (enforced in the final report):
- REAL:        CWRU, IMS (measured vibration)
- SIMULATED:   C-MAPSS (NASA high-fidelity engine simulation), AI4I
               (synthetic-by-design with documented generative rules)
- SYNTHETIC:   pressure cycles (generated in-repo, no public dataset exists)
"""
