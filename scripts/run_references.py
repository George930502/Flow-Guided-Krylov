#!/usr/bin/env python3
"""Run FCI, CCSD, CCSD(T), SCI on all completed molecules for comparison."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.molecules import get_molecule
from src.solvers.fci import FCISolver
from src.solvers.ccsd import CCSDSolver, CCSDTSolver
from src.solvers.sci import CIPSISolver

molecules = [
    "LiH", "H2O", "BeH2", "NH3", "CH4", "N2", "HCN", "H2S",
]

# HI+NQS+SQD results (from previous runs)
nqs_results = {
    "LiH":  -7.8823243789,
    "H2O":  -75.0131547015,
    "BeH2": -15.5951175626,
    "NH3":  -55.5178160561,
    "CH4":  -39.8060351767,
    "N2":   -107.6541224475,
    "HCN":  -91.8422041211,
    "H2S":  -394.3547416894,
}

print(f"{'Mol':<6} {'Q':>3} {'FCI':>18} {'CCSD':>18} {'CCSD(T)':>18} {'SCI':>18} {'HI+NQS+SQD':>18}")
print(f"{'':6} {'':>3} {'':>18} {'err(mHa)':>18} {'err(mHa)':>18} {'err/basis':>18} {'err/basis':>18}")
print("=" * 105)

for mol_name in molecules:
    H, info = get_molecule(mol_name)
    nq = info["n_qubits"]
    is_cas = info.get("is_cas", False)

    # FCI
    fci_r = FCISolver().solve(H, info)
    fci_e = fci_r.energy

    # CCSD
    ccsd_e = None
    ccsd_err = None
    if not is_cas:
        ccsd_r = CCSDSolver().solve(H, info)
        ccsd_e = ccsd_r.energy
        if ccsd_e and fci_e:
            ccsd_err = (ccsd_e - fci_e) * 1000

    # CCSD(T)
    ccsdt_e = None
    ccsdt_err = None
    if not is_cas:
        ccsdt_r = CCSDTSolver().solve(H, info)
        ccsdt_e = ccsdt_r.energy
        if ccsdt_e and fci_e:
            ccsdt_err = (ccsdt_e - fci_e) * 1000

    # SCI
    sci_r = CIPSISolver(max_basis_size=10000).solve(H, info)
    sci_e = sci_r.energy
    sci_err = None
    if sci_e and fci_e:
        sci_err = (sci_e - fci_e) * 1000

    # HI+NQS+SQD
    nqs_e = nqs_results.get(mol_name)
    nqs_err = None
    if nqs_e and fci_e:
        nqs_err = (nqs_e - fci_e) * 1000

    # Print
    fci_s = f"{fci_e:.10f}" if fci_e else "N/A"
    ccsd_s = f"{ccsd_err:.3f}" if ccsd_err is not None else "N/A"
    ccsdt_s = f"{ccsdt_err:.3f}" if ccsdt_err is not None else "N/A"
    sci_s = f"{sci_err:.3f}/{sci_r.diag_dim}" if sci_err is not None else f"/{sci_r.diag_dim}"
    nqs_s = f"{nqs_err:.3f}" if nqs_err is not None else "N/A"
    nqs_basis = nqs_results.get(mol_name + "_basis", "")

    print(f"{mol_name:<6} {nq:>3} {fci_s:>18} {ccsd_s:>18} {ccsdt_s:>18} {sci_s:>18} {nqs_s:>18}")

print("\n" + "=" * 105)
print("\nError = (E_method - E_FCI) × 1000 [mHa]")
print("Chemical accuracy: < 1.6 mHa")
