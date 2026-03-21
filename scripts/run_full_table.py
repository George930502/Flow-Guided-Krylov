#!/usr/bin/env python3
"""Generate complete comparison table with timing for all methods."""
import sys, time, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.molecules import get_molecule
from src.solvers.fci import FCISolver
from src.solvers.ccsd import CCSDSolver, CCSDTSolver
from src.solvers.sci import CIPSISolver

molecules = [
    "LiH", "H2O", "BeH2", "NH3", "CH4", "N2", "HCN", "C2H2", "H2S", "C2H4", "Benzene",
]

# HI+NQS+SQD results (from previous GPU runs)
nqs_data = {
    "LiH":     {"energy": -7.8823243789,   "basis": 225,    "time": 3.9},
    "H2O":     {"energy": -75.0131547015,   "basis": 441,    "time": 1.7},
    "BeH2":    {"energy": -15.5951175626,   "basis": 1200,   "time": 2.1},
    "NH3":     {"energy": -55.5178160561,   "basis": 3074,   "time": 5.5},
    "CH4":     {"energy": -39.8060351767,   "basis": 11129,  "time": 54.4},
    "N2":      {"energy": -107.6541224475,  "basis": 12632,  "time": 43.7},
    "HCN":     {"energy": -91.8422041211,   "basis": 15000,  "time": 67.7},
    "C2H2":    {"energy": -76.0245739783,   "basis": 10000,  "time": 496.2},
    "H2S":     {"energy": -394.3547416894,  "basis": 3013,   "time": 11.3},
    "C2H4":    {"energy": -77.2350820819,   "basis": 15000,  "time": 1473.8},
    "Benzene": {"energy": -227.9564531849,  "basis": 10000,  "time": 101.6},
}

results = []

for mol_name in molecules:
    H, info = get_molecule(mol_name)
    nq = info["n_qubits"]
    is_cas = info.get("is_cas", False)

    row = {"mol": mol_name, "nq": nq}

    # FCI
    t0 = time.time()
    fci_r = FCISolver().solve(H, info)
    row["fci_e"] = fci_r.energy
    row["fci_time"] = time.time() - t0

    # CCSD
    if not is_cas:
        t0 = time.time()
        ccsd_r = CCSDSolver().solve(H, info)
        row["ccsd_e"] = ccsd_r.energy
        row["ccsd_time"] = time.time() - t0
    else:
        row["ccsd_e"] = None
        row["ccsd_time"] = None

    # CCSD(T)
    if not is_cas:
        t0 = time.time()
        ccsdt_r = CCSDTSolver().solve(H, info)
        row["ccsdt_e"] = ccsdt_r.energy
        row["ccsdt_time"] = time.time() - t0
    else:
        row["ccsdt_e"] = None
        row["ccsdt_time"] = None

    # SCI
    t0 = time.time()
    sci_r = CIPSISolver(max_basis_size=10000).solve(H, info)
    row["sci_e"] = sci_r.energy
    row["sci_basis"] = sci_r.diag_dim
    row["sci_time"] = time.time() - t0

    # HI+NQS+SQD
    nd = nqs_data.get(mol_name, {})
    row["nqs_e"] = nd.get("energy")
    row["nqs_basis"] = nd.get("basis")
    row["nqs_time"] = nd.get("time")

    results.append(row)

    # Print progress
    fci_e = row["fci_e"]
    print(f"{mol_name} ({nq}Q) done", flush=True)

# Print table
print(f"\n{'='*130}")
print(f"{'Mol':<8} {'Q':>3} | {'FCI':>14} {'t':>6} | {'CCSD err':>10} {'t':>6} | {'CCSD(T) err':>12} {'t':>6} | {'SCI err':>10} {'basis':>6} {'t':>8} | {'HI+NQS err':>10} {'basis':>6} {'t':>8}")
print(f"{'':8} {'':>3} | {'(Ha)':>14} {'(s)':>6} | {'(mHa)':>10} {'(s)':>6} | {'(mHa)':>12} {'(s)':>6} | {'(mHa)':>10} {'':>6} {'(s)':>8} | {'(mHa)':>10} {'':>6} {'(s)':>8}")
print(f"{'='*130}")

for r in results:
    fci_e = r["fci_e"]

    # FCI
    fci_s = f"{fci_e:.6f}" if fci_e else "SKIP"
    fci_t = f"{r['fci_time']:.1f}" if r["fci_time"] else ""

    # CCSD error
    if r["ccsd_e"] and fci_e:
        ccsd_err = f"{(r['ccsd_e'] - fci_e)*1000:.3f}"
        ccsd_t = f"{r['ccsd_time']:.1f}"
    else:
        ccsd_err = "—"
        ccsd_t = "—"

    # CCSD(T) error
    if r["ccsdt_e"] and fci_e:
        ccsdt_err = f"{(r['ccsdt_e'] - fci_e)*1000:.3f}"
        ccsdt_t = f"{r['ccsdt_time']:.1f}"
    else:
        ccsdt_err = "—"
        ccsdt_t = "—"

    # SCI error
    ref_e = fci_e or r["sci_e"]
    if r["sci_e"] and fci_e:
        sci_err = f"{(r['sci_e'] - fci_e)*1000:.3f}"
    else:
        sci_err = "ref"
    sci_b = f"{r['sci_basis']}"
    sci_t = f"{r['sci_time']:.1f}"

    # HI+NQS+SQD error
    if r["nqs_e"] and fci_e:
        nqs_err = f"{(r['nqs_e'] - fci_e)*1000:.3f}"
    elif r["nqs_e"] and r["sci_e"]:
        nqs_err = f"{(r['nqs_e'] - r['sci_e'])*1000:.3f}*"
    else:
        nqs_err = "—"
    nqs_b = f"{r['nqs_basis']}" if r["nqs_basis"] else "—"
    nqs_t = f"{r['nqs_time']:.1f}" if r["nqs_time"] else "—"

    print(f"{r['mol']:<8} {r['nq']:>3} | {fci_s:>14} {fci_t:>6} | {ccsd_err:>10} {ccsd_t:>6} | {ccsdt_err:>12} {ccsdt_t:>6} | {sci_err:>10} {sci_b:>6} {sci_t:>8} | {nqs_err:>10} {nqs_b:>6} {nqs_t:>8}")

print(f"{'='*130}")
print("* = error vs SCI (FCI not available)")
print("Chemical accuracy: < 1.6 mHa")

# Save JSON
with open("results/full_comparison.json", "w") as f:
    json.dump(results, f, indent=2, default=str)
