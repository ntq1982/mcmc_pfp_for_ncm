'''
    This code performs a calculation of thermodynamic properties of LiNixCoyMnzO2 layered oxide.

    Energy calculations are done by using PFP within Atomic Simulation Environment (ASE).
    Random walks in energy space are done using Wang-Landau sampling method.
    This script can only be run online using Matlantis servers.

    Created on Nov 16, 2022 at RISM (Shinshu University)
    Last update: Jul 30, 2026 14:40 JST

    Copyright © 2022-2026 Quang Nguyen. All rights reserved.
'''

import os
import time
import warnings
import numpy as np
import pfp_api_client
from pfp_api_client.pfp.calculators.ase_calculator import ASECalculator
from pfp_api_client.pfp.estimator import Estimator, EstimatorCalcMode
from ase.io import read

# Create a logfile to save simulation status
start_time = time.time()
A, B, C    = 'Ni', 'Co', 'Mn'
cif_dir    = '../SA_Structures'
compound   = 'NCM523'
logfile    = compound+'_pfp+wl.log'
try:
    os.remove(logfile)
except OSError:
    pass

# Specify estimator and calculator for simulations
estimator = Estimator(calc_mode=EstimatorCalcMode.CRYSTAL_PLUS_D3, model_version='v3.0.0')
calculator = ASECalculator(estimator)
with open(logfile, 'a') as f:
    print(f"PFP client version: {pfp_api_client.__version__}", file=f)
    print(f"Model version: {estimator.model_version}", file=f)
    print(f"Calculation mode: {str(estimator.calc_mode).split('.')[1]}", file=f)
    print(f"****************************************", file=f)

# Import the most (known) stable and unstable structures
with open(logfile, 'a') as f:
    print(f"Examining {compound} ...", file=f)
    print(f"  Importing initial (known) stable/unstable structures ...", file=f)
SPC_stable   = read(cif_dir + '/' + compound+'_5x4x1_SAunopt_Most.cif')
SPC_unstable = read(cif_dir + '/' + compound+'_5x4x1_SAunopt_Least.cif')
Na, Nb, Nc   = 5, 4, 1
a            = SPC_stable.cell.lengths()[0]
b            = SPC_stable.cell.lengths()[1]
c            = SPC_stable.cell.lengths()[2]
with open(logfile, 'a') as f:
    print(f"  Supercell size: {Na} × {Nb} × {Nc}", file=f)
    print(f"  Number of atoms (perfect): {len(SPC_stable)}", file=f)
    print(f"  Cell parameters (a, b, c): {a / Na:.6f}, {b / Nb:.6f}, {c / Nc:.6f}", file=f)

# Estimate energy range for the compound
with open(logfile, 'a') as f:
    print(f"Estimating energy range ...", file=f)
SPC_stable.calc   = calculator
SPC_unstable.calc = calculator
Emin              = SPC_stable.get_potential_energy()
Emax              = SPC_unstable.get_potential_energy()
NEcalc            = 2
with open(logfile, 'a') as f:
    print(f"  NOTE: Energy range is estimated from imported structures!", file=f)
    print(f"  Energy range (min, max): {Emin:.6f}, {Emax:.6f}", file=f)

# Perform Wang-Landau MCMC simulation
with open(logfile, 'a') as f:
    print(f"Performing Wang-Landau MCMC sampling ...", file=f)
np.random.seed(12345)
NE             = 60 
dE             = (Emax - Emin) / (NE - 1)
En             = [Emin + iE * dE for iE in range(NE)]
ln_gE          = [0.0] * NE
ln_f           = 1.0
ln_ftol        = 10**(-6)
TrialMove      = 'swap'
Flatness       = 0.8
CheckFreq      = 100
PrintFreq      = 10000
iteration      = 0
SPC            = SPC_stable.copy()
SPC.calc       = calculator
E_SPC          = SPC.get_potential_energy()
NEcalc        += 1
E_Most_Stable  = Emin
E_Least_Stable = Emax
idx_TMs        = [atom.index for atom in SPC if atom.symbol == A or atom.symbol == B or atom.symbol == C]
with open(logfile, 'a') as f:
    print(f"  Number of energy states: {NE}", file=f)
    print(f"  Energy spacing (eV): {dE:.6f}", file=f)
    print(f"  Histogram flatness threshold: {Flatness:.1f}", file=f)
    print(f"  Initial modification factor: {np.exp(ln_f):.6f}", file=f)
    print(f"  Modification factor threshold: {ln_ftol:.2e}", file=f)
    print(f"  Number of F-iterations: 0 + {-np.fix(np.log2(ln_ftol)).astype(int)}", file=f)
while (ln_f > ln_ftol):
    with open(logfile, 'a') as f:
        print(f"  F-Iteration: {iteration} \tF = {np.exp(ln_f):.6f}", file=f)
    Hist  = [0.0] * NE
    Flat  = False
    sweep = 0
    while not Flat:
        InRange = False
        while not InRange:
            idx_1, idx_2 = np.random.choice(idx_TMs, 2, replace=False)
            while (SPC.symbols[idx_1] == SPC.symbols[idx_2]):
                idx_2 = np.random.choice(idx_TMs, 1, replace=False)
            tmp_symbol1        = SPC.symbols[idx_1]
            tmp_symbol2        = SPC.symbols[idx_2]
            SPC.symbols[idx_1] = tmp_symbol2
            SPC.symbols[idx_2] = tmp_symbol1
            E_SPC_new = SPC.get_potential_energy()
            NEcalc   += 1
            E_InRange = (E_SPC_new >= Emin - dE / 2) and (E_SPC_new < Emax + dE / 2)
            if E_InRange:
                InRange = True
            else:
                InRange = False
        for iE in range(NE):
            if (E_SPC >= En[iE] - dE / 2) and (E_SPC < En[iE] + dE / 2):
                iE_old = iE
            if (E_SPC_new >= En[iE] - dE / 2) and (E_SPC_new < En[iE] + dE / 2):
                iE_new = iE
        ln_gE_old = ln_gE[iE_old]
        ln_gE_new = ln_gE[iE_new]
        ln_gRatio = ln_gE_old - ln_gE_new
        P = 1.0
        if (ln_gRatio < 0):
            P = np.exp(ln_gRatio)
        if (P > np.random.rand()):
            ln_gE[iE_new] += ln_f
            Hist[iE_new]  += 1.0
            E_SPC          = E_SPC_new
        else:
            ln_gE[iE_old]     += ln_f
            Hist[iE_old]      += 1.0
            SPC.symbols[idx_1] = tmp_symbol1
            SPC.symbols[idx_2] = tmp_symbol2
        if (sweep % CheckFreq == 0):
            Hist_avg = np.average(Hist)
            Hist_min = np.min(Hist)
            if (Hist_min >= Flatness * Hist_avg): Flat = True
        if (sweep == 0) or (sweep % PrintFreq == 0) or Flat:
            current_time = time.time()
            elapsed_time = current_time - start_time
            with open(logfile, 'a') as f:
                print(f"  F-Iteration: {iteration} \tMC-Sweep: {sweep} \tTime(s): {elapsed_time:.6f}", file=f)
        sweep += 1
    with open(logfile, 'a') as f:
        print(f"  Cumulative number of energy calculations: {NEcalc}", file=f)
    ln_f /= 2.0
    iteration += 1

# Save the final histogram and DOS
with open(logfile, 'a') as f:
    print(f"Saving final MCMC simulation data ...", file=f)
datafile = compound+'_HnDOSvsE.dat'
try:
    os.remove(datafile)
except OSError:
    pass
with open(datafile, 'a') as f:
    for iE in range(NE):
        print(f"E: {En[iE]:.6f} \tH(E): {Hist[iE]:.6f} \tln(g(E)): {ln_gE[iE]:.6f}", file=f)

# Everything is ok if you can reach this point
end_time     = time.time()
elapsed_time = end_time - start_time
with open(logfile, 'a') as f:
    print(f"****************************************", file=f)
    print(f"Done successfully!", file=f)
    print(f"Elapsed time: {elapsed_time} seconds", file=f)