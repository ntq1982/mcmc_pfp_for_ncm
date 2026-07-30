'''
    This code performs a calculation of thermodynamic properties of LiNixCoyMnzO2 layered oxide.

    Energy calculations are done by using PFP within Atomic Simulation Environment (ASE).
    Random walks in energy space are done using Wang-Landau (WL) sampling method.
    This script can only be run online using Matlantis servers.

    Created on Nov 16, 2022 at RISM (Shinshu University)
    Last update: Jul 30, 2026 16:18 JST

    Copyright © 2022-2026 Quang Nguyen. All rights reserved.
'''

import os
import time
import datetime
import warnings
import csv
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pfp_api_client
from pfp_api_client.pfp.calculators.ase_calculator import ASECalculator
from pfp_api_client.pfp.estimator import Estimator, EstimatorCalcMode, EstimatorMethodType
from ase.io import read
from ase.neighborlist import NeighborList, NewPrimitiveNeighborList

# HELPER FUNCTION: Calculate 1NN bond fraction
def BndFrc(atoms, InElems, ExElems, cutoff):
    system = atoms.copy()
    for ExElem in ExElems:
        del system[[atom.symbol == ExElem for atom in system]]
    cutoffs = [cutoff / 2.0] * len(system)
    nl = NeighborList(cutoffs=cutoffs, skin=0.0, self_interaction=False, bothways=True, \
                      primitive=NewPrimitiveNeighborList)
    nl.update(system)
    N_Pairs = 6
    Pairs = np.zeros(N_Pairs, dtype=int)
    for i in range(len(system)):
        indices, _ = nl.get_neighbors(i)
        for j in indices:
            if system.symbols[i] == InElems[0]:
                if system.symbols[j] == InElems[0]: Pairs[0] += 1
                if system.symbols[j] == InElems[1]: Pairs[1] += 1
                if system.symbols[j] == InElems[2]: Pairs[2] += 1
            if system.symbols[i] == InElems[1]:
                if system.symbols[j] == InElems[0]: Pairs[1] += 1
                if system.symbols[j] == InElems[1]: Pairs[3] += 1
                if system.symbols[j] == InElems[2]: Pairs[4] += 1
            if system.symbols[i] == InElems[2]:
                if system.symbols[j] == InElems[0]: Pairs[2] += 1
                if system.symbols[j] == InElems[1]: Pairs[4] += 1
                if system.symbols[j] == InElems[2]: Pairs[5] += 1
    Pairs = np.multiply(Pairs, 1/2).astype(int)
    Tot_Pairs = np.sum(Pairs)
    results = np.insert(Pairs / Tot_Pairs, 0, Tot_Pairs)
    return results.tolist()

# -----------------------------------------------------------------------------

# Initialize simulation parameters
start_time      = time.time()
compound        = 'NCM523'
InElems         = ['Ni', 'Co', 'Mn']
ExElems         = ['Li', 'O']
bond_names      = {1: "AA", 2: "AB", 3: "AC", 4: "BB", 5: "BC", 6: "CC",}
bondtype        = 1
Na, Nb, Nc      = 5, 4, 1
predosfile      = '../MCMC_Stage1/'+compound+'_HnDOSvsE.dat'
struct_stable   = '../SA_Structures/'+compound+'_5x4x1_SAunopt_Most.cif'
struct_unstable = '../SA_Structures/'+compound+'_5x4x1_SAunopt_Least.cif'
histdos2Ddat    = f"{compound}_{bond_names[bondtype]}_HistDOS2D.dat"
logfile         = f"{compound}_{bond_names[bondtype]}_pfp+wl.log"

# Specify estimator and calculator for simulations
estimator  = Estimator(method_type=EstimatorMethodType.PFVM, calc_mode=EstimatorCalcMode.CRYSTAL_PLUS_D3, model_version='v3.0.0')
calculator = ASECalculator(estimator)
with open(logfile, 'w') as f:
    print(f"Executed on: {datetime.datetime.now()}", file=f)
    print(f"PFP client version: {pfp_api_client.__version__}", file=f)
    print(f"Model version: {estimator.model_version}", file=f)
    print(f"Calculation mode: {str(estimator.calc_mode).split('.')[1]}", file=f)
    print(f"Method type: {str(estimator.method_type).split('.')[1]}", file=f)
    print(f"****************************************", file=f)

# Import the most (known) stable and unstable structures
with open(logfile, 'a') as f:
    print(f"Examining {compound} ...", file=f)
    print(f"  Importing initial (known) stable/unstable structures ...", file=f)
SPC_stable   = read(struct_stable)
SPC_unstable = read(struct_unstable)
a            = SPC_stable.cell.lengths()[0]
b            = SPC_stable.cell.lengths()[1]
c            = SPC_stable.cell.lengths()[2]
cutoff       = np.max([a / Na, b / Nb]) + 0.1
BFmin, BFmax = 0.0, 0.5
NBF          = 60
dBF          = (BFmax - BFmin) / (NBF - 1)
BFn          = np.array([BFmin + iBF * dBF for iBF in range(NBF)])
with open(logfile, 'a') as f:
    print(f"    Supercell size: {Na} × {Nb} × {Nc}", file=f)
    print(f"    Number of atoms (perfect): {len(SPC_stable)}", file=f)
    print(f"    Cell parameters (a, b, c): {a / Na:.6f}, {b / Nb:.6f}, {c / Nc:.6f}", file=f)
    print(f"    Structural parameter range (min, max): {BFmin:.6f}, {BFmax:.6f}", file=f)
    print(f"    Number of discrete structural parameters: {NBF}", file=f)
    print(f"    Structural parameter spacing: {dBF:.6f}", file=f)

# Import energies and 1-D DOS from previous WL simlation
with open(logfile, 'a') as f:
    print(f"  Importing energies and 1-D DOS from previous WL simlation ...", file=f)
data       = np.loadtxt(predosfile, usecols=[1,5])
En, ln_gE  = data[:, 0], data[:, 1]
Emin, Emax = En[0], En[-1]
NE         = len(En)
dE         = (Emax - Emin) / (NE - 1)
with open(logfile, 'a') as f:
    print(f"    Energy range (min, max): {Emin:.6f}, {Emax:.6f}", file=f)
    print(f"    Number of energy states: {NE}", file=f)
    print(f"    Energy spacing (eV): {dE:.6f}", file=f)

# Create an initial random configuration from the (stable) structure above
SPC        = SPC_stable.copy()
idx_TMs    = [atom.index for atom in SPC if atom.symbol == InElems[0] \
                                         or atom.symbol == InElems[1] \
                                         or atom.symbol == InElems[2]]
symbol_TMs = SPC.symbols[idx_TMs]
np.random.seed(12345)
np.random.shuffle(symbol_TMs)
SPC.symbols[idx_TMs] = symbol_TMs
SPC.calc   = calculator

# Perform multicanonical Monte Carlo simulation
with open(logfile, 'a') as f:
    print(f"  Performing multicanonical Monte Carlo (MCMC) sampling ...", file=f)
TrialMove      = 'swap'
MaxSweep       = 10000000
PrintFreq      = 10000
E_SPC          = SPC.get_potential_energy()
NEcalc         = 1
BF_SPC_all     = BndFrc(SPC, InElems, ExElems, cutoff)
BF_SPC         = BF_SPC_all[bondtype]
E_InRange      = (E_SPC >= Emin - dE / 2) and (E_SPC < Emax + dE / 2)
BF_InRange     = (BF_SPC >= BFmin - dBF / 2) and (BF_SPC < BFmax + dBF / 2)
if E_InRange and BF_InRange:
    InRange = True
else:
    InRange = False
Sweep          = 0
Hist           = np.array([[0.0] * NBF for _ in range(NE)])
with open(logfile, 'a') as f:
    print(f"    Number of NNP-neighbors: {SPC.calc.results['calc_stats']['n_neighbors']}", file=f)
    print(f"    Maximum number of Monte Carlo steps: {MaxSweep}", file=f)
    print(f"    Trial move style: {TrialMove}", file=f)
while (Sweep <= MaxSweep):
    InRange = False
    while not InRange:
        if (TrialMove == 'swap'):
            idx_1, idx_2 = np.random.choice(idx_TMs, 2, replace=False)
            while (SPC.symbols[idx_1] == SPC.symbols[idx_2]):
                idx_2 = np.random.choice(idx_TMs)
            tmp_symbol_TM1 = SPC.symbols[idx_1]
            tmp_symbol_TM2 = SPC.symbols[idx_2]
            SPC.symbols[idx_1] = tmp_symbol_TM2
            SPC.symbols[idx_2] = tmp_symbol_TM1
            symbol_TMs = SPC.symbols[idx_TMs]
        elif (TrialMove == 'shuffle'):
            tmp_symbol_TMs = SPC.symbols[idx_TMs]
            np.random.shuffle(symbol_TMs)
            SPC.symbols[idx_TMs] = symbol_TMs
        E_SPC_new  = SPC.get_potential_energy()
        NEcalc    += 1
        BF_SPC_all = BndFrc(SPC, InElems, ExElems, cutoff)
        BF_SPC_new = BF_SPC_all[bondtype]
        E_InRange  = (E_SPC_new >= Emin - dE / 2) and (E_SPC_new < Emax + dE / 2)
        BF_InRange = (BF_SPC_new >= BFmin - dBF / 2) and (BF_SPC_new < BFmax + dBF / 2)
        if E_InRange and BF_InRange:
            InRange = True
    for iE in range(NE):
        if (E_SPC >= En[iE] - dE / 2) and (E_SPC < En[iE] + dE / 2):
            iE_old = iE
        if (E_SPC_new >= En[iE] - dE / 2) and (E_SPC_new < En[iE] + dE / 2):
            iE_new = iE
    for iBF in range(NBF):
        if (BF_SPC >= BFn[iBF] - dBF / 2) and (BF_SPC < BFn[iBF] + dBF / 2):
            iBF_old = iBF
        if (BF_SPC_new >= BFn[iBF] - dBF / 2) and (BF_SPC_new < BFn[iBF] + dBF / 2):
            iBF_new = iBF
    ln_gE_old = ln_gE[iE_old]
    ln_gE_new = ln_gE[iE_new]
    ln_gE_ratio = ln_gE_old - ln_gE_new
    P = np.min([1, np.exp(ln_gE_ratio)])
    if (P > np.random.rand()): 
        Hist[iE_new, iBF_new] += 1.0
        E_SPC = E_SPC_new
        BF_SPC = BF_SPC_new
    else:
        Hist[iE_old, iBF_old] += 1.0
        if (TrialMove == 'swap'):
            SPC.symbols[idx_1] = tmp_symbol_TM1
            SPC.symbols[idx_2] = tmp_symbol_TM2
        elif (TrialMove == 'shuffle'):
            SPC.symbols[idx_TMs] = tmp_symbol_TMs
    if (Sweep == 0) or (Sweep % PrintFreq == 0):
        current_time = time.time()
        elapsed_time = current_time - start_time
        sSweep = str(Sweep).zfill(len(str(MaxSweep)))
        with open(logfile, 'a') as f:
            print(f"    -- MCMC Step: {sSweep} \tTime(s): {elapsed_time:.6f}", file=f)
    Sweep += 1

# Calculate 2-D DOS from normalized 1-D DOS and 2-D histogram
ln_gEmin = np.min(ln_gE)
ln_gEBF  = np.zeros(np.shape(Hist))
for iE in range(NE):
    for iBF in range(NBF):
        if Hist[iE, iBF] == 0.0:
            ln_gEBF[iE, iBF] = 0
        else:
            ln_gEBF[iE, iBF] = (ln_gE[iE] - ln_gEmin) + np.log(Hist[iE, iBF])

# Save the final 2-D histogram and DOS
with open(logfile, 'a') as f:
    print(f"  Saving final MCMC simulation data ...", file=f)
try:
    os.remove(histdos2Ddat)
except OSError:
    pass
with open(histdos2Ddat, 'a') as f:
    for iE in range(NE):
        for iBF in range(NBF):
            print(f"E: {En[iE]:.6f} \tBF: {BFn[iBF]:.6f} \tH(E,BF): {Hist[iE, iBF]:.6f}", file=f) 

# Everything is ok if you can reach this point
end_time = time.time()
elapsed_time = end_time - start_time
with open(logfile, 'a') as f:
    print(f"****************************************", file=f)
    print(f"Done successfully!", file=f)
    print(f"Cumulative number of energy calculations: {NEcalc}", file=f)
    print(f"Elapsed time: {elapsed_time} seconds", file=f)
    print(f"Finished on: {datetime.datetime.now()}", file=f)
