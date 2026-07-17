"""
    This program is to search for the most stable/unstable atomic arrangement of transition metals in NCM.
    The structure is used for the Multi-canonical Monte Carlo simulation.

    Created on Jul 31, 2024 at Shinshu University
    Last update: Aug 05, 2024 14:02 JST

    Copyright © 2024 Quang Nguyen. All rights reserved.
"""

import os
import time
import datetime
import numpy as np
import pfp_api_client
from pfp_api_client.pfp.calculators.ase_calculator import ASECalculator
from pfp_api_client.pfp.estimator import Estimator, EstimatorCalcMode, EstimatorMethodType
from itertools import combinations
from ase.io import read, write
from ase.spacegroup import crystal
from ase.optimize import LBFGS
from ase.constraints import ExpCellFilter
from ase_annealing import SimAnn

# User settings
start_time = time.time()
compound   = 'NCM811'
TMs        = ['Ni', 'Co', 'Mn']
conc_TMs   = [8/10, 1/10, 1/10]
Na, Nb, Nc = 5, 4, 1
inputfile  = compound+'_StableIni_Most.cif'
logfile    = compound+'_SAsearch.log'
with open(logfile, 'w') as f:
    print(f"Executed on: {datetime.datetime.now()}", file=f)

# Specify estimator and calculator for simulations
estimator  = Estimator(method_type=EstimatorMethodType.PFVM, calc_mode=EstimatorCalcMode.CRYSTAL_PLUS_D3, model_version='v3.0.0')
calculator = ASECalculator(estimator)
with open(logfile, 'a') as f:
    print(f"PFP client version: {pfp_api_client.__version__}", file=f)
    print(f"Model version: {estimator.model_version}", file=f)
    print(f"Calculation mode: {str(estimator.calc_mode).split('.')[1]}", file=f)
    print(f"Method type: {str(estimator.method_type).split('.')[1]}", file=f)
    print(f"****************************************", file=f)

# Create a conventional cell of hexagonal LiNiO2 (R-3m)
a, b, c            = 2.878153, 2.878153, 14.150990
alpha, beta, gamma = 90, 90, 120
LiNiO2             = crystal(['Li', 'Ni', 'O'],
                             basis=[(0, 0, 0.5), (0, 0, 0), (0, 0, 0.26)],
                             spacegroup=166,
                             cellpar=[a, b, c, alpha, beta, gamma],
                             size=(1, 1, 1),
                             pbc=(True, True, True),
                             primitive_cell=False)

# Create a supercell of NCM to host cations
with open(logfile, 'a') as f:
    print(f"Creating a supercell for NCM from conventional cell...", file=f)
np.random.seed(12345)
SPC        = LiNiO2.repeat((Na, Nb, Nc))
idx_TMs    = [atom.index for atom in SPC if atom.symbol == 'Ni']
Natm_TMs   = [np.floor(len(idx_TMs) * conc_TMs[i]).astype(int) for i in range(len(conc_TMs))]
symbol_TMs = [TM for TM, count in zip(TMs, Natm_TMs) for _ in range(count)]
np.random.shuffle(symbol_TMs)
SPC.symbols[idx_TMs] = symbol_TMs
with open(logfile, 'a') as f:
    print(f"  Supercell size: {Na} × {Nb} × {Nc}", file=f)
    print(f"  Total number of atoms: {len(SPC)}", file=f)
    print(f"  Number of Ni, Co, Mn atoms: {Natm_TMs[0]} {Natm_TMs[1]} {Natm_TMs[2]}", file=f)

# Try to find optimal cell parameters for the compound
with open(logfile, 'a') as f:
    print(f"Searching for optimal lattice constant...", file=f)
nconfig             = 100
a_opt, b_opt, c_opt = [], [], []
for i in range(nconfig):
    SPCx = SPC.copy()
    np.random.shuffle(symbol_TMs)
    SPCx.symbols[idx_TMs] = symbol_TMs
    SPCx.calc = calculator
    opt = LBFGS(ExpCellFilter(SPCx, mask=[1, 1, 1, 0, 0, 0]), logfile=None)
    opt.run(fmax=0.05, steps=1000)
    a_opt.append(SPCx.cell.lengths()[0])
    b_opt.append(SPCx.cell.lengths()[1])
    c_opt.append(SPCx.cell.lengths()[2])
a_new, b_new, c_new = np.mean(a_opt), np.mean(b_opt), np.mean(c_opt)
ab_ave = (a_new / Na + b_new / Nb) / 2
SPC.set_cell([Na * ab_ave, Nb * ab_ave, c_new, alpha, beta, gamma], scale_atoms=True)
with open(logfile, 'a') as f:
    print(f"  Number of random configurations: {nconfig}", file=f)
    print(f"  Old supercell parameters: {Na * a : .6f} {Nb * b : .6f} {Nc * c : .6f}", file=f)
    print(f"  New supercell parameters: {Na * ab_ave : .6f} {Nb * ab_ave : .6f} {c_new : .6f} (average)", file=f)

# Perform global search for stable atomic arrangement
with open(logfile, 'a') as f:
    print(f"Searching for optimal arrangement of metals...", file=f)
N_configs_max       = np.math.factorial(sum(Natm_TMs))
for i in range(len(TMs)):
    N_configs_max  /= np.math.factorial(Natm_TMs[i])
N_configs_max       = int(N_configs_max)
N_steps             = 3000000
PrintFrequency      = 1000
SA_mode             = 'unopt'
SA_free_swap        = 'on'
struct_stable_most  = compound+'_'+str(Na)+'x'+str(Nb)+'x'+str(Nc)+'_SA'+SA_mode+'_Most.cif'
struct_stable_least = compound+'_'+str(Na)+'x'+str(Nb)+'x'+str(Nc)+'_SA'+SA_mode+'_Least.cif'
with open(logfile, 'a') as f:
    print(f"  Max number of configurations: {N_configs_max : .3E}", file=f)
    print(f"  Number of provided configurations: {N_steps}", file=f)
    print(f"  Frequency to print SA status: {PrintFrequency}", file=f)
    if SA_mode=='opt':
        Optimization = True
        print(f"  Optimization mode: ON", file=f)
    else:
        Optimization = False
        print(f"  Optimization mode: OFF", file=f)
    if SA_free_swap=='on':
        Layered = False
        print(f"  Free swapping mode: ON", file=f)
    else:
        Layered = True
        print(f"  Free swapping mode: OFF", file=f)
    print(f"  Most  stable configuration is saved to: {struct_stable_most}", file=f)
    print(f"  Least stable configuration is saved to: {struct_stable_least}", file=f)
SPC_best, SPC_worst = SimAnn(Structure=SPC, SwappingList=TMs, Calculator=calculator,
                             Target='Stable', Optimization=Optimization,
                             CoolingType='Exponential', T_start=1.0e4, T_stop=1.0e1,
                             N_steps=N_steps, PrintFrequency=PrintFrequency, RandomSeed=123, LogFile=logfile)
write(struct_stable_most, SPC_best, format='cif')
write(struct_stable_least, SPC_worst, format='cif')

# Everything is ok if you can reach this point
end_time = time.time()
elapsed_time = end_time - start_time
with open(logfile, 'a') as f:
    print(f"****************************************", file=f)
    print(f"Done successfully!", file=f)
    print(f"Elapsed time: {elapsed_time:.6f} seconds", file=f)
    print(f"Finished on: {datetime.datetime.now()}", file=f)