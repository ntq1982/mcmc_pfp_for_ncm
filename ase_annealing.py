"""
    This module contains a function based on the Simulatated Annealing algorithm
    for finding the most stable/unstable atomic arrangement in multi-component systems.

    Created on Jul 02, 2023 at Shinshu University
    Last update: Jul 10, 2024 09:21 JST

    Copyright © 2023 Quang Nguyen. All rights reserved.
"""

import numpy as np
from typing import List
from ase import Atoms
from ase.optimize import LBFGS
from ase.constraints import UnitCellFilter

def SimAnn(Structure: Atoms, SwappingList: List[str], Calculator,
           Target: str = 'Stable', Optimization: bool = False,
           CoolingType: str = 'Exponential', T_start: float = 1.0e4, T_stop: float = 1.0e-3,
           N_steps: int = 1000000, PrintFrequency: int = 1000, RandomSeed: int = None,
           LogFile: str = None) -> Atoms:
    """
        Parameters:
            Structure      : An ASE Atoms object
                             Initial structure of the alloying system.
            SwappingList   : A list of chemical symbols
                             List of elements to be included the swapping process.
            Calculator     : An ASE Calculator
                             Name of calculator use to do energy calculation.
            Target         : A string ('Stable' or 'Unstable')
                             Search target (the most stable or unstable arrangement)
                             Default value: 'Stable'
            Optimization   : A boolean
                             Either to perform full structural optimization or not.
                             Default value: False
            CoolingType    : A string ('Linear' or 'Exponential')
                             Method of cooling the annealing temperature.
                             Default value: 'Exponential'
            T_start        : A single float
                             Artificial temperature at which annealing is started.
                             Default value: 5.0
            T_stop         : A single float
                             Artificial temperature at which annealing is stopped.
                             Default value: 0.001
            N_steps        : An integer
                             Number of steps to take in the annealing simulation.
                             Default value: 1000000
            PrintFrequency : An integer
                             Frequency to print out the simulation progress.
                             Default value: 1000
            LogFile        : A string
                             Name of the log file that keeps track of SA steps
                             Default value: None -> LogFile='SimAnn.log'

        Returns: ASE Atoms objects
                 If Target = 'Stable' : The most stable arrangement and the known unstable one.
                 If Target = 'Unstable' : The most unstable arrangement and the known stable one.
    """

    if Target == 'Stable':
        sign = 1
    elif Target == 'Unstable':
        sign = -1

    if RandomSeed is None:
        RandomSeed = np.random.randint(0, int(2**32 - 1))
    np.random.seed(RandomSeed)

    if LogFile is None:
        LogFile = 'SimAnn.log'

    indices = [atom.index for atom in Structure if atom.symbol in SwappingList]

    VacanciesExist = False
    for symbol in SwappingList:
        if symbol == 'X':
            VacanciesExist = True

    with open(LogFile, 'a') as f:
        #print(f"Searching for stable structure using SA method ...", file=f)
        print(f"    Number of Monte Carlo steps: {N_steps}", file=f)
        print(f"    Starting temperature: {T_start : .6f}", file=f)
        print(f"    Stopping temperature: {T_stop : .6f}", file=f)

    step = 0
    accepted_trials = 0
    Str_current = Structure.copy()
    if VacanciesExist:
        Str_current_X = Str_current.copy()
        del Str_current_X[[atom.index for atom in Str_current_X if atom.symbol=='X']]
        Str_current_X.calc = Calculator
        if Optimization:
            opt = LBFGS(UnitCellFilter(Str_current_X, mask=[1, 1, 1, 1, 1, 1]), logfile=None)
            opt.run(fmax=0.05, steps=100000)
        E_current = sign * Str_current_X.get_potential_energy()
    else:
        Str_current.calc = Calculator
        if Optimization:
            opt = LBFGS(UnitCellFilter(Str_current, mask=[1, 1, 1, 1, 1, 1]), logfile=None)
            opt.run(fmax=0.05, steps=100000)
        E_current = sign * Str_current.get_potential_energy()
    Str_best, E_best = Str_current.copy(), E_current
    Str_worst, E_worst = Str_current.copy(), E_current

    while step <= N_steps:

        if CoolingType == 'Linear':
            T_current = T_start + (T_stop - T_start) * step / (N_steps)
        elif CoolingType == 'Exponential':
            T_current = T_start + (T_stop - T_start) * np.log(step + 1) / np.log(N_steps + 1)

        if step % PrintFrequency == 0:
            with open(LogFile, 'a') as f:
                print(f"    -- step: {step}/{N_steps} ({accepted_trials} accepted)  T: {T_current:.3f}"
                      f"  E_current: {E_current:.6f}  E_best: {E_best:.6f}  E_worst: {E_worst:.6f}", file=f)

        Str_candidate = Str_current.copy()
        index_1, index_2 = np.random.choice(indices, size=2, replace=False)
        while (Str_candidate.symbols[index_1] == Str_candidate.symbols[index_2]):
            index_2 = np.random.choice(indices)
        Str_candidate.positions[[index_1, index_2]] = Str_candidate.positions[[index_2, index_1]]

        if VacanciesExist:
            Str_candidate_X = Str_candidate.copy()
            del Str_candidate_X[[atom.index for atom in Str_candidate_X if atom.symbol=='X']]
            Str_candidate_X.calc = Calculator
            if Optimization:
                opt = LBFGS(UnitCellFilter(Str_candidate_X, mask=[1, 1, 1, 1, 1, 1]), logfile=None)
                opt.run(fmax=0.05, steps=100000)
            E_candidate = sign * Str_candidate_X.get_potential_energy()
        else:
            Str_candidate.calc = Calculator
            if Optimization:
                opt = LBFGS(UnitCellFilter(Str_candidate, mask=[1, 1, 1, 1, 1, 1]), logfile=None)
                opt.run(fmax=0.05, steps=100000)
            E_candidate = sign * Str_candidate.get_potential_energy()

        if E_candidate < E_best:
            Str_best, E_best = Str_candidate.copy(), E_candidate

        if E_candidate > E_worst:
            Str_worst, E_worst = Str_candidate.copy(), E_candidate

        kB = 8.617333262e-5    # Boltzmann constant in eV·K^-1
        Probability = np.exp(- (E_candidate - E_current) / (kB * T_current))
        if (E_candidate - E_current) < 0.0 or np.random.rand() < Probability:
            Str_current, E_current = Str_candidate.copy(), E_candidate
            accepted_trials += 1

        step += 1

    return Str_best, Str_worst