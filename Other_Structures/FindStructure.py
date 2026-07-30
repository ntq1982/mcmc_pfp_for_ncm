'''
    Find structure from given bond fractions using NCM database generated during the sampling process.
    Due to the huge size of the database, please contact me if you want to run this script.

    Created on Oct 02, 2023 at RISM (Shinshu University)
    Last update: Jul 30, 2026 15:19 JST

    Copyright © 2022-2026 Quang Nguyen. All rights reserved.
'''

import numpy as np
import pandas as pd
from ase.io import read, write

# Place to extract data
csv_path = 'NCM_Database'
compound = 'NCM523'
bond     = 'AA'
ref_path = '../SA_Structures'

# Check header of csv
data = pd.read_csv(csv_path+'/'+compound+'_'+bond+'.csv', nrows=0)
column_names = data.columns.tolist()
print("Column names:", column_names)

# Target bond fractions (For NCM523, NCM622, and NCM811, respectively)
temp     = '50K'
P_target = {'σ_AA' : 0.18451488532719165, 'σ_AB' : 0.15030403342323956, 'σ_AC' : 0.46520724552790094,
            'σ_BB' : 0.07370039414461348, 'σ_BC' : 0.09896018245740157, 'σ_CC' : 0.02089758725799434}
# temp = '600K'
# P_target = {'σ_AA' : 0.18752896645887180, 'σ_AB' : 0.17743146199387440, 'σ_AC' : 0.44777841857910480,
#             'σ_BB' : 0.05730899166615489, 'σ_BC' : 0.10848158115556569, 'σ_CC' : 0.02207401860591004}
# temp = '50K'
# P_target = {'σ_AA' : 0.33343006452791330, 'σ_AB' : 0.15450829892933043, 'σ_AC' : 0.38011360549594636,
#             'σ_BB' : 0.11014780308995840, 'σ_BC' : 0.02482909147186453, 'σ_CC' : 0.00055260754923995}
# temp = '600K'
# P_target = {'σ_AA' : 0.31506937761144005, 'σ_AB' : 0.23261026534078720, 'σ_AC' : 0.33675890765972170,
#             'σ_BB' : 0.05491293851999015, 'σ_BC' : 0.05990115692389664, 'σ_CC' : 0.00238499006859354}
# temp = '50K'
# P_target = {'σ_AA' : 0.61148463077397130, 'σ_AB' : 0.17651807814195516, 'σ_AC' : 0.20338985947079594,
#             'σ_BB' : 0.01350306728595789, 'σ_BC' : 0.00000010978568277, 'σ_CC' : 0.00000000026326223}
# temp = '600K'
# P_target = {'σ_AA' : 0.62123208226955510, 'σ_AB' : 0.16886274702465148, 'σ_AC' : 0.18843513836640624,
#             'σ_BB' : 0.01018366051007998, 'σ_BC' : 0.01206235960522070, 'σ_CC' : 0.00022974358236846}

# Initialize variables to track the minimum difference and corresponding row
min_diff = float('inf')
best_row = None
best_structure = None

# Retrieve the symbols of structure with the minimum total difference
chunk_size = 10**5
for chunk in pd.read_csv(csv_path+'/'+compound+'_'+bond+'.csv', chunksize=chunk_size):
    chunk['Total_Difference'] = (np.abs(chunk['σ_AA'] - P_target['σ_AA']) +
                                 np.abs(chunk['σ_AB'] - P_target['σ_AB']) +
                                 np.abs(chunk['σ_AC'] - P_target['σ_AC']) +
                                 np.abs(chunk['σ_BB'] - P_target['σ_BB']) +
                                 np.abs(chunk['σ_BC'] - P_target['σ_BC']) +
                                 np.abs(chunk['σ_CC'] - P_target['σ_CC']) )
    local_min_idx  = chunk['Total_Difference'].idxmin()
    local_min_diff = chunk.loc[local_min_idx, 'Total_Difference']
    if local_min_diff < min_diff:
        min_diff = local_min_diff
        best_row = chunk.loc[local_min_idx]
        best_structure = best_row['TM_Arrangement']
symbol_TMs = best_structure
P_values   = best_row[['σ_AA', 'σ_AB', 'σ_AC', 'σ_BB', 'σ_BC', 'σ_CC']]
print("Best matching structure:", symbol_TMs)
print("Bond fractions:")
for key, value in P_values.items():
    print(f"  {key:6}: {value: .6f}")

# Save the strucuture to CIF file
SPC     = read(ref_path+'/'+compound+'_5x4x1_SAunopt_Most.cif')
idx_TMs = [atom.index for atom in SPC if atom.symbol == 'Ni' or atom.symbol == 'Co' or atom.symbol == 'Mn']
SPC.symbols[idx_TMs] = symbol_TMs
write(compound+'_5x4x1_'+temp+'.cif', SPC, format='cif')
