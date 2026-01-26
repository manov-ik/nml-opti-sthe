import pandas as pd
import numpy as np

# =============================
# LOAD DATASET
# =============================
df = pd.read_csv("final_dual_fluid_shell_tube_dataset.csv")
print("Dataset shape:", df.shape)

# =============================
# FLUID PROPERTY DICTIONARY
# =============================
fluid_props = {
    "water":  {"cp": 4180.0},
    "oil":    {"cp": 2100.0},
    "glycol": {"cp": 2400.0},
    "steam":  {"cp": 2010.0}
}

# =============================
# ENERGY BALANCE VALIDATION
# =============================
Q_tube = []
Q_shell = []

for _, row in df.iterrows():
    cp_t = fluid_props[row["tube_fluid"]]["cp"]
    cp_s = fluid_props[row["shell_fluid"]]["cp"]

    Qt = (
        row["tube_mass_flow_rate"]
        * cp_t
        * (row["tube_inlet_temp"] - row["tube_outlet_temp"])
    )

    Qs = (
        row["shell_mass_flow_rate"]
        * cp_s
        * (row["shell_outlet_temp"] - row["shell_inlet_temp"])
    )

    Q_tube.append(Qt)
    Q_shell.append(Qs)

df["Q_tube_calc"] = Q_tube
df["Q_shell_calc"] = Q_shell

# Percentage difference
df["energy_error_pct"] = (
    abs(df["Q_tube_calc"] - df["Q_shell_calc"])
    / df["Q_tube_calc"]
    * 100
)

print("\n--- ENERGY BALANCE CHECK ---")
print("Mean % error:", df["energy_error_pct"].mean())
print("Max  % error:", df["energy_error_pct"].max())

# =============================
# LMTD-BASED VALIDATION
# =============================
A = (
    np.pi
    * df["tube_outer_diameter"]
    * df["tube_length"]
    * df["number_of_tubes"]
)

DT1 = df["tube_inlet_temp"] - df["shell_outlet_temp"]
DT2 = df["tube_outlet_temp"] - df["shell_inlet_temp"]

# Avoid log singularities
mask = (DT1 > 0) & (DT2 > 0) & (abs(DT1 - DT2) > 1e-6)

LMTD = np.zeros(len(df))
LMTD[mask] = (DT1[mask] - DT2[mask]) / np.log(DT1[mask] / DT2[mask])
LMTD[~mask] = DT1[~mask]

Q_lmtd_calc = df["heat_flux"] * A  # since q'' = Q / A

df["lmtd_error_pct"] = (
    abs(Q_lmtd_calc - df["heat_transfer_rate"])
    / df["heat_transfer_rate"]
    * 100
)

print("\n--- LMTD CONSISTENCY CHECK ---")
print("Mean % error:", df["lmtd_error_pct"].mean())
print("Max  % error:", df["lmtd_error_pct"].max())


"""
Output:
Dataset shape: (10000, 19)

--- ENERGY BALANCE CHECK ---
Mean % error: 5.731419094933186e-13
Max  % error: 3.7502319140006795e-11

--- LMTD CONSISTENCY CHECK ---
Mean % error: 1.9339096419314019e-13
Max  % error: 4.680649543595265e-13
"""

'''
new 
Dataset shape: (10000, 19)

--- ENERGY BALANCE CHECK ---
Mean % error: 4.669726264807962e-13
Max  % error: 3.000131744427258e-11

--- LMTD CONSISTENCY CHECK ---
Mean % error: 1.933267354683774e-13
Max  % error: 4.641153144094196e-13
'''