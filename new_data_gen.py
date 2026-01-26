import numpy as np
import pandas as pd

# =============================
# DATASET SIZE
# =============================
N = 10000
data = []

# =============================
# FLUID PROPERTY DICTIONARY
# =============================
fluid_properties = {
    "water":  {"rho": 997.0,  "mu": 0.001,   "cp": 4180.0, "k": 0.60},
    "oil":    {"rho": 850.0,  "mu": 0.015,   "cp": 2100.0, "k": 0.13},
    "glycol": {"rho": 1110.0, "mu": 0.004,   "cp": 2400.0, "k": 0.25},
    "steam":  {"rho": 0.6,    "mu": 1.3e-5,  "cp": 2010.0, "k": 0.03}
}

# =============================
# TEMPERATURE & FLOW RANGES
# =============================
temp_ranges = {
    "water":  (300, 370),
    "oil":    (330, 550),
    "glycol": (280, 390),
    "steam":  (380, 520)
}

flow_ranges = {
    "water":  (0.5, 5.0),
    "oil":    (0.3, 3.0),
    "glycol": (0.4, 4.0),
    "steam":  (0.05, 1.0)
}

# =============================
# DATA GENERATION LOOP
# =============================
while len(data) < N:

    # -----------------------------
    # FLUID SELECTION
    # -----------------------------
    tube_fluid = np.random.choice(list(fluid_properties.keys()))
    shell_fluid = np.random.choice(list(fluid_properties.keys()))

    tf = fluid_properties[tube_fluid]
    sf = fluid_properties[shell_fluid]

    rho_t, mu_t, cp_t, k_tfluid = tf["rho"], tf["mu"], tf["cp"], tf["k"]
    rho_s, cp_s, k_sfluid = sf["rho"], sf["cp"], sf["k"]

    Pr_t = cp_t * mu_t / k_tfluid

    # -----------------------------
    # GEOMETRY
    # -----------------------------
    shell_d = np.random.uniform(0.50, 0.60)

    tube_do = np.random.uniform(0.022, 0.028)
    wall_t = np.random.uniform(0.0015, 0.0035)
    tube_di = tube_do - 2 * wall_t
    if tube_di <= 0:
        continue

    tube_L = np.random.uniform(2.5, 3.5)
    n_tubes = np.random.randint(18, 27)
    tube_pitch = np.random.uniform(1.25 * tube_do, 1.5 * tube_do)

    baffle_spacing = np.random.uniform(0.20 * shell_d, 0.45 * shell_d)
    if baffle_spacing >= tube_L:
        continue

    n_baffles = int(tube_L / baffle_spacing) - 1
    if n_baffles < 2:
        continue

    # -----------------------------
    # MATERIAL
    # -----------------------------
    k_tube = np.random.choice([16.0, 205.0, 385.0])  # Steel, Al, Cu

    # -----------------------------
    # OPERATING CONDITIONS
    # -----------------------------
    T_t_in = np.random.uniform(*temp_ranges[tube_fluid])
    T_s_in = np.random.uniform(*temp_ranges[shell_fluid])

    if T_t_in <= T_s_in:
        T_t_in, T_s_in = T_s_in, T_t_in
    
    m_dot_t = np.random.uniform(*flow_ranges[tube_fluid])
    m_dot_s = np.random.uniform(*flow_ranges[shell_fluid])

    DT = np.random.uniform(5.0, 40.0)
    # DT = np.random.uniform(10, 50) * (1 / (1 + m_dot_t)) # new
    # DT = np.clip(DT, 5, 40) #new
    T_t_out = T_t_in - DT

    

    # -----------------------------
    # TUBE-SIDE HEAT TRANSFER
    # -----------------------------
    flow_area = n_tubes * (np.pi * tube_di**2 / 4)
    velocity = m_dot_t / (rho_t * flow_area)

    Re = rho_t * velocity * tube_di / mu_t

    if Re < 2300:
        Nu = 3.66
    elif Re >= 3000:
        Nu = 0.023 * Re**0.8 * Pr_t**0.4
    else:
        Nu = 3.66

    h_tube = Nu * k_tfluid / tube_di

    # -----------------------------
    # SHELL-SIDE HEAT TRANSFER
    # -----------------------------
    h_shell = (
        220.0
        * (shell_d / 0.55)
        * (0.25 / baffle_spacing)**0.6
        * (k_sfluid / 0.6)
    )

    # -----------------------------
    # OVERALL U
    # -----------------------------
    R_wall = (tube_do * np.log(tube_do / tube_di)) / (2 * k_tube)
    U = 1.0 / ((1 / h_tube) + R_wall + (1 / h_shell))
    U = min(U, 650.0)

    # -----------------------------
    # ENERGY BALANCE
    # -----------------------------
    Q = m_dot_t * cp_t * (T_t_in - T_t_out)
    T_s_out = T_s_in + Q / (m_dot_s * cp_s)

    if T_s_out >= T_t_out:
        continue

    # -----------------------------
    # LMTD (COUNTER-FLOW)
    # -----------------------------
    DT1 = T_t_in - T_s_out
    DT2 = T_t_out - T_s_in
    if DT1 <= 0 or DT2 <= 0:
        continue

    if abs(DT1 - DT2) < 1e-6:
        LMTD = DT1
    else:
        LMTD = (DT1 - DT2) / np.log(DT1 / DT2)

    A = np.pi * tube_do * tube_L * n_tubes
    Q_lmtd = U * A * LMTD

    # Add realism noise
    Q_lmtd *= np.random.normal(1.0, 0.03)

    q_flux = Q_lmtd / A

    # -----------------------------
    # STORE ROW
    # -----------------------------
    data.append([
        tube_fluid, shell_fluid,
        shell_d, tube_do, tube_di, tube_L, n_tubes, tube_pitch,
        baffle_spacing, n_baffles,
        m_dot_t, m_dot_s,
        T_t_in, T_t_out,
        T_s_in, T_s_out,
        k_tube,
        Q_lmtd, q_flux
    ])

# =============================
# DATAFRAME
# =============================
columns = [
    "tube_fluid", "shell_fluid",
    "shell_diameter", "tube_outer_diameter", "tube_inner_diameter",
    "tube_length", "number_of_tubes", "tube_pitch",
    "baffle_spacing", "number_of_baffles",
    "tube_mass_flow_rate", "shell_mass_flow_rate",
    "tube_inlet_temp", "tube_outlet_temp",
    "shell_inlet_temp", "shell_outlet_temp",
    "tube_thermal_conductivity",
    "heat_transfer_rate", "heat_flux"
]

df = pd.DataFrame(data, columns=columns)
df.to_csv("final_dual_fluid_shell_tube_dataset.csv", index=False)

print("FINAL DATASET SHAPE:", df.shape)