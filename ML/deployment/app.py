import gradio as gr
import pandas as pd
import numpy as np
import joblib

# ── Load model artifacts ──────────────────────────────────────────────────────
model       = joblib.load("random_forest_heat_exchanger_model.pkl")
preprocessor = joblib.load("preprocessor_dual_fluid.pkl")

# ── Training-derived ranges for optimization ──────────────────────────────────
RANGES = {
    "shell_diameter":        (0.52, 0.58),
    "tube_outer_diameter":   (0.023, 0.027),
    "tube_inner_diameter":   (0.019, 0.023),
    "tube_length":           (2.5, 3.5),
    "number_of_tubes":       (18, 25),
    "tube_pitch":            (0.030, 0.035),
    "baffle_spacing":        (0.15, 0.25),
    "number_of_baffles":     (8, 12),
    "tube_mass_flow_rate":   (0.5, 4.0),
    "shell_mass_flow_rate":  (0.5, 3.0),
    "tube_inlet_temp":       (400, 450),
    "tube_outlet_temp":      (360, 400),
    "shell_inlet_temp":      (300, 350),
    "shell_outlet_temp":     (350, 380),
    "tube_thermal_conductivity": [16, 205, 385],   # discrete material choices
}

INTEGER_KEYS = {"number_of_tubes", "number_of_baffles"}
DISCRETE_KEYS = {"tube_thermal_conductivity"}

# ── Helpers ───────────────────────────────────────────────────────────────────
def needs_opt(value) -> bool:
    """True if this parameter should be AI-sampled."""
    if value is None:
        return True
    if isinstance(value, (int, float)) and value < 0:
        return True
    return False

def sample_param(key):
    r = RANGES[key]
    if key in DISCRETE_KEYS:
        return int(np.random.choice(r))
    lo, hi = r
    if key in INTEGER_KEYS:
        return int(np.random.randint(lo, hi + 1))
    return float(np.random.uniform(lo, hi))

# ── Core function ─────────────────────────────────────────────────────────────
def process_request(
    tube_fluid, shell_fluid,
    target_Q, n_trials,
    # all optional — leave None / blank to let AI optimize
    shell_diameter       = None,
    tube_outer_diameter  = None,
    tube_inner_diameter  = None,
    tube_length          = None,
    number_of_tubes      = None,
    tube_pitch           = None,
    baffle_spacing       = None,
    number_of_baffles    = None,
    tube_mass_flow_rate  = None,
    shell_mass_flow_rate = None,
    tube_inlet_temp      = None,
    tube_outlet_temp     = None,
    shell_inlet_temp     = None,
    shell_outlet_temp    = None,
    tube_thermal_conductivity = None,
):
    numeric_inputs = {
        "shell_diameter":           shell_diameter,
        "tube_outer_diameter":      tube_outer_diameter,
        "tube_inner_diameter":      tube_inner_diameter,
        "tube_length":              tube_length,
        "number_of_tubes":          number_of_tubes,
        "tube_pitch":               tube_pitch,
        "baffle_spacing":           baffle_spacing,
        "number_of_baffles":        number_of_baffles,
        "tube_mass_flow_rate":      tube_mass_flow_rate,
        "shell_mass_flow_rate":     shell_mass_flow_rate,
        "tube_inlet_temp":          tube_inlet_temp,
        "tube_outlet_temp":         tube_outlet_temp,
        "shell_inlet_temp":         shell_inlet_temp,
        "shell_outlet_temp":        shell_outlet_temp,
        "tube_thermal_conductivity": tube_thermal_conductivity,
    }

    to_optimize = [k for k, v in numeric_inputs.items() if needs_opt(v)]
    user_set    = {k: v for k, v in numeric_inputs.items() if not needs_opt(v)}

    target = float(target_Q or 0)
    n      = int(n_trials or 1000)

    candidates = []

    for _ in range(n):
        row = {"tube_fluid": tube_fluid, "shell_fluid": shell_fluid}
        for key in numeric_inputs:
            row[key] = sample_param(key) if key in to_optimize else user_set[key]

        X    = pd.DataFrame([row])
        Xp   = preprocessor.transform(X)
        Q    = float(np.expm1(model.predict(Xp))[0])

        if target <= 0 or Q >= target:
            candidates.append({**row, "Q_pred": Q})

    if not candidates:
        return {
            "status":  "no_match",
            "message": (
                f"No design reached {target:,.0f} W in {n} trials. "
                "Try lowering your target Q or increasing search trials."
            ),
            "candidates": [],
            "optimized_params": to_optimize,
            "user_params": list(user_set.keys()),
        }

    top3 = sorted(candidates, key=lambda x: x["Q_pred"], reverse=True)[:3]

    return {
        "status":           "ok",
        "candidates":       top3,
        "optimized_params": to_optimize,
        "user_params":      list(user_set.keys()),
    }

# ── Gradio interface ──────────────────────────────────────────────────────────
# NOTE: input ORDER here must match the data[] array sent from the frontend.
inputs = [
    # --- required ---
    gr.Dropdown(["water", "oil"], label="Tube Fluid", value="water"),
    gr.Dropdown(["water", "oil"], label="Shell Fluid", value="oil"),
    gr.Number(label="Target Q (W) — set 0 to maximize", value=0),
    gr.Slider(minimum=500, maximum=5000, step=500, label="Search Trials", value=1000),
    # --- optional geometry ---
    gr.Number(label="Shell Diameter (m)  [opt: 0.52–0.58]",       value=None),
    gr.Number(label="Tube Outer Ø (m)    [opt: 0.023–0.027]",     value=None),
    gr.Number(label="Tube Inner Ø (m)    [opt: 0.019–0.023]",     value=None),
    gr.Number(label="Tube Length (m)     [opt: 2.5–3.5]",         value=None),
    gr.Number(label="No. of Tubes        [opt: 18–25]",            value=None),
    gr.Number(label="Tube Pitch (m)      [opt: 0.030–0.035]",     value=None),
    gr.Number(label="Baffle Spacing (m)  [opt: 0.15–0.25]",       value=None),
    gr.Number(label="No. of Baffles      [opt: 8–12]",             value=None),
    # --- optional flow ---
    gr.Number(label="Tube Mass Flow (kg/s) [opt: 0.5–4.0]",       value=None),
    gr.Number(label="Shell Mass Flow (kg/s)[opt: 0.5–3.0]",       value=None),
    # --- optional temperature ---
    gr.Number(label="Tube Inlet Temp (K)  [opt: 400–450]",        value=None),
    gr.Number(label="Tube Outlet Temp (K) [opt: 360–400]",        value=None),
    gr.Number(label="Shell Inlet Temp (K) [opt: 300–350]",        value=None),
    gr.Number(label="Shell Outlet Temp (K)[opt: 350–380]",        value=None),
    # --- optional material ---
    gr.Number(label="Thermal Conductivity (W/mK) — 16/205/385", value=None),
]

output = gr.JSON(label="Optimization Result")

demo = gr.Interface(
    fn=process_request,
    inputs=inputs,
    outputs=output,
    title="HEXP.io — Heat Exchanger Optimizer",
    description=(
        "Fill in the parameters you know. Leave the rest blank — "
        "the AI will sample from its training ranges and find the best design."
    ),
)

if __name__ == "__main__":
    demo.launch()