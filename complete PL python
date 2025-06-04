import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import matplotlib
from matplotlib import pyplot as plt

# add plotting later

# --- Settings ---
excel_path = "C:/Users/mw/OneDrive/Documents/b_Cambridge/b_Cam_term_2/e_persistance_results/a_FC_1.1_50x25_monomers.xlsx"  # Change this to your actual Excel file path
delta_start = 2  # Skip delta=1 since it's always 1

# --- Exponential decay model ---
def exp_decay(x, Lp):
    return np.exp(-x / Lp)

# --- Load Excel data ---
df = pd.read_excel(excel_path)

# --- Initialize storage for persistence lengths ---
Lp_values = []

# --- Loop over each chain column (excluding delta) ---
for column in df.columns:
    if "chain" in column.lower():
        delta = df["delta"].values
        cos_theta = df[column].values

        # Remove delta=1 (autocorrelation = 1) for fitting
        delta = delta[delta >= delta_start]
        cos_theta = cos_theta[delta_start - 1:]

        # Use only positive values for stable fitting
        positive_indices = cos_theta > 0
        delta_fit = delta[positive_indices]
        cos_theta_fit = cos_theta[positive_indices]

        if len(delta_fit) >= 3:  # Require at least a few points for reliable fit
            try:
                popt, _ = curve_fit(exp_decay, delta_fit, cos_theta_fit, bounds=(0.1, 1000))
                Lp = popt[0]
                Lp_values.append(Lp)
            except RuntimeError:
                print(f"Fit failed for {column}")

# --- Output results ---
Lp_mean = np.mean(Lp_values) if Lp_values else None
Lp_std = np.std(Lp_values) if Lp_values else None

print(f"Persistence Lengths: {Lp_values}")
print(f"Average Lp: {Lp_mean:.4f} ± {Lp_std:.4f}" if Lp_mean else "No valid persistence lengths.")
