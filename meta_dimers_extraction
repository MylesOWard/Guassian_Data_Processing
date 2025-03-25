import math 
import pandas as pd
import matplotlib.pyplot as plt

# Define file paths and structure names
file_paths = {
    "MPP1": "C:/Users/mw/University of Cambridge/James Elliott - Myles Ward/gaussview/c_results/e_meta_para_para/a_meta_para_para_results_1.txt",
    "MPP2": "C:/Users/mw/University of Cambridge/James Elliott - Myles Ward/gaussview/c_results/e_meta_para_para/b_meta_para_para_results_2_amalgamated.txt",
    "MPM1": "C:/Users/mw/University of Cambridge/James Elliott - Myles Ward/gaussview/c_results/f_meta_para_meta/a_meta_para_meta_results_1.txt",
    "MPM2": "C:/Users/mw/University of Cambridge/James Elliott - Myles Ward/gaussview/c_results/f_meta_para_meta/b_meta_para_meta_results_2.txt",
    "MMP1": "C:/Users/mw/University of Cambridge/James Elliott - Myles Ward/gaussview/c_results/g_meta_meta_para/meta_meta_para_1_amalgamated.txt",
    "MMP2": "C:/Users/mw/University of Cambridge/James Elliott - Myles Ward/gaussview/c_results/g_meta_meta_para/meta_meta_para_2_amalgamated.txt",
    "MMM1": "C:/Users/mw/University of Cambridge/James Elliott - Myles Ward/gaussview/c_results/h_meta_meta_meta/a_meta_meta_meta_results_1.txt",
    "MMM2": "C:/Users/mw/University of Cambridge/James Elliott - Myles Ward/gaussview/c_results/h_meta_meta_meta/b_mmm_2_amalgamated.txt",
}

HARTREE_TO_KJ_MOL = 2625.5

# Read data into a dictionary
data = {}
for key, path in file_paths.items():
    try:
        df = pd.read_csv(path, delim_whitespace=True, skiprows=4, names=["X", "Y"])
        if not df.empty:
            df["X"] = (df.index) * 10  #first point is set to 0, increments are by 10 degrees
            df["Y"] = df["Y"] * HARTREE_TO_KJ_MOL  #Hartree to kj/mol
        data[key] = df
    except Exception as e:
        print(f"Error reading {key}: {e}")

# Find the global minimum Y value across all datasets
global_min_y = min(df["Y"].min() for df in data.values() if not df.empty)

# Plot all datasets with Y values shifted so the lowest value is 0
plt.figure(figsize=(10, 6))
for key, df in data.items():
    if not df.empty:  # Ensure we only plot non-empty data
        plt.plot(df["X"], df["Y"] - global_min_y, label=key)

plt.xlabel("Rotation (degrees)")
plt.ylabel("Conformation Energy (kJ/mol)")
plt.title("Conformation Energies, Pemion Meta-Dimers")
plt.legend(loc='upper left', bbox_to_anchor=(1, 0.75), fontsize=12)
plt.grid(True)
plt.show()
