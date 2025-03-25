import math 
import pandas as pd
import matplotlib.pyplot as plt

# Define file paths and structure names
file_paths = {
    "PPP1": "C:/Users/mw/University of Cambridge/James Elliott - Myles Ward/gaussview/c_results/a_para_para_para/a_para_para_para_results_1_amalgamated.txt",
    "PPP2": "C:/Users/mw/University of Cambridge/James Elliott - Myles Ward/gaussview/c_results/a_para_para_para/b_para_para_para_results_2.txt",
    "PPM1": "C:/Users/mw/University of Cambridge/James Elliott - Myles Ward/gaussview/c_results/b_para_para_meta/a_para_para_meta_reults_1_(amalgamated).txt",
    "PPM2": "C:/Users/mw/University of Cambridge/James Elliott - Myles Ward/gaussview/c_results/b_para_para_meta/para_para_meta_results_2_amalgamated.txt",
    "PMP1": "C:/Users/mw/University of Cambridge/James Elliott - Myles Ward/gaussview/c_results/c_para_meta_para/a_para_meta_para_results_1.txt",
    "PMP2": "C:/Users/mw/University of Cambridge/James Elliott - Myles Ward/gaussview/c_results/c_para_meta_para/b_para_meta_para_results_2_amalgamated.txt",
    "PMM1": "C:/Users/mw/University of Cambridge/James Elliott - Myles Ward/gaussview/c_results/d_para_meta_meta/para_meta_meta_1_amalgamated.txt",
    "PMM2": "C:/Users/mw/University of Cambridge/James Elliott - Myles Ward/gaussview/c_results/d_para_meta_meta/b_pmm_2_amalgamated.txt",
}

HARTREE_TO_KJ_MOL = 2625.5

# Read data into a dictionary
data = {}
for key, path in file_paths.items():
    try:
        df = pd.read_csv(path, delim_whitespace=True, skiprows=4, names=["X", "Y"])
        if not df.empty:
            df["X"] = (df.index) * 10  # Reset X-axis: first point is 0, increments by 10 degrees
            df["Y"] = df["Y"] * HARTREE_TO_KJ_MOL  # Convert Hartree to kJ/mol
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



# Place the legend outside the plot to the right


plt.xlabel("Rotation (degrees)")
plt.ylabel("Conformation Energy (kJ/mol)")
plt.title("Conformation Energies, Pemion Para-Dimers")
plt.legend(loc='upper left', bbox_to_anchor=(1, 0.75), fontsize=12)
plt.grid(True)
plt.show()
